"""lightsmith backend that runs the ADDA executable (GPL-3.0-or-later) as a subprocess.

One ADDA run per wavelength. The scene is voxelised here and passed with
``-shape read`` in ADDA's multi-domain format, so layered spheres and clusters of
different materials work. A host medium is handled as the ADDA manual (§4.2)
prescribes: relative refractive indices m / n_medium and the wavelength in the
medium, lambda / n_medium. All lengths are given in nm, so ADDA's cross sections
come back in nm² (ADDA treats the length unit as arbitrary; the manual says µm).

Conventions: ADDA uses exp(-i omega t), so absorbing materials have Im(m) > 0
(manual §6.3, footnote 34), the same as lightsmith; no conjugation is needed.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
from lightsmith import Result, Simulation
from lightsmith.backends import Backend, unsupported_outputs
from lightsmith.device import Device
from pydantic import BaseModel, ConfigDict, Field

from .geometry import Voxels, default_dipoles_per_diameter, incident_frame, voxelise, write_shape_file
from .output import avg_params_text, parse_version, read_cross_sections, read_log

ENV_VAR = "LIGHTSMITH_ADDA"
EXECUTABLES = ("adda", "adda_ocl", "adda_mpi")
MAX_POINTS = 20_000  # dipole positions listed by discretization(), as lightsmith.discretize: subsampled beyond


class AddaOptions(BaseModel):
    """Options under ``Simulation.options["adda"]`` (strict). Shared keys of the
    method-level ``options["dda"]`` are mapped where ADDA has an equivalent."""

    model_config = ConfigDict(extra="forbid")

    dipoles_per_diameter: int | None = Field(
        None, ge=2, description="Lattice sites across the smallest particle's largest dimension. "
        "Default: max(16, enough for |m| k d <= 0.5), as for the native DDA.")
    volume_correction: bool = Field(True, description="Rescale the dipole size so the dipole volume equals the "
                                    "particle volume (ADDA does this for its own shapes, not for read shapes).")
    polarizability: str | None = Field(
        None, description="ADDA -pol argument, e.g. 'ldr', 'cldr', 'cm', 'rrc', 'fcd', 'igt_so', 'dgf', 'lak'. "
        "Default: ADDA's (ldr).")
    interaction: str | None = Field(None, description="ADDA -int argument, e.g. 'poi', 'fcd', 'igt'. Default: poi.")
    iterative_solver: Literal["bcgs2", "bicg", "bicgstab", "cgnr", "csym", "qmr", "qmr2"] | None = Field(
        None, description="ADDA -iter. Default: qmr.")
    tol: float | None = Field(None, gt=0, lt=1, description="Relative residual target (ADDA -eps -log10(tol)). "
                              "Default: ADDA's 1e-5.")
    max_iter: int | None = Field(None, ge=1, description="ADDA -maxiter.")
    max_dipoles: int = Field(5_000_000, ge=1, description="Refuse problems larger than this.")
    avg_beta_jmax: int = Field(4, ge=2, le=10, description="Romberg refinements for beta in orientation averaging.")
    avg_gamma_jmax: int = Field(3, ge=2, le=10, description="Romberg refinements for gamma in orientation averaging.")
    avg_eps: float = Field(0.0, ge=0, description="Romberg tolerance for orientation averaging (0: use Jmax).")
    executable: str | None = Field(None, description=f"Path to the ADDA executable (overrides ${ENV_VAR} and PATH).")
    timeout_s: float | None = Field(None, gt=0, description="Kill an ADDA run after this many seconds.")
    extra_args: list[str] = Field(default_factory=list, description="Extra ADDA command-line arguments, verbatim.")
    keep_files: bool = Field(False, description="Keep the working directory and report it in provenance.")


# Keys of the method-level options["dda"] (native DDA's DDAOptions) and how they map.
_SHARED = {"dipoles_per_diameter", "tol", "max_iter", "max_dipoles", "polarizability"}
_POL_MAP = {"ldr": "ldr", "cm_rr": "rrc"}  # native name -> ADDA -pol


def find_executable(explicit: str | None = None) -> str | None:
    """ADDA executable: ``explicit``, else $LIGHTSMITH_ADDA, else adda/adda_ocl/adda_mpi on PATH."""
    for cand in (explicit, os.environ.get(ENV_VAR)):
        if cand:
            path = shutil.which(cand) or (cand if Path(cand).is_file() else None)
            return path
    for name in EXECUTABLES:
        path = shutil.which(name)
        if path:
            return path
    return None


def _fmt(x: float) -> str:
    return repr(float(x))


class AddaBackend(Backend):
    name = "adda"
    method = "dda"
    license = "GPL-3.0-or-later"
    citation = ("M. A. Yurkin and A. G. Hoekstra, JQSRT 112, 2234 (2011), doi:10.1016/j.jqsrt.2011.01.031; "
                "ADDA, https://github.com/adda-team/adda")
    gpu = False  # the OpenCL build (adda_ocl) picks its own GPU; lightsmith's device is not used

    @classmethod
    def available(cls):
        if find_executable() is not None:
            return True, ""
        if os.environ.get(ENV_VAR):
            return False, f"{ENV_VAR}={os.environ[ENV_VAR]!r} is not an executable file"
        return False, (f"ADDA executable not found: put 'adda' on PATH or set {ENV_VAR} to its path "
                       "(binaries: https://github.com/adda-team/adda/releases)")

    @classmethod
    def supports(cls, sim: Simulation):
        why = unsupported_outputs(sim, orientation_average=True)
        return (not why), why

    @classmethod
    def options(cls, sim: Simulation) -> tuple[AddaOptions, list[str]]:
        """Validated options plus warnings about method-level keys ADDA cannot honour."""
        shared = dict(sim.options.get(cls.method, {}))
        warnings = []
        ignored = sorted(k for k in shared if k not in _SHARED)
        if ignored:
            warnings.append(f"ADDA ignores dda options {ignored}")
        mapped = {k: v for k, v in shared.items() if k in _SHARED}
        if "polarizability" in mapped:
            mapped["polarizability"] = _POL_MAP.get(mapped["polarizability"], mapped["polarizability"])
        return AddaOptions.model_validate({**mapped, **sim.options.get(cls.name, {})}), warnings

    @classmethod
    def _voxels(cls, sim: Simulation, opts: AddaOptions) -> Voxels:
        npd = opts.dipoles_per_diameter or default_dipoles_per_diameter(sim)
        frame = None if sim.orientation_average else incident_frame(*sim.illumination.unit_vectors())
        return voxelise(sim, npd, frame, opts.volume_correction)

    @classmethod
    def estimate(cls, sim, device):
        opts, _ = cls.options(sim)
        vox = cls._voxels(sim, opts)
        nx, ny, nz = vox.grid
        # ADDA's FFT grid is about twice the box per axis; ~ (2nx)(2ny)(2nz) complex doubles x a few arrays.
        mem = 16 * 8 * nx * ny * nz * 12
        return {"dipoles": vox.n_dip, "grid": list(vox.grid), "dipole_spacing_nm": vox.d,
                "dipoles_per_diameter": opts.dipoles_per_diameter or default_dipoles_per_diameter(sim),
                "domains": len(vox.materials), "memory_bytes": int(mem), "memory_note": "rough upper bound",
                "solves": len(sim.wavelength_array()), "fits": vox.n_dip <= opts.max_dipoles,
                "executable": find_executable(opts.executable)}

    @classmethod
    def discretization(cls, sim: Simulation, device: Device) -> dict:
        """The dipole lattice ADDA will be given (``lightsmith.discretization``): the voxelisation written to the
        shape file, at ADDA's dipole size, in lightsmith coordinates. Never runs ADDA."""
        opts, warnings = cls.options(sim)
        npd = opts.dipoles_per_diameter or default_dipoles_per_diameter(sim)
        vox = cls._voxels(sim, opts)
        pts = vox.points_nm()
        stride = max(1, math.ceil(len(pts) / MAX_POINTS))
        grid = list(vox.grid)
        payload = {"spacing_nm": vox.d, "dipoles_per_diameter": int(npd), "grid": grid, "n_dipoles": vox.n_dip,
                   "stride": stride, "points_nm": np.round(pts[::stride], 3).tolist(),
                   "material_slot": (vox.domain[::stride] - 1).tolist(), "discretization": "center",
                   "extrapolate": None, "domains": len(vox.materials), "volume_correction": opts.volume_correction,
                   "incident_frame": vox.frame.tolist()}
        summary = {"dipoles": vox.n_dip, "spacing (nm)": round(vox.d, 4), "dipoles per diameter": int(npd),
                   "lattice": " × ".join(map(str, grid)), "discretization": "center", "domains": len(vox.materials)}
        error = None
        if vox.n_dip > opts.max_dipoles:
            error = (f"{vox.n_dip} dipoles exceeds max_dipoles={opts.max_dipoles}; "
                     "lower dipoles_per_diameter or raise max_dipoles")
        return {"layout_kind": "dipole_lattice", "summary": summary, "payload": payload, "warnings": warnings,
                "error": error}

    @classmethod
    def command(cls, exe: str, shape_file: str | Path, vox: Voxels, wavelength_nm: float, sim: Simulation,
                opts: AddaOptions, run_dir: str | Path, avg_file: str | Path | None = None,
                warnings: list[str] | None = None) -> list[str]:
        """ADDA command line for one wavelength (lengths in nm, scaled into the host medium)."""
        n0 = sim.medium_index
        args = [exe, "-shape", "read", str(shape_file), "-size", _fmt(vox.size_x_nm),
                "-lambda", _fmt(wavelength_nm / n0)]
        args.append("-m")
        for mat in vox.materials:
            m = complex(mat.n(np.array([wavelength_nm]))[0]) / n0
            if abs(m - 1) < 1e-12:
                # ADDA rejects m == 1 (manual §6.3); it suggests a value very close to 1 instead.
                m = complex(1.00001, m.imag)
                if warnings is not None:
                    warnings.append(f"relative index 1 at {wavelength_nm:g} nm replaced by 1.00001 (ADDA rejects 1)")
            args += [_fmt(m.real), _fmt(m.imag)]
        args += ["-dir", str(run_dir), "-scat_matr", "none"]
        if avg_file is not None:
            args += ["-orient", "avg", str(avg_file)]
        if opts.polarizability:
            args += ["-pol", *opts.polarizability.split()]
        if opts.interaction:
            args += ["-int", *opts.interaction.split()]
        if opts.iterative_solver:
            args += ["-iter", opts.iterative_solver]
        if opts.tol is not None:
            args += ["-eps", _fmt(-math.log10(opts.tol))]
        if opts.max_iter is not None:
            args += ["-maxiter", str(opts.max_iter)]
        return args + list(opts.extra_args)

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        t0 = time.perf_counter()
        opts, warnings = cls.options(sim)
        exe = find_executable(opts.executable)
        if exe is None:
            raise RuntimeError(cls.available()[1])
        vox = cls._voxels(sim, opts)
        if vox.n_dip > opts.max_dipoles:
            raise ValueError(f"{vox.n_dip} dipoles exceeds max_dipoles={opts.max_dipoles}; "
                             "lower dipoles_per_diameter or raise max_dipoles")
        if device.name != "cpu":
            warnings.append("ADDA runs as an external CPU (or its own OpenCL) process; device ignored")
        wl = sim.wavelength_array()
        workdir = Path(tempfile.mkdtemp(prefix="lightsmith-adda-"))
        try:
            shape_file = write_shape_file(workdir / "shape.geom", vox, comment=f"{len(vox.materials)} domain(s)")
            avg_file = None
            if sim.orientation_average:
                avg_file = workdir / "avg_params.dat"
                avg_file.write_text(avg_params_text(opts.avg_beta_jmax, opts.avg_gamma_jmax, opts.avg_eps),
                                    encoding="ascii")
            c_ext, c_abs, c_sca, commands, logs = [], [], [], [], []
            for i, lam in enumerate(wl):
                run_dir = workdir / f"run{i:03d}"
                cmd = cls.command(exe, shape_file, vox, lam, sim, opts, run_dir, avg_file, warnings)
                commands.append(subprocess.list2cmdline(cmd))
                proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=opts.timeout_s)
                if proc.returncode != 0:
                    tail = (proc.stderr.strip() or proc.stdout.strip())[-2000:]
                    raise RuntimeError(f"ADDA failed at {lam:g} nm (exit {proc.returncode}): {tail}\n"
                                       f"command: {commands[-1]}")
                cs = read_cross_sections(run_dir, sim.orientation_average)
                log = read_log(run_dir)
                logs.append(log)
                ext, ab = float(cs["Cext"]), float(cs.get("Cabs", 0.0))
                c_ext.append(ext)
                c_abs.append(ab)
                c_sca.append(float(cs["Csca"]) if "Csca" in cs else ext - ab)
                if log.get("dipoles") not in (None, vox.n_dip):
                    warnings.append(f"ADDA reports {log['dipoles']} dipoles, the shape file has {vox.n_dip}")
                if "dipole_size" in log and abs(log["dipole_size"] / vox.d - 1) > 1e-4:
                    warnings.append(f"ADDA used dipole size {log['dipole_size']:g} nm, expected {vox.d:g} nm "
                                    "(lengths are not rescaled for the medium, so these should match)")
        finally:
            if not opts.keep_files:
                shutil.rmtree(workdir, ignore_errors=True)

        version = next((g["version"] for g in logs if g.get("version")), None) or _version(exe)
        provenance = {
            "device": "cpu", "executable": exe, "adda_version": version, "commands": commands,
            "dipoles": vox.n_dip, "grid": list(vox.grid), "dipole_spacing_nm": vox.d, "domains": len(vox.materials),
            "volume_correction": opts.volume_correction,
            "incident_frame": vox.frame.tolist(), "polarization": "ADDA Y (CrossSec-Y)" if not
            sim.orientation_average else "orientation averaged (CrossSec)",
            "iterations": [g.get("iterations") for g in logs],
            "polarizability": next((g["polarizability"] for g in logs if g.get("polarizability")), None),
            "iterative_method": next((g["iterative_method"] for g in logs if g.get("iterative_method")), None),
            "runtime_s": time.perf_counter() - t0,
        }
        if opts.keep_files:
            provenance["workdir"] = str(workdir)
        return Result.from_arrays(
            method=cls.method, backend=cls.name, wavelengths_nm=wl, c_ext=c_ext, c_sca=c_sca, c_abs=c_abs,
            r_eq=sim.equivalent_radius_nm(), provenance=provenance, warnings=warnings)


def _version(exe: str) -> str | None:
    try:
        proc = subprocess.run([exe, "-V"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_version(proc.stdout + proc.stderr)
