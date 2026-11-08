"""Voxelise an lightsmith scene and write it as an ADDA geometry file.

ADDA geometry files (manual §B.5, ``-shape read``) list one occupied dipole per
line as integer lattice coordinates. The single-domain format is ``x y z``; the
multi-domain format starts with a ``Nmat=<K>`` line and appends the 1-based
domain number, ``x y z d``, which selects the refractive index given in the same
position of ``-m``. Lines starting with ``#`` before the data are comments.
ADDA places the smallest box around all dipoles, so coordinates may start
anywhere; the grid dimensions are ``max - min + 1`` along each axis.

The lattice is built in ADDA's particle frame. ``frame`` holds ADDA's x, y, z
axes (rows) expressed in lightsmith coordinates, so the incident wave can be
mapped onto ADDA's default propagation (+z) and Y polarization (+y).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lightsmith import Simulation


@dataclass
class Voxels:
    ijk: np.ndarray  # (N, 3) integer lattice coordinates of occupied dipoles
    domain: np.ndarray  # (N,) 1-based ADDA domain numbers
    materials: list  # Material per domain (index = domain - 1)
    d: float  # dipole size in nm
    frame: np.ndarray = field(default_factory=lambda: np.eye(3))
    centre_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))  # lattice centre, ADDA frame
    lattice: tuple[int, int, int] | None = None  # voxelised box (ijk index ranges), before trimming to dipoles

    @property
    def n_dip(self) -> int:
        return int(len(self.ijk))

    def points_nm(self) -> np.ndarray:
        """(N, 3) dipole centres in lightsmith coordinates, spaced by ADDA's dipole size ``d`` around the
        voxelised box's centre: the lattice ADDA simulates, placed where the particles are."""
        n = np.asarray(self.lattice if self.lattice is not None else self.ijk.max(axis=0) + 1, float)
        return (np.asarray(self.centre_nm, float) + (self.ijk - (n - 1) / 2) * self.d) @ self.frame

    @property
    def grid(self) -> tuple[int, int, int]:
        """ADDA's computational grid: the bounding box of the dipoles."""
        if not self.n_dip:
            return (0, 0, 0)
        span = self.ijk.max(axis=0) - self.ijk.min(axis=0) + 1
        return tuple(int(v) for v in span)

    @property
    def size_x_nm(self) -> float:
        """Grid extent along ADDA's x axis, the value passed to ``-size``."""
        return self.grid[0] * self.d


def incident_frame(direction, polarization) -> np.ndarray:
    """Rows: ADDA x, y, z axes in lightsmith coordinates, with z = k and y = E.

    ADDA's default incident wave travels along +z with its Y polarization along +y
    (manual §9.3 and the log's "Incident polarization Y(par): (0,1,0)"), so the
    result for our polarization is ADDA's CrossSec-Y, which is always computed."""
    k = np.asarray(direction, float)
    e = np.asarray(polarization, float)
    k, e = k / np.linalg.norm(k), e / np.linalg.norm(e)
    return np.stack([np.cross(e, k), e, k])


def default_dipoles_per_diameter(sim: Simulation) -> int:
    """Same rule as the native DDA: max(16, enough for |m| k d <= 0.5)."""
    wl = sim.wavelength_array()
    k_max = 2 * np.pi * sim.medium_index / wl.min()
    m_max = max(np.abs(m.n(wl) / sim.medium_index).max() for p in sim.particles for m in p.region_materials())
    diam = min(2 * p.shape.bounding_half_extent().max() for p in sim.particles)
    return max(16, math.ceil(diam * m_max * k_max / 0.5))


def _slots(sim: Simulation, pts: np.ndarray) -> tuple[np.ndarray, list]:
    """Material slot per point (-1 = outside) and the material of each slot.

    The first particle listed wins overlaps, as in the native DDA."""
    slot = np.full(pts.shape[:-1], -1, dtype=int)
    materials: list = []
    for p in sim.particles:
        region = p.region(pts - np.asarray(p.position_nm))
        mats = p.region_materials()
        for r in range(len(mats)):
            slot[(region == r) & (slot < 0)] = len(materials) + r
        materials.extend(mats)
    return slot, materials


def voxelise(sim: Simulation, dipoles_per_diameter: int, frame: np.ndarray | None = None,
             volume_correction: bool = True) -> Voxels:
    """Centre-in-shape voxelisation of all particles (and shells) on one cubic lattice.

    The lattice spacing is the smallest particle's largest dimension divided by
    ``dipoles_per_diameter``. With ``volume_correction`` the dipole size is then
    rescaled so the dipole volume equals the true particle volume, which is what
    ADDA itself does for its built-in shapes (manual §6.3) but not for read shapes.
    Identical materials share one ADDA domain."""
    frame = np.eye(3) if frame is None else np.asarray(frame, float)
    diam = min(2 * p.shape.bounding_half_extent().max() for p in sim.particles)
    d0 = diam / dipoles_per_diameter
    # Bounding boxes in the ADDA frame (a rotated box is enclosed by |R| @ half-extent).
    centres = [frame @ np.asarray(p.position_nm, float) for p in sim.particles]
    halves = [np.abs(frame) @ p.shape.bounding_half_extent() for p in sim.particles]
    lo = np.min([c - h for c, h in zip(centres, halves)], axis=0)
    hi = np.max([c + h for c, h in zip(centres, halves)], axis=0)
    n = np.maximum(np.round((hi - lo) / d0).astype(int), 1)
    centre = (lo + hi) / 2
    axes = [centre[i] + (np.arange(n[i]) - (n[i] - 1) / 2) * d0 for i in range(3)]
    pts_adda = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    pts = pts_adda @ frame  # back to lightsmith coordinates (frame is orthonormal)

    slot, slot_materials = _slots(sim, pts)
    occupied = slot >= 0
    n_dip = int(occupied.sum())
    if n_dip == 0:
        raise ValueError("no dipoles inside the particles; increase dipoles_per_diameter")

    # Merge identical materials into one domain, keep only domains that have dipoles.
    domains: list = []
    slot_to_domain = np.zeros(len(slot_materials), dtype=int)
    for s, mat in enumerate(slot_materials):
        if not np.any(slot == s):
            continue
        for i, known in enumerate(domains):
            if known == mat:
                slot_to_domain[s] = i + 1
                break
        else:
            domains.append(mat)
            slot_to_domain[s] = len(domains)

    ijk = np.argwhere(occupied)
    domain = slot_to_domain[slot[occupied]]
    d = (sum(p.shape.volume() for p in sim.particles) / n_dip) ** (1 / 3) if volume_correction else d0
    return Voxels(ijk=ijk, domain=domain, materials=domains, d=float(d), frame=frame, centre_nm=centre,
                  lattice=tuple(int(v) for v in n))


def write_shape_file(path: str | Path, vox: Voxels, comment: str = "") -> Path:
    """Write ``vox`` in ADDA's text format (single-domain) or text_ext format (multi-domain)."""
    path = Path(path)
    multi = len(vox.materials) > 1
    lines = ["# ADDA geometry written by lightsmith-adda"]
    lines += [f"# {c}" for c in comment.splitlines() if c.strip()]
    lines.append(f"# {vox.n_dip} dipoles, grid {'x'.join(map(str, vox.grid))}, dipole size {vox.d!r} nm")
    if multi:
        lines.append(f"Nmat={len(vox.materials)}")
        rows = np.column_stack([vox.ijk, vox.domain])
        fmt = "%d %d %d %d"
    else:
        rows = vox.ijk
        fmt = "%d %d %d"
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
        np.savetxt(f, rows, fmt=fmt)
    return path


def read_shape_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Parse an ADDA text/text_ext geometry file: (ijk, domain, Nmat or None).

    Mirrors ADDA's rules: leading ``#`` comments, optional ``Nmat=`` line, then
    integer rows; single-domain rows get domain 1."""
    nmat = None
    rows = []
    for raw in Path(path).read_text(encoding="ascii").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("Nmat="):
            nmat = int(s.split("=", 1)[1])
            continue
        rows.append([int(v) for v in s.split()])
    arr = np.asarray(rows, dtype=int).reshape(len(rows), -1)
    if arr.shape[1] == 3:
        return arr, np.ones(len(arr), dtype=int), nmat
    if arr.shape[1] == 4:
        return arr[:, :3], arr[:, 3], nmat
    raise ValueError(f"expected 3 or 4 integers per line, got {arr.shape[1]}")
