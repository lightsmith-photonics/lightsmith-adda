# lightsmith-adda

A [lightsmith](https://github.com/lightsmith-photonics/lightsmith) backend that runs [ADDA](https://github.com/adda-team/adda), the
discrete dipole approximation code by Yurkin and Hoekstra, as an external executable.

**License: GPL-3.0-or-later.** This plugin is kept outside the MIT lightsmith core. It talks to
ADDA (GPL-3) only through files and a subprocess, and it does not bundle ADDA.

## Install

1. Get ADDA (prebuilt Windows binaries are on the
   [releases page](https://github.com/adda-team/adda/releases); on Linux and macOS, build from source).
2. Put `adda` (or `adda_ocl` / `adda_mpi`) on `PATH`, or set `LIGHTSMITH_ADDA` to the full path of the executable.
3. Install this package next to lightsmith, e.g. `uv pip install --no-sources git+https://github.com/lightsmith-photonics/lightsmith-adda`
   (`--no-sources`: the `../lightsmith` path in `pyproject.toml` is for development; plain `pip install git+...` works too).

```console
$ lightsmith backends      # lists "adda" (method "dda"); "available" is false until step 2 is done
```

## Use

```python
import lightsmith as ls

sim = ls.Simulation(
    particles=[ls.Particle(shape=ls.LayeredSphere(core_radius_nm=25, shells=[
        ls.Shell(thickness_nm=10, material=ls.ConstantIndex(n_real=2.0))]),
        material=ls.ConstantIndex(n_real=1.5, n_imag=0.02))],
    medium_index=1.33,
    wavelengths=[500, 600],
    options={"dda": {"dipoles_per_diameter": 20}, "adda": {"tol": 1e-8}},
)
ls.estimate(sim, method="adda")
res = ls.solve(sim, method="adda")
print(res.c_ext_nm2, res.provenance["commands"])
```

## How the problem is mapped onto ADDA

| lightsmith | ADDA |
|---|---|
| particles, shells, clusters | voxelised here (dipole centre inside the shape) and written as an ADDA geometry file, `-shape read <file>`. One domain per distinct material (`Nmat=<K>` format, manual §B.5); identical materials share a domain. |
| dipole size | `-size <grid_x * d>` in nm. By default the dipole size is rescaled so the dipole volume equals the particle volume (`volume_correction`), as ADDA does for its built-in shapes. |
| `medium_index` n₀ | relative indices `-m <Re(n/n₀)> <Im(n/n₀)> ...` and `-lambda <λ/n₀>` (manual §4.2). |
| wavelengths | one ADDA run per wavelength, each in its own `-dir`. |
| `illumination` | the lattice is built in a rotated frame where k is ADDA's +z (default `-prop`) and E is ADDA's Y polarization (0,1,0). The result is read from `CrossSec-Y`, which ADDA always writes. |
| `orientation_average` | `-orient avg <avg_params.dat>` (β and γ by Romberg integration, `avg_beta_jmax`, `avg_gamma_jmax`); result from `CrossSec` (unpolarized light). |
| outputs | `Cext`, `Cabs` from CrossSec; `Csca = Cext - Cabs` unless ADDA reports `Csca`. `-scat_matr none` skips the Mueller matrix. |

Units: ADDA treats lengths in arbitrary consistent units. Everything is passed in nm, so cross sections
come back in nm². Efficiencies are recomputed by lightsmith with π r_eq² of the true particle volume,
not ADDA's dipole-volume a_eq.

Sign convention: ADDA uses exp(-iωt), with Im(m) > 0 for absorbing materials (manual §6.3), the same as
lightsmith, so indices are passed unchanged. ADDA rejects m = 1 exactly; such a domain is sent as
1.00001 (the manual's own suggestion) and a warning is added.

Options (`Simulation.options["adda"]`, validated strictly): `dipoles_per_diameter`, `volume_correction`,
`polarizability` (`-pol`), `interaction` (`-int`), `iterative_solver` (`-iter`), `tol` (`-eps`),
`max_iter` (`-maxiter`), `max_dipoles`, `avg_beta_jmax`, `avg_gamma_jmax`, `avg_eps`, `executable`,
`timeout_s`, `extra_args`, `keep_files`. From the method-level `options["dda"]`, `dipoles_per_diameter`,
`tol`, `max_iter`, `max_dipoles` and `polarizability` (`ldr` → `ldr`, `cm_rr` → `rrc`) are honoured.
Other keys are ignored with a warning.

Provenance records the executable, the ADDA version (from the log, else `adda -V`), every command line,
the dipole count and grid, iterations per run, and the polarizability and iterative method from ADDA's log.

`ls.discretization(sim, method="adda")` returns the dipole lattice written to the shape file, without running
ADDA: the occupied sites in lightsmith coordinates at ADDA's dipole size, with one material slot per domain.

Not supported: substrate (ADDA's `-surf` assumes vacuum above the surface), near-field and far-field outputs.

## Develop

```console
cd lightsmith-adda            # next to a lightsmith checkout in ../lightsmith (tool.uv.sources)
uv sync
uv run pytest                 # tests needing the real ADDA are skipped when it is absent
uv run lightsmith backends   # shows the "adda" backend
```

Tests check the geometry writer and output parsers against hand-written files in the manual's formats,
and run the whole pipeline against a stand-in executable (`tests/fake_adda.py`). Tests marked `adda` compare
against Mie and run only when a real ADDA is found.

## Cite

M. A. Yurkin and A. G. Hoekstra, "The discrete-dipole-approximation code ADDA: capabilities and known
limitations," JQSRT 112, 2234–2247 (2011), doi:10.1016/j.jqsrt.2011.01.031.
