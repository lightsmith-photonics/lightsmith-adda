import math
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends import all_backends, get_backend
from lightsmith_adda import AddaBackend, find_executable
from lightsmith_adda.geometry import incident_frame, read_shape_file, voxelise, write_shape_file
from lightsmith_adda.output import avg_params_text, parse_cross_sections, parse_log, parse_version

DATA = Path(__file__).parent / "data"
DIELECTRIC = ls.ConstantIndex(n_real=1.5, n_imag=0.02)
SHELL = ls.ConstantIndex(n_real=2.0, n_imag=0.0)
HAVE_ADDA = find_executable() is not None


def sphere_sim(material=DIELECTRIC, radius=40.0, wavelengths=(450.0, 550.0, 650.0), npd=12, medium=1.33, **kw):
    kw.setdefault("options", {"dda": {"dipoles_per_diameter": npd}})
    return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material)],
                          medium_index=medium, wavelengths=list(wavelengths), **kw)


def layered_sim(npd=16, shell_material=SHELL, **kw):
    shape = ls.LayeredSphere(core_radius_nm=25.0, shells=[ls.Shell(thickness_nm=15.0, material=shell_material)])
    kw.setdefault("options", {"dda": {"dipoles_per_diameter": npd}})
    return ls.Simulation(particles=[ls.Particle(shape=shape, material=DIELECTRIC)], medium_index=1.33,
                          wavelengths=[500.0, 600.0], **kw)


# --- registration -----------------------------------------------------------


def test_entry_point_registers_backend():
    assert "adda" in {b.name for b in all_backends()}
    b = get_backend("adda")
    assert b is AddaBackend
    info = b.info()
    assert info["method"] == "dda" and info["license"] == "GPL-3.0-or-later"


def test_unavailable_without_executable(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("LIGHTSMITH_ADDA", raising=False)
    ok, why = AddaBackend.available()
    assert not ok and "LIGHTSMITH_ADDA" in why
    monkeypatch.setenv("LIGHTSMITH_ADDA", str(tmp_path / "no-such-adda.exe"))
    ok, why = AddaBackend.available()
    assert not ok and "not an executable" in why
    with pytest.raises(ValueError, match="cannot run"):
        ls.solve(sphere_sim(), method="adda")


def test_supports():
    assert AddaBackend.supports(sphere_sim())[0]
    assert AddaBackend.supports(sphere_sim(orientation_average=True))[0]
    sub = sphere_sim(substrate=ls.Substrate(material=ls.ConstantIndex(n_real=1.5)))
    ok, why = AddaBackend.supports(sub)
    assert not ok and "substrate" in why


# --- geometry / shape files -------------------------------------------------


def test_single_domain_shape_file_roundtrip(tmp_path):
    vox = voxelise(sphere_sim(), 12)
    assert len(vox.materials) == 1
    path = write_shape_file(tmp_path / "s.geom", vox)
    lines = path.read_text().splitlines()
    data = [ln for ln in lines if not ln.startswith("#")]
    assert all(ln.startswith("#") for ln in lines[: len(lines) - len(data)])  # comments first
    assert not any(ln.startswith("Nmat") for ln in lines)
    assert all(len(ln.split()) == 3 for ln in data)
    ijk, dom, nmat = read_shape_file(path)
    assert nmat is None and len(ijk) == vox.n_dip and set(dom) == {1}
    np.testing.assert_array_equal(ijk, vox.ijk)
    # Volume correction: N d^3 equals the sphere volume.
    assert vox.n_dip * vox.d**3 == pytest.approx(4 / 3 * math.pi * 40.0**3, rel=1e-12)
    assert vox.size_x_nm == pytest.approx(vox.grid[0] * vox.d)
    assert vox.grid[0] in (11, 12, 13)


def test_layered_sphere_multi_domain(tmp_path):
    vox = voxelise(layered_sim(), 16)
    assert vox.materials == [DIELECTRIC, SHELL]
    path = write_shape_file(tmp_path / "c.geom", vox)
    data = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    assert data[0] == "Nmat=2"  # directly after the comments, no blank line (manual B.5)
    ijk, dom, nmat = read_shape_file(path)
    assert nmat == 2 and set(dom) == {1, 2}
    # Core dipoles sit closer to the centre than shell dipoles.
    r = np.linalg.norm(ijk - ijk.mean(axis=0), axis=1)
    assert r[dom == 1].max() < r[dom == 2].max()
    assert r[dom == 1].max() <= r[dom == 2].min() + 1.0
    # Core volume fraction close to (25/40)^3.
    assert (dom == 1).mean() == pytest.approx((25 / 40) ** 3, rel=0.15)


def test_identical_materials_share_a_domain():
    two = [ls.Particle(shape=ls.Sphere(radius_nm=10), material=DIELECTRIC, position_nm=(-12, 0, 0)),
           ls.Particle(shape=ls.Sphere(radius_nm=10), material=DIELECTRIC, position_nm=(12, 0, 0))]
    sim = ls.Simulation(particles=two, wavelengths=[500.0])
    assert len(voxelise(sim, 10).materials) == 1
    other = [two[0], two[1].model_copy(update={"material": SHELL})]
    vox = voxelise(ls.Simulation(particles=other, wavelengths=[500.0]), 10)
    assert vox.materials == [DIELECTRIC, SHELL]
    # The two spheres sit along x, so the dipoles of domain 2 are all at larger x.
    assert vox.ijk[vox.domain == 1, 0].max() < vox.ijk[vox.domain == 2, 0].min()
    # A layered sphere whose shell equals its core collapses to one domain.
    assert len(voxelise(layered_sim(shell_material=DIELECTRIC), 12).materials) == 1


def test_read_handwritten_shape_files():
    ijk, dom, nmat = read_shape_file(DATA / "coated_sample.geom")
    assert nmat == 2 and len(ijk) == 27 and dom.tolist().count(2) == 1
    assert ijk[dom == 2].tolist() == [[0, 0, 0]]
    ijk, dom, nmat = read_shape_file(DATA / "single_sample.geom")
    assert nmat is None and ijk.tolist() == [[0, 0, 1], [2, 1, 0]] and set(dom) == {1}


def test_incident_frame_maps_k_to_z_and_e_to_y():
    k, e = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    f = incident_frame(k, e)
    np.testing.assert_allclose(f @ f.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(f) == pytest.approx(1.0)
    np.testing.assert_allclose(f @ k, [0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(f @ e, [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(incident_frame((0, 0, 1), (0, 1, 0)), np.eye(3), atol=1e-15)


def test_rotated_voxelisation_follows_incidence():
    # Prolate spheroid along z, light along x polarized along z: in ADDA's frame the long axis is y (= E).
    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=30), material=DIELECTRIC)],
        illumination=ls.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1)), wavelengths=[500.0])
    vox = AddaBackend._voxels(sim, AddaBackend.options(sim)[0])
    nx, ny, nz = vox.grid
    assert ny > 2.5 * nx and ny > 2.5 * nz


# --- output files -----------------------------------------------------------


def test_parse_crosssec_files():
    y = parse_cross_sections((DATA / "CrossSec-Y_sample").read_text())
    assert y == {"Cext": 135.0449046, "Qext": 3.79114961, "Cabs": 0.0, "Qabs": 0.0}
    x = parse_cross_sections((DATA / "CrossSec-X_sample").read_text())
    assert x["Cext"] == 1.234567891e4 and x["Csca"] == 3580.24681
    assert x["g"] == (0.000419, -6.98e-07, 0.3354) and x["Csca.g"] == (1.5, -2.5e-3, 1200.75)
    avg = parse_cross_sections((DATA / "CrossSec_avg_sample").read_text())
    assert avg["Cext"] == 4321.5 and avg["Cabs"] == 1234.25
    with pytest.raises(ValueError, match="Cext"):
        parse_cross_sections("Qext\t= 1\n")


def test_parse_log_and_version():
    log = parse_log((DATA / "log_sample").read_text())
    assert log["version"] == "1.4.0"
    assert log["dipoles"] == 2176 and log["grid"] == [16, 16, 16]
    assert log["dipole_size"] == 5.46875 and log["iterations"] == 22
    assert log["polarizability"] == "Lattice Dispersion Relation"
    assert log["iterative_method"] == "QMR (complex symmetric)" and log["symmetries"] == "XYR"
    assert parse_version("ADDA v.1.4.0\n...") == "1.4.0"
    assert parse_version("ADDA v1.5.0-alpha3 (OpenCL)") == "1.5.0-alpha3"
    assert parse_log("nothing here") == {}


def test_avg_params_text():
    text = avg_params_text(beta_jmax=5, gamma_jmax=2)
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    keys = ["min", "max", "Jmin", "Jmax", "eps", "equiv", "periodic"]
    for start, name in zip((0, 8, 16), ("alpha", "beta", "gamma")):
        assert body[start] == f"{name}:"
        assert [ln.split("=")[0] for ln in body[start + 1:start + 8]] == keys
    assert body[8:16] == ["beta:", "min=0", "max=180", "Jmin=2", "Jmax=5", "eps=0", "equiv=false", "periodic=false"]
    assert body[16:24] == ["gamma:", "min=0", "max=360", "Jmin=2", "Jmax=2", "eps=0", "equiv=true", "periodic=true"]


# --- command line -----------------------------------------------------------


def _flag(cmd, name, n=1):
    i = cmd.index(name)
    return cmd[i + 1:i + 1 + n]


def test_command_line_scales_to_medium():
    sim = layered_sim(options={"dda": {"dipoles_per_diameter": 16, "polarizability": "cm_rr", "tol": 1e-8,
                                       "precision": "double"},
                               "adda": {"iterative_solver": "bicgstab", "max_iter": 500}})
    opts, warnings = AddaBackend.options(sim)
    assert opts.polarizability == "rrc" and any("precision" in w for w in warnings)
    vox = AddaBackend._voxels(sim, opts)
    cmd = AddaBackend.command("adda", "shape.geom", vox, 600.0, sim, opts, "run000", None)
    assert cmd[:4] == ["adda", "-shape", "read", "shape.geom"]
    assert float(_flag(cmd, "-lambda")[0]) == pytest.approx(600.0 / 1.33, rel=1e-15)
    assert float(_flag(cmd, "-size")[0]) == pytest.approx(vox.grid[0] * vox.d, rel=1e-15)
    m = [float(v) for v in _flag(cmd, "-m", 4)]
    assert m == pytest.approx([1.5 / 1.33, 0.02 / 1.33, 2.0 / 1.33, 0.0], rel=1e-15)
    assert _flag(cmd, "-dir") == ["run000"] and _flag(cmd, "-scat_matr") == ["none"]
    assert _flag(cmd, "-pol") == ["rrc"] and _flag(cmd, "-iter") == ["bicgstab"]
    assert float(_flag(cmd, "-eps")[0]) == pytest.approx(8.0) and _flag(cmd, "-maxiter") == ["500"]
    assert "-orient" not in cmd and "-prop" not in cmd  # incidence is built into the geometry


def test_command_line_orientation_average_and_unit_index():
    mat = ls.ConstantIndex(n_real=1.33)
    sim = sphere_sim(material=mat, orientation_average=True)
    opts, _ = AddaBackend.options(sim)
    vox = AddaBackend._voxels(sim, opts)
    warns = []
    cmd = AddaBackend.command("adda", "s.geom", vox, 500.0, sim, opts, "r", "avg.dat", warns)
    assert _flag(cmd, "-orient", 2) == ["avg", "avg.dat"]
    assert float(_flag(cmd, "-m")[0]) == 1.00001 and warns


def test_options_reject_unknown_adda_keys():
    with pytest.raises(Exception):
        AddaBackend.options(sphere_sim(options={"adda": {"no_such_option": 1}}))


# --- full pipeline with a stand-in executable --------------------------------


@pytest.fixture
def fake_adda(tmp_path, monkeypatch):
    script = Path(__file__).parent / "fake_adda.py"
    if sys.platform == "win32":
        exe = tmp_path / "adda.cmd"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        exe = tmp_path / "adda"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LIGHTSMITH_ADDA", str(exe))
    return exe


def test_solve_pipeline_with_fake_adda(fake_adda):
    sim = layered_sim()
    assert AddaBackend.available()[0]
    res = ls.solve(sim, method="adda")
    vox = AddaBackend._voxels(sim, AddaBackend.options(sim)[0])
    expected_ext = [vox.size_x_nm * lam / 1.33 for lam in (500.0, 600.0)]
    np.testing.assert_allclose(res.c_ext_nm2, expected_ext, rtol=1e-9)
    np.testing.assert_allclose(res.c_abs_nm2, [vox.n_dip] * 2)
    np.testing.assert_allclose(res.c_sca_nm2, np.array(expected_ext) - vox.n_dip, rtol=1e-9)
    assert res.backend == "adda" and res.method == "dda"
    assert res.equivalent_radius_nm == pytest.approx(40.0)
    p = res.provenance
    assert p["adda_version"] == "9.9.9-fake" and len(p["commands"]) == 2 and p["iterations"] == [7, 7]
    assert "-shape read" in p["commands"][0] and p["domains"] == 2
    assert not any("dipole size" in w for w in res.warnings), res.warnings


def test_discretization_is_the_lattice_written_for_adda(fake_adda, monkeypatch):
    def no_run(*a, **k):
        raise AssertionError("discretization() must not run ADDA")

    monkeypatch.setattr("lightsmith_adda.backend.subprocess.run", no_run)
    centre = np.array([10.0, -5.0, 3.0])
    sim = ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=40.0), material=DIELECTRIC,
                                               position_nm=tuple(centre))],
                        medium_index=1.33, wavelengths=[500.0], options={"adda": {"dipoles_per_diameter": 12}})
    d = ls.discretization(sim, "adda")
    assert d["backend"] == "adda" and d["layout_kind"] == "dipole_lattice" and d["error"] is None, d["error"]
    p = d["payload"]
    vox = AddaBackend._voxels(sim, AddaBackend.options(sim)[0])
    est = ls.estimate(sim, method="adda")
    assert p["n_dipoles"] == vox.n_dip == est["dipoles"] == d["summary"]["dipoles"]
    assert p["grid"] == list(vox.grid) == est["grid"] and p["dipoles_per_diameter"] == 12
    assert p["spacing_nm"] == pytest.approx(est["dipole_spacing_nm"]) and p["stride"] == 1
    pts = np.asarray(p["points_nm"])
    assert pts.shape == (vox.n_dip, 3) and set(p["material_slot"]) == {0}
    # Placed where the particle is, inside it, and spaced by ADDA's dipole size along the lattice axes.
    np.testing.assert_allclose(pts.mean(axis=0), centre, atol=0.5)
    assert np.linalg.norm(pts - centre, axis=1).max() < 40.0
    steps = np.diff(np.unique(np.round(pts[:, 0], 3)))
    np.testing.assert_allclose(steps, p["spacing_nm"], rtol=1e-3)


def test_discretization_layers_rotation_and_size_limit(fake_adda):
    vox_sim = layered_sim(npd=16)
    p = ls.discretization(vox_sim, "adda")["payload"]
    pts, slot = np.asarray(p["points_nm"]), np.asarray(p["material_slot"])
    r = np.linalg.norm(pts, axis=1)
    assert p["domains"] == 2 and set(slot) == {0, 1}
    assert r[slot == 0].max() < 25.0 + p["spacing_nm"] < r[slot == 1].max()
    # Light along x polarized along z: ADDA's frame has the long axis on y, but the points stay in lightsmith
    # coordinates, where the spheroid's long axis is z.
    rod = ls.Simulation(
        particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=30), material=DIELECTRIC)],
        illumination=ls.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1)), wavelengths=[500.0])
    pts = np.asarray(ls.discretization(rod, "adda")["payload"]["points_nm"])
    extent = pts.max(axis=0) - pts.min(axis=0)
    assert extent[2] > 2.5 * extent[0] and extent[2] > 2.5 * extent[1]
    # Over max_dipoles: the lattice is still shown, with the refusal solve() would give.
    big = sphere_sim(options={"dda": {"dipoles_per_diameter": 12}, "adda": {"max_dipoles": 10}})
    d = ls.discretization(big, "adda")
    assert d["layout_kind"] == "dipole_lattice" and "max_dipoles" in d["error"]


def test_orientation_average_pipeline_with_fake_adda(fake_adda):
    res = ls.solve(sphere_sim(orientation_average=True, wavelengths=[500.0]), method="adda")
    assert res.provenance["polarization"].startswith("orientation averaged")
    assert res.c_ext_nm2[0] > 0


_ORIGINAL_COMMAND = AddaBackend.__dict__["command"].__func__


def _one_index_only(cls, *a, **k):
    cmd = _ORIGINAL_COMMAND(cls, *a, **k)
    i = cmd.index("-m")
    return cmd[:i + 3] + cmd[i + 5:]  # keep only the first refractive index


def test_adda_failure_is_reported(fake_adda, monkeypatch):
    # Two domains but one refractive index: the stand-in rejects it, as ADDA does.
    monkeypatch.setattr(AddaBackend, "command", classmethod(_one_index_only))
    with pytest.raises(RuntimeError, match="ADDA failed"):
        ls.solve(layered_sim(), method="adda")


# --- against the real ADDA (skipped without the executable) ------------------

requires_adda = pytest.mark.skipif(not HAVE_ADDA, reason="ADDA executable not found (PATH or LIGHTSMITH_ADDA)")


@requires_adda
@pytest.mark.adda
def test_real_adda_matches_mie_for_dielectric_sphere():
    sim = sphere_sim(npd=20)
    ref = ls.solve(sim, method="miepython")
    res = ls.solve(sim, method="adda")
    np.testing.assert_allclose(res.c_ext_nm2, ref.c_ext_nm2, rtol=0.03)
    np.testing.assert_allclose(res.c_abs_nm2, ref.c_abs_nm2, rtol=0.05)
    assert res.provenance["adda_version"]


@requires_adda
@pytest.mark.adda
def test_real_adda_oblique_incidence_and_orientation_average_match_mie():
    ref = ls.solve(sphere_sim(npd=16, wavelengths=[550.0]), method="miepython")
    tilted = sphere_sim(npd=16, wavelengths=[550.0],
                        illumination=ls.PlaneWave(direction=(1, 1, 1), polarization=(1, -1, 0)))
    avg = sphere_sim(npd=16, wavelengths=[550.0], orientation_average=True,
                     options={"dda": {"dipoles_per_diameter": 16}, "adda": {"avg_beta_jmax": 2, "avg_gamma_jmax": 2}})
    for sim in (tilted, avg):
        res = ls.solve(sim, method="adda")
        np.testing.assert_allclose(res.c_ext_nm2, ref.c_ext_nm2, rtol=0.04)


@requires_adda
@pytest.mark.adda
def test_real_adda_layered_sphere_reduces_to_mie():
    # A shell of the core material must reproduce the homogeneous sphere; a real two-domain run must differ.
    same = layered_sim(npd=20, shell_material=DIELECTRIC)
    ref = ls.solve(sphere_sim(radius=40.0, wavelengths=[500.0, 600.0]), method="miepython")
    np.testing.assert_allclose(ls.solve(same, method="adda").c_ext_nm2, ref.c_ext_nm2, rtol=0.03)
    two = ls.solve(layered_sim(npd=20), method="adda")
    assert two.provenance["domains"] == 2 and two.c_ext_nm2[0] > ref.c_ext_nm2[0]
