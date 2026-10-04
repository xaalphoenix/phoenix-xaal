import numpy as np
import pytest

from conftest import is_valid_solid
from phoenix_stl.core.analyze import analyze
from phoenix_stl.core.cut import plane_cut, section_segments
from phoenix_stl.core.io_stl import load_stl, save_stl
from phoenix_stl.core.lod import display_mesh
from phoenix_stl.core.mesh import Mesh
from phoenix_stl.core.repair import RepairOptions, repair
from phoenix_stl.core.weld import weld_points, weld_soup


# ---------- IO / weld ----------

def test_binary_roundtrip(tmp_path, sphere):
    p = str(tmp_path / "s.stl")
    save_stl(sphere, p)
    back = load_stl(p)
    assert back.n_faces == sphere.n_faces
    assert back.n_vertices == sphere.n_vertices
    assert np.isclose(back.volume(), sphere.volume(), rtol=1e-6)


def test_ascii_load(tmp_path, cube):
    p = tmp_path / "c.stl"
    lines = ["solid test"]
    for tri in cube.vertices[cube.faces]:
        lines += ["facet normal 0 0 0", " outer loop"]
        lines += [f"  vertex {x:.6e} {y:.6e} {z:.6e}" for x, y, z in tri]
        lines += [" endloop", "endfacet"]
    lines.append("endsolid test")
    p.write_text("\n".join(lines))
    back = load_stl(str(p))
    assert back.n_faces == cube.n_faces and back.n_vertices == 8
    assert np.isclose(back.volume(), 8000, rtol=1e-5)


def test_weld_negative_zero_and_tolerance():
    pts = np.array([[0, 0, 0], [-0.0, 0, 0], [1, 1, 1], [1.0004, 1, 1]], dtype=np.float32)
    u, inv = weld_points(pts)
    assert len(u) == 3 and inv[0] == inv[1]
    u, inv = weld_points(pts, tol=0.01)
    assert len(u) == 2


def test_weld_soup_matches_unique():
    rng = np.random.default_rng(1)
    base = rng.random((500, 3)).astype(np.float32)
    tris = base[rng.integers(0, 500, (3000, 3))]
    m = weld_soup(tris)
    assert m.n_vertices == len(np.unique(tris.reshape(-1, 3), axis=0))
    assert np.array_equal(m.vertices[m.faces], tris)


# ---------- analyze ----------

def test_analyze_clean(sphere):
    r, segs = analyze(sphere)
    assert r.watertight and r.printable and r.shells == 1 and segs is None


def test_analyze_problems(sphere):
    f = sphere.faces.copy()
    f[:5] = f[:5][:, ::-1]  # flipped faces
    broken = Mesh(sphere.vertices, np.concatenate([f[:5], f[10:], f[20:21]]))  # hole + duplicate
    r, segs = analyze(broken)
    assert not r.watertight
    assert r.holes >= 1 and r.boundary_edges > 0
    assert r.duplicate_faces == 1
    assert r.flipped_edges > 0
    assert segs is not None and len(segs) == r.boundary_edges + r.nonmanifold_edges


def test_analyze_inside_out(sphere):
    r, _ = analyze(Mesh(sphere.vertices, sphere.faces[:, ::-1]))
    assert r.inside_out and not r.printable


# ---------- repair ----------

def test_repair_fills_holes_and_flips(sphere):
    f = sphere.faces.copy()
    f[100:400] = f[100:400][:, ::-1]
    broken = Mesh(sphere.vertices, np.concatenate([f[:50], f[80:]]))
    fixed, log = repair(broken)
    r, _ = analyze(fixed)
    assert r.printable, (r, log)
    assert log["filled_holes"] >= 1 and log["flipped_faces"] > 0
    assert is_valid_solid(fixed)
    assert np.isclose(fixed.volume(), sphere.volume(), rtol=0.02)


def test_repair_inside_out_keeps_hollow_inner_shell(hollow_sphere):
    inv = Mesh(hollow_sphere.vertices, hollow_sphere.faces[:, ::-1])
    fixed, _ = repair(inv)
    assert np.isclose(fixed.volume(), hollow_sphere.volume(), rtol=1e-6)
    assert analyze(fixed)[0].printable


def test_repair_removes_debris(sphere, cube):
    tiny = Mesh(cube.vertices * 0.001 + 50, cube.faces)
    fixed, log = repair(Mesh.concatenate([sphere, tiny]))
    assert log["removed_debris_shells"] == 1
    assert fixed.n_faces == sphere.n_faces


def test_repair_options_off(sphere):
    f = sphere.faces[10:]
    fixed, _ = repair(Mesh(sphere.vertices, f), RepairOptions(fill_holes=False))
    assert not analyze(fixed)[0].watertight


# ---------- cut ----------

@pytest.mark.parametrize("normal", [(0, 0, 1), (0.3, -0.4, 0.8), (1, 0, 0)])
def test_cut_sphere(sphere, normal):
    res = plane_cut(sphere, normal, (0.5, -0.3, 1.2))
    for part in (res.positive, res.negative):
        assert analyze(part)[0].printable
        assert is_valid_solid(part)
    assert np.isclose(res.positive.volume() + res.negative.volume(), sphere.volume(), rtol=1e-6)
    assert res.loops == 1 and res.open_loops == 0


def test_cut_torus_cap_with_hole(torus):
    # Horizontal cut: the cap is a ring (outer loop + hole).
    res = plane_cut(torus, (0, 0, 1), (0, 0, 0.7))
    assert res.loops == 2
    for part in (res.positive, res.negative):
        assert is_valid_solid(part)
    assert np.isclose(res.positive.volume() + res.negative.volume(), torus.volume(), rtol=1e-6)
    # Vertical cut through the ring: two separate disc caps.
    res = plane_cut(torus, (1, 0, 0), (0, 0, 0))
    assert res.loops == 2 and is_valid_solid(res.positive)


def test_cut_hollow(hollow_sphere):
    res = plane_cut(hollow_sphere, (0, 0, 1), (0, 0, 0.3))
    for part in (res.positive, res.negative):
        assert is_valid_solid(part)
        assert analyze(part)[0].printable
    assert np.isclose(res.positive.volume() + res.negative.volume(), hollow_sphere.volume(), rtol=1e-6)


def test_cut_through_vertices(cube):
    # Plane exactly through the cube's mid vertices/edges must stay closed.
    res = plane_cut(cube, (0, 0, 1), (0, 0, 0))
    for part in (res.positive, res.negative):
        assert analyze(part)[0].watertight
    assert np.isclose(res.positive.volume(), 4000, rtol=1e-6)


def test_cut_miss(sphere):
    res = plane_cut(sphere, (0, 0, 1), (0, 0, 50))
    assert res.positive.n_faces == 0 and res.negative.n_faces == sphere.n_faces


def test_section_segments(sphere):
    segs = section_segments(sphere, (0, 0, 1), (0, 0, 0))
    assert len(segs) > 50
    assert np.allclose(segs[..., 2], 0, atol=1e-4)
    assert np.allclose(np.linalg.norm(segs[..., :2], axis=-1), 10, atol=0.1)


def test_sequential_cuts(sphere):
    a = plane_cut(sphere, (0, 0, 1), (0, 0, 0)).positive
    b = plane_cut(a, (1, 0, 0), (0, 0, 0))
    c = plane_cut(b.negative, (0, 1, 0), (0, 2, 0))
    total = b.positive.volume() + c.positive.volume() + c.negative.volume()
    assert np.isclose(total, a.volume(), rtol=1e-6)
    assert all(is_valid_solid(m) for m in (b.positive, c.positive, c.negative))


# ---------- lod ----------

def test_display_mesh(sphere):
    small = display_mesh(sphere, 2000)
    assert 0 < small.n_faces <= 2500
    assert display_mesh(sphere, 10**9) is sphere


def test_cut_reimport_stays_manifold(tmp_path, sphere):
    # Symmetric primitives have vertices on x=0; exported parts must re-weld cleanly.
    a = plane_cut(sphere, (0, 0, 1), (0, 0, 0)).positive
    b = plane_cut(a, (1, 0, 0), (0, 0, 0))
    for i, part in enumerate((b.positive, b.negative)):
        p = str(tmp_path / f"p{i}.stl")
        save_stl(part, p)
        back = load_stl(p)
        assert back.n_vertices == part.n_vertices
        assert is_valid_solid(back) and analyze(back)[0].printable


def test_cut_grazing_face_is_not_a_cut(cube):
    res = plane_cut(cube, (0, 0, 1), (0, 0, -10))  # exactly the bottom face
    assert res.negative.n_faces == 0 and res.positive.n_faces == cube.n_faces


@pytest.fixture
def stepped():
    from conftest import from_manifold
    import manifold3d as m3d
    M = m3d.Manifold
    return from_manifold(M.cube((40, 40, 10)) + M.cube((20, 20, 10)).translate((10, 10, 10))
                         + M.cube((10, 10, 10)).translate((15, 15, 20)))


@pytest.fixture
def hollow_box():
    from conftest import from_manifold
    import manifold3d as m3d
    return from_manifold(m3d.Manifold.cube((30, 30, 30), True) - m3d.Manifold.cube((26, 26, 26), True))


@pytest.mark.parametrize("normal,origin", [((0, 0, 1), (0, 0, 10)), ((0, 0, 1), (0, 0, 20)),
                                           ((1, 0, 0), (10, 0, 0)), ((1, 0, 0), (15, 0, 0))])
def test_cut_exactly_at_cad_faces(tmp_path, stepped, normal, origin):
    # The plane lies exactly on model faces/edges: faces in the plane must go to the right side.
    res = plane_cut(stepped, normal, origin)
    assert np.isclose(res.positive.volume() + res.negative.volume(), stepped.volume(), rtol=1e-9)
    for part in (res.positive, res.negative):
        p = str(tmp_path / "p.stl")
        save_stl(part, p)
        assert is_valid_solid(load_stl(p)) and analyze(load_stl(p))[0].printable


@pytest.mark.parametrize("z", [-13.0, 0.0, 13.0, 14.0])
def test_cut_hollow_box_collinear_caps(hollow_box, z):
    # Box sides give collinear cap points; z = +-13 is exactly the cavity floor/ceiling.
    res = plane_cut(hollow_box, (0, 0, 1), (0, 0, z))
    assert "cap_fallback" not in res.warnings
    for part in (res.positive, res.negative):
        assert is_valid_solid(part) and analyze(part)[0].printable
    assert np.isclose(res.positive.volume() + res.negative.volume(), hollow_box.volume(), rtol=1e-9)
