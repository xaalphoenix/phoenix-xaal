import math

import manifold3d as m3d
import numpy as np
import pytest

from conftest import from_manifold, is_valid_solid
from phoenix_stl.core import connectors as C
from phoenix_stl.core.analyze import analyze
from phoenix_stl.core.boolean import to_manifold
from phoenix_stl.core.cut import plane_cut
from phoenix_stl.core.grid import grid_cut
from phoenix_stl.core.section import find_joint, flat_face_at, section_loops, to_cross_section
from phoenix_stl.core.surface_cut import Frame
from phoenix_stl.core.weld import weld_mesh


def split(mesh, z=0.3, normal=(0, 0, 1)):
    r = plane_cut(mesh, normal, (0, 0, z))
    return r.negative, r.positive


@pytest.fixture
def block():
    return from_manifold(m3d.Manifold.cube((40, 30, 40), True))


@pytest.fixture
def shell():
    return weld_mesh(from_manifold(m3d.Manifold.sphere(20, 96) - m3d.Manifold.sphere(17, 80)), 0)


def joint(a, b):
    j = find_joint(a, b)
    fr = Frame.from_normal(j["origin"], j["normal"])
    return fr, C.joint_shape({"A": a, "B": b}, fr)


def overlap(a, b) -> float:
    return (to_manifold(a) ^ to_manifold(b)).volume()


def test_find_joint_and_outline(block):
    a, b = split(block)
    j = find_joint(a, b)
    assert np.allclose(j["normal"], (0, 0, 1)) and abs(j["origin"][2] - 0.3) < 1e-6
    assert j["area"] == pytest.approx(1200, rel=1e-6)
    # parts that do not touch have no joint
    far = from_manifold(m3d.Manifold.cube((5, 5, 5)).translate((100, 0, 0)))
    assert find_joint(a, far) is None
    # clicked flat face, slightly wrong normal from a coarse preview
    f = flat_face_at(a, (3, 2, 0.3), (0.03, 0.0, 1.0))
    assert np.allclose(f["normal"], (0, 0, 1), atol=1e-6) and abs(f["origin"][2] - 0.3) < 1e-6
    loops = section_loops(block, (0, 0, 1), (0, 0, 0))
    assert len(loops) == 1
    assert to_cross_section(loops, Frame.from_normal((0, 0, 0), (0, 0, 1))).area() == pytest.approx(1200)


def test_grid_pieces_joint():
    box = from_manifold(m3d.Manifold.cube((60, 20, 20), True))
    pieces = dict(grid_cut(box, [(0, 0.0)]))
    a, b = pieces[(0, 0, 0)], pieces[(1, 0, 0)]
    j = find_joint(a, b)
    assert np.allclose(j["normal"], (1, 0, 0)) and j["area"] == pytest.approx(400, rel=1e-6)


@pytest.mark.parametrize("kind", C.TYPES)
def test_every_type_fits_and_prints(block, kind):
    a, b = split(block)
    fr, shape = joint(a, b)
    spec, fit = C.Spec.default(kind), C.Fit.preset("resin", "snug")
    pls, notes = C.auto_place(shape, spec, fit, 2)
    assert notes == []
    if kind in C.PLACED_TYPES:
        assert len(pls) == (1 if kind == "dovetail" else 2)
        assert all(C.placement_ok(shape, spec, fit, pls))
    pl = C.plan(spec, fit, pls, shape)
    out, pins, warnings = C.apply_plan({"A": a, "B": b}, pl, fr)
    assert warnings == []
    for m in out.values():
        assert is_valid_solid(m) and analyze(m)[0].printable
    # assembled in place, the two parts never collide (the clearance is real)
    assert overlap(out["A"], out["B"]) < 1e-6
    if kind in C.MALE_TYPES:
        assert out["A"].volume() > a.volume() and out["B"].volume() < b.volume()
    else:
        assert out["A"].volume() < a.volume() and out["B"].volume() < b.volume()
    if kind == "dowel":
        assert pins is not None and is_valid_solid(pins)
        assert pins.volume() == pytest.approx(2 * math.pi * 2 ** 2 * 12, rel=0.03)


def test_dowel_clearance_is_exact(block):
    a, b = split(block)
    fr, shape = joint(a, b)
    spec, fit = C.Spec.default("dowel"), C.Fit(clearance=0.1, depth_gap=0.3, chamfer=0.0, wall=1.0)
    p = C.Placement(0.0, 0.0)
    out, _, _ = C.apply_plan({"A": a, "B": b}, C.plan(spec, fit, [p], shape), fr)
    world = fr.matrix()[:3, :]

    def pin(r):  # a pin of radius r sitting in its holes
        return from_manifold(C.cylinder(r, -6.0, 6.0).transform(world))
    for side in "AB":
        assert overlap(out[side], pin(2.0 + 0.099)) < 1e-6  # just under the clearance: free
        assert overlap(out[side], pin(2.0 + 0.11)) > 1e-3  # just over it: touches the wall
    # holes are deeper than half the pin by the depth gap
    assert overlap(out["A"], from_manifold(C.cylinder(1.5, -6.29, 0.0).transform(world))) < 1e-6
    assert overlap(out["A"], from_manifold(C.cylinder(1.5, -6.35, 0.0).transform(world))) > 1e-4


def test_hollow_part_gets_bosses(shell):
    a, b = split(shell)
    fr, shape = joint(a, b)
    assert shape.hollow
    spec, fit = C.Spec("dowel", 3.0, 6.0), C.Fit.preset("resin", "snug")
    pls, notes = C.auto_place(shape, spec, fit, 3, boss=True)
    assert len(pls) == 3 and notes == []
    pl = C.plan(spec, fit, pls, shape)
    assert len(pl.boss["A"]) == 3 and len(pl.boss["B"]) == 3
    out, pins, warnings = C.apply_plan({"A": a, "B": b}, pl, fr)
    assert warnings == []
    for side, orig in (("A", a), ("B", b)):
        m = out[side]
        assert analyze(m)[0].printable
        # bosses stay inside the outer surface
        assert np.all(m.bounds[0] >= orig.bounds[0] - 1e-4) and np.all(m.bounds[1] <= orig.bounds[1] + 1e-4)
        assert (to_manifold(m) - m3d.Manifold.sphere(20.001, 96)).volume() < 1e-3
    assert overlap(out["A"], out["B"]) < 1e-6
    # without bosses the wall is too thin: nothing is placed
    pls2, notes2 = C.auto_place(shape, spec, fit, 3, boss=False)
    assert pls2 == [] and notes2 == [0]


def test_hollow_tongue_is_centred_in_the_wall(shell):
    a, b = split(shell)
    fr, shape = joint(a, b)
    spec, fit = C.Spec.default("tongue"), C.Fit.preset("resin", "snug")
    out, _, warnings = C.apply_plan({"A": a, "B": b}, C.plan(spec, fit, [], shape), fr)
    assert warnings == []
    assert all(analyze(m)[0].printable for m in out.values())
    added = out["A"].volume() - a.volume()
    removed = b.volume() - out["B"].volume()
    assert added > 100 and removed > added  # groove is wider and deeper than the tongue
    assert overlap(out["A"], out["B"]) < 1e-6


def test_breakout_is_reported(shell):
    a, b = split(shell)
    fr, shape = joint(a, b)
    spec, fit = C.Spec("magnet", 6.0, 3.0), C.Fit.preset("resin", "snug")
    inside = C.Placement(0.0, 0.0)  # over the cavity, no boss: nothing to hold it
    assert C.placement_ok(shape, spec, fit, [inside], boss=False) == [False]
    out, _, warnings = C.apply_plan({"A": a, "B": b}, C.plan(spec, fit, [inside], shape, boss=False), fr)
    assert {w["side"] for w in warnings} == {"A", "B"}
    assert all(w["connectors"] == [0] for w in warnings)


def test_single_part_holes(block):
    a, _ = split(block)
    f = flat_face_at(a, (0, 0, 0.3), (0, 0, 1))
    fr = Frame.from_normal(f["origin"], f["normal"])
    shape = C.joint_shape({"A": a}, fr)
    spec, fit = C.Spec("magnet", 6.0, 2.0), C.Fit.preset("fdm", "snug")
    pls, _ = C.auto_place(shape, spec, fit, 4)
    out, pins, warnings = C.apply_plan({"A": a}, C.plan(spec, fit, pls, shape, sides=("A",)), fr)
    assert len(pls) == 4 and warnings == [] and pins is None
    r, d = 3.0 + fit.clearance, 2.0 + fit.depth_gap
    hole = math.pi * r * r * d
    assert a.volume() - out["A"].volume() == pytest.approx(4 * hole, rel=0.04)


def test_male_on_b_flips(block):
    a, b = split(block)
    fr, shape = joint(a, b)
    spec, fit = C.Spec.default("peg"), C.Fit.preset("resin", "slide")
    pls, _ = C.auto_place(shape, spec, fit, 1)
    out, _, _ = C.apply_plan({"A": a, "B": b}, C.plan(spec, fit, pls, shape, male="B"), fr)
    assert out["B"].volume() > b.volume() and out["A"].volume() < a.volume()
    assert out["B"].bounds[0][2] < 0.3 - 4.9  # peg reaches down into A's side
    assert overlap(out["A"], out["B"]) < 1e-6


def test_tolerance_coupon():
    cl = C.coupon_clearances(0.1, 0.05)
    assert cl == [0.0, 0.05, 0.1, 0.15, 0.2]
    plate, pin = C.tolerance_coupon(4.0, cl)
    assert is_valid_solid(plate) and is_valid_solid(pin)
    assert analyze(plate)[0].printable and analyze(pin)[0].printable
    # the pin passes through every hole except the zero-clearance one, where it just touches
    b = plate.bounds
    thick = b[1][2] - b[0][2]
    for k, c in enumerate(cl):
        x = 1.0 + (4.0 + 2 * 0.2 + 5.0) * (k + 0.5)
        y = 3.5 + (4.0 + 0.4) / 2 + 1.0
        probe = from_manifold(C.cylinder(2.0 - 1e-3, -1, thick + 1).translate((x, y, 0)))
        assert overlap(plate, probe) < 1e-4


def test_hollow_ring_bosses_stay_out_of_the_middle_hole():
    """A hollow ring: its cavity may take bosses, the hole in the middle (outside air) never."""
    def annulus(r0, r1, h):
        return (m3d.CrossSection.circle(r1, 128) - m3d.CrossSection.circle(r0, 128)).extrude(h).translate((0, 0, -h / 2))
    outer = annulus(15, 35, 40)
    ring = weld_mesh(from_manifold(outer - annulus(17, 33, 36)), 0)
    a, b = split(ring, z=0.0)
    fr, shape = joint(a, b)
    assert shape.hollow
    cavity = math.pi * (33 ** 2 - 17 ** 2)
    assert shape.levels[0][1].area() == pytest.approx(shape.solid.area() + cavity, rel=0.01)
    assert not C.contains(shape.levels[0][1], [(0.0, 0.0)])[0]  # middle hole is not inside the part
    spec, fit = C.Spec("dowel", 3.0, 6.0), C.Fit.preset("resin", "snug")
    pls, notes = C.auto_place(shape, spec, fit, 2)
    assert len(pls) == 4 and notes == []  # both thin walls get connectors, via bosses in the cavity
    radii = sorted(math.hypot(*fr.to_world(np.array([[p.u, p.v, 0]]))[0][:2]) for p in pls)
    assert all(17 < r < 20 for r in radii[:2]) and all(30 < r < 33 for r in radii[2:])
    out, _, warnings = C.apply_plan({"A": a, "B": b}, C.plan(spec, fit, pls, shape), fr)
    assert warnings == []
    middle = m3d.Manifold.cylinder(60, 14.999, 14.999, 128).translate((0, 0, -30))
    for m in out.values():
        assert analyze(m)[0].printable
        assert (to_manifold(m) ^ middle).volume() < 1e-3
        assert (to_manifold(m) - outer).volume() < 1e-3
    assert overlap(out["A"], out["B"]) < 1e-6


def test_curved_shell_wall_gets_no_boss_that_would_poke_out():
    """Next to a strongly curved wall a boss would come out of the surface further down: none is placed."""
    outer = m3d.Manifold.revolve(m3d.CrossSection.circle(10, 64).translate((25, 0)), 128)
    inner = m3d.Manifold.revolve(m3d.CrossSection.circle(8, 48).translate((25, 0)), 96)
    a, b = split(weld_mesh(from_manifold(outer - inner), 0), z=0.0)
    fr, shape = joint(a, b)
    spec, fit = C.Spec("dowel", 3.0, 6.0), C.Fit.preset("resin", "snug")
    pls, notes = C.auto_place(shape, spec, fit, 2)
    for p in pls:
        out, _, warnings = C.apply_plan({"A": a, "B": b}, C.plan(spec, fit, [p], shape), fr)
        assert warnings == []
        for m in out.values():
            assert (to_manifold(m) - outer).volume() < 1e-3
