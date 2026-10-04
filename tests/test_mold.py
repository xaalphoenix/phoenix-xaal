import math

import manifold3d as m3d
import pytest

from conftest import from_manifold
from phoenix_stl.core import mold as M
from phoenix_stl.core.analyze import analyze
from phoenix_stl.core.boolean import to_manifold
from phoenix_stl.core.sdf import DistanceField, iso_surface, shell_values


def overlap(a, b) -> float:
    return (to_manifold(a) ^ to_manifold(b)).volume()


@pytest.fixture
def ball():
    return from_manifold(m3d.Manifold.sphere(10, 96))


@pytest.fixture
def pillars():
    """Two pillars of different height on a block: the low one needs a vent."""
    base = m3d.Manifold.cube((40, 16, 6)).translate((-20, -8, 0))
    tall = m3d.Manifold.cylinder(30, 4, 4, 48).translate((-12, 0, 5))
    low = m3d.Manifold.cylinder(18, 4, 4, 48).translate((12, 0, 5))
    return from_manifold(base + tall + low)


def test_field_and_offsets_are_exact(ball):
    f = DistanceField.build(ball, 0.75, pad=12)
    f.refine([5.0, 8.0], band=1.5)
    shell = f.snap(iso_surface(shell_values(f.dist, 5.0, 8.0), f.grid), [5.0, 8.0])
    assert analyze(shell)[0].printable
    d = f.exact(shell.vertices)
    inner, outer = d[d < 6.5], d[d > 6.5]
    assert abs(inner.min() - 5) < 0.005 and abs(inner.max() - 5) < 0.005
    assert abs(outer.min() - 8) < 0.005 and abs(outer.max() - 8) < 0.005
    # inside/outside of the voxel centres matches the solid volume
    assert f.inside.sum() * 0.75 ** 3 == pytest.approx(ball.volume(), rel=0.02)


def test_mold_for_a_ball(ball):
    p = M.MoldParams(thickness=5.0, wall=3.0, resolution=0.75, feet=3)
    res = M.build(ball, p)
    assert set(res.parts) == {"A", "B", "base"}
    for m in res.parts.values():
        assert analyze(m)[0].printable
    assert overlap(res.parts["A"], res.parts["B"]) < 1e-3
    for m in res.parts.values():
        assert overlap(m, ball) < 1e-3
    th = res.info["thickness"]
    assert abs(th["min"] - 5) < 0.01 and abs(th["max"] - 5) < 0.01
    # silicone: offset sphere r=15 above the base plane (z = -10) minus the ball
    r, z0 = 15.0, -10.0
    cap = math.pi * (r + z0) ** 2 * (3 * r - (r + z0)) / 3
    expected = (4 / 3 * math.pi * r ** 3 - cap - ball.volume()) / 1000
    assert res.info["silicone_ml"] == pytest.approx(expected, rel=0.01)
    assert res.info["silicone_g"] > res.info["silicone_ml"]
    assert res.info["sprue"] is not None and len(res.info["feet"]) == 3
    assert res.info["bolts"] >= 2 and res.info["keys"] >= 1


def test_mold_for_pillars_has_a_vent_and_stands(pillars):
    p = M.MoldParams(resolution=1.0, feet=4)
    res = M.build(pillars, p)
    assert res.warnings == []
    assert len(res.info["vents"]) == 1  # the low pillar; the tall one gets the pour funnel
    vx, vy = res.info["vents"][0]
    sx, sy = res.info["sprue"]
    assert vx > 5 and sx < -5
    for m in res.parts.values():
        assert analyze(m)[0].printable and overlap(m, pillars) < 1e-3
    # turned over, the mold stands on its feet: they all end at one height
    tops = [res.parts[k].bounds[1][2] for k in ("A", "B")]
    assert abs(tops[0] - tops[1]) < 1e-3


def test_options_off(ball):
    p = M.MoldParams(resolution=1.0, split=False, base_plate=False, feet=0, funnel=False, vents=[])
    res = M.build(ball, p)
    assert set(res.parts) == {"shell"}
    assert analyze(res.parts["shell"])[0].printable
    assert res.parts["shell"].bounds[0][2] == pytest.approx(-10.0, abs=0.05)  # no groove without a plate


def test_params_roundtrip():
    p = M.MoldParams(sprue=(1.0, 2.0), vents=[(3.0, 4.0)], bolt="M4")
    q = M.MoldParams.from_dict(p.to_dict())
    assert q == p


def test_preview_is_quick_and_rough(ball):
    r = M.preview(ball, M.MoldParams(resolution=2.0))
    assert r["shell"].n_faces > 0 and r["sprue"] is not None
    assert r["silicone_ml"] == pytest.approx(8.9, rel=0.15)
