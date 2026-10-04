import math

import manifold3d as m3d
import numpy as np
import pytest

from conftest import from_manifold
from phoenix_stl.core import decal as D
from phoenix_stl.core.analyze import analyze


def block_image():
    """A bar (left) and a T (right); 19.5 % of the image is on."""
    img = np.zeros((100, 200))
    img[20:80, 20:60] = 1.0
    img[20:30, 80:180] = 1.0
    img[30:80, 125:135] = 1.0
    return img


@pytest.fixture
def box():
    return from_manifold(m3d.Manifold.cube((60, 40, 20), True))


def test_frame_is_right_handed_and_follows_world_up():
    f = D.oriented_frame((0, 0, 0), (0, -1, 0), 0.0)
    assert np.allclose(f.v, (0, 0, 1)) and np.allclose(f.u, (1, 0, 0))  # seen from the front: right = +X
    assert np.allclose(np.cross(f.u, f.v), f.w)
    f = D.oriented_frame((0, 0, 0), (0, 0, 1), 90.0)  # top face, turned 90 degrees
    assert np.allclose(np.cross(f.u, f.v), f.w) and abs(f.u @ f.v) < 1e-12


def test_prepare_image():
    img = np.linspace(0, 255, 11)[None, :].repeat(3, 0)
    logo = D.prepare_image(img, D.DecalParams(threshold=0.5))
    assert logo[0, 0] == 0 and logo[0, -1] == 1 and logo[0, 5] == pytest.approx(0.5)
    inv = D.prepare_image(img, D.DecalParams(threshold=0.5, invert=True))
    assert inv[0, 0] == 1 and inv[0, -1] == 0
    tex = D.prepare_image(img, D.DecalParams(kind="texture"))
    assert np.allclose(tex[0], np.linspace(0, 1, 11))


def test_params_roundtrip():
    p = D.DecalParams(width=12, mode="deboss", kind="texture", projection="cylinder", tile=3)
    assert D.DecalParams.from_dict(p.to_dict()) == p


@pytest.mark.parametrize("mode", ["emboss", "deboss"])
def test_flat_logo(box, mode):
    p = D.DecalParams(width=40, height=20, depth=1.0, mode=mode, resolution=0.2)
    out, info = D.apply_decal(box, D.oriented_frame((0, 0, 10), (0, 0, 1), 0.0), block_image(), p)
    assert analyze(out)[0].printable
    sign = -1 if mode == "deboss" else 1
    expected = 0.195 * 40 * 20 * 1.0
    assert sign * info["volume_change"] == pytest.approx(expected, rel=0.06)
    if mode == "emboss":
        assert out.bounds[1][2] == pytest.approx(11.0, abs=0.02)
    else:
        top = out.vertices[np.abs(out.vertices[:, 0]) < 19]
        assert top[:, 2].min() == pytest.approx(9.0, abs=0.02)
    assert out.bounds[1][0] == pytest.approx(30.0) and out.bounds[0][2] == pytest.approx(-10.0)


def test_relief_height_is_measured_along_a_curved_surface():
    ball = from_manifold(m3d.Manifold.sphere(20, 128))
    p = D.DecalParams(width=24, height=12, depth=0.8, resolution=0.15)
    out, info = D.apply_decal(ball, D.oriented_frame((0, 0, 20), (0, 0, 1), 0.0), block_image(), p)
    assert analyze(out)[0].printable and info["volume_change"] > 0
    r = np.linalg.norm(out.vertices, axis=1)
    assert r.max() == pytest.approx(20.8, abs=0.03)  # 0.8 mm above the sphere everywhere
    # the relief is in the frame only
    far = out.vertices[r > 20.05]
    assert np.all(np.abs(far[:, 0]) < 12.5) and np.all(np.abs(far[:, 1]) < 6.5) and np.all(far[:, 2] > 15)


def test_wrap_around_a_cylinder():
    cyl = from_manifold(m3d.Manifold.cylinder(40, 15, 15, 128, True))
    p = D.DecalParams(width=40, height=20, depth=0.8, projection="cylinder", resolution=0.15)
    out, info = D.apply_decal(cyl, D.oriented_frame((15, 0, 0), (1, 0, 0), 0.0), block_image(), p)
    assert analyze(out)[0].printable
    assert info["radius"] == pytest.approx(15.0, rel=0.01)
    rho = np.hypot(out.vertices[:, 0], out.vertices[:, 1])
    assert rho.max() == pytest.approx(15.8, abs=0.03)
    # the image is on from 10 % to 90 % of the 40 mm width: 16 mm of arc each side, 61 degrees
    raised = out.vertices[rho > 15.4]
    ang = np.degrees(np.abs(np.arctan2(raised[:, 1], raised[:, 0])))
    assert ang.max() == pytest.approx(math.degrees(16 / 15), abs=0.6)
    # flat volume estimate, a little more because the relief sits on the outside of the curve
    assert info["volume_change"] == pytest.approx(0.195 * 40 * 20 * 0.8, rel=0.08)


def test_texture_keeps_smooth_low_points(box):
    yy, xx = np.mgrid[0:100, 0:100] / 100.0
    tex = 0.5 + 0.5 * np.sin(xx * 2 * math.pi) * np.cos(yy * 2 * math.pi)
    p = D.DecalParams(width=20, height=20, depth=0.6, kind="texture", tile=2, resolution=0.2)
    out, info = D.apply_decal(box, D.oriented_frame((0, 0, 10), (0, 0, 1), 0.0), tex, p)
    assert analyze(out)[0].printable
    assert 0.3 * 400 * 0.6 < info["volume_change"] < 0.7 * 400 * 0.6
    inside = out.vertices[(np.abs(out.vertices[:, 0]) < 9.5) & (np.abs(out.vertices[:, 1]) < 9.5)]
    assert inside[:, 2].min() > 10.0  # dark parts keep a thin layer: no pits down to the surface


def test_errors(box):
    p = D.DecalParams(resolution=0.3)
    with pytest.raises(ValueError):  # the frame is far from the model
        D.apply_decal(box, D.oriented_frame((0, 0, 200), (0, 0, 1), 0.0), block_image(), p)
    with pytest.raises(ValueError):  # nothing to raise
        D.apply_decal(box, D.oriented_frame((0, 0, 10), (0, 0, 1), 0.0), np.zeros((10, 10)), p)
