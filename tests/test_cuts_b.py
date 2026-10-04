import numpy as np

from conftest import is_valid_solid
from phoenix_stl.core.analyze import analyze
from phoenix_stl.core.grid import cell_name, fit_counts, fit_planes, grid_cut
from phoenix_stl.core.io_stl import load_stl, save_stl
from phoenix_stl.core.surface_cut import (Frame, curve_cutters, cut_with, curve_values, height_function,
                                          heightfield_solid, heightfield_values, level_segments, smooth_curve)


def test_fit_counts_saturn():
    vol = (211.68, 118.37, 220.0)
    assert fit_counts((100, 100, 100), vol, 5) == ((1, 1, 1), False)
    counts, rotated = fit_counts((300, 100, 100), vol, 5)
    assert counts[0] * counts[1] * counts[2] == 2
    counts, rotated = fit_counts((100, 200, 50), vol, 5)  # fits when turned 90 degrees
    assert counts == (1, 1, 1) and rotated
    assert fit_counts((100, 100, 500), vol, 5)[0][2] == 3


def test_grid_cut_volumes_and_validity(tmp_path, torus):
    planes = [(0, 0.0), (1, -3.0), (1, 4.0), (2, 0.5)]
    pieces = grid_cut(torus, planes)
    total = sum(m.volume() for _, m in pieces)
    assert np.isclose(total, torus.volume(), rtol=1e-6)
    cells = {idx for idx, _ in pieces}
    assert all(0 <= i <= 1 and 0 <= j <= 2 and 0 <= k <= 1 for i, j, k in cells)
    for idx, m in pieces:
        p = str(tmp_path / "c.stl")
        save_stl(m, p)
        assert is_valid_solid(load_stl(p)) and analyze(load_stl(p))[0].printable
    assert cell_name("ring", (1, 2, 0), (2, 3, 1)) == "ring_x2_y3"


def test_fit_planes_pieces_fit(sphere):
    big = type(sphere)(sphere.vertices * 12, sphere.faces)  # 240 mm ball
    vol = (211.68, 118.37, 220.0)
    planes, counts, rotated = fit_planes(big.bounds, vol, 5)
    pieces = grid_cut(big, planes)
    for _, m in pieces:
        s = sorted(m.size[:2])
        assert m.size[2] <= 215 + 1e-6 and s[0] <= 108.37 + 1e-6 and s[1] <= 201.68 + 1e-6


def test_curve_cut(tmp_path, sphere):
    frame = Frame.from_normal((0, 0, 0), (0, 0, 1))  # draw looking down
    pts = np.array([[-15, -3], [-5, 2], [0, -2], [5, 3], [15, -1]], float)
    curve = smooth_curve(pts, corners=[2])
    a_cut, b_cut = curve_cutters(curve, frame, reach=60.0, gap=0.0)
    a, b = cut_with(sphere, a_cut, b_cut)
    assert np.isclose(a.volume() + b.volume(), sphere.volume(), rtol=1e-4)
    for m in (a, b):
        assert analyze(m)[0].printable and is_valid_solid(m)
    # with a gap the two sides lose a thin band
    a2, b2 = cut_with(sphere, *curve_cutters(curve, frame, reach=60.0, gap=0.2))
    assert a2.volume() + b2.volume() < sphere.volume() - 1.0
    vals = curve_values(sphere.vertices, frame, curve, 60.0)
    assert (vals < 0).any() and (vals > 0).any()
    assert len(level_segments(sphere, vals)) > 20


def test_freeform_cut(tmp_path, sphere):
    frame = Frame.from_normal((0, 0, 0), (0, 0, 1))
    heights = np.zeros((4, 4))
    heights[1, 2] = 4.0
    heights[2, 1] = -3.0
    hf = height_function(heights, (-12, 12), (-12, 12))
    solid = heightfield_solid(hf, frame, (-12, 12), (-12, 12), depth=40.0, res=48)
    assert analyze(solid)[0].printable
    a, b = cut_with(sphere, solid)
    assert np.isclose(a.volume() + b.volume(), sphere.volume(), rtol=1e-4)
    for m in (a, b):
        assert analyze(m)[0].printable
    vals = heightfield_values(sphere.vertices, frame, hf)
    assert len(level_segments(sphere, vals)) > 20
