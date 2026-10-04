"""Cuts along a drawn curve or a free-form (bendable) surface.

Both build a closed "cutter" solid on one side of the cut surface; the part is
then intersected with it (side A) and the rest is kept (side B), with an
optional gap so curved joints fit after printing. Booleans run through
manifold3d, so both sides come out closed and printable.
"""
from __future__ import annotations

from dataclasses import dataclass

import manifold3d as m3d
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .boolean import from_manifold, to_manifold
from .mesh import Mesh, report


@dataclass
class Frame:
    """Orthonormal frame: u, v span the drawing/base plane, w is its normal."""
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray

    @classmethod
    def from_normal(cls, origin, normal) -> "Frame":
        w = np.asarray(normal, float)
        w /= np.linalg.norm(w)
        helper = np.array([1.0, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(helper, w)
        u /= np.linalg.norm(u)
        return cls(np.asarray(origin, float), u, np.cross(w, u), w)

    def to_local(self, pts) -> np.ndarray:
        d = np.asarray(pts, float) - self.origin
        return np.stack([d @ self.u, d @ self.v, d @ self.w], -1)

    def to_world(self, local) -> np.ndarray:
        local = np.asarray(local, float)
        return self.origin + local[..., :1] * self.u + local[..., 1:2] * self.v + local[..., 2:3] * self.w

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = self.u, self.v, self.w, self.origin
        return m


# -- curves ------------------------------------------------------------------------

def smooth_curve(points: np.ndarray, corners=None, samples: int = 12) -> np.ndarray:
    """Catmull-Rom spline through 2D points; corner points stay sharp."""
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return pts
    corners = set(corners or [])
    out = [pts[0]]
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 and i not in corners else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) and (i + 1) not in corners else pts[i + 1]
        t = np.linspace(0, 1, samples + 1)[1:, None]
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        out.extend(seg)
    return np.array(out)


def curve_region(curve: np.ndarray, reach: float) -> np.ndarray:
    """Closed 2D polygon covering the left side of an open curve.

    The ends are extended along their tangents far past the part, then the
    polygon closes far out on the left.
    """
    c = np.asarray(curve, float)
    t0 = c[1] - c[0]
    t1 = c[-1] - c[-2]
    t0 /= np.linalg.norm(t0)
    t1 /= np.linalg.norm(t1)
    start = c[0] - t0 * reach
    end = c[-1] + t1 * reach
    left = lambda t: np.array([-t[1], t[0]])  # noqa: E731
    far = 2.0 * reach
    return np.vstack([start, c, end, end + left(t1) * far, start + left(t0) * far])


def prism(poly2d: np.ndarray, frame: Frame, depth: float) -> m3d.Manifold:
    """Extrude a 2D region (frame u, v) along w from -depth to +depth."""
    cs = m3d.CrossSection([np.asarray(poly2d, float)], m3d.FillRule.NonZero)
    if cs.area() < 0:
        cs = m3d.CrossSection([np.asarray(poly2d, float)[::-1]], m3d.FillRule.NonZero)
    solid = cs.extrude(2 * depth).translate((0, 0, -depth))
    return solid.transform(frame.matrix()[:3, :])


def curve_cutters(curve2d: np.ndarray, frame: Frame, reach: float, gap: float = 0.0):
    """(cutter for side A, cutter for side B) for a curve drawn in frame's plane."""
    region = curve_region(curve2d, reach)
    a = prism(region, frame, reach)
    if gap <= 0:
        return a, a
    cs = m3d.CrossSection([region], m3d.FillRule.NonZero)
    grown = cs.offset(gap, m3d.JoinType.Round)
    b = grown.extrude(2 * reach).translate((0, 0, -reach)).transform(frame.matrix()[:3, :])
    return a, b


# -- free-form surface -----------------------------------------------------------------

def height_function(heights: np.ndarray, u_range, v_range):
    """Smooth h(u, v) through a k x k grid of control heights."""
    h = np.asarray(heights, float)
    ku, kv = h.shape
    us = np.linspace(u_range[0], u_range[1], ku)
    vs = np.linspace(v_range[0], v_range[1], kv)
    spline = RectBivariateSpline(us, vs, h, kx=min(3, ku - 1), ky=min(3, kv - 1))

    def f(u, v):
        uu = np.clip(u, u_range[0], u_range[1])
        vv = np.clip(v, v_range[0], v_range[1])
        return spline.ev(uu, vv)
    return f


def heightfield_solid(hf, frame: Frame, u_range, v_range, depth: float, res: int = 96,
                      shift: float = 0.0) -> Mesh:
    """Closed solid below the surface w = h(u, v) + shift, down to w = -depth."""
    us = np.linspace(u_range[0], u_range[1], res)
    vs = np.linspace(v_range[0], v_range[1], res)
    U, V = np.meshgrid(us, vs, indexing="ij")
    top = np.stack([U, V, hf(U, V) + shift], -1).reshape(-1, 3)
    bot = np.stack([U, V, np.full_like(U, -depth)], -1).reshape(-1, 3)
    n = res * res
    idx = np.arange(n).reshape(res, res)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    quads_top = [np.stack([a, b, c], -1), np.stack([a, c, d], -1)]
    faces = [q.reshape(-1, 3) for q in quads_top]  # top faces point +w
    faces += [q.reshape(-1, 3)[:, ::-1] + n for q in quads_top]  # bottom faces point -w
    # side walls along the four borders (outward)
    for line, flip in ((idx[0, :], False), (idx[-1, :], True), (idx[:, 0], True), (idx[:, -1], False)):
        t0, t1 = line[:-1], line[1:]
        quad = [np.stack([t0, t1, t1 + n], -1), np.stack([t0, t1 + n, t0 + n], -1)]
        for q in quad:
            faces.append(q if not flip else q[:, ::-1])
    mesh = Mesh(frame.to_world(np.vstack([top, bot])), np.concatenate(faces))
    if mesh.volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1])
    return mesh


# -- applying a cutter ------------------------------------------------------------------

def cut_with(mesh: Mesh, cutter_a, cutter_b=None, progress=None):
    """(side A = mesh ∩ cutter_a, side B = mesh − cutter_b)."""
    cutter_b = cutter_a if cutter_b is None else cutter_b
    ca = cutter_a if isinstance(cutter_a, m3d.Manifold) else to_manifold(cutter_a)
    cb = cutter_b if isinstance(cutter_b, m3d.Manifold) else to_manifold(cutter_b)
    report(progress, 0.1, "Preparing cut")
    part = to_manifold(mesh)
    report(progress, 0.35, "Cutting")
    a = from_manifold(part ^ ca)
    report(progress, 0.7, "Cutting")
    b = from_manifold(part - cb)
    return a, b


# -- live preview: zero level of a per-vertex value ---------------------------------------

def level_segments(mesh: Mesh, values: np.ndarray) -> np.ndarray:
    """Segments where a per-vertex scalar crosses zero, as (k, 2, 3) float32."""
    pos = values >= 0
    s = pos[mesh.faces]
    cnt = s.sum(1)
    st = (cnt == 1) | (cnt == 2)
    f = mesh.faces[st]
    s = s[st]
    lonely = np.where(cnt[st] == 1, np.argmax(s, 1), np.argmin(s, 1))
    rot = (lonely[:, None] + np.arange(3)[None, :]) % 3
    t = np.take_along_axis(f, rot, 1)
    v = mesh.vertices.astype(np.float64)

    def point(a, b):
        da, db = values[a], values[b]
        w = (da / (da - db))[:, None]
        return v[a] + w * (v[b] - v[a])

    return np.stack([point(t[:, 0], t[:, 1]), point(t[:, 0], t[:, 2])], 1).astype(np.float32)


def heightfield_values(points, frame: Frame, hf) -> np.ndarray:
    loc = frame.to_local(points)
    return loc[:, 2] - hf(loc[:, 0], loc[:, 1])


def curve_values(points, frame: Frame, curve2d: np.ndarray, reach: float) -> np.ndarray:
    """Positive on side B, negative on side A (left of the curve)."""
    loc = frame.to_local(points)[:, :2]
    poly = curve_region(curve2d, reach)
    inside = _points_in_polygon(loc, poly)
    return np.where(inside, -1.0, 1.0) * (_dist_to_polyline(loc, curve2d) + 1e-9)


def _points_in_polygon(p, poly) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    inside = np.zeros(len(p), dtype=bool)
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    for a, b, c, d in zip(xi, yi, xj, yj):
        cross = (b > y) != (d > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (c - a) * (y - b) / (d - b) + a
        inside ^= cross & (x < xint)
    return inside


def _dist_to_polyline(p, line) -> np.ndarray:
    best = np.full(len(p), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        L = float(ab @ ab) or 1e-12
        t = np.clip(((p - a) @ ab) / L, 0, 1)
        q = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(p - q, axis=1))
    return best
