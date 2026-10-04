"""Distance fields on a voxel grid around a mesh, and iso-surfaces of them.

Used for the silicone mother mold: the shell surfaces lie at exact distances
from the model. Steps:

1. inside/outside of every voxel centre, by counting surface crossings along
   vertical rays (exact, from the full mesh);
2. a quick distance estimate everywhere (Euclidean distance transform);
3. exact distances near the levels that matter, from a k-d tree of dense
   surface samples (error far below a voxel);
4. iso-surfaces by marching cubes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .mesh import Mesh, report
from .weld import weld_mesh

# Rays are shifted by these fractions of a voxel so they never run exactly
# through mesh edges or vertices (which would count a crossing twice).
_SHIFT = (0.000137 * math.pi, 0.000113 * math.e)


@dataclass
class Grid:
    origin: np.ndarray  # centre of voxel (0, 0, 0)
    h: float
    shape: tuple

    @classmethod
    def around(cls, bounds, h: float, pad: float) -> "Grid":
        b = np.asarray(bounds, float)
        # shifted by odd fractions of a voxel: iso-surface vertices lie on grid planes,
        # and those must not line up with round coordinates (parting planes, flanges)
        lo = b[0] - pad - h * np.array([0.3719, 0.6137, 0.2531])
        n = np.ceil((b[1] - lo + pad) / h).astype(int) + 1
        return cls(lo, float(h), tuple(int(k) for k in n))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    def axis(self, k: int) -> np.ndarray:
        return self.origin[k] + self.h * np.arange(self.shape[k])

    def points(self, idx) -> np.ndarray:
        """World positions of voxel centres given as (k, 3) integer indices."""
        return self.origin + self.h * np.asarray(idx, float)


def estimate_bytes(bounds, h: float, pad: float) -> int:
    """Rough peak memory of building a field (bool grid, distance transform, floats)."""
    g = Grid.around(bounds, h, pad)
    return int(g.n_voxels * 26)


# -- inside / outside -------------------------------------------------------------------

def occupancy(mesh: Mesh, grid: Grid, progress=None, max_pairs: int = 4_000_000) -> np.ndarray:
    """Boolean grid: True where the voxel centre is inside the (closed) mesh."""
    nx, ny, nz = grid.shape
    ox, oy, oz = grid.origin
    h = grid.h
    sx, sy = _SHIFT[0] * h, _SHIFT[1] * h
    hits = np.zeros((nx, ny, nz + 1), dtype=np.uint8)
    v = mesh.vertices.astype(np.float64)
    f = mesh.faces
    # triangles per chunk so that (triangle, column) pairs stay bounded
    tri = v[f]
    xmin, xmax = tri[:, :, 0].min(1), tri[:, :, 0].max(1)
    ymin, ymax = tri[:, :, 1].min(1), tri[:, :, 1].max(1)
    i0 = np.ceil((xmin - ox - sx) / h).astype(np.int64)
    i1 = np.floor((xmax - ox - sx) / h).astype(np.int64)
    j0 = np.ceil((ymin - oy - sy) / h).astype(np.int64)
    j1 = np.floor((ymax - oy - sy) / h).astype(np.int64)
    ni = np.clip(i1 - i0 + 1, 0, None)
    nj = np.clip(j1 - j0 + 1, 0, None)
    count = ni * nj
    keep = np.nonzero(count)[0]
    csum = np.cumsum(count[keep])
    start = 0
    total = int(csum[-1]) if len(csum) else 0
    while start < len(keep):
        base = csum[start - 1] if start else 0
        end = int(np.searchsorted(csum, base + max_pairs, side="right"))
        end = max(end, start + 1)
        t_idx = keep[start:end]
        c = count[t_idx]
        rep = np.repeat(np.arange(len(t_idx)), c)
        local = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        nj_r = nj[t_idx][rep]
        ii = i0[t_idx][rep] + local // nj_r
        jj = j0[t_idx][rep] + local % nj_r
        px = ox + ii * h + sx
        py = oy + jj * h + sy
        T = tri[t_idx][rep]
        ax, ay = T[:, 0, 0], T[:, 0, 1]
        e0x, e0y = T[:, 1, 0] - ax, T[:, 1, 1] - ay
        e1x, e1y = T[:, 2, 0] - ax, T[:, 2, 1] - ay
        qx, qy = px - ax, py - ay
        den = e0x * e1y - e1x * e0y
        ok = np.abs(den) > 1e-300
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = (qx * e1y - e1x * qy) / den
            l2 = (e0x * qy - qx * e0y) / den
            l0 = 1.0 - l1 - l2
            inside = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            z = l0 * T[:, 0, 2] + l1 * T[:, 1, 2] + l2 * T[:, 2, 2]
        ii, jj, z = ii[inside], jj[inside], z[inside]
        valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        iz = np.clip(np.ceil((z[valid] - oz) / h).astype(np.int64), 0, nz)
        np.add.at(hits, (ii[valid], jj[valid], iz), 1)
        start = end
        report(progress, 0.6 * min(1.0, float(csum[end - 1]) / max(total, 1)), "Measuring the model")
    np.cumsum(hits, axis=2, dtype=np.uint8, out=hits)
    return (hits[:, :, :nz] & 1).astype(bool)


# -- distances ----------------------------------------------------------------------------

def surface_samples(mesh: Mesh, spacing: float, limit: int = 8_000_000) -> np.ndarray:
    """Points covering the surface with gaps of about `spacing` (count follows the area,
    so long thin triangles cost little)."""
    v = mesh.vertices.astype(np.float64)
    tri = v[mesh.faces]
    # lattice along the two edges from the corner opposite the longest edge
    lens = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2)  # edge i: corner i -> i+1
    corner = (np.argmax(lens, axis=1) + 2) % 3
    rot = (corner[:, None] + np.arange(3)[None, :]) % 3
    tri = np.take_along_axis(tri, rot[:, :, None], axis=1)
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
    est = len(v) + float(area.sum()) / (0.5 * spacing ** 2)
    if est > limit:  # keep memory bounded on huge surfaces
        spacing *= math.sqrt(est / limit)
    k1 = np.clip(np.ceil(np.linalg.norm(e0, axis=1) / spacing), 1, 256).astype(np.int64)
    k2 = np.clip(np.ceil(np.linalg.norm(e1, axis=1) / spacing), 1, 256).astype(np.int64)
    out = [v]
    big = (k1 > 1) | (k2 > 1)
    keys = k1[big] * 1000 + k2[big]
    sel_all = np.nonzero(big)[0]
    order = np.argsort(keys, kind="stable")
    keys, sel_all = keys[order], sel_all[order]
    bounds = np.r_[0, np.nonzero(np.diff(keys))[0] + 1, len(keys)]
    for a0, a1 in zip(bounds[:-1], bounds[1:]):
        sel = sel_all[a0:a1]
        if not len(sel):
            continue
        n1, n2 = int(k1[sel[0]]), int(k2[sel[0]])
        A, B = np.meshgrid(np.arange(n1 + 1) / n1, np.arange(n2 + 1) / n2, indexing="ij")
        m = (A + B) <= 1 + 1e-9
        wa, wb = A[m], B[m]
        pts = (tri[sel, 0][:, None, :] + wa[None, :, None] * e0[sel][:, None, :]
               + wb[None, :, None] * e1[sel][:, None, :])
        out.append(pts.reshape(-1, 3))
    return np.vstack(out)


@dataclass
class DistanceField:
    """Distance from each voxel centre to the model surface (0 inside the model)."""
    grid: Grid
    inside: np.ndarray
    dist: np.ndarray  # float32, mm
    tree: cKDTree | None = None

    @classmethod
    def build(cls, mesh: Mesh, h: float, pad: float, exact: bool = True, progress=None) -> "DistanceField":
        grid = Grid.around(mesh.bounds, h, pad)
        inside = occupancy(mesh, grid, progress)
        report(progress, 0.65, "Measuring distances")
        # distance from outside voxel centres to the nearest inside centre; the surface
        # lies on average half a voxel closer
        d = ndimage.distance_transform_edt(~inside, sampling=h).astype(np.float32)
        d = np.where(inside, 0.0, np.maximum(d - 0.5 * h, 0.0)).astype(np.float32)
        tree = None
        if exact:
            report(progress, 0.8, "Measuring distances")
            tree = cKDTree(surface_samples(mesh, 0.5 * h))
        return cls(grid, inside, d, tree)

    def refine(self, levels, band: float | None = None, progress=None) -> None:
        """Exact distances for voxels within `band` of any of the given levels."""
        if self.tree is None:
            return
        band = 1.75 * self.grid.h if band is None else band
        sel = np.zeros(self.dist.shape, dtype=bool)
        for lv in levels:
            sel |= np.abs(self.dist - lv) < band
        sel &= ~self.inside
        idx = np.argwhere(sel)
        bound = float(max(levels)) + band + 2.0 * self.grid.h
        for s in range(0, len(idx), 2_000_000):
            chunk = idx[s:s + 2_000_000]
            dd, _ = self.tree.query(self.grid.points(chunk), workers=-1, distance_upper_bound=bound)
            dd = np.where(np.isfinite(dd), dd, self.dist[tuple(chunk.T)])
            self.dist[tuple(chunk.T)] = dd.astype(np.float32)
            report(progress, 0.85 + 0.15 * min(1.0, (s + len(chunk)) / max(len(idx), 1)), "Measuring distances")

    def snap(self, mesh: Mesh, levels, max_move: float | None = None) -> Mesh:
        """Move every vertex of an iso-surface exactly onto the nearest of `levels`.

        A point at distance d from its nearest surface point p is moved along
        the line from p, to distance `level`: exact offsets regardless of the
        voxel size (moves are capped at about a voxel).
        """
        if self.tree is None or mesh.n_vertices == 0:
            return mesh
        levels = np.asarray(sorted(levels), float)
        max_move = 0.75 * self.grid.h if max_move is None else max_move
        v = mesh.vertices.astype(np.float64)
        # every vertex is within about a voxel of a level: bounding the search makes it fast
        d, idx = self.tree.query(v, workers=-1, distance_upper_bound=float(levels[-1]) + 2.5 * self.grid.h)
        miss = ~np.isfinite(d)
        if miss.any():
            d[miss], idx[miss] = self.tree.query(v[miss], workers=-1)
        near = self.tree.data[idx]
        target = levels[np.argmin(np.abs(d[:, None] - levels[None, :]), axis=1)]
        ok = d > 1e-9
        move = np.clip(target - d, -max_move, max_move)
        direction = np.zeros_like(v)
        direction[ok] = (v[ok] - near[ok]) / d[ok, None]
        out = v + direction * move[:, None]
        return Mesh(out, mesh.faces)

    def exact(self, pts) -> np.ndarray:
        """Exact distance to the model surface for arbitrary points."""
        if self.tree is None:
            raise ValueError("field was built without exact distances")
        return self.tree.query(np.asarray(pts, float).reshape(-1, 3), workers=-1)[0]

    def column_top(self, x: float, y: float, level: float) -> float | None:
        """Highest z in the column at (x, y) where the distance is still below `level`."""
        g = self.grid
        i = int(round((x - g.origin[0]) / g.h))
        j = int(round((y - g.origin[1]) / g.h))
        if not (0 <= i < g.shape[0] and 0 <= j < g.shape[1]):
            return None
        col = self.dist[i, j]
        below = np.nonzero(col < level)[0]
        if not len(below):
            return None
        k = below[-1]
        z = g.origin[2] + k * g.h
        if k + 1 < len(col):  # interpolate to the crossing
            a, b = float(col[k]), float(col[k + 1])
            if b != a:
                z += g.h * (level - a) / (b - a)
        return float(z)


# -- iso-surfaces ---------------------------------------------------------------------------

def iso_surface(values: np.ndarray, grid: Grid, level: float = 0.0) -> Mesh:
    """Closed surface where `values` crosses `level`, oriented so the solid is where values < level."""
    from skimage.measure._marching_cubes_lewiner import marching_cubes

    v = np.pad(values.astype(np.float32), 1, constant_values=float(level) + 1e3)
    if not (v < level).any():
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3)))
    verts, faces, _, _ = marching_cubes(v, level=level, spacing=(grid.h,) * 3, gradient_direction="ascent",
                                        allow_degenerate=False)
    verts += grid.origin - grid.h
    mesh = weld_mesh(Mesh(verts, faces), 0.0)
    if mesh.volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1])
    return mesh


def shell_values(dist: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """Negative between the two distances (the shell), positive elsewhere."""
    c = 0.5 * (inner + outer)
    return np.abs(dist - c) - 0.5 * (outer - inner)
