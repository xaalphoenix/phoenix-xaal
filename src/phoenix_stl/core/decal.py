"""Images and text pressed into or raised out of a model's surface.

A frame is placed on the surface (centre, normal, direction, width and
height). The image is projected onto the surface along the frame's normal
(or wrapped around a cylinder), and its brightness becomes a height above
(emboss) or depth below (deboss) the surface, measured along the surface
itself, so the relief is equally deep on curved parts.

Built on a fine local voxel grid around the frame only: exact distances to
the model's triangles, inside/outside by ray parity, marching cubes, then a
boolean with the model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .boolean import from_manifold, to_manifold
from .mesh import Mesh, report
from .sdf import Grid, iso_surface, surface_samples
from .surface_cut import Frame


@dataclass
class DecalParams:
    width: float = 30.0
    height: float = 15.0
    rotation: float = 0.0  # degrees around the surface normal
    depth: float = 0.8  # mm above (emboss) or below (deboss) the surface
    mode: str = "emboss"  # "emboss" or "deboss"
    kind: str = "logo"  # "logo": sharp two-level relief; "texture": brightness is height
    threshold: float = 0.5
    invert: bool = False
    blur: float = 0.0  # pixels
    tile: int = 1  # repeats across the frame
    projection: str = "flat"  # "flat" or "cylinder"
    radius: float = 0.0  # cylinder radius; 0: measured from the surface
    resolution: float = 0.1  # voxel size (mm)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecalParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def oriented_frame(origin, normal, rotation: float, up_hint=(0.0, 0.0, 1.0)) -> Frame:
    """Frame on the surface: w = outward normal; u = image right, v = image up.

    Image up follows the world up direction where possible (then rotated).
    """
    w = np.asarray(normal, float)
    w = w / np.linalg.norm(w)
    up = np.asarray(up_hint, float)
    if abs(up @ w) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    v = up - (up @ w) * w
    v /= np.linalg.norm(v)
    u = np.cross(v, w)
    a = math.radians(rotation)
    u, v = math.cos(a) * u + math.sin(a) * v, -math.sin(a) * u + math.cos(a) * v
    return Frame(np.asarray(origin, float), u, v, w)


def prepare_image(img: np.ndarray, p: DecalParams) -> np.ndarray:
    """Grayscale image (rows top to bottom, 0..1) to relief heights 0..1."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2) if a.shape[2] >= 3 else a[..., 0]
    if a.max() > 1.0:
        a = a / 255.0
    if p.invert:
        a = 1.0 - a
    if p.blur > 0:
        a = ndimage.gaussian_filter(a, p.blur)
    if p.kind == "logo":
        # two levels with a one-pixel soft edge (smooth walls after meshing)
        edge = 0.08
        a = np.clip((a - p.threshold) / edge + 0.5, 0.0, 1.0)
    return np.clip(a, 0.0, 1.0)


def estimate_radius(local_pts: np.ndarray, width: float) -> float:
    """Radius of the surface across the frame (a-c plane), by fitting a circle through the
    centre that touches the frame plane there: a^2 + (c + R)^2 = R^2."""
    a, b, c = local_pts.T
    sel = (np.abs(b) < 0.15 * width + 1.0) & (np.abs(a) < 0.5 * width) & (c < -1e-6) & (c > -width)
    if sel.sum() < 6:
        return 0.0
    a, c = a[sel], c[sel]
    R = -float(np.sum(c * (a * a + c * c))) / (2.0 * float(np.sum(c * c)))
    return R if R > 0 else 0.0


def _sample(img: np.ndarray, x, y) -> np.ndarray:
    rows, cols = img.shape
    r = (1.0 - y) * (rows - 1)
    q = x * (cols - 1)
    vals = ndimage.map_coordinates(img, [r.ravel(), q.ravel()], order=1, mode="constant", cval=0.0)
    vals = vals.reshape(np.shape(x))
    out = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    return np.where(out, vals, 0.0)


def voxel_count(p: DecalParams, surface_range: float = 2.0) -> int:
    """Rough size of the voxel grid, for a surface whose height varies by `surface_range` across the frame."""
    h = p.resolution
    z = surface_range + 2 * (p.depth + max(0.3, 3 * h) + 2 * h)
    return int((p.width / h + 4) * (p.height / h + 4) * (z / h))


def _outer_layer(x, y, z, cell: float) -> np.ndarray:
    """Heights of the outermost surface in each small column (the side the image is put on)."""
    if not len(z):
        return np.array([0.0])
    key = np.floor(x / cell).astype(np.int64) * 1_000_003 + np.floor(y / cell).astype(np.int64)
    order = np.lexsort((-z, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    return z[order][first]


def _signed_distance(P, tree, sample_tri, tris, tri_n, k: int = 16):
    """(signed distance, surface normal there) for points: exact distance to the nearby
    triangles, negative inside.

    The sign comes from the normals of the closest triangles; where several are
    equally close (edges, corners) their normals are averaged, which gives the
    right side for convex and concave edges alike.
    """
    from .sdf import closest_on_triangles
    out = np.empty(len(P))
    nrm = np.empty((len(P), 3))
    for s0 in range(0, len(P), 300_000):
        X = P[s0:s0 + 300_000]
        _, idx = tree.query(X, k=k, workers=-1)
        cand = np.sort(sample_tri[idx], axis=1)
        dup = np.zeros(cand.shape, dtype=bool)
        dup[:, 1:] = cand[:, 1:] == cand[:, :-1]  # many samples share one triangle: test it once
        D = np.full(cand.shape, np.inf)
        V = np.zeros(cand.shape + (3,))
        for j in range(k):
            rows = np.nonzero(~dup[:, j])[0]
            if not len(rows):
                continue
            t = tris[cand[rows, j]]
            q = closest_on_triangles(X[rows], t[:, 0], t[:, 1], t[:, 2])
            V[rows, j] = X[rows] - q
            D[rows, j] = np.linalg.norm(V[rows, j], axis=1)
        best = D.min(axis=1)
        tie = np.isfinite(D) & (D <= best[:, None] + 1e-6 + 1e-6 * best[:, None])
        nsum = np.einsum("nk,nkd->nd", tie.astype(float), tri_n[cand])
        jb = np.argmin(D, axis=1)
        vb = V[np.arange(len(X)), jb]
        side = np.einsum("nd,nd->n", vb, nsum)
        out[s0:s0 + 300_000] = np.where(side < 0, -best, best)
        nrm[s0:s0 + 300_000] = nsum / np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-12)
    return out, nrm


def apply_decal(mesh: Mesh, frame: Frame, image: np.ndarray, p: DecalParams, progress=None,
                max_voxels: int = 40_000_000) -> tuple[Mesh, dict]:
    """The model with the image raised out of / pressed into its surface. Returns (mesh, info)."""
    h = p.resolution
    m = max(0.3, 3 * h)  # how far the relief solid reaches into the model (or out of it)
    eps = 0.5 * h  # where the image is empty the relief solid stays this far off the surface
    img = prepare_image(image, p)
    report(progress, 0.05, "Placing the image")

    loc = frame.to_local(mesh.vertices)  # (N, 3) float64
    half_w = 0.5 * p.width + 2 * h
    half_h = 0.5 * p.height + 2 * h
    reach = 0.5 * max(p.width, p.height) + p.depth + 1.0
    # triangles whose box meets the region around the frame (one axis at a time: big models)
    near = np.ones(mesh.n_faces, dtype=bool)
    for k, (a, b) in enumerate(((-half_w - reach, half_w + reach),
                                (-half_h - p.depth - 1.5, half_h + p.depth + 1.5), (-2 * reach, reach))):
        c = loc[:, k][mesh.faces]
        near &= (c.max(axis=1) >= a) & (c.min(axis=1) <= b)
        del c
    if not near.any():
        raise ValueError("the frame does not touch the model")
    patch = mesh.faces[near]
    ptri = loc[patch]
    tri_n = np.cross(ptri[:, 1] - ptri[:, 0], ptri[:, 2] - ptri[:, 0])
    dbl_area = np.linalg.norm(tri_n, axis=1)
    tri_n /= np.maximum(dbl_area, 1e-300)[:, None]
    spacing, limit = max(h, 0.25), 3_000_000
    pts, tri_id = surface_samples(Mesh(loc, patch), spacing, limit=limit)
    est = len(patch) + 0.5 * float(dbl_area.sum()) / (0.5 * spacing ** 2)
    gap = 1.5 * spacing * math.sqrt(max(1.0, est / limit))  # no surface point is further from a sample
    tree = cKDTree(pts)

    wrap = p.projection == "cylinder"
    radius = p.radius if p.radius > 0 else (estimate_radius(pts, p.width) if wrap else 0.0)
    wrap = wrap and radius > 0
    if wrap:
        # unwrapped space: x = arc length at the radius, y = along the axis, z = radial offset
        rho = np.hypot(pts[:, 0], pts[:, 2] + radius)
        ux = radius * np.arctan2(pts[:, 0], pts[:, 2] + radius)
        uy, uz = pts[:, 1], rho - radius
    else:
        ux, uy, uz = pts[:, 0], pts[:, 1], pts[:, 2]
    inframe = (np.abs(ux) <= half_w) & (np.abs(uy) <= half_h) & (np.abs(uz) <= reach)
    cz = _outer_layer(ux[inframe], uy[inframe], uz[inframe], max(1.0, 4 * h))
    z_lo = float(cz.min()) - p.depth - m - 2 * h
    z_hi = float(cz.max()) + p.depth + m + 2 * h
    origin = np.array([-half_w, -half_h, z_lo])
    shape = tuple(int(k) for k in np.ceil((np.array([half_w, half_h, z_hi]) - origin) / h).astype(int) + 1)
    grid = Grid(origin, h, shape)
    if grid.n_voxels > max_voxels:
        raise MemoryError("decal grid too large")

    def to_local(q):
        """Grid space -> frame-local coordinates."""
        if not wrap:
            return q
        th = q[:, 0] / radius
        r = radius + q[:, 2]
        return np.stack([r * np.sin(th), q[:, 1], r * np.cos(th) - radius], axis=1)

    # signed distance of every voxel centre to the model
    report(progress, 0.2, "Measuring the surface")
    # Exact distances are needed only near the relief's two surfaces. Further away the
    # relief function is positive whatever the side, so the distance to the nearest
    # sample (unsigned) stands in for it.
    band = max(p.depth, m) + eps + 2 * h + gap
    total = grid.n_voxels
    sd = np.empty(total)
    facing = np.ones(total, dtype=np.float32)
    for s0 in range(0, total, 1_500_000):
        chunk = np.stack(np.unravel_index(np.arange(s0, min(total, s0 + 1_500_000)), grid.shape), axis=1)
        q = to_local(grid.points(chunk))
        d, _ = tree.query(q, k=1, workers=-1)
        near_s = np.nonzero(d <= band)[0]
        if len(near_s):
            q = q[near_s]
            d[near_s], n = _signed_distance(q, tree, tri_id, ptri, tri_n)
            if wrap:
                out_dir = np.stack([q[:, 0], np.zeros(len(q)), q[:, 2] + radius], axis=1)
                out_dir /= np.maximum(np.linalg.norm(out_dir, axis=1, keepdims=True), 1e-12)
            else:
                out_dir = np.array([0.0, 0.0, 1.0])
            facing[s0 + near_s] = np.sum(n * out_dir, axis=1)
        sd[s0:s0 + len(chunk)] = d
        report(progress, 0.2 + 0.5 * min(1.0, (s0 + len(chunk)) / total), "Measuring the surface")
    sd = sd.reshape(grid.shape)
    facing = facing.reshape(grid.shape)

    # relief heights from the image (image x, y = grid x, y across the frame)
    report(progress, 0.72, "Shaping the relief")
    X = grid.axis(0)[:, None] * np.ones(grid.shape[1])[None, :]
    Y = np.ones(grid.shape[0])[:, None] * grid.axis(1)[None, :]
    x, y = X / p.width + 0.5, Y / p.height + 0.5
    if p.tile > 1:
        inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        x = np.where(inside, (x * p.tile) % 1.0, x)
        y = np.where(inside, (y * p.tile) % 1.0, y)
    I = _sample(img, x, y)[:, :, None]
    facing_ok = facing > 0.15  # surfaces turned away (the back of thin parts) stay as they are
    I = np.where(facing_ok, I, 0.0)
    # where the image starts, the relief starts with a small step: a relief that rose from
    # the surface at a grazing angle would leave slivers in the boolean
    start = 0.03
    if p.kind == "texture":
        # a texture covers the whole frame: dark parts keep a thin layer instead of
        # dropping to the surface, so its low points stay smooth
        inframe = ((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1))[:, :, None] & facing_ok
        on = inframe
        I = np.where(on, np.maximum(I, max(start, 0.3 * h / p.depth)), 0.0)
    else:
        on = I > start
    if p.mode == "deboss":
        bottom = np.where(on, -p.depth * I, eps)
        g = np.maximum(bottom - sd, sd - m)
    else:
        top = np.where(on, p.depth * I, -eps)
        g = np.maximum(sd - top, -m - sd)
    if not on.any() or not (g < 0).any():
        raise ValueError("the image is empty here")

    report(progress, 0.8, "Shaping the relief")
    relief = iso_surface(g.astype(np.float32), grid)
    relief = Mesh(frame.to_world(to_local(relief.vertices.astype(np.float64))), relief.faces)
    if relief.volume() < 0:
        relief = Mesh(relief.vertices, relief.faces[:, ::-1])
    report(progress, 0.88, "Adding the relief")
    model = to_manifold(mesh)
    rm = to_manifold(relief)
    out = from_manifold(model - rm if p.mode == "deboss" else model + rm)
    report(progress, 1.0, "Adding the relief")
    return out, {"radius": radius if wrap else 0.0, "voxels": grid.n_voxels, "relief_faces": relief.n_faces,
                 "volume_change": out.volume() - mesh.volume()}
