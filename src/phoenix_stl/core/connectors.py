"""Connectors between cut parts: pins, pegs, magnets, rods, keys, tongue & groove, dovetail.

Everything is built in the joint frame: (u, v) span the joint face, w is its
normal pointing from part A into part B (A lies on the -w side). Sizes are in
mm. The clearance is the gap per side between mating surfaces; holes are
also made deeper by depth_gap (room for glue and resin overcure).

Round holes are built as polygons whose flat sides touch the nominal radius
(never smaller than asked), pins as polygons whose corners touch it (never
larger), so facets can only make a fit looser, never tighter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import manifold3d as m3d
import numpy as np
from matplotlib.path import Path as MplPath

from .boolean import from_manifold, to_manifold
from .mesh import CHUNK, Mesh, report
from .section import cross_section_of, fill_holes, polygons, section_stack, to_cross_section
from .surface_cut import Frame

TYPES = ("dowel", "peg", "magnet", "rod", "key", "tongue", "dovetail")
FITS = ("press", "snug", "slide", "loose")

# Gap per side (mm). Starting points: print the tolerance coupon to tune them for your printer.
FIT_PRESETS = {
    "resin": {"press": 0.05, "snug": 0.10, "slide": 0.15, "loose": 0.25},
    "fdm": {"press": 0.10, "snug": 0.15, "slide": 0.25, "loose": 0.35},
}
KIND_DEFAULTS = {
    "resin": {"depth_gap": 0.3, "chamfer": 0.3, "wall": 1.0},
    "fdm": {"depth_gap": 0.4, "chamfer": 0.5, "wall": 1.6},
}
MAGNETS = [(3, 1), (3, 2), (4, 2), (5, 2), (5, 3), (6, 2), (6, 3), (8, 2), (8, 3), (10, 2), (10, 3)]

# Default sizes per type (mm, degrees).
DEFAULTS = {
    "dowel": {"diameter": 4.0, "length": 12.0},
    "peg": {"diameter": 4.0, "length": 5.0},
    "magnet": {"diameter": 6.0, "length": 3.0},
    "rod": {"diameter": 2.0, "length": 20.0},
    "key": {"diameter": 5.0, "length": 5.0},
    "tongue": {"diameter": 2.0, "length": 2.0, "edge": 1.5},
    "dovetail": {"diameter": 6.0, "length": 4.0, "edge": 1.5, "flare": 15.0},
}
MALE_TYPES = ("peg", "key", "tongue", "dovetail")
PLACED_TYPES = ("dowel", "peg", "magnet", "rod", "key", "dovetail")

OVER = 0.05  # cutters stick out this far past the face, so no faces coincide
# Outlines are simplified to this (mm) before extruding: points closer than an
# STL float can tell apart would otherwise merge into broken edges.
SIMPLIFY = 1e-3
EMBED = 0.6  # male parts start this deep inside their own part, so they fuse
SECTION_INSET = 0.02  # outlines are measured this far inside each part
BOSS_INSET = 0.05  # bosses stop this far inside the outer surface
BOSS_RECESS = 0.02  # and this far below the joint face


@dataclass
class Fit:
    clearance: float = 0.10
    depth_gap: float = 0.3
    chamfer: float = 0.3
    wall: float = 1.0

    @classmethod
    def preset(cls, kind: str, fit: str) -> "Fit":
        kind = kind if kind in FIT_PRESETS else "resin"
        return cls(FIT_PRESETS[kind][fit], **KIND_DEFAULTS[kind])


@dataclass
class Spec:
    """One connector type with its sizes.

    diameter: pin/peg/rod/magnet diameter, key width, tongue width, dovetail narrow width.
    length: dowel/rod total length, peg/key/tongue/dovetail height, magnet thickness.
    edge: tongue/dovetail distance from the outline. flare: dovetail side angle.
    """
    type: str = "dowel"
    diameter: float = 4.0
    length: float = 12.0
    edge: float = 1.5
    flare: float = 15.0

    @classmethod
    def default(cls, type_: str) -> "Spec":
        return cls(type_, **DEFAULTS[type_])


@dataclass
class Placement:
    u: float
    v: float
    angle: float = 0.0  # degrees, for keys and dovetails


# -- sizes --------------------------------------------------------------------------------

def hole_radius(spec: Spec, fit: Fit) -> float:
    if spec.type == "key":
        return spec.diameter / math.sqrt(2) + fit.clearance
    return spec.diameter / 2 + fit.clearance


def depth_into(spec: Spec, fit: Fit, side: str, male: str) -> float:
    """How deep the connector goes into one side, measured from the face."""
    t = spec.type
    if t in ("dowel", "rod"):
        return spec.length / 2 + fit.depth_gap
    if t == "magnet":
        return spec.length + fit.depth_gap
    if t in MALE_TYPES:
        return EMBED if side == male else spec.length + fit.depth_gap
    raise ValueError(t)


def footprint(spec: Spec, fit: Fit) -> float:
    """Radius around a connector centre that must be solid (hole plus wall)."""
    if spec.type == "dovetail":
        top = spec.diameter + 2 * spec.length * math.tan(math.radians(spec.flare))
        return top / 2 + fit.clearance + fit.wall
    return hole_radius(spec, fit) + fit.wall


# -- primitive solids (local frame, unit placement at the origin) -----------------------------

def _segments(r: float) -> int:
    n = int(math.ceil(2 * math.pi * r / 0.25))
    return max(24, min(160, (n + 3) // 4 * 4))


def _revolve(profile, r_max: float, circumscribe: bool) -> m3d.Manifold:
    n = _segments(r_max)
    k = 1.0 / math.cos(math.pi / n) if circumscribe else 1.0
    pts = np.array([(x * k, z) for x, z in profile], dtype=np.float64)
    return m3d.CrossSection([pts], m3d.FillRule.EvenOdd).revolve(n)


def round_hole(r: float, depth: float, chamfer: float) -> m3d.Manifold:
    """Hole going down into -w from the face, with an optional 45 degree mouth chamfer."""
    ch = min(chamfer, 0.5 * r, 0.5 * depth)
    if ch > 0:
        prof = [(0, OVER), (r + ch + OVER, OVER), (r, -ch), (r, -depth), (0, -depth)]
    else:
        prof = [(0, OVER), (r, OVER), (r, -depth), (0, -depth)]
    return _revolve(prof, r + ch, circumscribe=True)


def round_peg(r: float, height: float, chamfer: float, embed: float = EMBED) -> m3d.Manifold:
    """Peg standing up into +w from the face, chamfered tip."""
    ch = min(chamfer, 0.5 * r, 0.5 * height)
    prof = [(0, -embed), (r, -embed), (r, height - ch), (r - ch, height), (0, height)]
    return _revolve(prof, r, circumscribe=False)


def dowel_pin(r: float, length: float, chamfer: float) -> m3d.Manifold:
    """Loose pin standing on z=0, chamfered at both ends."""
    ch = min(chamfer, 0.5 * r, 0.25 * length)
    prof = [(0, 0), (r - ch, 0), (r, ch), (r, length - ch), (r - ch, length), (0, length)]
    return _revolve(prof, r, circumscribe=False)


def square_key(width: float, height: float, chamfer: float, embed: float = EMBED) -> m3d.Manifold:
    """Square peg with a chamfered top, built as one convex hull (two touching pieces would
    leave duplicate points behind)."""
    ch = min(chamfer, 0.25 * width, 0.5 * height)
    h = width / 2
    t = h - ch
    pts = [(x, y, -embed) for x in (-h, h) for y in (-h, h)]
    pts += [(x, y, height - ch) for x in (-h, h) for y in (-h, h)]
    pts += [(x, y, height) for x in (-t, t) for y in (-t, t)]
    return m3d.Manifold.hull_points(np.array(pts, dtype=np.float64))


def square_key_hole(width: float, clearance: float, depth: float) -> m3d.Manifold:
    sq = m3d.CrossSection.square((width, width), True).offset(clearance, m3d.JoinType.Round)
    return sq.extrude(depth + OVER).translate((0, 0, -depth))


def cylinder(r: float, z0: float, z1: float) -> m3d.Manifold:
    """Plain cylinder between two heights (used for bosses and checks)."""
    return m3d.Manifold.cylinder(z1 - z0, r, r, _segments(r)).translate((0, 0, z0))


def _flip(m: m3d.Manifold) -> m3d.Manifold:
    """Mirror across the face (w -> -w): a cutter for side A becomes one for side B."""
    return m.mirror((0, 0, 1))


def _place(m: m3d.Manifold, p: Placement) -> m3d.Manifold:
    if p.angle:
        m = m.rotate((0, 0, p.angle))
    return m.translate((p.u, p.v, 0))


# -- outlines of the joint face ------------------------------------------------------------------

BOSS_LEVELS = tuple(0.5 * k for k in range(1, 17)) + (9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0)


@dataclass
class JointShape:
    """The joint face: solid outline, plus where bosses may go inside a hollow part.

    inside[side] = [(depth, outline)]: material or cavity of that side at each
    depth below the face (outside air excluded). levels = [(depth, outline)]:
    the face's material-or-cavity intersected over both sides and every depth
    down to `depth`; a straight hole of that depth inside it never breaks out.
    """
    solid: m3d.CrossSection
    levels: list
    inside: dict = field(default_factory=dict)
    solid_levels: list = field(default_factory=list)  # material only, both sides, down to each depth

    @property
    def hollow(self) -> bool:
        return bool(self.levels) and self.levels[0][1].area() > self.solid.area() * 1.02 + 1e-6

    def filled(self, depth: float) -> m3d.CrossSection:
        for d, cs in self.levels:
            if d >= depth - 1e-9:
                return cs
        return m3d.CrossSection()

    def material(self, depth: float) -> m3d.CrossSection:
        """Joint outline where there is material on both sides all the way down to `depth`."""
        if depth <= SECTION_INSET or not self.solid_levels:
            return self.solid
        for d, cs in self.solid_levels:
            if d >= depth - 1e-9:
                return cs
        return m3d.CrossSection()

    def to_dict(self) -> dict:
        def enc(items):
            return [(d, [p.tolist() for p in polygons(cs)]) for d, cs in items]
        return {"solid": [p.tolist() for p in polygons(self.solid)], "levels": enc(self.levels),
                "inside": {k: enc(v) for k, v in self.inside.items()}, "solid_levels": enc(self.solid_levels)}

    @classmethod
    def from_dict(cls, d: dict) -> "JointShape":
        def dec(items):
            return [(float(k), cross_section_of(v)) for k, v in items]
        return cls(cross_section_of(d["solid"]), dec(d["levels"]),
                   {k: dec(v) for k, v in d.get("inside", {}).items()}, dec(d.get("solid_levels", [])))


def rays_hit(mesh: Mesh, frame: Frame, pts2d, sign: float, start: float = 0.05) -> np.ndarray:
    """For points on the face: does a ray going into the part (along sign*w) hit its surface?"""
    pts = np.asarray(pts2d, dtype=np.float64).reshape(-1, 2)
    hit = np.zeros(len(pts), dtype=bool)
    if not len(pts) or mesh.n_faces == 0:
        return hit
    loc = frame.to_local(mesh.vertices)
    for s0 in range(0, mesh.n_faces, CHUNK):
        tri = loc[mesh.faces[s0:s0 + CHUNK]]
        umin, umax = tri[:, :, 0].min(1), tri[:, :, 0].max(1)
        vmin, vmax = tri[:, :, 1].min(1), tri[:, :, 1].max(1)
        for k, (pu, pv) in enumerate(pts):
            if hit[k]:
                continue
            m = (umin <= pu) & (pu <= umax) & (vmin <= pv) & (pv <= vmax)
            if not m.any():
                continue
            t = tri[m]
            a, b, c = t[:, 0, :2], t[:, 1, :2], t[:, 2, :2]
            e0, e1 = b - a, c - a
            q = np.array([pu, pv]) - a
            den = e0[:, 0] * e1[:, 1] - e1[:, 0] * e0[:, 1]
            ok = np.abs(den) > 1e-18
            with np.errstate(divide="ignore", invalid="ignore"):
                l1 = (q[:, 0] * e1[:, 1] - e1[:, 0] * q[:, 1]) / den
                l2 = (e0[:, 0] * q[:, 1] - q[:, 0] * e0[:, 1]) / den
            l0 = 1 - l1 - l2
            inside = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            w = l0 * t[:, 0, 2] + l1 * t[:, 1, 2] + l2 * t[:, 2, 2]
            hit[k] = bool(np.any(inside & (sign * w > start)))
    return hit


def cavities(solid: m3d.CrossSection, ray_mesh: Mesh, frame: Frame, sign: float, samples: int = 12):
    """Holes of the joint outline that are inner cavities (closed off further inside the part).

    A hole the part surrounds only at the face, like the middle of a ring or a
    gap under an arm, lets rays pass straight through; a cavity of a hollow
    model has a ceiling or floor, so most rays from it hit the part.
    """
    out = []
    for hole in (fill_holes(solid) - solid).decompose():
        if hole.area() < 1e-6:
            continue
        pts = spread_points(hole, samples)
        if pts and rays_hit(ray_mesh, frame, pts, sign).mean() >= 0.6:
            out.append(hole)
    return out


def joint_shape(meshes: dict, frame: Frame, max_depth: float = BOSS_LEVELS[-1], ray_meshes: dict | None = None,
                progress=None) -> JointShape:
    """Outlines for {'A': mesh} or {'A': mesh, 'B': mesh} at the frame's face.

    ray_meshes (optional, e.g. light previews) are used to tell cavities from
    through-holes; the full meshes are used otherwise.
    """
    levels = [d for d in BOSS_LEVELS if d <= max_depth + 1e-9]
    solid, filled, inside, material = None, None, {}, None
    for k, (side, mesh) in enumerate(meshes.items()):
        report(progress, k / len(meshes), "Measuring the joint")
        sign = -1.0 if side == "A" else 1.0
        stack = section_stack(mesh, frame.w, frame.origin, [sign * SECTION_INSET] + [sign * d for d in levels])
        s = to_cross_section(stack[0], frame, SIMPLIFY)
        rm = (ray_meshes or {}).get(side) or mesh
        cav = cavities(s, rm, frame, sign)
        cav_all = m3d.CrossSection.compose(cav) if cav else None
        f = s + cav_all if cav else s
        mat = s
        side_levels, side_inside, side_material = [], [], []
        for d, loops in zip(levels, stack[1:]):
            st = to_cross_section(loops, frame, SIMPLIFY)
            mat = mat ^ st
            side_material.append(mat)
            # deeper down, only material or the same cavity counts as inside: a hole
            # that widens below the face (outside air) must not take a boss
            here = _inside_at(st, cav_all)
            side_inside.append((d, here))
            f = f ^ here
            side_levels.append(f)
        inside[side] = [(0.0, s + cav_all if cav else s)] + side_inside
        solid = s if solid is None else solid ^ s
        filled = side_levels if filled is None else [a ^ b for a, b in zip(filled, side_levels)]
        material = side_material if material is None else [a ^ b for a, b in zip(material, side_material)]
    return JointShape(solid, list(zip(levels, filled)), inside, list(zip(levels, material)))


def _inside_at(section: m3d.CrossSection, cavity) -> m3d.CrossSection:
    """Material plus the holes that continue a face cavity (a hole mostly over it)."""
    if cavity is None:
        return section
    keep = [h for h in (fill_holes(section) - section).decompose()
            if h.area() > 1e-9 and (h ^ cavity).area() > 0.5 * h.area()]
    return section + m3d.CrossSection.compose(keep) if keep else section


def clipped_boss(shape: JointShape, side: str, center, radius: float, depth: float) -> m3d.Manifold:
    """A boss column going `depth` into a side (canonical: into -w), trimmed layer by layer
    to the inside of the part so it follows curved walls instead of poking out."""
    levels = shape.inside.get(side) or []
    disk = m3d.CrossSection.circle(radius, _segments(radius)).translate(tuple(center))
    pieces = []
    for (t0, a), (t1, b) in zip(levels, levels[1:]):
        if t0 >= depth - 1e-9:
            break
        top = min(t1, depth)
        t0 = max(t0, BOSS_RECESS)  # never flush with the face: near-coplanar faces leave slivers
        # trimmed a little inside the outer surface: coinciding with it would leave slivers
        layer = (disk ^ (a ^ b).offset(-BOSS_INSET, m3d.JoinType.Round)).simplify(SIMPLIFY)
        layer = m3d.CrossSection.compose([q for q in layer.decompose() if q.area() > 0.01]) \
            if not layer.is_empty() else layer
        if layer.area() > 1e-6:
            # layers overlap a hair (downwards, the top stays flush with the face) so they fuse
            pieces.append(layer.extrude(top - t0 + 1e-3).translate((0, 0, -top - 1e-3)))
    if not pieces:
        return m3d.Manifold()
    boss = m3d.Manifold.batch_boolean(pieces, m3d.OpType.Add) if len(pieces) > 1 else pieces[0]
    # neighbouring layers differ by fractions of a micron where they follow the wall;
    # collapse those steps so no points closer than an STL float can tell apart remain
    return boss.simplify(SIMPLIFY)


def boss_depth(spec: Spec, fit: Fit, male: str = "A") -> float:
    """Deepest boss any side of this connector needs (hole plus the wall under it)."""
    d = max(depth_into(spec, fit, side, male) for side in ("A", "B"))
    if spec.type in MALE_TYPES:
        d = max(d, spec.length)
    return d + fit.wall


def hole_depth(spec: Spec, fit: Fit, male: str = "A") -> float:
    """Deepest hole of this connector: it must stay inside the part all the way down."""
    return max(depth_into(spec, fit, side, male) for side in ("A", "B"))


# -- point tests and automatic placement -----------------------------------------------------------

def contains(cs: m3d.CrossSection, pts) -> np.ndarray:
    """Even-odd point-in-outline test for (k, 2) points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(len(pts), dtype=bool)
    for poly in polygons(cs):
        inside ^= MplPath(poly, closed=False).contains_points(pts)
    return inside


def distance_to_edges(cs: m3d.CrossSection, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    best = np.full(len(pts), np.inf)
    for poly in polygons(cs):
        a = poly
        b = np.roll(poly, -1, axis=0)
        ab = b - a
        L = np.maximum((ab ** 2).sum(1), 1e-18)
        for s in range(0, len(a), 2048):
            aa, abb, LL = a[s:s + 2048], ab[s:s + 2048], L[s:s + 2048]
            t = np.clip(((pts[:, None, :] - aa[None]) * abb[None]).sum(2) / LL[None], 0, 1)
            q = aa[None] + t[..., None] * abb[None]
            best = np.minimum(best, np.sqrt(((pts[:, None, :] - q) ** 2).sum(2)).min(1))
    return best


def boss_region(shape: JointShape, radius: float, overlap: float, depth: float, near=None) -> m3d.CrossSection:
    """Centres where a boss fits inside the part (material or cavity, never outside air)
    and still overlaps the wall (of `near`, default any) by `overlap`."""
    near = shape.solid if near is None else near
    return shape.filled(depth).offset(-radius, m3d.JoinType.Round) ^ near.offset(
        max(radius - overlap, 1e-3), m3d.JoinType.Round)


def safe_regions(shape: JointShape, radius: float, boss: bool, overlap: float, depth: float):
    """Per island of the joint face: (island, region where a centre fits, needs_boss)."""
    out = []
    deep = shape.material(depth)
    for island in shape.solid.decompose():
        region = (island ^ deep).offset(-radius, m3d.JoinType.Round)
        if region.area() > 1e-9:
            out.append((island, region, False))
            continue
        if boss and shape.hollow:
            region = boss_region(shape, radius, overlap, depth, island)
            if region.area() > 1e-9:
                out.append((island, region, True))
                continue
        out.append((island, region, False))
    return out


def _candidates(region: m3d.CrossSection, n: int = 900, max_grid: int = 12_000) -> np.ndarray:
    """Candidate centres: a grid over the region plus points along its outline (thin regions)."""
    b = np.asarray(region.bounds(), dtype=np.float64).reshape(2, 2)
    size = np.maximum(b[1] - b[0], 1e-9)
    h = max(math.sqrt(max(region.area(), 1e-12) / n), math.sqrt(size[0] * size[1] / max_grid))
    xs = np.arange(b[0, 0] + h / 2, b[1, 0], h)
    ys = np.arange(b[0, 1] + h / 2, b[1, 1], h)
    pts = np.zeros((0, 2))
    if len(xs) and len(ys):
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        grid = np.c_[X.ravel(), Y.ravel()]
        pts = grid[contains(region, grid)]
    edge = [q for q in polygons(region)]
    if edge:
        ring = np.vstack(edge)
        step = max(1, len(ring) // 1500)
        pts = np.vstack([pts, ring[::step]])
    return pts


def spread_points(region: m3d.CrossSection, count: int) -> list[tuple[float, float]]:
    """count points spread evenly over a region (one point: its most interior spot)."""
    inner = region.offset(-0.01, m3d.JoinType.Round)  # keep clear of the exact boundary
    if inner.area() < 1e-4:
        return []  # a sliver a few microns wide is no room at all
    cand = _candidates(inner)
    if not len(cand) or count <= 0:
        return []
    depth = distance_to_edges(region, cand) * np.where(contains(region, cand), 1.0, 0.0)
    first = int(np.argmax(depth))
    if count == 1:
        return [tuple(cand[first])]
    count = min(count, len(cand))
    chosen = [first]
    dist = np.linalg.norm(cand - cand[first], axis=1)
    for _ in range(count - 1):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(cand - cand[i], axis=1))
    centers = cand[chosen]
    for _ in range(12):  # Lloyd relaxation restricted to candidate points
        lab = np.argmin(((cand[:, None] - centers[None]) ** 2).sum(2), axis=1)
        new = []
        for k in range(len(centers)):
            grp = cand[lab == k]
            if not len(grp):
                new.append(centers[k])
                continue
            mean = grp.mean(0)
            new.append(grp[np.argmin(((grp - mean) ** 2).sum(1))])
        new = np.array(new)
        if np.allclose(new, centers):
            break
        centers = new
    return [tuple(p) for p in centers]


def auto_place(shape: JointShape, spec: Spec, fit: Fit, count: int, boss: bool = True, male: str = "A"):
    """Placements spread over every island of the joint face, plus notes for islands left empty."""
    if spec.type == "tongue":
        return [], []
    r = footprint(spec, fit)
    overlap = max(0.8, 0.35 * r)
    out, notes = [], []
    for i, (island, region, _) in enumerate(safe_regions(shape, r, boss, overlap, hole_depth(spec, fit, male))):
        n = 1 if spec.type == "dovetail" else count
        pts = spread_points(region, n) if region.area() > 1e-9 else []
        if not pts and island.area() > 1e-6:
            notes.append(i)
        out.extend(Placement(float(x), float(y), 0.0) for x, y in pts)
    return out, notes


def placement_ok(shape: JointShape, spec: Spec, fit: Fit, placements, boss: bool = True,
                 male: str = "A") -> list[bool]:
    """Whether each placement has enough wall around it (with a boss, if allowed)."""
    if not placements:
        return []
    r = footprint(spec, fit)
    pts = np.array([(p.u, p.v) for p in placements])
    ok = contains(shape.material(hole_depth(spec, fit, male)).offset(-r, m3d.JoinType.Round), pts)
    if boss and shape.hollow and not ok.all():
        ok |= contains(boss_region(shape, r, max(0.8, 0.35 * r), hole_depth(spec, fit, male)), pts)
    return ok.tolist()


# -- building the connectors -------------------------------------------------------------------------

@dataclass
class Plan:
    """Solids per side, in the joint frame."""
    add: dict = field(default_factory=lambda: {"A": [], "B": []})
    sub: dict = field(default_factory=lambda: {"A": [], "B": []})
    boss: dict = field(default_factory=lambda: {"A": [], "B": []})
    check: dict = field(default_factory=lambda: {"A": [], "B": []})  # (index, solid that must be material)
    pins: list = field(default_factory=list)  # loose parts (dowel pins), standing at the origin


def _perimeter(cs: m3d.CrossSection) -> float:
    return float(sum(np.linalg.norm(np.diff(np.vstack([q, q[:1]]), axis=0), axis=1).sum() for q in polygons(cs)))


def _band(shape: JointShape, edge: float, width: float, grow: float = 0.0, depth: float = 0.0) -> m3d.CrossSection:
    """Tongue footprint: a strip `width` wide, `edge` inside the outline of every island.

    Where a wall is too thin for that (hollow models), the strip is centred in
    the wall instead and narrowed to a third of the wall so both groove walls
    stay printable. The tongue (grow=0) and its groove (grow=clearance) always
    use the same rule, so they match.
    """
    keep = []
    base = shape.material(depth)
    for island in base.decompose():
        if island.offset(-edge - width, m3d.JoinType.Round).area() > 1e-6:
            band = (island.offset(-edge + grow, m3d.JoinType.Round)
                    - island.offset(-edge - width - grow, m3d.JoinType.Round))
        else:
            wall = 2 * island.area() / max(_perimeter(island), 1e-9)
            w = min(width, wall / 3)
            if w <= 0.2:
                continue
            band = island.offset(-(wall - w) / 2 + grow, m3d.JoinType.Round)
        keep.extend(b for b in band.decompose() if b.area() > 0.05)
    return m3d.CrossSection.compose(keep).simplify(SIMPLIFY) if keep else m3d.CrossSection()


def _dovetail_profile(spec: Spec, height: float, embed: float) -> m3d.CrossSection:
    """Trapezoid in the (across, w) plane: narrow at the face, wider at the top."""
    wn = spec.diameter
    wt = wn + 2 * height * math.tan(math.radians(spec.flare))
    pts = [(-wn / 2, -embed), (wn / 2, -embed), (wn / 2, 0), (wt / 2, height), (-wt / 2, height), (-wn / 2, 0)]
    if embed <= 0:
        pts = [(-wn / 2, 0), (wn / 2, 0), (wt / 2, height), (-wt / 2, height)]
    return m3d.CrossSection([np.array(pts, float)], m3d.FillRule.EvenOdd)


def _along(profile: m3d.CrossSection, p: Placement, reach: float) -> m3d.Manifold:
    """Extrude a (across, w) profile along the in-plane direction `angle`, centred on the placement."""
    a = math.radians(p.angle)
    t = np.array([math.cos(a), math.sin(a), 0.0])
    s = np.array([-math.sin(a), math.cos(a), 0.0])
    w = np.array([0.0, 0.0, 1.0])
    m = np.zeros((3, 4))
    m[:, 0], m[:, 1], m[:, 2] = s, w, t
    m[:, 3] = np.array([p.u, p.v, 0.0]) - reach * t
    return profile.extrude(2 * reach).transform(m)


def plan(spec: Spec, fit: Fit, placements, shape: JointShape, sides=("A", "B"), male: str = "A",
         boss: bool = True, reach: float = 1000.0) -> Plan:
    """Decide every solid to add to / cut from each side."""
    pl = Plan()
    t = spec.type
    c, ch = fit.clearance, fit.chamfer
    female = "B" if male == "A" else "A"
    sides = tuple(sides)

    def oriented(m, side):
        return m if side == "A" else _flip(m)

    if t == "tongue":
        h = spec.length
        gd = h + fit.depth_gap  # groove depth: the strip is measured where both sides hold that deep
        band = _band(shape, spec.edge, spec.diameter, depth=gd)
        if band.is_empty():
            return pl
        if male in sides:
            pl.add[male].append(oriented(band.extrude(h + EMBED).translate((0, 0, -EMBED)), male))
            pl.check[male].append((0, oriented(band.extrude(EMBED - OVER).translate((0, 0, -EMBED)), male)))
        if female in sides:
            groove = _band(shape, spec.edge, spec.diameter, grow=c, depth=gd)
            pl.sub[female].append(oriented(groove.extrude(gd + OVER).translate((0, 0, -gd)), female))
            pl.check[female].append((0, oriented(band.extrude(gd - OVER).translate((0, 0, -gd)), female)))
        return pl

    pts = [(p.u, p.v) for p in placements]
    ok_solid = contains(shape.material(hole_depth(spec, fit, male)).offset(-footprint(spec, fit),
                                                                           m3d.JoinType.Round), pts) if pts else []
    ok_boss = (contains(shape.filled(hole_depth(spec, fit, male)).offset(-footprint(spec, fit), m3d.JoinType.Round),
                        pts) if pts and boss and shape.hollow else [False] * len(pts))
    for i, p in enumerate(placements):
        needs_boss = boss and shape.hollow and not ok_solid[i] and ok_boss[i]
        for side in sides:
            depth = depth_into(spec, fit, side, male)
            if t in ("dowel", "rod", "magnet"):
                r = hole_radius(spec, fit)
                pl.sub[side].append(oriented(_place(round_hole(r, depth, ch), p), side))
                pl.check[side].append((i, oriented(_place(cylinder(spec.diameter / 2, -depth, -OVER), p), side)))
            elif t == "peg":
                if side == male:
                    pl.add[side].append(oriented(_place(round_peg(spec.diameter / 2, spec.length, ch), p), side))
                else:
                    r = hole_radius(spec, fit)
                    pl.sub[side].append(oriented(_place(round_hole(r, depth, ch), p), side))
                pl.check[side].append((i, oriented(_place(cylinder(spec.diameter / 2, -depth, -OVER), p), side)))
            elif t == "key":
                if side == male:
                    pl.add[side].append(oriented(_place(square_key(spec.diameter, spec.length, ch), p), side))
                else:
                    pl.sub[side].append(oriented(_place(square_key_hole(spec.diameter, c, depth), p), side))
                sq = m3d.CrossSection.square((spec.diameter, spec.diameter), True)
                pl.check[side].append((i, oriented(_place(sq.extrude(depth - OVER).translate((0, 0, -depth)), p),
                                                   side)))
            elif t == "dovetail":
                h = spec.length
                if side == male:
                    rail = _along(_dovetail_profile(spec, h, EMBED), p, reach)
                    clip = shape.material(h + fit.depth_gap).offset(-spec.edge, m3d.JoinType.Round).simplify(SIMPLIFY)
                    rail = rail ^ clip.extrude(h + EMBED + 2).translate((0, 0, -EMBED - 1))
                    pl.add[side].append(oriented(rail, side))
                else:
                    # built going into -w like every cutter, then flipped for side B
                    prof = _dovetail_profile(spec, h + fit.depth_gap, 0.0).offset(c, m3d.JoinType.Miter)
                    pl.sub[side].append(oriented(_along(prof.mirror((0, 1)), p, reach), side))
                continue
            if needs_boss:
                bd = max(depth, spec.length if t in MALE_TYPES else 0.0) + fit.wall
                boss_ = clipped_boss(shape, side, (p.u, p.v), footprint(spec, fit), bd)
                if not boss_.is_empty():
                    pl.boss[side].append(oriented(boss_, side))
        if t == "dowel":
            pl.pins.append(dowel_pin(spec.diameter / 2, spec.length, ch))
    return pl


def _compose(items) -> m3d.Manifold | None:
    items = [m for m in items if not m.is_empty()]
    if not items:
        return None
    return m3d.Manifold.batch_boolean(items, m3d.OpType.Add) if len(items) > 1 else items[0]


def _to_mesh(man: m3d.Manifold):
    return from_manifold(man)


def apply_plan(meshes: dict, pl: Plan, frame: Frame, progress=None):
    """Apply a plan to {'A': mesh, 'B': mesh}. Returns ({side: Mesh}, pins Mesh | None, warnings)."""
    to_world = frame.matrix()[:3, :]
    out, warnings = {}, []
    sides = list(meshes)
    for k, side in enumerate(sides):
        report(progress, 0.1 + 0.8 * k / len(sides), "Adding connectors")
        man = to_manifold(meshes[side])
        bosses = _compose(pl.boss[side])
        if bosses is not None:
            man = man + bosses.transform(to_world)
        adds = _compose(pl.add[side])
        if adds is not None:
            man = man + adds.transform(to_world)
        # every hole / peg foot must sit in material: report the ones that break out
        bad = []
        checks = [(i, m.transform(to_world)) for i, m in pl.check[side] if not m.is_empty()]
        whole = _compose([m for _, m in checks])
        if whole is not None and (whole ^ man).volume() < 0.995 * whole.volume():
            for i, m in checks:
                if (m ^ man).volume() < 0.99 * m.volume():
                    bad.append(i)
        if bad:
            warnings.append({"side": side, "connectors": sorted(set(bad))})
        subs = _compose(pl.sub[side])
        if subs is not None:
            man = man - subs.transform(to_world)
        out[side] = _to_mesh(man)
    pins = None
    if pl.pins:
        gap = 3.0
        row = []
        x = 0.0
        for pin in pl.pins:
            b = np.asarray(pin.bounding_box()).reshape(2, 3)
            row.append(pin.translate((x - b[0][0], -b[0][1], 0)))
            x += (b[1][0] - b[0][0]) + gap
        pins = _to_mesh(_compose(row))
    report(progress, 1.0, "Adding connectors")
    return out, pins, warnings


# -- tolerance coupon -----------------------------------------------------------------------------------

def coupon_clearances(center: float, step: float, n: int = 5) -> list[float]:
    return [round(max(0.0, center + (i - n // 2) * step), 3) for i in range(n)]


def tolerance_coupon(diameter: float, clearances, chamfer: float = 0.3, thickness: float | None = None):
    """Test plate with one through hole per clearance (hole k is marked with k dimples) and a test pin.

    Returns (plate Mesh, pin Mesh). Push the pin into each hole and keep the
    clearance of the hole that fits the way you want.
    """
    r = diameter / 2
    clearances = list(clearances)
    cmax = max(clearances)
    pitch = diameter + 2 * cmax + 5.0
    thick = thickness or max(4.0, min(8.0, diameter * 1.5))
    width = pitch * len(clearances) + 2.0
    depth = diameter + 2 * cmax + 9.0
    plate = m3d.Manifold.cube((width, depth, thick)).translate((0, 0, 0))
    cut = []
    dot_r = 0.6
    for k, c in enumerate(clearances):
        x = 1.0 + pitch * (k + 0.5)
        y = 3.5 + (diameter + 2 * cmax) / 2 + 1.0
        hole = _revolve([(0, -OVER), (r + c, -OVER), (r + c, thick + OVER), (0, thick + OVER)], r + c, True)
        cut.append(hole.translate((x, y, 0)))
        if chamfer > 0:
            ch = min(chamfer, 0.5 * r)
            mouth = _revolve([(0, thick - ch), (r + c, thick - ch), (r + c + ch + OVER, thick + OVER),
                              (0, thick + OVER)], r + c + ch, True)
            cut.append(mouth.translate((x, y, 0)))
        for j in range(k + 1):  # k+1 dimples in front of hole k
            dx = (j - k / 2) * 1.8
            cut.append(m3d.Manifold.sphere(dot_r, 24).translate((x + dx, 1.8, thick)))
    plate = plate - _compose(cut)
    pin = dowel_pin(r, thick + 6.0, chamfer).translate((width + 4.0, depth / 2, 0))
    return from_manifold(plate), from_manifold(pin)
