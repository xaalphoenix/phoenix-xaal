"""Silicone mother mold: a printed shell around the model at an exact distance.

Silicone is poured into the gap between the model and the shell. The shell
is open at the bottom, where it stands in a groove of the base plate (the
model sits on the plate). It has a pour funnel at the top, vents at the
places where air would be trapped, feet so it stands level when turned over
for casting, and is split into two halves with flanges, bolt holes and
alignment keys so the cured silicone can be taken out.

All distances are measured from the model's surface; the gap is exact to a
few micrometres (iso-surface vertices are snapped to it).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import manifold3d as m3d
import numpy as np
from scipy.spatial import cKDTree

from .boolean import from_manifold, to_manifold
from .cut import plane_cut
from .mesh import Mesh, report
from .sdf import DistanceField, estimate_bytes, iso_surface, shell_values
from .section import fill_holes, polygons, section_loops, to_cross_section
from .surface_cut import Frame

BOLTS = {  # clearance hole radius, nut across flats, nut thickness
    "M3": (1.7, 5.5, 2.4),
    "M4": (2.25, 7.0, 3.2),
}
OVER = 0.05
GAP = 0.02  # each flange stops this far from the parting plane (no coplanar faces)


@dataclass
class MoldParams:
    thickness: float = 5.0  # silicone between model and shell
    wall: float = 3.0  # shell wall
    resolution: float = 0.5  # voxel size (mm)
    base_z: float | None = None  # base plane height; None: the model's bottom
    # two halves
    split: bool = True
    split_angle: float | None = None  # direction of the parting plane's normal, degrees around Z
    split_offset: float = 0.0
    flange_width: float = 10.0
    flange_thickness: float = 4.0
    bolt: str = "M3"  # "M3", "M4" or "none"
    bolt_spacing: float = 35.0
    nut_traps: bool = True
    keys: bool = True
    # pouring
    sprue: tuple | None = None  # (x, y); None: the highest point of the cavity
    sprue_diameter: float = 8.0
    chimney: float = 8.0  # how far pour and vent tubes stick out of the shell
    funnel: bool = True
    vents: list | None = None  # [(x, y)]; None: found automatically
    vent_diameter: float = 2.5
    # feet (on top: the mold stands on them when turned over for casting)
    feet: int = 4
    feet_height: float = 5.0  # above the highest tube
    feet_diameter: float = 8.0
    feet_spread: float = 0.6
    # base plate
    base_plate: bool = True
    plate_margin: float = 8.0
    plate_thickness: float = 3.0
    groove_depth: float = 1.5
    locator: bool = True
    rod: bool = False
    rod_diameter: float = 4.0
    clearance: float = 0.15
    density: float = 1.1  # g/ml, for the silicone estimate

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MoldParams":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if known.get("sprue") is not None:
            known["sprue"] = tuple(known["sprue"])
        if known.get("vents") is not None:
            known["vents"] = [tuple(v) for v in known["vents"]]
        return cls(**known)


@dataclass
class MoldResult:
    parts: dict = field(default_factory=dict)  # name suffix -> Mesh
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


# -- helpers ----------------------------------------------------------------------------------

def _segments(r: float) -> int:
    return max(24, min(128, int(math.ceil(2 * math.pi * r / 0.4)) // 4 * 4))


def _cyl(x, y, z0, z1, r) -> m3d.Manifold:
    return m3d.Manifold.cylinder(z1 - z0, r, r, _segments(r)).translate((x, y, z0))


def _cone(x, y, z0, z1, r0, r1) -> m3d.Manifold:
    return m3d.Manifold.cylinder(z1 - z0, r0, r1, _segments(max(r0, r1))).translate((x, y, z0))


def _union(items) -> m3d.Manifold | None:
    items = [m for m in items if m is not None and not m.is_empty()]
    if not items:
        return None
    return m3d.Manifold.batch_boolean(items, m3d.OpType.Add) if len(items) > 1 else items[0]


def _closed_above(mesh: Mesh, z: float) -> Mesh:
    """The part of a closed mesh above height z, capped (still closed)."""
    if mesh.bounds[0][2] >= z:
        return mesh
    return plane_cut(mesh, (0, 0, 1), (0, 0, z)).positive


def default_base(model: Mesh) -> float:
    return float(model.bounds[0][2])


def default_split_angle(model: Mesh) -> float:
    """Parting plane along the longer horizontal side (its normal along the shorter one)."""
    s = model.size
    return 90.0 if s[0] >= s[1] else 0.0


def field_pad(p: MoldParams) -> float:
    return p.thickness + p.wall + 3.0 * p.resolution


def memory_estimate(model: Mesh, p: MoldParams) -> int:
    return estimate_bytes(model.bounds, p.resolution, field_pad(p))


# -- where to pour and vent -------------------------------------------------------------------

def cavity_top(inner: Mesh, z_min: float):
    """Highest point of the cavity's outer surface (pour position)."""
    v = inner.vertices
    keep = v[:, 2] > z_min
    if not keep.any():
        return None
    i = int(np.argmax(np.where(keep, v[:, 2], -np.inf)))
    return tuple(float(c) for c in v[i])


def find_vents(inner: Mesh, sprue_xy, z_min: float, radius: float | None = None) -> list[tuple]:
    """Local high points of the cavity ceiling where air would be trapped: [(x, y, z)]."""
    if inner.n_faces == 0:
        return []
    v = inner.vertices.astype(np.float64)
    f = inner.faces
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    size = float(np.linalg.norm(inner.size))
    radius = radius or max(10.0, 0.12 * size)
    cand = np.nonzero((vn[:, 2] > 0.5) & (v[:, 2] > z_min + 1.0))[0]
    if not len(cand):
        return []
    # one candidate per small cell: the highest
    cell = np.floor(v[cand] / (0.5 * radius)).astype(np.int64)
    order = np.lexsort((-v[cand, 2], cell[:, 2], cell[:, 1], cell[:, 0]))
    cs = cell[order]
    first = np.r_[True, np.any(np.diff(cs, axis=0) != 0, axis=1)]
    cand = cand[order[first]]
    tree = cKDTree(v)
    cand = cand[np.argsort(-v[cand, 2])]
    picked = []
    for i in cand:
        if picked and np.min(np.linalg.norm(v[picked] - v[i], axis=1)) < radius:
            continue
        nb = tree.query_ball_point(v[i], radius)
        if v[nb, 2].max() > v[i, 2] + 1e-6:
            continue
        picked.append(i)
    out = []
    for i in picked:
        if sprue_xy is not None and math.hypot(v[i, 0] - sprue_xy[0], v[i, 1] - sprue_xy[1]) < radius:
            continue
        out.append(tuple(float(c) for c in v[i]))
    return out


def feet_positions(outer_footprint: m3d.CrossSection, count: int, spread: float) -> list[tuple]:
    """Foot positions (x, y) spread around the middle of the shell's footprint."""
    if count <= 0 or outer_footprint.is_empty():
        return []
    b = np.asarray(outer_footprint.bounds(), float).reshape(2, 2)
    c = b.mean(0)
    half = 0.5 * (b[1] - b[0])
    start = math.pi / count if count in (2, 4) else math.pi / 2
    out = []
    for k in range(count):
        a = start + 2 * math.pi * k / count
        out.append((float(c[0] + spread * half[0] * math.cos(a)), float(c[1] + spread * half[1] * math.sin(a))))
    return out


# -- the mold ---------------------------------------------------------------------------------

def build_field(model: Mesh, p: MoldParams, exact: bool = True, progress=None) -> DistanceField:
    return DistanceField.build(model, p.resolution, field_pad(p), exact=exact, progress=progress)


def surfaces(fld: DistanceField, p: MoldParams, snap: bool = True):
    """(shell, inner solid, outer solid) iso-surfaces, uncut.

    The shell's surface has two sides: the outer one is the outer solid, the
    inner one (turned inside out) is the inner solid, so only one surface is
    extracted and snapped.
    """
    t, w = p.thickness, p.wall
    if snap and fld.tree is not None:
        fld.refine([t, t + w], band=2.0 * fld.grid.h)
    shell = iso_surface(shell_values(fld.dist, t, t + w), fld.grid)
    if snap and fld.tree is not None:
        shell = fld.snap(shell, [t, t + w])
    inner, outer = _split_sides(shell, fld, t, w)
    return shell, inner, outer


def _split_sides(shell: Mesh, fld: DistanceField, t: float, w: float):
    """Inner and outer solids from the shell: components classed by their distance to the model."""
    from .analyze import edge_info, vertex_components
    if shell.n_faces == 0:
        empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3)))
        return empty, empty
    e = edge_info(shell)
    _, labels = vertex_components(shell.n_vertices, e.lo, e.hi)
    face_label = labels[shell.faces[:, 0]]
    # distance level of each component, from its vertices (the field estimate is enough)
    g = fld.grid
    idx = np.clip(np.round((shell.vertices - g.origin) / g.h).astype(np.int64), 0, np.array(g.shape) - 1)
    dv = fld.dist[idx[:, 0], idx[:, 1], idx[:, 2]]
    comp_d = np.bincount(labels, weights=dv) / np.maximum(np.bincount(labels), 1)
    inner_faces = comp_d[face_label] < t + 0.5 * w
    inner = Mesh(shell.vertices, shell.faces[inner_faces][:, ::-1]).compact()
    outer = Mesh(shell.vertices, shell.faces[~inner_faces]).compact()
    return inner, outer


def preview(model: Mesh, p: MoldParams, progress=None) -> dict:
    """Quick look: the shell (coarse), where pour and vents would go, silicone estimate."""
    z0 = default_base(model) if p.base_z is None else p.base_z
    fld = build_field(model, p, exact=False, progress=progress)
    shell, inner, outer = surfaces(fld, p, snap=False)
    zb = z0 - (p.groove_depth if p.base_plate else 0.0)
    shell = _closed_above(shell, zb) if shell.n_faces else shell
    inner_c = _closed_above(inner, z0) if inner.n_faces else inner
    top = cavity_top(inner_c, z0)
    sprue = p.sprue or (top[:2] if top else None)
    vents = [tuple(v[:2]) for v in find_vents(inner_c, sprue, z0)] if p.vents is None else list(p.vents)
    outer_c = _closed_above(outer, z0) if outer.n_faces else outer
    foot = to_cross_section(section_loops(outer_c, (0, 0, 1), (0, 0, z0 + 0.05)),
                            Frame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])))
    model_above = _closed_above(model, z0)
    vol = max(inner_c.volume() - model_above.volume(), 0.0) if inner_c.n_faces else 0.0
    t, w = p.thickness, p.wall
    shell_top = float(shell.bounds[1][2]) if shell.n_faces else z0

    def at_top(xy):  # marker where the tube leaves the shell
        z = fld.column_top(xy[0], xy[1], t + w)
        return (float(xy[0]), float(xy[1]), float(z if z is not None else shell_top))
    tubes = [at_top(v) for v in ([sprue] if sprue else []) + vents]
    z_feet = max([shell_top] + [z + p.chimney for _, _, z in tubes]) + p.feet_height
    feet = [(x, y, z_feet) for x, y in feet_positions(fill_holes(foot), p.feet, p.feet_spread)
            if fld.column_top(x, y, t + w) is not None]
    return {"shell": shell, "z0": z0, "zb": zb, "sprue": at_top(sprue) if sprue else None,
            "vents": [at_top(v) for v in vents], "feet": feet,
            "silicone_ml": vol / 1000.0, "split_angle": default_split_angle(model) if p.split_angle is None
            else p.split_angle, "size": (shell.size.tolist() if shell.n_faces else [0, 0, 0])}


def build(model: Mesh, p: MoldParams, progress=None) -> MoldResult:
    res = MoldResult()
    t, w = p.thickness, p.wall
    z0 = default_base(model) if p.base_z is None else p.base_z
    zb = z0 - (p.groove_depth if p.base_plate else 0.0)
    report(progress, 0.0, "Measuring the model")
    fld = build_field(model, p, exact=True, progress=lambda f, m: report(progress, 0.35 * f, m))
    report(progress, 0.36, "Building the shell")
    shell, inner, outer = surfaces(fld, p)
    if shell.n_faces == 0:
        raise ValueError("empty shell")
    shell = _closed_above(shell, zb)
    inner_c = _closed_above(inner, z0)
    outer_c = _closed_above(outer, zb)
    xy = Frame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    footprint = fill_holes(to_cross_section(section_loops(outer_c, (0, 0, 1), (0, 0, z0 + 0.05)), xy))

    # pour funnel and vents --------------------------------------------------------------------
    report(progress, 0.45, "Adding pour and vents")
    top = cavity_top(inner_c, z0)
    sprue_xy = tuple(p.sprue) if p.sprue else (top[:2] if top else None)
    vents_xy = ([tuple(v[:2]) for v in find_vents(inner_c, sprue_xy, z0)] if p.vents is None
                else [tuple(v) for v in p.vents])
    adds, subs = [], []
    tube_tops = []
    used_vents = []

    def column(x, y):
        """(cavity top, shell outer top) at (x, y), or None if the column misses the cavity."""
        zc = fld.column_top(x, y, t)
        zo = fld.column_top(x, y, t + w)
        if zc is None or zo is None or zc < z0 or zo <= zc:
            return None
        return zc, zo

    def clear_above(x, y, z_from, z_to) -> bool:
        """No other part of the mold is in the way between two heights."""
        g = fld.grid
        i = int(round((x - g.origin[0]) / g.h))
        j = int(round((y - g.origin[1]) / g.h))
        k0 = int(math.ceil((z_from - g.origin[2]) / g.h)) + 1
        col = fld.dist[i, j, max(k0, 0):]
        return bool(np.all(col > t + w - 1e-6))

    if sprue_xy is not None:
        col = column(*sprue_xy)
        if col is None:
            res.warnings.append({"code": "sprue_outside"})
        else:
            zc, zo = col
            rs = p.sprue_diameter / 2
            ztop = zo + p.chimney
            tw = max(1.6, 0.6 * w)
            adds.append(_cyl(sprue_xy[0], sprue_xy[1], zo - 0.5 * w, ztop, rs + tw))
            subs.append(_cyl(sprue_xy[0], sprue_xy[1], zc - 1.0, ztop + 1.0, rs))
            if p.funnel:
                fh = min(p.chimney, 1.5 * p.sprue_diameter)
                adds.append(_cone(sprue_xy[0], sprue_xy[1], ztop - fh, ztop, rs + tw, rs + fh + tw))
                subs.append(_cone(sprue_xy[0], sprue_xy[1], ztop - fh, ztop + OVER, rs, rs + fh + OVER))
            tube_tops.append(ztop)
            if not clear_above(sprue_xy[0], sprue_xy[1], zo, ztop):
                res.warnings.append({"code": "sprue_blocked"})
    rv = p.vent_diameter / 2
    for (x, y) in vents_xy:
        col = column(x, y)
        if col is None or not clear_above(x, y, col[1], col[1] + p.chimney):
            res.warnings.append({"code": "vent_skipped", "at": [x, y]})
            continue
        zc, zo = col
        ztop = zo + p.chimney
        adds.append(_cyl(x, y, zo - 0.5 * w, ztop, rv + 1.2))
        subs.append(_cyl(x, y, zc - 1.0, ztop + 1.0, rv))
        tube_tops.append(ztop)
        used_vents.append((x, y))

    # feet ----------------------------------------------------------------------------------------
    shell_top = float(shell.bounds[1][2])
    z_feet = max([shell_top] + tube_tops) + p.feet_height
    feet_xy = []
    for (x, y) in feet_positions(footprint, p.feet, p.feet_spread):
        zo = fld.column_top(x, y, t + w)
        if zo is None:
            continue
        adds.append(_cyl(x, y, zo - 0.6 * w, z_feet, p.feet_diameter / 2))
        feet_xy.append((x, y))

    report(progress, 0.55, "Adding pour and vents")
    man = to_manifold(shell)
    a = _union(adds)
    if a is not None:
        # tubes and feet must not reach into the cavity
        man = man + (a - to_manifold(inner_c))
    s = _union(subs)
    if s is not None:
        man = man - s
    shell_full = from_manifold(man)

    # split into halves with flanges, bolts and keys ----------------------------------------------
    names = {}
    if p.split:
        report(progress, 0.65, "Splitting the shell")
        ang = math.radians(default_split_angle(model) if p.split_angle is None else p.split_angle)
        n = np.array([math.cos(ang), math.sin(ang), 0.0])
        c = model.bounds.mean(0)
        origin = np.array([c[0], c[1], z0]) + p.split_offset * n
        fr = Frame(origin, np.array([-n[1], n[0], 0.0]), np.array([0.0, 0.0, 1.0]), n)
        cut = plane_cut(shell_full, n, origin)
        halves = {"A": cut.negative, "B": cut.positive}
        if not halves["A"].n_faces or not halves["B"].n_faces:
            raise ValueError("parting plane misses the shell")
        sec = to_cross_section(section_loops(outer_c, n, origin), fr, 1e-3)
        above = m3d.CrossSection.square((1e5, 1e5)).translate((-5e4, z0 - origin[2]))
        # the flange reaches halfway into the shell wall: meeting the outer surface exactly
        # would leave near-tangent faces (it is kept out of the silicone gap below)
        flange2d = ((fill_holes(sec).offset(p.flange_width, m3d.JoinType.Round)
                     - fill_holes(sec).offset(-0.5 * w, m3d.JoinType.Round)) ^ above)
        flange2d = flange2d.simplify(1e-3)
        world = fr.matrix()[:3, :]
        inner_man = to_manifold(inner_c)
        ft = p.flange_thickness
        bolts = _flange_points(sec, p, z0 - origin[2], 0.5) if p.bolt in BOLTS else []
        key_pts = _flange_points(sec, p, z0 - origin[2], 0.0)[1::2] if p.keys else []
        r_b, nut_af, nut_t = BOLTS.get(p.bolt, (0, 0, 0))
        for side, sign in (("A", -1.0), ("B", 1.0)):
            report(progress, 0.7 + (0.1 if side == "B" else 0.0), "Adding flanges")
            hm = to_manifold(halves[side])
            if not flange2d.is_empty():
                slab = flange2d.extrude(ft - GAP).translate((0, 0, GAP if sign > 0 else -ft))
                hm = hm + (slab.transform(world) - inner_man)
            cutters = []
            for (u, v) in bolts if r_b else []:
                cutters.append(_local_cyl(u, v, -ft - 1, ft + 1, r_b))
                if p.nut_traps and side == "B":
                    r_hex = (nut_af + 2 * p.clearance) / math.sqrt(3)
                    hexa = m3d.CrossSection.circle(r_hex, 6).rotate(30).translate((u, v))
                    cutters.append(hexa.extrude(nut_t + 1.0).translate((0, 0, ft - nut_t)))
            peg_r, peg_h = 2.0, 2.5
            for (u, v) in key_pts:
                if side == "A":  # peg standing out of A's flange into B's
                    peg = m3d.Manifold.cylinder(peg_h + GAP + 0.6, peg_r, peg_r, 32).translate((u, v, -GAP - 0.6))
                    hm = hm + peg.transform(world)
                else:
                    cutters.append(m3d.Manifold.cylinder(peg_h + 0.3 + OVER, peg_r + p.clearance,
                                                         peg_r + p.clearance, 32)
                                   .translate((u, v, GAP - OVER)))
            cu = _union(cutters)
            if cu is not None:
                hm = hm - cu.transform(world)
            names[side] = from_manifold(hm)
        res.info["bolts"] = len(bolts) if r_b else 0
        res.info["keys"] = len(key_pts)
    else:
        names["shell"] = shell_full

    # base plate ----------------------------------------------------------------------------------
    if p.base_plate:
        report(progress, 0.85, "Making the base plate")
        names["base"] = _base_plate(model, shell, footprint, p, z0, zb)

    for k, m in names.items():
        res.parts[k] = m

    # numbers -------------------------------------------------------------------------------------
    report(progress, 0.95, "Checking the mold")
    model_above = _closed_above(model, z0)
    channels = 0.0
    if sprue_xy is not None and column(*sprue_xy) is not None:
        zc, zo = column(*sprue_xy)
        channels += math.pi * (p.sprue_diameter / 2) ** 2 * (zo + p.chimney - zc)
    for (x, y) in used_vents:
        zc, zo = column(x, y)
        channels += math.pi * rv ** 2 * (zo + p.chimney - zc)
    vol = max(inner_c.volume() - model_above.volume(), 0.0)
    res.info.update({
        "silicone_ml": vol / 1000.0, "silicone_ml_with_tubes": (vol + channels) / 1000.0,
        "silicone_g": (vol + channels) / 1000.0 * p.density,
        "thickness": _thickness_check(fld, inner, z0, t),
        "sprue": list(sprue_xy) if sprue_xy else None, "vents": [list(v) for v in used_vents],
        "feet": [list(f) for f in feet_xy], "z0": z0, "params": p.to_dict(),
    })
    report(progress, 1.0, "Checking the mold")
    return res


def _local_cyl(u, v, w0, w1, r) -> m3d.Manifold:
    return m3d.Manifold.cylinder(w1 - w0, r, r, _segments(r)).translate((u, v, w0))


def _flange_points(sec: m3d.CrossSection, p: MoldParams, v_min: float, phase: float) -> list[tuple]:
    """Points along the middle of the flange, evenly spaced (bolts at phase 0.5, keys between)."""
    mid = fill_holes(sec).offset(p.flange_width / 2, m3d.JoinType.Round)
    pts = []
    for poly in polygons(mid):
        if len(poly) < 3:
            continue
        loop = np.vstack([poly, poly[:1]])
        seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        total = s[-1]
        n = max(2, int(total // max(p.bolt_spacing, 5.0)))
        for k in range(n):
            target = (k + phase) * total / n
            i = min(max(int(np.searchsorted(s, target)) - 1, 0), len(seg) - 1)
            f = (target - s[i]) / max(seg[i], 1e-12)
            q = loop[i] + f * (loop[i + 1] - loop[i])
            if q[1] > v_min + p.flange_width / 2 + 2.0:
                pts.append((float(q[0]), float(q[1])))
    return pts


def _base_plate(model: Mesh, shell: Mesh, footprint: m3d.CrossSection, p: MoldParams, z0: float, zb: float) -> Mesh:
    xy = Frame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    outline = footprint.offset(max(p.plate_margin, p.flange_width + 2.0), m3d.JoinType.Round).simplify(1e-3)
    floor = zb - p.plate_thickness
    plate = outline.extrude(z0 - floor).translate((0, 0, floor))
    cuts = []
    ring = to_cross_section(section_loops(shell, (0, 0, 1), (0, 0, zb + 0.1)), xy, 1e-3)
    if not ring.is_empty():
        groove = ring.offset(p.clearance, m3d.JoinType.Round)
        cuts.append(groove.extrude(z0 - zb + OVER).translate((0, 0, zb)))
    if p.locator:
        seat = to_cross_section(section_loops(model, (0, 0, 1), (0, 0, z0 + 0.2)), xy, 1e-3)
        if seat.area() > 4.0:
            cuts.append(seat.offset(p.clearance, m3d.JoinType.Round).extrude(0.6 + OVER).translate((0, 0, z0 - 0.6)))
    if cuts:
        plate = plate - _union(cuts)
    if p.rod:
        v = model.vertices
        low = v[np.argmin(v[:, 2])]
        if low[2] > z0 + 0.5:
            plate = plate + _cyl(float(low[0]), float(low[1]), z0 - 0.5, float(low[2]) + 1.0, p.rod_diameter / 2)
    return from_manifold(plate)


def _thickness_check(fld: DistanceField, inner: Mesh, z0: float, t: float) -> dict:
    """Silicone thickness measured on the finished inner surface (away from the base)."""
    v = inner.vertices
    keep = v[:, 2] > z0 + t
    if not keep.any() or fld.tree is None:
        return {}
    d = fld.exact(v[keep], t)
    return {"min": float(d.min()), "max": float(d.max()), "mean": float(d.mean())}
