"""Operations the engine process runs. Each returns plain picklable data.

Every op takes the PartStore and a progress(frac, msg) callback; results carry
small preview meshes for drawing, never the full-resolution data.

Parts are immutable. An edit returns a change set {"removed": [ids],
"added": [{"part": info, "preview": (V, F)}]}; the old files stay on disk so
the UI can undo, and are deleted later with "purge".
"""
from __future__ import annotations

import os
import uuid

import numpy as np

from ..core import hardware
from ..core.analyze import analyze as _analyze
from ..core import connectors as conn
from ..core.boolean import NotSolid, union
from ..core.cut import plane_cut
from ..core.grid import cell_name, grid_cut
from ..core.surface_cut import Frame, curve_cutters, cut_with, height_function, heightfield_solid
from ..core.io_stl import load_stl, save_stl, triangle_count
from ..core.lod import display_mesh
from ..core.mesh import Mesh, report
from ..core.repair import RepairOptions, repair as _repair
from ..core.section import find_joint, flat_face_at
from ..core.transform import transform_mesh, translation


class EngineError(Exception):
    """Error with a stable code the UI translates (e.g. 'memory', 'empty')."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _ensure_memory(op: str, n_faces: int) -> None:
    ok, need, avail = hardware.check_memory(op, n_faces)
    if not ok:
        raise EngineError("memory", f"{need / 2**30:.1f}|{avail / 2**30:.1f}")


def part_info(pid: str, name: str, mesh: Mesh) -> dict:
    b = mesh.bounds
    return {"id": pid, "name": name, "n_faces": mesh.n_faces, "n_vertices": mesh.n_vertices,
            "bounds": b.tolist(), "size": (b[1] - b[0]).tolist()}


def _sub(progress, a, b):
    return lambda f, m: report(progress, a + (b - a) * f, m)


def _store_new(store, mesh: Mesh, name: str, preview_faces: int, progress=None,
               preview: Mesh | None = None, **extra) -> dict:
    """Save a new part (and its preview); returns its 'added' entry."""
    pid = _new_id()
    if preview is None:
        preview = display_mesh(mesh, max(int(preview_faces), 50_000), progress)
    store.put(pid, mesh, preview)
    return {"part": part_info(pid, name, mesh), "preview": (preview.vertices, preview.faces), **extra}


def op_load(store, progress, path: str, preview_faces: int = 2_000_000):
    _ensure_memory("load", triangle_count(path))
    mesh = load_stl(path, _sub(progress, 0.0, 0.85))
    if mesh.n_faces == 0:
        raise EngineError("empty")
    name = os.path.splitext(os.path.basename(path))[0]
    return {"removed": [], "added": [_store_new(store, mesh, name, preview_faces, _sub(progress, 0.85, 1.0))]}


def op_analyze(store, progress, pid: str):
    mesh = store.get(pid)
    _ensure_memory("analyze", mesh.n_faces)
    rep, segs = _analyze(mesh, progress)
    return {"id": pid, "report": rep.to_dict(), "segments": segs}


def op_repair(store, progress, pid: str, name: str, options: dict | None = None,
              preview_faces: int = 2_000_000):
    mesh = store.get(pid)
    _ensure_memory("repair", mesh.n_faces)
    fixed, log = _repair(mesh, RepairOptions.from_dict(options), _sub(progress, 0.0, 0.85))
    return {"removed": [pid], "log": log,
            "added": [_store_new(store, fixed, name, preview_faces, _sub(progress, 0.85, 1.0))]}


def op_cut(store, progress, pid: str, name: str, normal, origin, keep: str = "both",
           preview_faces: int = 2_000_000):
    mesh = store.get(pid)
    _ensure_memory("cut", mesh.n_faces)
    res = plane_cut(mesh, normal, origin, _sub(progress, 0.0, 0.7))
    sides = [("positive", "A", res.positive), ("negative", "B", res.negative)]
    sides = [s for s in sides if s[2].n_faces and keep in ("both", s[0])]
    if not sides or (len(sides) == 1 and sides[0][2].n_faces == mesh.n_faces):
        raise EngineError("no_cut")
    added = []
    for i, (side, tag, part) in enumerate(sides):
        a = 0.7 + 0.3 * i / len(sides)
        added.append(_store_new(store, part, f"{name}_{tag}", preview_faces,
                                _sub(progress, a, a + 0.3 / len(sides)), side=side))
    return {"removed": [pid], "added": added, "loops": res.loops,
            "open_loops": res.open_loops, "warnings": res.warnings}


def op_grid_cut(store, progress, pid: str, name: str, planes: list, preview_faces: int = 2_000_000):
    """planes: [(axis, position)] -> one part per grid cell."""
    mesh = store.get(pid)
    _ensure_memory("cut", 2 * mesh.n_faces)
    pieces = sorted(grid_cut(mesh, [(int(a), float(p)) for a, p in planes], _sub(progress, 0.0, 0.75)),
                    key=lambda x: x[0])
    if len(pieces) < 2:
        raise EngineError("no_cut")
    counts = [1 + sum(1 for a, _ in planes if int(a) == k) for k in range(3)]
    each = max(50_000, preview_faces // len(pieces))
    added = []
    for i, (idx, part) in enumerate(pieces):
        a = 0.75 + 0.25 * i / len(pieces)
        added.append(_store_new(store, part, cell_name(name, idx, counts), each,
                                _sub(progress, a, a + 0.25 / len(pieces))))
    return {"removed": [pid], "added": added}


def op_surface_cut(store, progress, pid: str, name: str, kind: str, frame: dict, params: dict,
                   gap: float = 0.0, keep: str = "both", preview_faces: int = 2_000_000):
    """Cut along a drawn curve (kind "curve") or a bent surface (kind "freeform")."""
    mesh = store.get(pid)
    _ensure_memory("boolean", 2 * mesh.n_faces)
    fr = _frame_of(frame)
    b = mesh.bounds
    reach = 2.0 * float(np.linalg.norm(b[1] - b[0])) + float(np.linalg.norm((b[0] + b[1]) / 2 - fr.origin))
    report(progress, 0.05, "Preparing cut")
    if kind == "curve":
        ca, cb = curve_cutters(np.asarray(params["curve"], float), fr, reach, gap)
    else:
        hf = height_function(np.asarray(params["heights"], float), params["u_range"], params["v_range"])
        res = int(params.get("res", 96))
        ca = heightfield_solid(hf, fr, params["u_range"], params["v_range"], reach, res)
        cb = heightfield_solid(hf, fr, params["u_range"], params["v_range"], reach, res, shift=gap) if gap else ca
    try:
        side_a, side_b = cut_with(mesh, ca, cb, _sub(progress, 0.05, 0.75))
    except NotSolid as e:
        raise EngineError("not_solid_cut", str(e)) from None
    sides = [("positive", "A", side_a), ("negative", "B", side_b)]
    sides = [s for s in sides if s[2].n_faces and keep in ("both", s[0])]
    if not sides or all(abs(s[2].volume() - mesh.volume()) < 1e-6 * abs(mesh.volume()) for s in sides):
        raise EngineError("no_cut")
    added = []
    for i, (side, tag, part) in enumerate(sides):
        a = 0.75 + 0.25 * i / len(sides)
        added.append(_store_new(store, part, f"{name}_{tag}", preview_faces,
                                _sub(progress, a, a + 0.25 / len(sides)), side=side))
    return {"removed": [pid], "added": added, "open_loops": 0, "loops": 0, "warnings": []}


def _frame_of(frame: dict) -> Frame:
    return Frame(*(np.asarray(frame[k], float) for k in ("origin", "u", "v", "w")))


def _frame_dict(fr: Frame) -> dict:
    return {k: np.asarray(getattr(fr, k), float).tolist() for k in ("origin", "u", "v", "w")}


def _previews(store, pids) -> dict:
    """Light meshes per side, for ray tests that need no full detail."""
    out = {}
    for side, pid in zip("AB", pids):
        prev = store.get_preview(pid)
        if prev is not None:
            out[side] = prev
    return out


def op_joint(store, progress, pids: list, point=None, normal=None):
    """The joint face of two touching parts, or the flat face of one part through a clicked point.

    Returns the face frame (w points out of the first part) and its outlines.
    """
    report(progress, 0.05, "Finding the joint")
    meshes = [store.get(pid) for pid in pids]
    if len(meshes) == 2:
        j = find_joint(meshes[0], meshes[1])
        if j is None:
            raise EngineError("no_joint")
    else:
        j = flat_face_at(meshes[0], point, normal)
        if j is None:
            raise EngineError("no_face")
    fr = Frame.from_normal(j["origin"], j["normal"])
    shape = conn.joint_shape(dict(zip("AB", meshes)), fr, ray_meshes=_previews(store, pids),
                             progress=_sub(progress, 0.3, 1.0))
    if shape.solid.area() <= 1e-9:
        raise EngineError("no_joint")
    return {"pids": list(pids), "frame": _frame_dict(fr), "shape": shape.to_dict(),
            "area": shape.solid.area(), "hollow": shape.hollow}


def op_connect(store, progress, pids: list, names: list, frame: dict, spec: dict, fit: dict,
               placements: list, male: str = "A", boss: bool = True, preview_faces: int = 2_000_000):
    """Add connectors to one part or to both sides of a joint. Dowel pins come out as an extra part."""
    sides = "AB"[:len(pids)]
    meshes = {side: store.get(pid) for side, pid in zip(sides, pids)}
    _ensure_memory("boolean", sum(m.n_faces for m in meshes.values()))
    fr = _frame_of(frame)
    shape = conn.joint_shape(meshes, fr, ray_meshes=_previews(store, pids), progress=_sub(progress, 0.0, 0.15))
    reach = 2.0 * max(float(np.linalg.norm(m.size)) for m in meshes.values()) + 10.0
    pl = conn.plan(conn.Spec(**spec), conn.Fit(**fit), [conn.Placement(**p) for p in placements], shape,
                   sides, male, boss, reach)
    try:
        out, pins, warnings = conn.apply_plan(meshes, pl, fr, _sub(progress, 0.15, 0.8))
    except NotSolid as e:
        raise EngineError("not_solid_cut", str(e)) from None
    added = []
    for i, side in enumerate(sides):
        a = 0.8 + 0.15 * i / len(sides)
        added.append(_store_new(store, out[side], names[i], preview_faces, _sub(progress, a, a + 0.15 / len(sides))))
    if pins is not None:
        lo = np.min([m.bounds[0] for m in out.values()], axis=0)
        hi = np.max([m.bounds[1] for m in out.values()], axis=0)
        pb = pins.bounds
        pins = transform_mesh(pins, translation((hi[0] + 10 - pb[0][0], lo[1] - pb[0][1], lo[2] - pb[0][2])))
        added.append(_store_new(store, pins, f"{names[0]}_pins", preview_faces, pins=True))
    return {"removed": list(pids), "added": added, "warnings": warnings}


def op_coupon(store, progress, diameter: float, clearances: list, chamfer: float = 0.3, name: str = "coupon",
              preview_faces: int = 2_000_000):
    """Tolerance test plate (one hole per clearance) and its test pin."""
    report(progress, 0.1, "Making the coupon")
    plate, pin = conn.tolerance_coupon(diameter, clearances, chamfer)
    return {"removed": [], "added": [_store_new(store, plate, name, preview_faces),
                                     _store_new(store, pin, f"{name}_pin", preview_faces)]}


def op_transform(store, progress, items: list, on_bed: bool = False, centered: bool = False):
    """items: [(pid, name, 4x4 matrix)] -> each part replaced by its moved copy.

    on_bed / centered then shift the whole group exactly (full-resolution
    bounds): lowest point to Z = 0, bounding-box centre to X = Y = 0.
    """
    moved = []
    for i, (pid, name, matrix) in enumerate(items):
        m = np.asarray(matrix, dtype=np.float64)
        mesh = store.get(pid)
        _ensure_memory("transform", mesh.n_faces)
        prev = store.get_preview(pid)
        moved.append((name, transform_mesh(mesh, m), transform_mesh(prev, m) if prev is not None else None))
        store.drop_cache()
        report(progress, 0.8 * (i + 1) / len(items), "Moving parts")
    if on_bed or centered:
        lo = np.min([mm.bounds[0] for _, mm, _ in moved], axis=0)
        hi = np.max([mm.bounds[1] for _, mm, _ in moved], axis=0)
        shift = np.zeros(3)
        if centered:
            shift[:2] = -(lo[:2] + hi[:2]) / 2
        if on_bed:
            shift[2] = -lo[2]
        tm = translation(shift)
        moved = [(n, transform_mesh(mm, tm), transform_mesh(pp, tm) if pp is not None else None)
                 for n, mm, pp in moved]
    added = [_store_new(store, mm, n, 1_000_000, preview=pp) for n, mm, pp in moved]
    report(progress, 1.0, "Moving parts")
    return {"removed": [pid for pid, _, _ in items], "added": added}


def op_merge(store, progress, pids: list, name: str, mode: str = "union",
             preview_faces: int = 2_000_000):
    meshes = [store.get(pid) for pid in pids]
    total = sum(m.n_faces for m in meshes)
    report(progress, 0.05, "Merging parts")
    if mode == "union":
        _ensure_memory("boolean", total)
        try:
            merged = union(meshes)
        except NotSolid as e:
            raise EngineError("not_solid", str(e)) from None
    else:
        merged = Mesh.concatenate(meshes)
    store.drop_cache()
    return {"removed": list(pids),
            "added": [_store_new(store, merged, name, preview_faces, _sub(progress, 0.8, 1.0))]}


def op_restore(store, progress, pids: list):
    """Previews of parts brought back by undo/redo."""
    out = []
    for pid in pids:
        prev = store.get_preview(pid)
        if prev is None:
            prev = display_mesh(store.get(pid), 1_000_000)
        out.append({"id": pid, "preview": (prev.vertices, prev.faces)})
    return {"parts": out}


def op_export(store, progress, items: list):
    """items: [(pid, path)] -> written paths."""
    out = []
    for i, (pid, path) in enumerate(items):
        mesh = store.get(pid)
        save_stl(mesh, path, _sub(progress, i / len(items), (i + 1) / len(items)))
        out.append(path)
    return {"paths": out}


def op_purge(store, progress, pids: list):
    """Delete parts no undo step can bring back any more."""
    for pid in pids:
        store.delete(pid)
    return {"purged": list(pids)}


def op_sysinfo(store, progress):
    return hardware.system_info()


OPS = {
    "load": op_load,
    "analyze": op_analyze,
    "repair": op_repair,
    "cut": op_cut,
    "grid_cut": op_grid_cut,
    "surface_cut": op_surface_cut,
    "joint": op_joint,
    "connect": op_connect,
    "coupon": op_coupon,
    "transform": op_transform,
    "merge": op_merge,
    "restore": op_restore,
    "export": op_export,
    "purge": op_purge,
    "sysinfo": op_sysinfo,
}
