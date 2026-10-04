"""Operations the engine process runs. Each returns plain picklable data.

Every op takes the PartStore and a progress(frac, msg) callback; results carry
small preview meshes for drawing, never the full-resolution data.
"""
from __future__ import annotations

import os
import uuid

from ..core import hardware
from ..core.analyze import analyze as _analyze
from ..core.cut import plane_cut
from ..core.io_stl import load_stl, save_stl, triangle_count
from ..core.lod import display_mesh
from ..core.mesh import Mesh, report
from ..core.repair import RepairOptions, repair as _repair


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


def _preview(mesh: Mesh, target: int, progress=None):
    p = display_mesh(mesh, max(int(target), 50_000), progress)
    return p.vertices, p.faces


def _sub(progress, a, b):
    return lambda f, m: report(progress, a + (b - a) * f, m)


def op_load(store, progress, path: str, preview_faces: int = 2_000_000):
    _ensure_memory("load", triangle_count(path))
    mesh = load_stl(path, _sub(progress, 0.0, 0.85))
    if mesh.n_faces == 0:
        raise EngineError("empty")
    pid = _new_id()
    store.put(pid, mesh)
    name = os.path.splitext(os.path.basename(path))[0]
    return {"part": part_info(pid, name, mesh),
            "preview": _preview(mesh, preview_faces, _sub(progress, 0.85, 1.0))}


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
    store.put(pid, fixed)
    return {"part": part_info(pid, name, fixed), "log": log,
            "preview": _preview(fixed, preview_faces, _sub(progress, 0.85, 1.0))}


def op_cut(store, progress, pid: str, name: str, normal, origin, keep: str = "both",
           preview_faces: int = 2_000_000):
    mesh = store.get(pid)
    _ensure_memory("cut", mesh.n_faces)
    res = plane_cut(mesh, normal, origin, _sub(progress, 0.0, 0.7))
    sides = [("positive", "A", res.positive), ("negative", "B", res.negative)]
    sides = [s for s in sides if s[2].n_faces and keep in ("both", s[0])]
    if not sides or (len(sides) == 1 and sides[0][2].n_faces == mesh.n_faces):
        raise EngineError("no_cut")
    parts = []
    for i, (side, tag, part) in enumerate(sides):
        new = _new_id()
        store.put(new, part)
        a = 0.7 + 0.3 * i / len(sides)
        parts.append({"part": part_info(new, f"{name}_{tag}", part), "side": side,
                      "preview": _preview(part, preview_faces, _sub(progress, a, a + 0.3 / len(sides)))})
    store.delete(pid)
    return {"removed": pid, "parts": parts, "loops": res.loops,
            "open_loops": res.open_loops, "warnings": res.warnings}


def op_export(store, progress, items: list):
    """items: [(pid, path)] -> written paths."""
    out = []
    for i, (pid, path) in enumerate(items):
        mesh = store.get(pid)
        save_stl(mesh, path, _sub(progress, i / len(items), (i + 1) / len(items)))
        out.append(path)
    return {"paths": out}


def op_delete(store, progress, pid: str):
    store.delete(pid)
    return {"removed": pid}


def op_sysinfo(store, progress):
    return hardware.system_info()


OPS = {
    "load": op_load,
    "analyze": op_analyze,
    "repair": op_repair,
    "cut": op_cut,
    "export": op_export,
    "delete": op_delete,
    "sysinfo": op_sysinfo,
}
