"""Grid cutting (several axis planes at once) and Fit to Printer."""
from __future__ import annotations

import math

import numpy as np

from .cut import plane_cut
from .mesh import Mesh, report

AXES = "xyz"


def even_planes(lo: float, hi: float, count: int) -> list[float]:
    """Positions splitting [lo, hi] into `count` equal slabs."""
    if count <= 1:
        return []
    step = (hi - lo) / count
    return [lo + step * i for i in range(1, count)]


def fit_counts(size, volume, margin: float = 0.0):
    """Fewest pieces per axis so each fits the printer; also tries a 90° turn.

    Returns (counts (nx, ny, nz), rotated) where rotated means the pieces fit
    when turned 90 degrees on the bed.
    """
    w, d, h = volume
    usable = [max(w - 2 * margin, 1e-6), max(d - 2 * margin, 1e-6), max(h - margin, 1e-6)]
    best = None
    for rotated in (False, True):
        ux, uy = (usable[1], usable[0]) if rotated else (usable[0], usable[1])
        counts = (max(1, math.ceil(size[0] / ux - 1e-9)), max(1, math.ceil(size[1] / uy - 1e-9)),
                  max(1, math.ceil(size[2] / usable[2] - 1e-9)))
        key = (counts[0] * counts[1] * counts[2], counts[2], rotated)
        if best is None or key < best[0]:
            best = (key, counts, rotated)
    return best[1], best[2]


def fit_planes(bounds, volume, margin: float = 0.0):
    """Evenly spaced planes [(axis, position)] for Fit to Printer."""
    b = np.asarray(bounds, float)
    counts, rotated = fit_counts(b[1] - b[0], volume, margin)
    planes = [(k, p) for k in range(3) for p in even_planes(b[0][k], b[1][k], counts[k])]
    return planes, counts, rotated


def grid_cut(mesh: Mesh, planes, progress=None):
    """Cut by axis planes [(axis, position)].

    Returns [((i, j, k) cell index, Mesh)] for every non-empty cell.
    """
    by_axis = {k: sorted(p for a, p in planes if a == k) for k in range(3)}
    pieces = [((0, 0, 0), mesh)]
    total = max(1, sum(len(v) for v in by_axis.values()))
    done = 0
    for k in range(3):
        nxt = []
        for idx, piece in pieces:
            rest = piece
            cell = 0
            for pos in by_axis[k]:
                lo, hi = rest.bounds[0][k], rest.bounds[1][k]
                if rest.n_faces and lo < pos < hi:
                    normal = np.eye(3)[k]
                    origin = np.zeros(3)
                    origin[k] = pos
                    res = plane_cut(rest, normal, origin)
                    if res.negative.n_faces:
                        nxt.append((_with(idx, k, cell), res.negative))
                    rest = res.positive
                elif rest.n_faces and hi <= pos:
                    break
                cell += 1
            if rest.n_faces:
                nxt.append((_with(idx, k, _cell_of(rest, k, by_axis[k])), rest))
        done += len(by_axis[k])
        report(progress, done / total, "Cutting grid")
        pieces = nxt
    return pieces


def _with(idx, k, value):
    out = list(idx)
    out[k] = value
    return tuple(out)


def _cell_of(mesh: Mesh, k: int, positions) -> int:
    center = (mesh.bounds[0][k] + mesh.bounds[1][k]) / 2
    return int(sum(1 for p in positions if p < center))


def cell_name(base: str, idx, counts) -> str:
    tags = [f"{AXES[k]}{idx[k] + 1}" for k in range(3) if counts[k] > 1]
    return base + ("_" + "_".join(tags) if tags else "")
