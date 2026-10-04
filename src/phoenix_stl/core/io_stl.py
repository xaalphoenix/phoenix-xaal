"""Fast STL reading/writing.

Binary files are memory-mapped and copied block-wise into one float32 soup,
which is welded in place; a 700 MB file (14M triangles) peaks around 2 GB.
"""
from __future__ import annotations

import os
import re

import numpy as np

from .mesh import Mesh, report
from .weld import weld_points, weld_soup

STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
WRITE_CHUNK = 1_000_000
READ_CHUNK = 1_000_000


class STLError(Exception):
    pass


def is_binary(path: str) -> bool:
    size = os.path.getsize(path)
    if size < 84:
        with open(path, "rb") as f:
            return not f.read(5).lower().startswith(b"solid")
    with open(path, "rb") as f:
        head = f.read(84)
    n = int.from_bytes(head[80:84], "little")
    if 84 + 50 * n == size:
        return True
    # Some exporters write "solid" into binary headers; the size check above wins.
    return not head[:5].lower().startswith(b"solid")


def triangle_count(path: str) -> int:
    """Cheap estimate used for memory checks before loading."""
    if is_binary(path):
        with open(path, "rb") as f:
            f.seek(80)
            return int.from_bytes(f.read(4), "little")
    return os.path.getsize(path) // 250


def load_stl(path: str, progress=None) -> Mesh:
    if not os.path.isfile(path):
        raise STLError(f"File not found: {path}")
    if is_binary(path):
        return _load_binary(path, progress)
    return _load_ascii(path, progress)


def _load_binary(path: str, progress) -> Mesh:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(80)
        n = int.from_bytes(f.read(4), "little")
    avail = (size - 84) // 50
    if n == 0 or n > avail:
        # Header count is wrong (it happens); trust the file size.
        n = avail
    if n == 0:
        raise STLError("STL file contains no triangles")
    data = np.memmap(path, dtype=STL_DTYPE, mode="r", offset=84, shape=(n,))
    soup = np.empty((3 * n, 3), dtype=np.float32)
    for s in range(0, n, READ_CHUNK):
        e = min(s + READ_CHUNK, n)
        soup[3 * s:3 * e] = data["v"][s:e].reshape(-1, 3)
        report(progress, 0.25 * e / n, "Reading STL")
    del data
    verts, inv = weld_points(soup, 0.0, lambda f, m: report(progress, 0.25 + 0.75 * f, "Welding vertices"))
    del soup
    return Mesh(verts, inv.reshape(-1, 3))


_FLOAT = rb"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
_VERTEX_RE = re.compile(rb"vertex\s+" + _FLOAT + rb"\s+" + _FLOAT + rb"\s+" + _FLOAT)


def _load_ascii(path: str, progress) -> Mesh:
    report(progress, 0.05, "Reading ASCII STL")
    with open(path, "rb") as f:
        text = f.read()
    vals = _VERTEX_RE.findall(text)
    del text
    if not vals or len(vals) % 3:
        raise STLError("Could not parse ASCII STL (vertex count is not a multiple of 3)")
    pts = np.array(vals, dtype=np.float32)
    del vals
    report(progress, 0.4, "Welding vertices")
    return weld_soup(pts.reshape(-1, 3, 3),
                     progress=lambda f, m: report(progress, 0.4 + 0.6 * f, "Welding vertices"))


def save_stl(mesh: Mesh, path: str, progress=None, header: str = "PHOENIX STL Studio") -> None:
    """Write binary STL (mm), streaming in blocks."""
    n = mesh.n_faces
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii", "replace")[:80].ljust(80, b" "))
        f.write(np.uint32(n).tobytes())
        for s in range(0, n, WRITE_CHUNK):
            tri = mesh.vertices[mesh.faces[s:s + WRITE_CHUNK]]
            rec = np.zeros(len(tri), dtype=STL_DTYPE)
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            np.divide(nrm, ln, out=nrm, where=ln > 0)
            rec["normal"] = nrm
            rec["v"] = tri
            rec.tofile(f)
            report(progress, min(s + WRITE_CHUNK, n) / max(n, 1), "Writing STL")
    os.replace(tmp, path)
