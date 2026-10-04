"""Mesh health check: is it a closed, consistently oriented solid?"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .mesh import CHUNK, Mesh, face_metrics, report
from .weld import group_rows

MAX_PROBLEM_SEGMENTS = 200_000


@dataclass
class EdgeInfo:
    """Undirected edge table built from the 3*m half-edges of a mesh."""
    lo: np.ndarray  # (k,) int32 smaller vertex index of each unique edge, non-decreasing
    hi: np.ndarray  # (k,) int32
    count: np.ndarray  # (k,) how many half-edges use the edge
    consistent: np.ndarray  # (k,) bool: count == 2 and the two half-edges are opposite
    start: np.ndarray  # (k,) position of each edge's first half-edge in sorted order
    half_order: np.ndarray | None  # half-edge ids sorted by edge (only if requested)

    def keys(self, n_vertices: int) -> np.ndarray:
        return self.lo.astype(np.int64) * np.int64(n_vertices) + self.hi


def _half_edge_keys(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Undirected key << 1 | direction, for each half-edge (face-major order)."""
    n = np.int64(max(n_vertices, 1))
    key = np.empty(faces.size, dtype=np.int64)
    for s in range(0, len(faces), CHUNK):
        a = faces[s:s + CHUNK].astype(np.int64)
        b = np.roll(a, -1, axis=1)
        k = np.minimum(a, b) * n + np.maximum(a, b)
        k <<= 1
        k |= a > b
        key[3 * s:3 * s + k.size] = k.ravel()
    return key


def edge_info(mesh: Mesh, with_half_edges: bool = False) -> EdgeInfo:
    key = _half_edge_keys(mesh.faces, mesh.n_vertices)
    n = np.int64(max(mesh.n_vertices, 1))
    if with_half_edges:
        half_order = np.argsort(key, kind="stable")
        key = key[half_order]
    else:
        half_order = None
        key.sort()
    direction = (key & 1).astype(bool)
    key >>= 1
    starts = np.empty(len(key), dtype=bool)
    if len(key):
        starts[0] = True
        np.not_equal(key[1:], key[:-1], out=starts[1:])
    pos = np.nonzero(starts)[0]
    del starts
    count = np.diff(np.append(pos, len(key))).astype(np.int32)
    uk = key[pos]
    del key
    lo = (uk // n).astype(np.int32)
    hi = (uk - lo.astype(np.int64) * n).astype(np.int32)
    consistent = np.zeros(len(pos), dtype=bool)
    two = count == 2
    p2 = pos[two]
    consistent[two] = direction[p2] != direction[p2 + 1]
    return EdgeInfo(lo, hi, count, consistent, pos, half_order)


def boundary_half_edges(mesh: Mesh, e: EdgeInfo):
    """Directed (a, b) of every half-edge on an open boundary."""
    bnd = e.count == 1
    if not bnd.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    bkeys = e.keys(mesh.n_vertices)[bnd]  # sorted
    key = _half_edge_keys(mesh.faces, mesh.n_vertices) >> 1
    i = np.clip(np.searchsorted(bkeys, key), 0, len(bkeys) - 1)
    h = np.nonzero(bkeys[i] == key)[0]
    f, corner = h // 3, h % 3
    a = mesh.faces[f, corner].astype(np.int64)
    b = mesh.faces[f, (corner + 1) % 3].astype(np.int64)
    return a, b


def vertex_components(n_vertices: int, lo: np.ndarray, hi: np.ndarray):
    """(n_components, label per vertex) of the graph given by edges lo-hi.

    lo must be non-decreasing (as edge_info returns it), so the CSR structure
    can be built directly without scipy re-sorting 20M+ entries.
    """
    indptr = np.searchsorted(lo, np.arange(n_vertices + 1)).astype(np.int64)
    g = csr_matrix((np.ones(len(lo), dtype=np.int8), hi.astype(np.int32), indptr),
                   shape=(n_vertices, n_vertices))
    return connected_components(g, directed=False)


def degenerate_mask(mesh: Mesh, areas: np.ndarray) -> np.ndarray:
    f = mesh.faces
    bad = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
    diag = float(np.linalg.norm(mesh.size)) or 1.0
    return bad | (areas <= (1e-9 * diag) ** 2)


def duplicate_mask(mesh: Mesh, e: EdgeInfo, progress=None) -> np.ndarray:
    """True for every face that repeats the vertex set of an earlier face.

    A duplicate always makes its edges non-manifold (used 3+ times), so only
    faces touching such edges need to be compared.
    """
    dup = np.zeros(mesh.n_faces, dtype=bool)
    nm = e.count > 2
    if not nm.any():
        return dup
    nm_keys = e.keys(mesh.n_vertices)[nm]
    key = (_half_edge_keys(mesh.faces, mesh.n_vertices) >> 1).reshape(-1, 3)
    hit = np.zeros(mesh.n_faces, dtype=bool)
    for k in range(3):
        i = np.clip(np.searchsorted(nm_keys, key[:, k]), 0, len(nm_keys) - 1)
        hit |= nm_keys[i] == key[:, k]
    del key
    cand = np.nonzero(hit)[0]
    f = np.sort(mesh.faces[cand], axis=1).astype(np.float64)
    first, _ = group_rows(len(f), lambda ix: f[ix], progress)
    keep = np.zeros(len(cand), dtype=bool)
    keep[first] = True
    dup[cand[~keep]] = True
    return dup


@dataclass
class Topology:
    """Everything analyze and repair need, computed in one pass."""
    edges: EdgeInfo
    areas: np.ndarray
    vols: np.ndarray
    n_shells: int
    face_shell: np.ndarray  # compact shell id per face

    @classmethod
    def of(cls, mesh: Mesh, progress=None) -> "Topology":
        report(progress, 0.05, "Measuring")
        areas, vols = face_metrics(mesh)
        report(progress, 0.25, "Checking edges")
        e = edge_info(mesh)
        report(progress, 0.6, "Counting shells")
        _, labels = vertex_components(mesh.n_vertices, e.lo, e.hi)
        fs = labels[mesh.faces[:, 0]] if mesh.n_faces else np.zeros(0, np.int64)
        uniq, fs = np.unique(fs, return_inverse=True)
        return cls(e, areas, vols, len(uniq), fs.astype(np.int64).ravel())

    def shell_volumes(self) -> np.ndarray:
        return np.bincount(self.face_shell, weights=self.vols, minlength=self.n_shells)


@dataclass
class Report:
    n_faces: int = 0
    n_vertices: int = 0
    size: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounds: list = field(default_factory=lambda: [[0.0] * 3, [0.0] * 3])
    volume: float = 0.0
    area: float = 0.0
    boundary_edges: int = 0
    holes: int = 0
    nonmanifold_edges: int = 0
    flipped_edges: int = 0
    degenerate_faces: int = 0
    duplicate_faces: int = 0
    shells: int = 0
    inside_out: bool = False

    @property
    def watertight(self) -> bool:
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0

    @property
    def printable(self) -> bool:
        return (self.watertight and self.flipped_edges == 0 and not self.inside_out
                and self.n_faces > 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["watertight"] = self.watertight
        d["printable"] = self.printable
        return d


def analyze(mesh: Mesh, progress=None, with_segments: bool = True, topo: Topology | None = None):
    """Returns (Report, problem_segments (k, 2, 3) float32 or None)."""
    r = Report(n_faces=mesh.n_faces, n_vertices=mesh.n_vertices)
    if mesh.n_faces == 0:
        return r, None
    b = mesh.bounds
    r.bounds = b.tolist()
    r.size = (b[1] - b[0]).tolist()
    t = topo or Topology.of(mesh, lambda f, m: report(progress, 0.85 * f, m))
    e = t.edges
    r.area = float(t.areas.sum())
    r.volume = float(t.vols.sum())
    r.degenerate_faces = int(degenerate_mask(mesh, t.areas).sum())
    boundary = e.count == 1
    nonman = e.count > 2
    r.boundary_edges = int(boundary.sum())
    r.nonmanifold_edges = int(nonman.sum())
    r.flipped_edges = int(((e.count == 2) & ~e.consistent).sum())
    report(progress, 0.88, "Finding holes")
    if r.boundary_edges:
        bl, bh = e.lo[boundary], e.hi[boundary]
        verts, local = np.unique(np.concatenate([bl, bh]), return_inverse=True)
        k = len(bl)
        order = np.argsort(local[:k], kind="stable")
        r.holes = vertex_components(len(verts), local[:k][order], local[k:][order])[0]
    r.shells = t.n_shells
    sv = t.shell_volumes()
    r.inside_out = bool(sv[np.argmax(np.abs(sv))] < 0)
    report(progress, 0.93, "Looking for duplicates")
    r.duplicate_faces = int(duplicate_mask(mesh, e).sum())
    segs = None
    if with_segments and (r.boundary_edges or r.nonmanifold_edges):
        idx = np.nonzero(boundary | nonman)[0][:MAX_PROBLEM_SEGMENTS]
        segs = np.stack([mesh.vertices[e.lo[idx]], mesh.vertices[e.hi[idx]]], axis=1)
    report(progress, 1.0, "Done")
    return r, segs
