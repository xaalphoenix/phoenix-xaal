"""Full-resolution parts live on disk (.npy) so a killed engine loses nothing.

Parts are immutable: every edit writes a new part id and the old files stay
until the UI's undo history no longer needs them (then they are purged).
Each part also keeps its display preview, so undo can bring it back at once.
"""
from __future__ import annotations

import os
import shutil
from collections import OrderedDict

import numpy as np

from ..core.mesh import Mesh

CACHE_PARTS = 2
SUFFIXES = (".v.npy", ".f.npy", ".pv.npy", ".pf.npy")


def _save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


class PartStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._cache: OrderedDict[str, Mesh] = OrderedDict()

    def _path(self, pid: str, suffix: str) -> str:
        return os.path.join(self.root, pid + suffix)

    def put(self, pid: str, mesh: Mesh, preview: Mesh | None = None) -> None:
        _save(self._path(pid, ".v.npy"), mesh.vertices)
        _save(self._path(pid, ".f.npy"), mesh.faces)
        if preview is not None:
            self.put_preview(pid, preview)
        self._remember(pid, mesh)

    def put_preview(self, pid: str, preview: Mesh) -> None:
        _save(self._path(pid, ".pv.npy"), preview.vertices)
        _save(self._path(pid, ".pf.npy"), preview.faces)

    def get(self, pid: str) -> Mesh:
        if pid in self._cache:
            self._cache.move_to_end(pid)
            return self._cache[pid]
        pv = self._path(pid, ".v.npy")
        if not os.path.exists(pv):
            raise KeyError(f"unknown part {pid}")
        mesh = Mesh(np.load(pv), np.load(self._path(pid, ".f.npy")))
        self._remember(pid, mesh)
        return mesh

    def get_preview(self, pid: str) -> Mesh | None:
        pv = self._path(pid, ".pv.npy")
        if not os.path.exists(pv):
            return None
        return Mesh(np.load(pv), np.load(self._path(pid, ".pf.npy")))

    def delete(self, pid: str) -> None:
        self._cache.pop(pid, None)
        for s in SUFFIXES:
            p = self._path(pid, s)
            if os.path.exists(p):
                os.remove(p)

    def exists(self, pid: str) -> bool:
        return os.path.exists(self._path(pid, ".v.npy"))

    def drop_cache(self) -> None:
        self._cache.clear()

    def _remember(self, pid: str, mesh: Mesh) -> None:
        self._cache[pid] = mesh
        self._cache.move_to_end(pid)
        while len(self._cache) > CACHE_PARTS:
            self._cache.popitem(last=False)

    def destroy(self) -> None:
        self._cache.clear()
        shutil.rmtree(self.root, ignore_errors=True)
