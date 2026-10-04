"""Full-resolution parts live on disk (.npy) so a killed engine loses nothing."""
from __future__ import annotations

import os
import shutil
from collections import OrderedDict

import numpy as np

from ..core.mesh import Mesh

CACHE_PARTS = 2


class PartStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._cache: OrderedDict[str, Mesh] = OrderedDict()

    def _paths(self, pid: str):
        return os.path.join(self.root, f"{pid}.v.npy"), os.path.join(self.root, f"{pid}.f.npy")

    def put(self, pid: str, mesh: Mesh) -> None:
        pv, pf = self._paths(pid)
        for path, arr in ((pv, mesh.vertices), (pf, mesh.faces)):
            tmp = path + ".tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, path)
        self._remember(pid, mesh)

    def get(self, pid: str) -> Mesh:
        if pid in self._cache:
            self._cache.move_to_end(pid)
            return self._cache[pid]
        pv, pf = self._paths(pid)
        if not os.path.exists(pv):
            raise KeyError(f"unknown part {pid}")
        mesh = Mesh(np.load(pv), np.load(pf))
        self._remember(pid, mesh)
        return mesh

    def delete(self, pid: str) -> None:
        self._cache.pop(pid, None)
        for p in self._paths(pid):
            if os.path.exists(p):
                os.remove(p)

    def exists(self, pid: str) -> bool:
        return os.path.exists(self._paths(pid)[0])

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
