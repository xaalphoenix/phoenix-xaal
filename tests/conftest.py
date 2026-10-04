import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import manifold3d as m3d  # noqa: E402

from phoenix_stl.core.mesh import Mesh  # noqa: E402


def from_manifold(man) -> Mesh:
    mg = man.to_mesh()
    return Mesh(np.asarray(mg.vert_properties)[:, :3], np.asarray(mg.tri_verts))


def to_manifold(mesh: Mesh):
    return m3d.Manifold(m3d.Mesh(vert_properties=np.array(mesh.vertices, dtype=np.float32, order="C"),
                                 tri_verts=np.array(mesh.faces, dtype=np.uint32, order="C")))


def is_valid_solid(mesh: Mesh) -> bool:
    return to_manifold(mesh).status() == m3d.Error.NoError


@pytest.fixture
def cube():
    return from_manifold(m3d.Manifold.cube((20, 20, 20), True))


@pytest.fixture
def sphere():
    return from_manifold(m3d.Manifold.sphere(10, 96))


@pytest.fixture
def hollow_sphere():
    outer = m3d.Manifold.sphere(10, 64)
    inner = m3d.Manifold.sphere(8, 48)
    return from_manifold(outer - inner)


@pytest.fixture
def torus():
    # Revolve a circle: a genus-1 solid, so cuts produce caps with holes.
    circle = m3d.CrossSection.circle(4, 48).translate((10, 0))
    return from_manifold(m3d.Manifold.revolve(circle, 96))
