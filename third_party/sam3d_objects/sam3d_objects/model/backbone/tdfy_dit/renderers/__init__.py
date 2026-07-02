# Copyright (c) Meta Platforms, Inc. and affiliates.
from .gaussian_render import GaussianRenderer

def __getattr__(name):
    if name == "OctreeRenderer":
        from .octree_renderer import OctreeRenderer

        return OctreeRenderer
    if name == "MeshRenderer":
        try:
            from .mesh_renderer import MeshRenderer
        except ImportError:
            return None
        return MeshRenderer
    raise AttributeError(name)
