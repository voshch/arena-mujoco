import os

import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')


@pytest.fixture
def store():
    from arena_mujoco.scene import SceneStore

    scene = SceneStore()
    scene.ensure_env(0)
    return scene
