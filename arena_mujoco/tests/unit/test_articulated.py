import math
from pathlib import Path

import mujoco
import pytest
import trimesh
from arena_mujoco_msgs.msg import ArticulatedLink, JointCommand, JointDynamics, Prim, Scale
from arena_mujoco_msgs.srv import DriveArticulated, SpawnArticulated
from geometry_msgs.msg import Point, Pose, Vector3

from arena_mujoco import context
from arena_mujoco.scene import SceneStore
from arena_mujoco.services.Articulated import _drive_articulated_cb, _spawn_articulated_cb, forget_articulated
from arena_mujoco.services.SpawnPrims import _add_prim

NAME = 'env_0/Obstacles/cabinet'
STEPS_PER_SECOND = 500


def _box(tmp_path: Path, name: str, extents: tuple[float, float, float], center: tuple[float, float, float]) -> str:
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    path = tmp_path / f'{name}.obj'
    box.export(str(path))
    return str(path)


def _cabinet(tmp_path: Path, x: float = 0.0) -> SpawnArticulated.Request:
    """A 1 m cube carcass with a 1 m wide door hinged at its front right edge, 0.1 m proud of the carcass."""
    body = ArticulatedLink(name='body', mesh_path=_box(tmp_path, 'body', (1.0, 1.0, 1.0), (0.0, 0.0, 0.5)), mass=20.0, center=Point(z=0.5), moments=Vector3(x=3.0, y=3.0, z=3.0))
    door = ArticulatedLink(
        name='door',
        parent='body',
        mesh_path=_box(tmp_path, 'door', (1.0, 0.04, 1.0), (0.0, -0.57, 0.5)),
        mass=2.0,
        center=Point(y=-0.57, z=0.5),
        moments=Vector3(x=0.2, y=0.4, z=0.2),
        joint='door',
        type=ArticulatedLink.REVOLUTE,
        origin=Point(x=0.5, y=-0.57),
        axis=Vector3(z=1.0),
        lower=0.0,
        upper=1.6,
        dynamics=JointDynamics(damping=0.5, friction=0.2, armature=0.01, effort=10.0, stiffness=100.0, drive_damping=2.0 * math.sqrt(100.0 * 0.7)),
    )
    pose = Pose()
    pose.position.x = x
    pose.orientation.w = 1.0
    return SpawnArticulated.Request(env_id=0, name=NAME, pose=pose, scale=Scale(x=1.0, y=1.0, z=1.0), links=[body, door])


def _drive(reference: float, drive: bool, place: bool = False) -> list[float]:
    request = DriveArticulated.Request(env_id=0, commands=[JointCommand(name=NAME, joints=['door'], reference=[reference], drive=[drive], place=[place])])
    (reading,) = _drive_articulated_cb(request, DriveArticulated.Response()).readings
    assert reading.found
    assert list(reading.joints) == ['door']
    return list(reading.position)


def _run(store: SceneStore, seconds: float) -> None:
    store.step(int(seconds * STEPS_PER_SECOND))


@pytest.fixture
def store() -> SceneStore:
    store = SceneStore()
    store.ensure_env(0)
    context.set_scene_store(store)
    yield store
    forget_articulated(0, '/')


def test_an_articulated_object_is_one_body_tree_with_a_damped_hinge_and_a_drive_that_starts_off(store: SceneStore, tmp_path: Path):
    assert _spawn_articulated_cb(_cabinet(tmp_path, x=2.0), SpawnArticulated.Response()).ret
    model = store.get_model(0)
    root, door = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ('env_0_Obstacles_cabinet', 'env_0_Obstacles_cabinet__door'))
    assert model.body_parentid[door] == root
    assert model.body_jntnum[root] == 0
    assert tuple(model.body_pos[root]) == pytest.approx((2.0, 0.0, 0.0))
    assert model.body_mass[door] == pytest.approx(2.0)
    joint = model.joint('env_0_Obstacles_cabinet__door')
    assert joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert tuple(joint.range) == pytest.approx((0.0, 1.6))
    assert model.dof_damping[joint.dofadr[0]] == pytest.approx(0.5)
    assert model.dof_frictionloss[joint.dofadr[0]] == pytest.approx(0.2)
    actuator = model.actuator('env_0_Obstacles_cabinet__door__drive')
    assert actuator.gainprm[0] == 0.0
    assert tuple(actuator.forcerange) == pytest.approx((-10.0, 10.0))
    _run(store, 1.0)
    assert _drive(0.0, False) == pytest.approx([0.0], abs=1e-3)


def test_the_drive_swings_the_door_to_its_reference_without_the_carcass_in_the_way(store: SceneStore, tmp_path: Path):
    assert _spawn_articulated_cb(_cabinet(tmp_path), SpawnArticulated.Response()).ret
    for step in range(1, 31):
        _drive(min(1.2, 0.05 * step), True)
        _run(store, 1 / 30)
    _run(store, 1.0)
    (angle,) = _drive(1.2, False)
    assert angle == pytest.approx(1.2, abs=0.05)
    door = store.get_data(0).xipos[mujoco.mj_name2id(store.get_model(0), mujoco.mjtObj.mjOBJ_BODY, 'env_0_Obstacles_cabinet__door')]
    assert tuple(door) == pytest.approx((0.5 - 0.5 * math.cos(angle), -0.57 - 0.5 * math.sin(angle), 0.5), abs=1e-6)


def test_a_released_door_swings_under_a_push_and_its_friction_holds_it_after(store: SceneStore, tmp_path: Path):
    assert _spawn_articulated_cb(_cabinet(tmp_path), SpawnArticulated.Response()).ret
    model, data = store.get_model(0), store.get_data(0)
    dof = model.joint('env_0_Obstacles_cabinet__door').dofadr[0]
    data.qfrc_applied[dof] = 3.0
    _run(store, 0.3)
    data.qfrc_applied[dof] = 0.0
    (pushed,) = _drive(0.0, False)
    assert pushed > 0.1
    _run(store, 2.0)
    (rested,) = _drive(0.0, False)
    _run(store, 2.0)
    assert _drive(0.0, False) == pytest.approx([rested], abs=1e-3)
    assert rested >= pushed


def test_placing_sets_the_joint_at_rest_and_a_recompile_keeps_the_angle_and_the_drive(store: SceneStore, tmp_path: Path):
    assert _spawn_articulated_cb(_cabinet(tmp_path), SpawnArticulated.Response()).ret
    assert _drive(0.8, True, place=True) == pytest.approx([0.8])
    assert _add_prim(0, Prim(name='crate', scale=Scale(x=0.5, y=0.5, z=0.5)))
    store.recompile(0)
    model = store.get_model(0)
    assert model.actuator('env_0_Obstacles_cabinet__door__drive').gainprm[0] == pytest.approx(100.0)
    _run(store, 0.5)
    assert _drive(0.8, False) == pytest.approx([0.8], abs=0.02)
    assert store.get_model(0).actuator('env_0_Obstacles_cabinet__door__drive').gainprm[0] == 0.0


def test_deleting_the_object_takes_its_drives_and_excludes_and_frees_its_name(store: SceneStore, tmp_path: Path):
    assert _spawn_articulated_cb(_cabinet(tmp_path), SpawnArticulated.Response()).ret
    assert not _spawn_articulated_cb(_cabinet(tmp_path), SpawnArticulated.Response()).ret
    forget_articulated(0, NAME)
    assert store.remove(0, NAME)
    store.recompile(0)
    model = store.get_model(0)
    assert (model.nu, model.nexclude, model.njnt) == (0, 0, 0)
    request = DriveArticulated.Request(env_id=0, commands=[JointCommand(name=NAME, joints=['door'], reference=[0.0], drive=[False], place=[False])])
    assert not _drive_articulated_cb(request, DriveArticulated.Response()).readings[0].found
    assert _spawn_articulated_cb(_cabinet(tmp_path), SpawnArticulated.Response()).ret


def test_an_object_that_does_not_compile_leaves_nothing_behind(store: SceneStore, tmp_path: Path):
    from arena_mujoco import hooks

    class Node:
        def get_logger(self):
            return self

        def warning(self, message: str) -> None:
            self.message = message

    node = Node()
    hooks.set_node(node)
    unbuildable = _cabinet(tmp_path)
    unbuildable.links[1].mesh_path = str(tmp_path / 'missing.obj')
    uncompilable = _cabinet(tmp_path)
    uncompilable.links[1].lower, uncompilable.links[1].upper = 1.0, -1.0
    for request in (unbuildable, uncompilable):
        assert not _spawn_articulated_cb(request, SpawnArticulated.Response()).ret
        assert 'cabinet' in node.message
        model = store.get_model(0)
        assert (model.nbody, model.nu, model.nexclude, model.nmesh) == (1, 0, 0, 0)
    assert _spawn_articulated_cb(_cabinet(tmp_path), SpawnArticulated.Response()).ret
