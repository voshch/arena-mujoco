from pathlib import Path

import mujoco
import pytest

from arena_mujoco.control import _add_velocity_actuator, _pre_compile
from arena_mujoco.scene import SceneStore
from arena_mujoco.services.SpawnUrdf import RobotEntry, _armature_floor, _attach_robot, _mujoco_compat

_INERTIAL = '<inertial><mass value="0.03"/><inertia ixx="4e-06" iyy="4e-06" izz="7e-06" ixy="0" ixz="0" iyz="0"/></inertial>'


def _wheel(tmp_path: Path, limit: str) -> mujoco.MjSpec:
    urdf = tmp_path / 'r.urdf'
    urdf.write_text(f'<robot name="r"><link name="base">{_INERTIAL}</link><link name="wheel">{_INERTIAL}</link><joint name="drive" type="continuous"><parent link="base"/><child link="wheel"/><axis xyz="0 0 1"/>{limit}</joint></robot>')
    compat, _, _ = _mujoco_compat(str(urdf))
    return mujoco.MjSpec.from_file(compat)


@pytest.mark.parametrize(('limit', 'floor'), [('', 0.2), ('<limit effort="100" velocity="10"/>', 0.2), ('<limit effort="0.035" velocity="20"/>', 0.035 / 50.0)])
def test_the_armature_floor_leaves_a_weak_motor_room_to_accelerate(tmp_path: Path, limit: str, floor: float):
    robot = _wheel(tmp_path, limit)
    assert _armature_floor(robot.joint('drive')) == pytest.approx(floor)


def test_a_velocity_servo_on_a_35_mnm_motor_spins_its_wheel_up_within_a_second(tmp_path: Path):
    robot = _wheel(tmp_path, '<limit effort="0.035" velocity="20"/>')
    drive = robot.joint('drive')
    drive.armature = _armature_floor(drive)
    _add_velocity_actuator(robot, drive, 'v')
    robot.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    robot.option.gravity = [0.0, 0.0, 0.0]
    model = robot.compile()
    data = mujoco.MjData(model)
    data.ctrl[0] = 12.55
    while data.time < 1.0:
        mujoco.mj_step(model, data)
    assert data.qvel[model.joint('drive').dofadr[0]] == pytest.approx(12.55, rel=0.05)


def test_a_weak_motor_spawned_into_a_compiled_scene_keeps_its_force_limit(tmp_path: Path):
    robot = _wheel(tmp_path, '<limit effort="0.035" velocity="20"/>')
    drive = robot.joint('drive')
    drive.armature = _armature_floor(drive)
    store = SceneStore()
    store.ensure_env(0)
    entry = RobotEntry(
        robot_model='r',
        tf_prefix='env_0/r/',
        base_frame='base',
        prim_name='r_base',
        localization=True,
        joint_names=('drive',),
        joint_states_topic='',
        cmd_vel_topic='',
        odom_topic='',
        command_interfaces={'drive': ('velocity',)},
        urdf_sensors={},
        sensors=(),
    )

    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        root = _attach_robot(spec, store.asset_cache(0), robot, 'r_', 'base', (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), True)
        _pre_compile(0, spec, root, entry, ['drive'])
        return root

    store.add_body(0, build)
    store.recompile(0)
    model = store.get_model(0)
    data = store.get_data(0)
    actuator = model.actuator('r_drive_vel')
    assert list(actuator.forcerange) == pytest.approx([-0.035, 0.035])
    data.ctrl[actuator.id] = 12.55
    while data.time < 1.0:
        mujoco.mj_step(model, data)
    assert data.qvel[model.joint('r_drive').dofadr[0]] == pytest.approx(12.55, rel=0.05)
