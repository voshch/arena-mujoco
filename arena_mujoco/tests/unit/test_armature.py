from pathlib import Path

import mujoco
import pytest

from arena_mujoco.control import _add_velocity_actuator
from arena_mujoco.services.SpawnUrdf import _armature_floor, _mujoco_compat

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
    _add_velocity_actuator(robot, 'drive', 'v')
    robot.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    robot.option.gravity = [0.0, 0.0, 0.0]
    model = robot.compile()
    data = mujoco.MjData(model)
    data.ctrl[0] = 12.55
    while data.time < 1.0:
        mujoco.mj_step(model, data)
    assert data.qvel[model.joint('drive').dofadr[0]] == pytest.approx(12.55, rel=0.05)
