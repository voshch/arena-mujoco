from pathlib import Path

import mujoco
import pytest

from arena_mujoco import context, control
from arena_mujoco.scene import SceneStore
from arena_mujoco.services.SpawnUrdf import RobotEntry, _attach_robot, _initial_positions, _mujoco_compat

_INERTIAL = '<inertial><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>'
_URDF = f"""<robot name="r">
  <link name="base">{_INERTIAL}</link>
  <link name="shin">{_INERTIAL}<collision><origin xyz="0 0 -0.2"/><geometry><box size="0.05 0.05 0.4"/></geometry></collision></link>
  <joint name="knee" type="revolute"><parent link="base"/><child link="shin"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="50" velocity="10"/></joint>
  <ros2_control name="r_system" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="knee">
      <command_interface name="position"/>
      <state_interface name="position"><param name="initial_value">0.7</param></state_interface>
    </joint>
  </ros2_control>
</robot>
"""


def test_a_robot_spawns_at_its_ros2_control_initial_positions_and_holds_them(tmp_path: Path):
    urdf = tmp_path / 'r.urdf'
    urdf.write_text(_URDF)
    compat, command_interfaces, _ = _mujoco_compat(str(urdf))
    robot_spec = mujoco.MjSpec.from_file(compat)
    store = SceneStore()
    store.ensure_env(0)
    context.set_scene_store(store)
    entry = RobotEntry(
        robot_model='r',
        tf_prefix='env_0/r/',
        base_frame='base',
        prim_name='r_base',
        localization=False,
        joint_names=('knee',),
        joint_states_topic='',
        cmd_vel_topic='',
        odom_topic='',
        command_interfaces=command_interfaces,
        urdf_sensors={},
        sensors=(),
        initial_positions=_initial_positions(str(urdf)),
    )

    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        root = _attach_robot(spec, store.asset_cache(0), robot_spec, 'r_', 'base', (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), False)
        control._pre_compile(0, spec, root, entry, ['knee'])
        return root

    store.add_body(0, build)
    store.recompile(0)
    robot = control._Robot(0, ('knee',), control._pending.pop('r_base'), 'r_base')
    robot.mjcf_joints = control._pending_mjcf.pop('r_base')
    control._apply_initial_positions(0, robot, entry.initial_positions)
    model = store.get_model(0)
    data = store.get_data(0)
    knee = model.joint(robot.mjcf_joints['knee']).qposadr[0]
    assert entry.initial_positions == {'knee': 0.7}
    assert data.qpos[knee] == pytest.approx(0.7)
    store.step(500)
    assert data.qpos[knee] == pytest.approx(0.7, abs=0.05)
