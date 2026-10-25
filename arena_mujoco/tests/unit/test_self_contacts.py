from pathlib import Path

import mujoco

from arena_mujoco.services.SpawnUrdf import _exclude_self_contacts, _mujoco_compat

_INERTIAL = '<inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>'


def _steered_wheel_robot(tmp_path: Path) -> mujoco.MjSpec:
    urdf = tmp_path / 'r.urdf'
    urdf.write_text(
        '<robot name="r">'
        f'<link name="base">{_INERTIAL}<collision><geometry><box size="0.6 0.4 0.1"/></geometry></collision></link>'
        f'<link name="caster">{_INERTIAL}</link>'
        f'<link name="wheel">{_INERTIAL}<collision><geometry><cylinder radius="0.08" length="0.04"/></geometry></collision></link>'
        '<joint name="steer" type="revolute"><parent link="base"/><child link="caster"/><origin xyz="0.25 0.15 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="100" velocity="5"/></joint>'
        '<joint name="drive" type="continuous"><parent link="caster"/><child link="wheel"/><origin xyz="0 0 -0.05" rpy="1.5708 0 0"/><axis xyz="0 0 1"/></joint>'
        '</robot>'
    )
    compat, _, _ = _mujoco_compat(str(urdf))
    return mujoco.MjSpec.from_file(compat)


def _robot_contacts(robot: mujoco.MjSpec) -> int:
    world = mujoco.MjSpec()
    world.attach(robot, prefix='r_', frame=world.worldbody.add_frame(pos=[0.0, 0.0, 1.0]))
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return sum(1 for contact in data.contact[: data.ncon] if model.geom_bodyid[contact.geom1] and model.geom_bodyid[contact.geom2])


def test_a_steered_wheel_overlapping_the_chassis_touches_it_without_the_exclusion(tmp_path: Path):
    assert _robot_contacts(_steered_wheel_robot(tmp_path)) > 0


def test_a_robot_never_collides_with_itself(tmp_path: Path):
    robot = _steered_wheel_robot(tmp_path)
    _exclude_self_contacts(robot)
    assert _robot_contacts(robot) == 0
