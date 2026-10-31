import dataclasses
import importlib
import logging
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from arena_people_msgs.msg import Pedestrian, SpawnPedestrian
from arena_people_msgs.srv import SpawnPedestrians
from arena_assets.rig import Rig as BodyRig
from arena_assets.rig import write_rig
from arena_simulation_setup.testing import make_rig
from geometry_msgs.msg import Quaternion

from arena_mujoco import context
from arena_mujoco.humans import Rig, SkinPart, Wire, _quaternions, _sole_pairs, _soles, actor_rig, clip_frames, joint_poses, quaternion_matrix, read_actor
from arena_mujoco.scene import SceneStore
from arena_mujoco.sensors.laser import cast_rays
from arena_mujoco.services.SpawnPedestrians import reset_env, spawn_pedestrians_service, update_pedestrians

_IDENTITY = '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'
_KNEE_REST = '1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 1'
_KNEE_BENT = '0 0 1 0 0 1 0 0 -1 0 0 1 0 0 0 1'
_HIPS_MOVED = '1 0 0 5 0 1 0 0 0 0 1 0 0 0 0 1'
_HIPS_STEP = '1 0 0 6 0 1 0 0 0 0 1 0.1 0 0 0 1'
_HIPS_ARRIVED = '1 0 0 7 0 1 0 0 0 0 1 0 0 0 0 1'
_HEAD = '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><asset><unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>'

_SKIN = f"""{_HEAD}
<library_effects><effect id="fx"><profile_COMMON><technique sid="common"><lambert>
  <diffuse><color>0.2 0.4 0.6 1</color></diffuse></lambert></technique></profile_COMMON></effect></library_effects>
<library_materials><material id="mat"><instance_effect url="#fx"/></material></library_materials>
<library_geometries><geometry id="g"><mesh>
  <source id="g-pos"><float_array id="g-pos-a" count="12">0 0 0 0 0 1 0 0 2 0.1 0 1</float_array>
    <technique_common><accessor source="#g-pos-a" count="4" stride="3">
      <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
  <vertices id="g-v"><input semantic="POSITION" source="#g-pos"/></vertices>
  <triangles material="mat" count="2"><input semantic="VERTEX" source="#g-v" offset="0"/><p>0 1 3 1 2 3</p></triangles>
</mesh></geometry></library_geometries>
<library_controllers><controller id="c"><skin source="#g">
  <bind_shape_matrix>{_IDENTITY}</bind_shape_matrix>
  <source id="c-j"><Name_array id="c-j-a" count="2">Hips LeftLeg</Name_array>
    <technique_common><accessor source="#c-j-a" count="2" stride="1"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="c-ibm"><float_array id="c-ibm-a" count="32">{_IDENTITY} 1 0 0 0 0 1 0 0 0 0 1 -1 0 0 0 1</float_array>
    <technique_common><accessor source="#c-ibm-a" count="2" stride="16"><param name="TRANSFORM" type="float4x4"/></accessor></technique_common></source>
  <source id="c-w"><float_array id="c-w-a" count="1">1</float_array>
    <technique_common><accessor source="#c-w-a" count="1" stride="1"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints><input semantic="JOINT" source="#c-j"/><input semantic="INV_BIND_MATRIX" source="#c-ibm"/></joints>
  <vertex_weights count="4"><input semantic="JOINT" source="#c-j" offset="0"/><input semantic="WEIGHT" source="#c-w" offset="1"/>
    <vcount>1 1 1 1</vcount><v>0 0 0 0 1 0 1 0</v></vertex_weights>
</skin></controller></library_controllers>
<library_visual_scenes><visual_scene id="s">
  <node id="Armature"><node id="Hips" sid="Hips" type="JOINT"><matrix sid="transform">{_IDENTITY}</matrix>
    <node id="LeftLeg" sid="LeftLeg" type="JOINT"><matrix sid="transform">{_KNEE_REST}</matrix></node></node></node>
  <node id="skin"><instance_controller url="#c"><skeleton>#Hips</skeleton>
    <bind_material><technique_common><instance_material symbol="mat" target="#mat"/></technique_common></bind_material>
  </instance_controller></node>
</visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#s"/></scene></COLLADA>"""


def _channel(joint: str, *matrices: str) -> str:
    times = ' '.join(str(0.5 * key) for key in range(len(matrices)))
    return f"""<animation id="clip_{joint}">
  <source id="{joint}-in"><float_array id="{joint}-in-a" count="{len(matrices)}">{times}</float_array></source>
  <source id="{joint}-out"><float_array id="{joint}-out-a" count="{16 * len(matrices)}">{' '.join(matrices)}</float_array></source>
  <sampler id="{joint}-s"><input semantic="INPUT" source="#{joint}-in"/><input semantic="OUTPUT" source="#{joint}-out"/></sampler>
  <channel source="#{joint}-s" target="{joint}/transform"/></animation>"""


_IDLE = f'{_HEAD}<library_animations>{_channel("Hips", _HIPS_MOVED)}{_channel("LeftLeg", _KNEE_BENT)}</library_animations></COLLADA>'
_WALK = f'{_HEAD}<library_animations>{_channel("Hips", _HIPS_MOVED, _HIPS_STEP, _HIPS_ARRIVED)}{_channel("LeftLeg", _KNEE_REST, _KNEE_BENT, _KNEE_REST)}</library_animations></COLLADA>'

_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_PELVIS = {'pelvis': (None, [('Hips', 1.0, _IDENTITY)])}
_PELVIS_KNEE = {**_PELVIS, 'left_knee': ('pelvis', [('LeftLeg', 1.0, _IDENTITY)])}

_STANDING = {(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (0.1, 0.0, 1.0)}
_KNEE_UP = {(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 0.9)}


@pytest.fixture
def actor_sdf(tmp_path: Path) -> str:
    (tmp_path / 'meshes').mkdir()
    (tmp_path / 'clips').mkdir()
    (tmp_path / 'meshes' / 'person.dae').write_text(_SKIN)
    (tmp_path / 'clips' / 'idle.dae').write_text(_IDLE)
    (tmp_path / 'clips' / 'walk.dae').write_text(_WALK)
    write_rig(tmp_path / 'person.rig', make_rig(_PELVIS_KNEE))
    sdf = tmp_path / 'person.sdf'
    sdf.write_text('<sdf version="1.9"><actor name="person"><pose>0 0 -0.04 0 0 0</pose><skin><filename>meshes/person.dae</filename></skin><animation name="idle"><filename>clips/idle.dae</filename></animation><animation name="walk"><filename>clips/walk.dae</filename></animation></actor></sdf>')
    return str(sdf)


def _skinned(rig: Rig, positions: np.ndarray, quaternions: np.ndarray) -> set[tuple[float, ...]]:
    (part,) = rig.parts
    posed = np.zeros_like(part.vertices)
    for vertices, weights, bind, position, quaternion in zip(part.bone_vertices, part.bone_weights, rig.bind_positions, positions, quaternions, strict=True):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        posed[vertices] += weights[:, None] * ((part.vertices[vertices] - bind) @ rotation.reshape(3, 3).T + position)
    return {tuple(vertex) for vertex in (posed + 0.0).round(4).tolist()}


def test_actor_sdf_names_skin_clips_and_offset(actor_sdf: str):
    actor = read_actor(actor_sdf)
    assert actor.skin.name == 'person.dae'
    assert {name: path.name for name, path in actor.clips.items()} == {'idle': 'idle.dae', 'walk': 'walk.dae'}
    assert actor.z_offset == pytest.approx(-0.04)
    times, frames = clip_frames(actor.clips['walk'])
    assert times.tolist() == [0.0, 0.5, 1.0]
    assert frames['LeftLeg'].shape == (3, 4, 4)
    assert frames['LeftLeg'][1, 2].tolist() == [-1.0, 0.0, 0.0, 1.0]


def test_rig_poses_the_skin_by_its_clips_with_the_root_advance_removed(actor_sdf: str):
    rig = actor_rig(actor_sdf)
    (part,) = rig.parts
    assert rig.bones == ('Hips', 'LeftLeg')
    assert rig.z_offset == pytest.approx(-0.04)
    assert part.rgba == pytest.approx((0.2, 0.4, 0.6, 1.0))
    assert _skinned(rig, *rig.clips['idle'].pose(0.0)) == _KNEE_UP

    walk = rig.clips['walk']
    assert (walk.duration, walk.stride) == pytest.approx((1.0, 2.0))
    assert _skinned(rig, *walk.pose(0.0)) == _STANDING
    assert _skinned(rig, *walk.pose(0.5)) == {(x, y, round(z + 0.1, 4)) for x, y, z in _KNEE_UP}
    assert walk.pose(0.25)[0][0] == pytest.approx((0.0, 0.0, 0.05))
    assert _skinned(rig, *walk.pose(1.0)) == _STANDING


def _ry(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]])


def _rz(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])


def _stood(hips: np.ndarray, shin: np.ndarray) -> set[tuple[float, ...]]:
    """The fixture's vertices with the Hips bone at world rotation hips and the LeftLeg bone at shin, which hangs from the hips by (0, 0, 1)."""
    knee = hips @ (0.0, 0.0, 1.0)
    points = [np.zeros(3), knee, shin @ (0.0, 0.0, 1.0) + knee, shin @ (0.1, 0.0, 0.0) + knee]
    return {tuple((point + 0.0).round(4).tolist()) for point in points}


def _knee_turned(angle: float) -> set[tuple[float, ...]]:
    """The fixture's vertices with the LeftLeg bone turned by the left_knee rotation from its bind rest."""
    return _stood(np.eye(3), _ry(angle))


def _knee_wire(angle: float) -> Wire:
    return Wire(rotations={'left_knee': _ry(angle)})


def _rigged(actor_sdf: str, rig: BodyRig) -> Rig:
    write_rig(Path(actor_sdf).with_suffix('.rig'), rig)
    return actor_rig(actor_sdf)


def test_wire_joint_rotations_turn_leg_bones_from_their_bind_rest(actor_sdf: str):
    rig = actor_rig(actor_sdf)
    standing = np.eye(4)[None]

    positions, quaternions = joint_poses([rig], standing, [Wire()])
    assert _skinned(rig, positions[0], quaternions[0]) == _STANDING == _knee_turned(0.0)

    positions, quaternions = joint_poses([rig, rig], np.concatenate([standing, standing]), [Wire(rotations={**_knee_wire(-0.8).rotations, 'unknown': _ry(1.0)}), _knee_wire(0.5)])
    assert _skinned(rig, positions[0], quaternions[0]) == _knee_turned(-0.8)
    assert _skinned(rig, positions[1], quaternions[1]) == _knee_turned(0.5)
    assert _knee_turned(-0.8) != _STANDING


def test_a_mapped_bone_takes_its_joints_world_rotation_times_the_offset_down_a_two_bone_chain(actor_sdf: str):
    half = math.sqrt(0.5)
    rig = _rigged(actor_sdf, make_rig({**_PELVIS, 'left_knee': ('pelvis', [('LeftLeg', 1.0, (0.0, 0.0, half, half))])}))
    wire = Wire(rotations={'pelvis': _rz(0.3), 'left_knee': _ry(0.5)})
    positions, quaternions = joint_poses([rig], np.eye(4)[None], [wire])
    hips = _rz(0.3)
    assert _skinned(rig, positions[0], quaternions[0]) == _stood(hips, hips @ _ry(0.5) @ _rz(math.pi / 2))


def test_a_spread_joint_gives_each_bone_its_share_of_the_angle_and_the_last_the_whole(actor_sdf: str):
    rig = _rigged(actor_sdf, make_rig({'pelvis': (None, [('Hips', 0.5, _IDENTITY), ('LeftLeg', 0.5, _IDENTITY)])}))
    positions, quaternions = joint_poses([rig], np.eye(4)[None], [Wire(rotations={'pelvis': _ry(0.8)})])
    assert _skinned(rig, positions[0], quaternions[0]) == _stood(_ry(0.4), _ry(0.8))


def test_a_joint_missing_from_the_wire_keeps_its_stance(actor_sdf: str):
    half = math.sqrt(0.5)
    stance = {'pelvis': _IDENTITY, 'left_knee': (0.0, half, 0.0, half)}
    rig = _rigged(actor_sdf, make_rig(_PELVIS_KNEE, stance))
    wire = Wire(rotations={'pelvis': np.eye(3)})
    held = joint_poses([rig], np.eye(4)[None], [wire])
    posed = joint_poses([rig], np.eye(4)[None], [Wire(rotations={'pelvis': np.eye(3), 'left_knee': quaternion_matrix((0.0, half, 0.0, half))})])
    assert held[1] == pytest.approx(posed[1])


def test_a_two_joint_rig_with_no_human_names_poses_its_bones(actor_sdf: str):
    half = math.sqrt(0.5)
    rig = _rigged(actor_sdf, make_rig({'chassis': (None, [('Hips', 1.0, _IDENTITY)]), 'boom': ('chassis', [('LeftLeg', 1.0, _IDENTITY)])}))
    still = joint_poses([rig], np.eye(4)[None], [Wire(rotations={'chassis': np.eye(3), 'boom': np.eye(3)})])
    turned = joint_poses([rig], np.eye(4)[None], [Wire(rotations={'chassis': np.eye(3), 'boom': quaternion_matrix((0.0, half, 0.0, half))})])
    index = rig.bones.index('LeftLeg')
    assert not np.allclose(still[1][0, index], turned[1][0, index])
    assert np.allclose(still[1][0, rig.bones.index('Hips')], turned[1][0, rig.bones.index('Hips')])


def test_bones_the_rig_does_not_name_hold_their_neutral_stance_under_their_parent(actor_sdf: str):
    rig = _rigged(actor_sdf, make_rig(_PELVIS))
    positions, quaternions = joint_poses([rig], np.eye(4)[None], [Wire()])
    assert _skinned(rig, positions[0], quaternions[0]) == _KNEE_UP
    turn = _rz(math.pi / 2)
    positions, quaternions = joint_poses([rig], np.eye(4)[None], [Wire(rotations={'pelvis': turn})])
    assert _skinned(rig, positions[0], quaternions[0]) == _stood(turn, turn @ _ry(math.pi / 2))


def test_the_root_offset_moves_the_top_bone_and_with_it_the_skeleton(actor_sdf: str):
    rig = actor_rig(actor_sdf)
    positions, quaternions = joint_poses([rig], np.eye(4)[None], [Wire(offset=np.array([0.2, 0.0, 0.3]))])
    assert _skinned(rig, positions[0], quaternions[0]) == {(round(x + 0.2, 4), y, round(z + 0.3, 4)) for x, y, z in _STANDING}


def test_a_driver_bone_turns_from_its_neutral_about_its_axis_by_the_joint_state_angle(actor_sdf: str):
    rig = _rigged(actor_sdf, make_rig(_PELVIS, drivers={'shin_turn': {'bone': 'LeftLeg', 'from': 'angle', 'axis': [0.0, 2.0, 0.0]}}))
    for angles, expected in (({}, _KNEE_UP), ({'shin_turn': 0.5}, _stood(np.eye(3), _ry(math.pi / 2 + 0.5)))):
        positions, quaternions = joint_poses([rig], np.eye(4)[None], [Wire(angles=angles)])
        assert _skinned(rig, positions[0], quaternions[0]) == expected


def test_quaternion_matrix_matches_mujoco_for_any_rotation():
    rng = np.random.default_rng(1)
    for wxyz in rng.normal(size=(50, 4)):
        wxyz /= np.linalg.norm(wxyz)
        expected = np.empty(9)
        mujoco.mju_quat2Mat(expected, wxyz)
        assert quaternion_matrix(np.roll(wxyz, -1)) == pytest.approx(expected.reshape(3, 3))
    assert quaternion_matrix((0.0, 0.0, 0.0, 0.0)) == pytest.approx(np.eye(3))


_FEET = ('Hips', 'LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase')
_FEET_BINDS = np.array([[0.0, 0.0, 0.9], [0.0, 0.1, 0.1], [0.15, 0.1, 0.03], [0.0, -0.1, 0.1], [0.15, -0.1, 0.03]])
_SHOE = (
    ((-0.05, 0.1, 0.004), 1),
    ((-0.04, 0.1, 0.0), 1),
    ((0.0, 0.1, 0.02), 1),
    ((0.18, 0.1, 0.002), 2),
    ((0.20, 0.1, 0.006), 2),
    ((0.0, 0.0, 0.9), 0),
    ((-0.04, -0.1, 0.0), 3),
    ((0.19, -0.1, 0.001), 4),
)


def _shoe_part() -> SkinPart:
    owners = np.array([bone for _, bone in _SHOE])
    return SkinPart(
        vertices=np.array([point for point, _ in _SHOE]),
        faces=np.zeros((0, 3), dtype=int),
        uv=None,
        rgba=None,
        texture=None,
        bone_vertices=tuple(np.flatnonzero(owners == bone) for bone in range(len(_FEET))),
        bone_weights=tuple(np.ones(int((owners == bone).sum())) for bone in range(len(_FEET))),
    )


_FOOT_PAIRS = (('LeftFoot', 'LeftToeBase'), ('RightFoot', 'RightToeBase'))

_FEET_JOINTS = {
    'left_knee': (None, []),
    'left_ankle': ('left_knee', [('LeftFoot', 1.0, _IDENTITY)]),
    'left_foot': ('left_ankle', [('LeftToeBase', 1.0, _IDENTITY)]),
    'right_knee': (None, []),
    'right_ankle': ('right_knee', [('RightFoot', 1.0, _IDENTITY)]),
    'right_foot': ('right_ankle', [('RightToeBase', 1.0, _IDENTITY)]),
}
_SOLE_JOINTS = ['left_ankle', 'left_foot', 'right_ankle', 'right_foot']


def test_soles_are_found_through_the_joints_the_rig_names():
    pairs = _sole_pairs(make_rig(_FEET_JOINTS, soles=_SOLE_JOINTS))
    assert pairs == _FOOT_PAIRS
    bones, _ = _soles(_FEET, [_shoe_part()], _FEET_BINDS, pairs)
    assert bones.tolist() == [1, 2, 3, 4]
    assert _sole_pairs(None) == _sole_pairs(make_rig(_PELVIS)) == ()


def test_soles_take_the_lowest_rear_and_front_sole_vertex_of_each_foot():
    bones, points = _soles(_FEET, [_shoe_part()], _FEET_BINDS, _FOOT_PAIRS)
    assert bones.tolist() == [1, 2, 3, 4]
    assert points.tolist() == [[-0.04, 0.1, 0.0], [0.18, 0.1, 0.002], [-0.04, -0.1, 0.0], [0.19, -0.1, 0.001]]


def test_a_rig_with_a_distance_driver_has_no_soles():
    wheel = {'l_wheel': {'bone': 'WheelL', 'from': 'distance', 'radius': 0.3, 'lateral': 0.288, 'axis': [0.0, 1.0, 0.0]}}
    pairs = _sole_pairs(make_rig(_FEET_JOINTS, soles=_SOLE_JOINTS, drivers=wheel))
    assert pairs == ()
    bones, points = _soles((*_FEET, 'WheelL'), [_shoe_part()], np.vstack([_FEET_BINDS, np.zeros(3)]), pairs)
    assert len(bones) == 0 and points.shape == (0, 3)


def test_joint_poses_stand_the_lowest_sole_point_at_the_root_height(actor_sdf: str):
    rig = dataclasses.replace(actor_rig(actor_sdf), sole_bones=np.array([1]), sole_points=np.array([[0.1, 0.0, 1.0]]))
    root = np.eye(4)[None]
    root[0, 2, 3] = 0.25
    for wire in (Wire(), _knee_wire(-0.8)):
        positions, quaternions = joint_poses([rig], root, [wire])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternions[0, 1])
        sole = rotation.reshape(3, 3) @ (rig.sole_points[0] - rig.bind_positions[1]) + positions[0, 1]
        assert sole[2] == pytest.approx(0.25)
        unplanted, _ = joint_poses([actor_rig(actor_sdf)], root, [wire])
        assert positions[0, :, :2] == pytest.approx(unplanted[0, :, :2])


def test_joint_poses_plant_the_soles_along_a_tipped_root_up_axis(actor_sdf: str):
    rig = dataclasses.replace(actor_rig(actor_sdf), sole_bones=np.array([1]), sole_points=np.array([[0.1, 0.0, 1.0]]))
    root = np.eye(4)[None]
    root[0, :3, :3] = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    root[0, :3, 3] = (1.0, 2.0, 0.6)
    positions, quaternions = joint_poses([rig], root, [Wire()])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternions[0, 1])
    sole = rotation.reshape(3, 3) @ (rig.sole_points[0] - rig.bind_positions[1]) + positions[0, 1]
    assert (sole - root[0, :3, 3]) @ root[0, :3, 2] == pytest.approx(0.0)
    unplanted, _ = joint_poses([actor_rig(actor_sdf)], root, [Wire()])
    assert positions[0, :, 1:] == pytest.approx(unplanted[0, :, 1:])


def test_quaternions_match_mujoco_for_any_rotation():
    rng = np.random.default_rng(0)
    quaternions = rng.normal(size=(200, 4))
    quaternions[:20, 0] = 1e-9
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    rotations = np.empty((200, 9))
    for rotation, quaternion in zip(rotations, quaternions, strict=True):
        mujoco.mju_quat2Mat(rotation, quaternion)
    converted = _quaternions(rotations.reshape(200, 3, 3))
    assert np.abs(np.sum(converted * quaternions, axis=1)) == pytest.approx(np.ones(200), abs=1e-9)


_ISAAC = Path(__file__).parents[4] / 'arena_isaac' / 'arena_isaac'


@pytest.fixture
def isaac_peds():
    """Isaac's pure-Python pedestrian package, importable for the duration of a test."""
    if not (_ISAAC / 'peds').is_dir():
        pytest.skip('arena_isaac is not checked out')
    sys.path.insert(0, str(_ISAAC))
    try:
        yield importlib.import_module('peds.convert'), importlib.import_module('peds.providers.external')
    except ImportError as exc:
        pytest.skip(f'arena_isaac peds not importable: {exc}')
    finally:
        sys.path.remove(str(_ISAAC))


def _axis_quaternion(axis: int, angle: float) -> list[float]:
    quaternion = [0.0, 0.0, 0.0, math.cos(angle / 2)]
    quaternion[axis] = math.sin(angle / 2)
    return quaternion


_HALF = math.sqrt(0.5)

_ISAAC_CASES = {
    'knee': (make_rig(_PELVIS_KNEE), {'left_knee': _axis_quaternion(1, -0.7)}, (0.0, 0.0, 0.0), {}),
    'chain': (
        make_rig({**_PELVIS, 'left_knee': ('pelvis', [('LeftLeg', 1.0, (0.0, 0.0, _HALF, _HALF))])}),
        {'pelvis': _axis_quaternion(2, 0.3), 'left_knee': _axis_quaternion(1, 0.5)},
        (0.1, 0.0, 0.2),
        {},
    ),
    'spread': (
        make_rig({'pelvis': (None, [('Hips', 0.5, _IDENTITY), ('LeftLeg', 0.5, _IDENTITY)])}),
        {'pelvis': _axis_quaternion(1, 0.8)},
        (0.0, 0.0, 0.0),
        {},
    ),
    'driver': (
        make_rig(_PELVIS, drivers={'shin_turn': {'bone': 'LeftLeg', 'from': 'angle', 'axis': [0.0, 1.0, 0.0]}}),
        {'pelvis': _axis_quaternion(2, 0.4)},
        (0.0, 0.0, 0.0),
        {'shin_turn': 0.5},
    ),
}


@pytest.mark.parametrize('case', list(_ISAAC_CASES))
def test_body_pose_matches_isaacs_pose_provider(actor_sdf: str, isaac_peds, case: str):
    convert, external = isaac_peds
    body_rig, rotations, offset, angles = _ISAAC_CASES[case]
    rig = _rigged(actor_sdf, body_rig)

    actor = read_actor(actor_sdf)
    document = convert._Collada(str(actor.skin))
    joints = convert._parse_skeleton(document)
    (skinned,) = convert._parse_skinned_meshes(document)
    neutral, translations = convert._neutral_pose(joints, convert._parse_animations(convert._Collada(str(actor.clips['idle']))))
    provider = external.ExternalPoseProvider(tuple(joint.path for joint in joints), np.array(neutral), np.array(translations), body_rig)
    provider.push(0.0, list(rotations), list(rotations.values()), offset, list(angles), list(angles.values()))
    pose = provider.evaluate(0.0, 1.0)

    worlds: list[np.ndarray] = []
    for index, joint in enumerate(joints):
        local = np.eye(4)
        local[:3, :3] = quaternion_matrix(pose.rotations[index])
        local[:3, 3] = pose.translations[index]
        worlds.append(local if joint.parent < 0 else worlds[joint.parent] @ local)
    by_name = {joint.name: world for joint, world in zip(joints, worlds, strict=True)}
    points = np.c_[skinned.mesh.points, np.ones(len(skinned.mesh.points))] @ skinned.skin.bind_shape.T
    isaac = set()
    for point, influences in zip(points, skinned.skin.influences, strict=True):
        posed = sum(weight * (by_name[skinned.skin.joint_names[joint]] @ skinned.skin.inv_bind[joint] @ point) for joint, weight in influences)
        isaac.add(tuple((posed[:3] + 0.0).round(4).tolist()))

    wire = Wire(rotations={name: quaternion_matrix(quaternion) for name, quaternion in rotations.items()}, offset=np.array(offset), angles=angles)
    positions, quaternions = joint_poses([rig], np.eye(4)[None], [wire])
    assert _skinned(rig, positions[0], quaternions[0]) == isaac != _STANDING


def _pedestrian(name: str, model_uri: str, x: float) -> Pedestrian:
    pedestrian = Pedestrian(name=name, model_uri=model_uri)
    pedestrian.pose.position.x = x
    return pedestrian


def _spawn(name: str, model_uri: str, x: float) -> int:
    request = SpawnPedestrians.Request(pedestrians=[SpawnPedestrian(pedestrian=_pedestrian(name, model_uri, x))])
    response = spawn_pedestrians_service.kwargs['callback'](request, SpawnPedestrians.Response())
    return response.results[0]


def _skin_vertices(store: SceneStore, env_id: int) -> set[tuple[float, ...]]:
    """World vertices of every skin as the renderer draws them."""
    model, data = store.get_model(env_id), store.get_data(env_id)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
    return {tuple(vertex) for vertex in (np.array(scene.skinvert, dtype=np.float64).reshape(-1, 3) + 0.0).round(4).tolist()}


def _shifted(vertices: set[tuple[float, ...]], x: float, z: float) -> set[tuple[float, ...]]:
    return {(round(vx + x, 4), vy, round(vz + z, 4)) for vx, vy, vz in vertices}


@pytest.fixture
def store() -> SceneStore:
    store = SceneStore()
    context.set_scene_store(store)
    reset_env(7)
    return store


def test_spawned_pedestrian_wears_its_skin_for_cameras_and_keeps_the_capsule_for_rays(store: SceneStore, actor_sdf: str):
    assert _spawn('env_7/peds/doctor', actor_sdf, 3.0) == SpawnPedestrians.Response.SUCCESS
    assert _spawn('env_7/peds/plain', '', 6.0) == SpawnPedestrians.Response.SUCCESS
    model, data = store.get_model(7), store.get_data(7)
    mujoco.mj_forward(model, data)
    assert (model.nskin, model.nskinbone) == (1, 2)
    assert _skin_vertices(store, 7) == _shifted(_KNEE_UP, 3.0, -0.04)

    def capsule_group(x: float) -> int:
        body = next(b for b in range(model.nbody) if model.body_geomnum[b] and abs(data.xpos[b][0] - x) < 1e-6)
        (geom,) = np.flatnonzero(model.geom_bodyid == body)
        return int(model.geom_group[geom])

    assert (capsule_group(3.0), capsule_group(6.0)) == (4, 0)

    store.add_body(7, lambda spec: _probe(spec))
    store.recompile(7)
    _spawn('env_7/peds/doctor_2', actor_sdf, 3.0)
    model, data = store.get_model(7), store.get_data(7)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'probe')
    ranges = cast_rays(model, data, site, np.array([[1.0, 0.0, 0.0]]), np.array([], dtype=int), 0.01, 10.0)
    assert ranges[0] == pytest.approx(3.0 - 0.25, abs=1e-3)


def test_walking_pedestrian_plays_the_walk_clip_by_distance_and_stands_in_idle(store: SceneStore, actor_sdf: str):
    _spawn('env_7/peds/doctor', actor_sdf, 3.0)
    assert _skin_vertices(store, 7) == _shifted(_KNEE_UP, 3.0, -0.04)

    store.step(500)
    update_pedestrians([_pedestrian('env_7/peds/doctor', actor_sdf, 4.0)])
    assert _skin_vertices(store, 7) == _shifted(_STANDING, 4.0, -0.04)
    store.step(500)
    update_pedestrians([_pedestrian('env_7/peds/doctor', actor_sdf, 5.0)])
    assert _skin_vertices(store, 7) == _shifted(_KNEE_UP, 5.0, -0.04 + 0.1)

    store.step(500)
    update_pedestrians([_pedestrian('env_7/peds/doctor', actor_sdf, 5.0)])
    assert _skin_vertices(store, 7) == _shifted(_KNEE_UP, 5.0, -0.04)


def _with_body(pedestrian: Pedestrian, **rotations: tuple[float, float, float, float]) -> Pedestrian:
    pedestrian.body.joints = list(rotations)
    pedestrian.body.rotations = [Quaternion(x=x, y=y, z=z, w=w) for x, y, z, w in rotations.values()]
    return pedestrian


def _knee_quaternion(angle: float) -> tuple[float, float, float, float]:
    return 0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)


def test_pedestrian_with_a_body_pose_on_the_wire_follows_it_instead_of_a_clip(store: SceneStore, actor_sdf: str):
    _spawn('env_7/peds/doctor', actor_sdf, 3.0)
    walking = _with_body(_pedestrian('env_7/peds/doctor', actor_sdf, 4.0), left_knee=_knee_quaternion(-0.8))
    store.step(500)
    update_pedestrians([walking])
    assert _skin_vertices(store, 7) == _shifted(_knee_turned(-0.8), 4.0, -0.04)

    turned = _with_body(_pedestrian('env_7/peds/doctor', actor_sdf, 4.0), left_knee=(0.0, 0.0, 0.0, 1.0))
    turned.pose.orientation.z = turned.pose.orientation.w = 0.5**0.5
    update_pedestrians([turned])
    assert _skin_vertices(store, 7) == {(round(4.0 - y, 4), round(x, 4), round(z - 0.04, 4)) for x, y, z in _STANDING}


def test_a_missing_rig_plays_clips_and_is_logged_once_per_model(store: SceneStore, actor_sdf: str, caplog: pytest.LogCaptureFixture):
    Path(actor_sdf).with_suffix('.rig').unlink()
    with caplog.at_level(logging.WARNING):
        assert actor_rig(actor_sdf).joints is None
        _spawn('env_7/peds/doctor', actor_sdf, 3.0)
        _spawn('env_7/peds/nurse', actor_sdf, 6.0)
    assert caplog.text.count('rig_human') == 1

    update_pedestrians([_with_body(_pedestrian('env_7/peds/doctor', actor_sdf, 3.0), left_knee=_knee_quaternion(-0.8))])
    assert _skin_vertices(store, 7) == _shifted(_KNEE_UP, 3.0, -0.04) | _shifted(_KNEE_UP, 6.0, -0.04)


def test_a_rig_without_joints_plays_its_clips(store: SceneStore, actor_sdf: str):
    write_rig(Path(actor_sdf).with_suffix('.rig'), make_rig({}, settings={'root_offset': [0.0, 0.0, -0.3]}))
    assert actor_rig(actor_sdf).joints is None
    _spawn('env_7/peds/doctor', actor_sdf, 3.0)
    update_pedestrians([_with_body(_pedestrian('env_7/peds/doctor', actor_sdf, 3.0), left_knee=_knee_quaternion(-0.8))])
    assert _skin_vertices(store, 7) == _shifted(_KNEE_UP, 3.0, -0.04)


def test_deleting_a_pedestrian_body_takes_its_bones_and_skins_along(store: SceneStore, actor_sdf: str):
    _spawn('env_7/peds/doctor', actor_sdf, 3.0)
    assert store.get_model(7).nskin == 1
    assert store.remove_by_prefix(7, '') == 64
    store.compile_dirty()
    model = store.get_model(7)
    assert (model.nbody, model.nskin, model.nmat) == (1, 0, 0)


def _probe(spec: mujoco.MjSpec) -> mujoco.MjsBody:
    body = spec.worldbody.add_body(name='probe_body', pos=[0.0, 0.0, 0.85])
    body.add_site(name='probe')
    return body
