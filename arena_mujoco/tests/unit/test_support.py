import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from arena_people_msgs.msg import Pedestrian

from arena_mujoco.support import LIE_ON, SIT_ON, TYPICAL_CLEARANCE, PelvisClearance, Support, pelvis_clearance, rest, support_of

CLEARANCE = PelvisClearance(sit=0.1, lie=0.12)
# a body standing at the root: feet on the floor, pelvis 0.9 m up, head 1.7 m up, a toe 0.2 m in front
STANDING = np.array([[0.0, 0.0, 0.9], [0.0, 0.0, 0.0], [0.0, 0.0, 1.7], [0.2, 0.0, 0.0]])
PELVIS = 0
HEAD = 2
TOE = 3


def _root(x: float, y: float, yaw: float, floor: float = 0.0) -> np.ndarray:
    root = np.eye(4)
    root[:3, :3] = ((math.cos(yaw), -math.sin(yaw), 0.0), (math.sin(yaw), math.cos(yaw), 0.0), (0.0, 0.0, 1.0))
    root[:3, 3] = (x, y, floor)
    return root


def _posed(root: np.ndarray, points: np.ndarray = STANDING) -> tuple[np.ndarray, np.ndarray]:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, root[:3, :3].flatten())
    return points @ root[:3, :3].T + root[:3, 3], np.tile(quaternion, (len(points), 1))


def _pedestrian(interaction: int, height: float) -> Pedestrian:
    pedestrian = Pedestrian(interaction_type=interaction)
    pedestrian.support_height = height
    return pedestrian


def test_only_a_sit_or_lie_over_a_known_surface_is_a_support():
    assert support_of(_pedestrian(SIT_ON, 0.45)) == Support(SIT_ON, 0.45)
    assert support_of(_pedestrian(LIE_ON, 0.6)) == Support(LIE_ON, 0.6)
    assert support_of(_pedestrian(SIT_ON, Pedestrian.NO_SUPPORT)) is None
    assert support_of(_pedestrian(0, 0.45)) is None


def test_a_seated_pelvis_rests_its_clearance_above_the_seat():
    root = _root(2.0, 1.0, 0.7, floor=0.3)
    seated = STANDING * (1.0, 1.0, 0.5)  # legs folded: the pelvis 0.45 m above the soles
    positions, quaternions = rest(Support(SIT_ON, 0.4), CLEARANCE, root, root[2, 3], PELVIS, *_posed(root, seated))
    assert positions[PELVIS, 2] == pytest.approx(0.3 + 0.4 + 0.1)
    assert positions[:, :2] == pytest.approx(_posed(root, seated)[0][:, :2])
    assert quaternions == pytest.approx(_posed(root, seated)[1])


def test_a_seated_body_never_pushes_its_feet_below_the_floor():
    root = _root(0.0, 0.0, 0.0)
    positions, _ = rest(Support(SIT_ON, 0.4), CLEARANCE, root, root[2, 3], PELVIS, *_posed(root))  # legs not yet folded
    assert positions == pytest.approx(STANDING)


@pytest.mark.parametrize('yaw', [0.0, math.pi / 2, -2.5])
def test_a_lying_body_is_on_its_back_over_its_spot_head_away_from_its_facing(yaw: float):
    root = _root(3.0, -1.0, yaw, floor=0.2)
    positions, quaternions = rest(Support(LIE_ON, 0.6), CLEARANCE, root, root[2, 3], PELVIS, *_posed(root))
    facing = np.array((math.cos(yaw), math.sin(yaw), 0.0))
    assert positions[PELVIS] == pytest.approx((3.0, -1.0, 0.2 + 0.6 + 0.12))
    assert positions[HEAD] - positions[PELVIS] == pytest.approx(-0.8 * facing)  # the head 0.8 m beyond, away from the facing
    assert positions[TOE, 2] - positions[PELVIS, 2] == pytest.approx(0.2)  # the toes point up
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternions[HEAD])
    assert rotation.reshape(3, 3) @ (0.0, 0.0, 1.0) == pytest.approx(-facing)  # every bone tipped with the body


def test_clearance_comes_from_the_annotation_beside_the_sdf(tmp_path: Path):
    measured, bare = tmp_path / 'measured', tmp_path / 'bare'
    measured.mkdir()
    bare.mkdir()
    (measured / 'annotation.yaml').write_text('pelvis_clearance: {sit: 0.11, lie: 0.13}\n')
    (bare / 'annotation.yaml').write_text('name: bare\n')
    assert pelvis_clearance(str(measured / 'person.sdf')) == PelvisClearance(sit=0.11, lie=0.13)
    assert pelvis_clearance(str(bare / 'person.sdf')) == TYPICAL_CLEARANCE
    assert pelvis_clearance(str(tmp_path / 'missing' / 'person.sdf')) == TYPICAL_CLEARANCE
