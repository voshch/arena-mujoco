"""Pedestrians resting on furniture: sitting on a seat or lying on a bed.

The wire carries a floor position, a yaw and joint angles. A sit clip folds the legs but keeps the hips at their
standing height, and no clip tips a whole body onto its back. While a ped holds a sit or lie interaction, the adapter
reports how high the surface under it is (Pedestrian.support_height); `rest` moves the posed bones so the pelvis rests
that character's measured clearance above the surface (annotation `pelvis_clearance`, measured by the arena_assets
human build). The same placement as arena_isaac's peds/support.py.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import yaml

if TYPE_CHECKING:
    from arena_people_msgs.msg import Pedestrian

SIT_ON = 3  # humansim InteractionType values, as Pedestrian.interaction_type carries them
LIE_ON = 4
PELVIS_BONE = 'Hips'  # cmu_mb rig, the skeleton's root joint
NO_SUPPORT = -1.0  # Pedestrian.NO_SUPPORT

# local +x (facing) -> up, local +z (head) -> -x: a quarter turn about local y
_SUPINE = np.array(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)))


@dataclass(frozen=True)
class Support:
    """Furniture a ped rests on: how (SIT_ON or LIE_ON), and the surface height above the ped's floor in m."""

    kind: int
    height: float


@dataclass(frozen=True)
class PelvisClearance:
    """Height of a character's pelvis joint above the surface it rests on, in m: seated, and lying on its back."""

    sit: float
    lie: float


# Median over the 83 characters the arena_assets human build measured, for a character built before the measurement.
TYPICAL_CLEARANCE = PelvisClearance(sit=0.122, lie=0.117)


def support_of(pedestrian: Pedestrian) -> Support | None:
    """The seat or bed the ped rests on, None while it is on its feet or the surface is unknown."""
    if pedestrian.support_height == NO_SUPPORT or pedestrian.interaction_type not in (SIT_ON, LIE_ON):
        return None
    return Support(kind=int(pedestrian.interaction_type), height=float(pedestrian.support_height))


@functools.cache
def pelvis_clearance(sdf_path: str) -> PelvisClearance:
    """The clearance in the annotation beside a character's SDF, the typical one when the asset was built without it."""
    try:
        annotation = yaml.safe_load((Path(sdf_path).parent / 'annotation.yaml').read_text()) or {}
    except (OSError, yaml.YAMLError):
        return TYPICAL_CLEARANCE
    clearance = annotation.get('pelvis_clearance')
    return PelvisClearance(sit=float(clearance['sit']), lie=float(clearance['lie'])) if clearance else TYPICAL_CLEARANCE


def rest(support: Support, clearance: PelvisClearance, root: np.ndarray, floor: float, pelvis: int, positions: np.ndarray, quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bone positions (bones, 3) and wxyz quaternions (bones, 4) of a body posed standing at root (4, 4) on the floor
    at height floor, set down on its supporting surface. pelvis is the index of the pelvis bone."""
    if support.kind == LIE_ON:
        origin = np.array((root[0, 3], root[1, 3], floor))
        return _lie(root[:3, :3], origin, positions, quaternions, positions[pelvis] - origin, floor + support.height + clearance.lie)
    positions = positions.copy()
    # the feet never go below the floor: while the clip still folds the legs the body stays up, and lowers onto the
    # seat as they fold. On a seat higher than the folded legs reach, the feet hang.
    positions[:, 2] += max(floor + support.height + clearance.sit - positions[pelvis, 2], 0.0)
    return positions, quaternions


def _lie(rotation: np.ndarray, origin: np.ndarray, positions: np.ndarray, quaternions: np.ndarray, pelvis: np.ndarray, pelvis_z: float) -> tuple[np.ndarray, np.ndarray]:
    """On its back with the pelvis at height pelvis_z, over origin (the ped's spot on the floor): the head points away
    from the facing, the feet toward it."""
    tip = rotation @ _SUPINE @ rotation.T  # the quarter turn about the body's own left-right axis, in world
    height = float(rotation.T[2] @ pelvis)  # pelvis height above the floor along the body's up axis
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    target = origin + (height * math.cos(yaw), height * math.sin(yaw), pelvis_z - origin[2])
    turn = np.empty(4)
    mujoco.mju_mat2Quat(turn, tip.flatten())
    return (positions - origin) @ tip.T + target, quat_mul(turn, quaternions)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of one wxyz quaternion with a stack (n, 4)."""
    w, x, y, z = left
    rw, rx, ry, rz = right.T
    return np.stack(
        [
            w * rw - x * rx - y * ry - z * rz,
            w * rx + x * rw + y * rz - z * ry,
            w * ry - x * rz + y * rw + z * rx,
            w * rz + x * ry - y * rx + z * rw,
        ],
        axis=1,
    )
