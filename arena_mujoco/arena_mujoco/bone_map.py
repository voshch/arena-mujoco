"""Wire joint to bone rotation map: each ROS4HRI joint angle turns its bones about an axis in the bone's own neutral frame."""

from __future__ import annotations

import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def _load(path: Path) -> dict[str, tuple[tuple[str, tuple[float, float, float], float], ...]]:
    """Bone targets per wire joint from arena_people_msgs' bone_map.json, gain as sign times scale."""
    return {joint: tuple((target['bone'], tuple(target['axis']), target['sign'] * target['scale']) for target in targets) for joint, targets in json.loads(path.read_text()).items()}


BONE_MAP = _load(Path(get_package_share_directory('arena_people_msgs')) / 'bone_map.json')

__all__ = ['BONE_MAP']
