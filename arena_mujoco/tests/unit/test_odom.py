import pytest
from arena_mujoco_msgs.srv import SetMapOffset
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Vector3

from arena_mujoco import odom
from arena_mujoco.services.SetMapOffset import _set_map_offset_cb

_JACKAL = 'env_0_Robots_jackal_base_link'


def _entry(env_id: int) -> odom._RobotOdom:
    return odom._RobotOdom(
        env_id=env_id,
        prim_name=_JACKAL,
        tf_prefix=f'env_{env_id}/jackal/',
        base_frame='base_link',
        odom_frame_id=f'env_{env_id}/jackal/odom',
        localization=True,
        odom_pub=None,
    )


@pytest.fixture
def robots():
    entries = (_entry(0), _entry(1))
    odom._entries.extend(entries)
    yield entries
    for entry in entries:
        odom._entries.remove(entry)


def _translation(entry: odom._RobotOdom) -> tuple[float, float, float]:
    translation = odom._map_transform(entry, Time(sec=2)).transform.translation
    return translation.x, translation.y, translation.z


def test_map_transform_is_the_identity_until_an_offset_is_set(robots):
    transform = odom._map_transform(robots[0], Time(sec=2))
    assert (transform.header.frame_id, transform.child_frame_id, transform.header.stamp.sec) == ('map', 'env_0/jackal/odom', 2)
    assert _translation(robots[0]) == (0.0, 0.0, 0.0)
    assert transform.transform.rotation.w == 1.0


def test_set_map_offset_translates_the_map_transform_of_that_robot_only(robots):
    jackal, twin = robots
    request = SetMapOffset.Request(env_id=0, name=_JACKAL, translation=Vector3(x=7.8, y=-1.5, z=-3.0))
    assert _set_map_offset_cb(request, SetMapOffset.Response()).ok
    assert _translation(jackal) == pytest.approx((7.8, -1.5, -3.0))
    assert _translation(twin) == (0.0, 0.0, 0.0)
    assert odom._map_transform(jackal, Time()).transform.rotation.w == 1.0


def test_set_map_offset_of_an_unknown_robot_answers_false(robots):
    assert not _set_map_offset_cb(SetMapOffset.Request(env_id=0, name='ghost', translation=Vector3(x=1.0)), SetMapOffset.Response()).ok
    assert not _set_map_offset_cb(SetMapOffset.Request(env_id=2, name=_JACKAL, translation=Vector3(x=1.0)), SetMapOffset.Response()).ok
    assert all(_translation(entry) == (0.0, 0.0, 0.0) for entry in robots)


def test_a_robot_frame_takes_its_robots_map_offset_and_other_frames_stay_zero(robots):
    assert odom.set_map_offset(1, _JACKAL, (7.8, 0.0, -3.0))
    assert odom.frame_map_offset(1, 'env_1/jackal/lidar') == (7.8, 0.0, -3.0)
    assert odom.frame_map_offset(1, '/env_1/jackal/lidar') == (7.8, 0.0, -3.0)
    assert odom.frame_map_offset(0, 'env_0/jackal/lidar') == (0.0, 0.0, 0.0)
    assert odom.frame_map_offset(0, 'env_1/jackal/lidar') == (0.0, 0.0, 0.0)
    assert odom.frame_map_offset(1, 'env_1/map') == (0.0, 0.0, 0.0)
