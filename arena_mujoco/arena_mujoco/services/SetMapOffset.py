"""SetMapOffset service handler: translate the map -> odom transform of one robot."""

from __future__ import annotations

from arena_mujoco_msgs.srv import SetMapOffset

from arena_mujoco import odom

from .utils import Service


def _set_map_offset_cb(request: SetMapOffset.Request, response: SetMapOffset.Response) -> SetMapOffset.Response:
    translation = request.translation
    response.ok = odom.set_map_offset(int(request.env_id), request.name, (float(translation.x), float(translation.y), float(translation.z)))
    return response


set_map_offset_service = Service(srv_type=SetMapOffset, srv_name='mujoco/SetMapOffset', callback=_set_map_offset_cb)

__all__ = ['set_map_offset_service']
