"""SpawnLights, DeleteLights and SetLights service handlers."""

from __future__ import annotations

from arena_mujoco_msgs.msg import Light as LightMsg
from arena_mujoco_msgs.srv import DeleteLights, SetLights, SpawnLights

from arena_mujoco.lights import Light, glow_material, lighting

from .utils import Service


def _light(msg: LightMsg) -> Light:
    return Light(
        shape=msg.shape,
        positions=[(float(p.x), float(p.y), float(p.z)) for p in msg.positions],
        direction=(float(msg.direction.x), float(msg.direction.y), float(msg.direction.z)),
        flux=float(msg.flux),
        color=(float(msg.color[0]), float(msg.color[1]), float(msg.color[2])),
        cone_deg=float(msg.cone_deg),
        cast_shadows=bool(msg.cast_shadows),
        intrinsic=bool(msg.intrinsic),
        level=float(msg.level),
        alive=[bool(alive) for alive in msg.alive],
        rig=bool(msg.rig),
        glow=glow_material(msg.owner, msg.glow) if msg.glow else '',
    )


def _spawn_lights_cb(request: SpawnLights.Request, response: SpawnLights.Response) -> SpawnLights.Response:
    env = lighting(int(request.env_id))
    for msg in request.lights:
        env.spawn(msg.name, _light(msg))
    response.ret = [True] * len(request.lights)
    return response


def _delete_lights_cb(request: DeleteLights.Request, response: DeleteLights.Response) -> DeleteLights.Response:
    env = lighting(int(request.env_id))
    response.ret = [env.remove(name) for name in request.names]
    return response


def _set_lights_cb(request: SetLights.Request, response: SetLights.Response) -> SetLights.Response:
    env = lighting(int(request.env_id))
    response.ret = [env.move(state.name, (float(state.position.x), float(state.position.y), float(state.position.z)), float(state.yaw)) if state.move else env.set_output(state.name, float(state.level), [bool(alive) for alive in state.alive]) for state in request.states]
    return response


spawn_lights_service = Service(srv_type=SpawnLights, srv_name='mujoco/SpawnLights', callback=_spawn_lights_cb)
delete_lights_service = Service(srv_type=DeleteLights, srv_name='mujoco/DeleteLights', callback=_delete_lights_cb)
set_lights_service = Service(srv_type=SetLights, srv_name='mujoco/SetLights', callback=_set_lights_cb)

__all__ = ['delete_lights_service', 'set_lights_service', 'spawn_lights_service']
