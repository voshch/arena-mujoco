"""SpawnArticulated and DriveArticulated: articulated objects as body trees with damped joints and switchable position drives."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import mujoco
from arena_mujoco_msgs.msg import ArticulatedLink, JointReading
from arena_mujoco_msgs.srv import DriveArticulated, SpawnArticulated

from arena_mujoco import hooks
from arena_mujoco.context import get_scene_store
from arena_mujoco.mesh_mat import add_mesh_geoms, add_visual_geoms, prepare_hulls
from arena_mujoco.scene import AssetCache, sanitize_id

from .utils import Service

SEPARATOR = '__'
DRIVE_SUFFIX = '__drive'
WORLD_BODY = 'world'
MIN_MASS = 1e-3
BOUNCE_FREQUENCY = 160.0
BOUNCE_SOLMIX = 1e3
BOUNCE_BISECTIONS = 30
MAX_RESTITUTION = 0.99


@dataclass
class _Joint:
    """One MJCF joint of an object, its drive actuator and gains, and whether the drive is on."""

    name: str
    actuator: str
    stiffness: float
    damping: float
    wraps: bool
    on: bool = False


@dataclass
class _Object:
    bodies: list[str]
    joints: dict[str, _Joint] = field(default_factory=dict)


_objects: dict[tuple[int, str], _Object] = {}


def bounce_damping(restitution: float) -> float:
    """Damping ratio, critical at most, of a push-only spring-damper contact that rebounds at this restitution."""
    low, high = 0.0, 1.0
    for _ in range(BOUNCE_BISECTIONS):
        ratio = (low + high) / 2.0
        if math.exp(-2.0 * ratio * math.acos(ratio) / math.sqrt(1.0 - ratio * ratio)) > min(restitution, MAX_RESTITUTION):
            low = ratio
        else:
            high = ratio
    return high


def _prm(*values: float) -> list[float]:
    return [*values, *([0.0] * (10 - len(values)))]


def _gains(joint: _Joint, on: bool) -> tuple[list[float], list[float]]:
    if not on:
        return _prm(0.0), _prm(0.0)
    return _prm(joint.stiffness), _prm(0.0, -joint.stiffness, -joint.damping)


def _add_link(spec: mujoco.MjSpec, cache: AssetCache, root: str, parent: mujoco.MjsBody, link: ArticulatedLink, scale: tuple[float, float, float], obj: _Object) -> mujoco.MjsBody:
    body = parent.add_body()
    body.name = root if not link.parent else f'{root}{SEPARATOR}{sanitize_id(link.name)}'
    body.explicitinertial = True
    body.mass = max(link.mass, MIN_MASS)
    body.ipos = [link.center.x, link.center.y, link.center.z]
    body.inertia = [link.moments.x, link.moments.y, link.moments.z]
    if link.mesh_path:
        add_mesh_geoms(spec, cache, body, link.mesh_path, scale)
        add_visual_geoms(spec, cache, body, root, [(visual.name, visual.mesh_path) for visual in link.visuals], scale)
    if link.has_material:
        for geom in body.geoms:
            geom.friction = [link.dynamic_friction, geom.friction[1], geom.friction[2]]
            if link.restitution > 0.0:
                ratio = bounce_damping(link.restitution)
                geom.solref = [1.0 / (BOUNCE_FREQUENCY * ratio), ratio]
                geom.solmix = BOUNCE_SOLMIX
    obj.bodies.append(body.name)
    if not link.parent or link.type == ArticulatedLink.FIXED:
        return body
    joint = body.add_joint()
    joint.name = f'{root}{SEPARATOR}{sanitize_id(link.joint)}'
    joint.type = mujoco.mjtJoint.mjJNT_SLIDE if link.type == ArticulatedLink.PRISMATIC else mujoco.mjtJoint.mjJNT_HINGE
    joint.pos = [link.origin.x, link.origin.y, link.origin.z]
    joint.axis = [link.axis.x, link.axis.y, link.axis.z]
    if link.type != ArticulatedLink.CONTINUOUS:
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        angular = link.type == ArticulatedLink.REVOLUTE and spec.compiler.degree
        joint.range = [math.degrees(link.lower), math.degrees(link.upper)] if angular else [link.lower, link.upper]
    dynamics = link.dynamics
    joint.damping = [dynamics.damping, 0.0, 0.0]
    joint.frictionloss = dynamics.friction
    joint.armature = dynamics.armature
    joint.stiffness = [dynamics.spring_stiffness, 0.0, 0.0]
    joint.springref = math.degrees(dynamics.spring_rest) if link.type != ArticulatedLink.PRISMATIC and spec.compiler.degree else dynamics.spring_rest
    entry = _Joint(joint.name, f'{joint.name}{DRIVE_SUFFIX}', dynamics.stiffness, dynamics.drive_damping, link.type == ArticulatedLink.CONTINUOUS)
    actuator = spec.add_actuator()
    actuator.name = entry.actuator
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.target = joint.name
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.gainprm, actuator.biasprm = _gains(entry, False)
    actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    actuator.forcerange = [-dynamics.effort, dynamics.effort]
    obj.joints[link.joint] = entry
    return body


def _build(spec: mujoco.MjSpec, cache: AssetCache, request: SpawnArticulated.Request, obj: _Object) -> mujoco.MjsBody:
    """The object's bodies nested link under parent link at the model frame, the root fixed at the request pose, contacts within the object and with the world body excluded."""
    root = sanitize_id(request.name)
    scale = (request.scale.x, request.scale.y, request.scale.z)
    bodies: dict[str, mujoco.MjsBody] = {}
    for link in request.links:
        bodies[link.name] = _add_link(spec, cache, root, bodies[link.parent] if link.parent else spec.worldbody, link, scale, obj)
    top = bodies[list(request.links)[0].name]
    pose = request.pose
    top.pos = [pose.position.x, pose.position.y, pose.position.z]
    top.quat = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    for first, second in itertools.combinations([*obj.bodies, WORLD_BODY], 2):
        spec.add_exclude(bodyname1=first, bodyname2=second)
    return top


def _set_initial(env_id: int, request: SpawnArticulated.Request, obj: _Object) -> None:
    store = get_scene_store()
    model, data = store.get_model(env_id), store.get_data(env_id)
    if model is None or data is None:
        return
    for link in request.links:
        joint = obj.joints.get(link.joint) if link.parent else None
        if joint is not None:
            data.qpos[model.jnt_qposadr[model.joint(joint.name).id]] = link.value
    mujoco.mj_forward(model, data)


def _drop(spec: mujoco.MjSpec, obj: _Object) -> None:
    actuators = {joint.actuator for joint in obj.joints.values()}
    bodies = set(obj.bodies)
    for actuator in [actuator for actuator in spec.actuators if actuator.name in actuators]:
        spec.delete(actuator)
    for exclude in [exclude for exclude in spec.excludes if exclude.bodyname1 in bodies or exclude.bodyname2 in bodies]:
        spec.delete(exclude)


def forget_articulated(env_id: int, name: str) -> None:
    """Drop the drives and contact excludes of the objects a DeletePrims name removes (a trailing / removes by prefix), before their bodies leave the spec."""
    spec = get_scene_store().get_spec(env_id)
    stem = name.removesuffix('/')
    key = sanitize_id(stem) if stem else ''

    def removed(item: str) -> bool:
        return item.startswith(key) if name.endswith('/') else item == key

    for item in [item for item in _objects if item[0] == env_id and removed(item[1])]:
        obj = _objects.pop(item)
        if spec is not None:
            _drop(spec, obj)


def _abandon(env_id: int, obj: _Object, reason: object, name: str) -> None:
    """Take a failed spawn's drives, excludes and bodies back out of the spec and compile without them."""
    store = get_scene_store()
    spec = store.get_spec(env_id)
    if spec is not None:
        _drop(spec, obj)
        if obj.bodies and (root := spec.body(obj.bodies[0])) is not None:
            spec.delete(root)
    store.discard_pending(env_id)
    hooks.get_node().get_logger().warning(f'articulated object {name!r} not spawned: {reason}')


def _spawn_articulated_cb(request: SpawnArticulated.Request, response: SpawnArticulated.Response) -> SpawnArticulated.Response:
    store = get_scene_store()
    env_id = int(request.env_id)
    store.ensure_env(env_id)
    key = sanitize_id(request.name)
    if (env_id, key) in _objects or not request.links:
        response.ret = False
        return response
    prepare_hulls([path for link in request.links for path in (link.mesh_path, *(visual.mesh_path for visual in link.visuals)) if path])
    obj = _Object(bodies=[])
    try:
        store.add_body(env_id, lambda spec: _build(spec, store.asset_cache(env_id), request, obj))
        store.recompile(env_id)
    except Exception as exc:  # noqa: BLE001 - one unbuildable object must not take the server down
        _abandon(env_id, obj, exc, request.name)
        response.ret = False
        return response
    _objects[(env_id, key)] = obj
    _set_initial(env_id, request, obj)
    response.ret = True
    return response


def _drive_articulated_cb(request: DriveArticulated.Request, response: DriveArticulated.Response) -> DriveArticulated.Response:
    store = get_scene_store()
    env_id = int(request.env_id)
    model, data, spec = store.get_model(env_id), store.get_data(env_id), store.get_spec(env_id)
    readings: list[JointReading] = []
    for command in request.commands:
        obj = _objects.get((env_id, sanitize_id(command.name)))
        reading = JointReading(name=command.name, found=obj is not None and model is not None and data is not None and spec is not None)
        readings.append(reading)
        if obj is None or model is None or data is None or spec is None:
            continue
        placed = False
        for name, reference, drive, place in zip(command.joints, command.reference, command.drive, command.place, strict=True):
            joint = obj.joints.get(name)
            if joint is None:
                continue
            joint_id = model.joint(joint.name).id
            qpos = int(model.jnt_qposadr[joint_id])
            if place:
                data.qpos[qpos] = reference
                data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
                placed = True
            if drive != joint.on:
                gain, bias = _gains(joint, drive)
                actuator = model.actuator(joint.actuator).id
                model.actuator_gainprm[actuator] = gain
                model.actuator_biasprm[actuator] = bias
                compiled = spec.actuator(joint.actuator)
                compiled.gainprm, compiled.biasprm = gain, bias
                joint.on = drive
                store.note_model_edit(env_id)
            current = float(data.qpos[qpos])
            data.ctrl[model.actuator(joint.actuator).id] = current + math.remainder(reference - current, 2 * math.pi) if joint.wraps else reference
        if placed:
            mujoco.mj_forward(model, data)
        reading.joints = list(obj.joints)
        reading.position = [float(data.qpos[model.jnt_qposadr[model.joint(joint.name).id]]) for joint in obj.joints.values()]
    response.readings = readings
    return response


spawn_articulated_service = Service(
    srv_type=SpawnArticulated,
    srv_name='mujoco/SpawnArticulated',
    callback=_spawn_articulated_cb,
)

drive_articulated_service = Service(
    srv_type=DriveArticulated,
    srv_name='mujoco/DriveArticulated',
    callback=_drive_articulated_cb,
)

__all__ = ['drive_articulated_service', 'forget_articulated', 'spawn_articulated_service']
