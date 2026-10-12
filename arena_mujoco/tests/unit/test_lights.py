import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from arena_mujoco_msgs.msg import Light as LightMsg
from arena_mujoco_msgs.msg import LightState, Prim, PrimVisual, Scale
from arena_mujoco_msgs.srv import DeleteLights, SetLights, SpawnLights
from geometry_msgs.msg import Point, Pose, Vector3

from arena_mujoco import context
from arena_mujoco.lights import DEFAULT_SUN, DOME_GAIN, FILL_GAIN, GLOW_RESPONSE, REFERENCE_LUX, REFLECTANCE, RIG_GAIN, SLOTS, Headlight, Light, Lighting, glow_material, lighting
from arena_mujoco.mesh_mat import COLLISION_GROUP
from arena_mujoco.scene import SceneStore
from arena_mujoco.services.Lights import _delete_lights_cb, _set_lights_cb, _spawn_lights_cb
from arena_mujoco.services.SpawnPrims import _add_prim

_WHITE = (1.0, 1.0, 1.0)
_DOWN = (0.0, 0.0, -1.0)
_SIZE = 64
_PER_LUMEN = 1.0 / REFERENCE_LUX


def _panel(position: tuple[float, float, float], lumens: float = 3600.0, **kwargs: object) -> Light:
    return Light(shape='rect', positions=[position], direction=_DOWN, flux=lumens, color=_WHITE, **kwargs)


def _rig(columns: int, rows: int, pitch: float = 2.4, **kwargs: object) -> Light:
    positions = [(x * pitch, y * pitch, 2.6) for x in range(columns) for y in range(rows)]
    return Light(shape='rect', positions=positions, direction=_DOWN, flux=3600.0, color=_WHITE, **kwargs)


def _scaled(color: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (color[0] * factor, color[1] * factor, color[2] * factor)


def _total(lamps: list) -> float:
    return sum(lamp.diffuse[0] for lamp in lamps)


def test_object_light_keeps_the_default_lighting_and_a_declared_one_replaces_it():
    env = Lighting()
    env.spawn('lamp', _panel((0.0, 0.0, 1.0), intrinsic=True))
    assert env.headlight((0.0, 0.0, 1.0)) == Headlight()
    assert env.lamps((0.0, 0.0, 1.0))[0] == DEFAULT_SUN
    assert len(env.lamps((0.0, 0.0, 1.0))) == 2

    env.spawn('ceiling', _panel((3.0, 0.0, 2.6)))
    assert env.headlight((0.0, 0.0, 1.0)) == Headlight(ambient=(0.0, 0.0, 0.0), diffuse=(0.0, 0.0, 0.0), specular=(0.0, 0.0, 0.0))
    assert DEFAULT_SUN not in env.lamps((0.0, 0.0, 1.0))

    assert env.remove('ceiling')
    assert env.lamps((0.0, 0.0, 1.0))[0] == DEFAULT_SUN
    assert not env.remove('ceiling')


def test_dome_fills_and_shines_down_and_a_sun_follows_its_direction_and_level():
    env = Lighting()
    env.spawn('sky', Light(shape='dome', positions=[], direction=_DOWN, flux=REFERENCE_LUX, color=(1.0, 0.5, 0.25)))
    env.spawn('sun', Light(shape='distant', positions=[], direction=(1.0, 0.0, -1.0), flux=2.0 * REFERENCE_LUX, color=_WHITE, cast_shadows=True))
    assert env.headlight((0.0, 0.0, 1.0)).ambient == pytest.approx(_scaled((0.5, 0.25, 0.125), DOME_GAIN))
    down, sun = sorted(env.lamps((0.0, 0.0, 1.0)), key=lambda lamp: lamp.diffuse[0])
    assert down.directional and down.direction == _DOWN and down.diffuse == pytest.approx(_scaled((0.5, 0.25, 0.125), DOME_GAIN))
    assert sun.directional and sun.castshadow
    assert sun.direction == pytest.approx((math.sqrt(0.5), 0.0, -math.sqrt(0.5)))
    assert sun.diffuse == pytest.approx((2.0, 2.0, 2.0))

    assert env.set_output('sun', 0.25, [])
    assert max(lamp.diffuse[0] for lamp in env.lamps((0.0, 0.0, 1.0))) == pytest.approx(0.5)
    assert env.set_output('sun', 0.0, [])
    assert len(env.lamps((0.0, 0.0, 1.0))) == 1


def test_a_large_rig_fits_the_slots_without_losing_light():
    env = Lighting()
    env.spawn('hall', _rig(8, 5))
    one = 3600.0 * _PER_LUMEN / math.tau
    for eye in ((0.0, 0.0, 1.0), (8.0, 5.0, 60.0), (300.0, 0.0, 1.0)):
        lamps = env.lamps(eye)
        assert len(lamps) == SLOTS
        assert _total(lamps) == pytest.approx(40 * one)


def test_fixtures_at_the_eye_merge_less_than_far_ones():
    env = Lighting()
    env.spawn('hall', _rig(8, 5))
    one = 3600.0 * _PER_LUMEN / math.tau
    near = env.lamps((0.0, 0.0, 1.0))
    closest = min(near, key=lambda lamp: math.dist(lamp.position, (0.0, 0.0, 2.6)))
    farthest = max(near, key=lambda lamp: math.dist(lamp.position, (0.0, 0.0, 2.6)))
    assert closest.diffuse[0] <= 2.0 * one + 1e-9
    assert closest.softening < 3.0
    assert farthest.diffuse[0] > 8 * one
    assert farthest.softening > 4.0


def test_dead_fixtures_and_a_lower_level_dim_a_rig():
    env = Lighting()
    env.spawn('room', _rig(2, 2))
    eye = (1.2, 1.2, 1.0)
    full = _total(env.lamps(eye))
    assert env.set_output('room', 0.5, [True, True, False, False])
    assert _total(env.lamps(eye)) == pytest.approx(full / 4.0)
    assert len(env.lamps(eye)) == 2
    assert env.set_output('room', 0.0, [True] * 4)
    assert env.lamps(eye) == []
    assert not env.set_output('hallway', 1.0, [])


def test_panel_shines_only_into_the_half_space_it_faces():
    env = Lighting()
    env.spawn('panel', _panel((0.0, 0.0, 2.6)))
    (panel,) = env.lamps((0.0, 0.0, 1.0))
    assert panel.cutoff == 90.0
    assert panel.direction == _DOWN
    assert panel.ambient == (0.0, 0.0, 0.0)


def test_rig_fills_its_room_from_its_center_with_the_light_its_surfaces_reflect():
    env = Lighting()
    env.spawn('hall', _rig(4, 3, rig=True))
    length, width, height = 3 * 2.4 + 2.4, 2 * 2.4 + 2.4, 2.6
    surface = 2.0 * (length * width + length * height + width * height)
    reflected = FILL_GAIN * REFLECTANCE * 12 * 3600.0 / (surface * (1.0 - REFLECTANCE)) / REFERENCE_LUX

    def bounces(eye: tuple[float, float, float]) -> list:
        return [lamp for lamp in env.lamps(eye) if lamp.flat]

    (bounce,) = bounces((3.6, 2.4, 1.0))
    assert bounce.position == pytest.approx((3.6, 2.4, 1.3))
    assert bounce.diffuse == pytest.approx((reflected,) * 3)
    assert env.headlight((3.6, 2.4, 1.0)).ambient == (0.0, 0.0, 0.0)
    assert bounces((3.6, 2.4, 40.0)) == [bounce]
    assert bounces((30.0, 2.4, 1.0)) == []
    direct = _total(env.lamps((3.6, 2.4, 1.0))) - bounce.diffuse[0]
    assert direct == pytest.approx(12 * RIG_GAIN * 3600.0 * _PER_LUMEN / math.tau)
    assert len(env.lamps((3.6, 2.4, 1.0))) == SLOTS

    env.set_output('hall', 0.5, [True] * 6 + [False] * 6)
    assert bounces((3.6, 2.4, 1.0))[0].diffuse == pytest.approx((reflected / 4.0,) * 3)
    env.spawn('lamp', _panel((3.6, 2.4, 2.6)))
    assert bounces((3.6, 2.4, 1.0))[0].diffuse == pytest.approx((reflected / 4.0,) * 3)


def test_flat_bounce_is_unattenuated_in_the_scene():
    store = SceneStore()
    store.ensure_env(0)
    model, data = store.get_model(0), store.get_data(0)
    env = Lighting()
    env.spawn('hall', _rig(2, 1, rig=True))
    assert env.dress(model, data, (1.2, 0.0, 1.0))
    flat = [index for index in range(model.nlight) if model.light_active[index] and tuple(model.light_attenuation[index]) == (1.0, 0.0, 0.0) and model.light_type[index] == int(mujoco.mjtLightType.mjLIGHT_SPOT)]
    assert len(flat) == 1
    assert tuple(model.light_pos[flat[0]]) == pytest.approx((1.2, 0.0, 1.3))


def test_spot_keeps_its_cone_and_a_moved_one_turns_with_the_yaw():
    env = Lighting()
    env.spawn('beam', Light(shape='disk', positions=[(0.0, 0.0, 0.5)], direction=(1.0, 0.0, 0.0), flux=1000.0, color=_WHITE, cone_deg=40.0))
    (beam,) = env.lamps((0.0, 0.0, 1.0))
    assert beam.cutoff == pytest.approx(20.0)
    assert beam.ambient == (0.0, 0.0, 0.0)
    assert beam.diffuse[0] == pytest.approx(1000.0 * _PER_LUMEN / (math.tau * (1.0 - math.cos(math.radians(20.0)))))

    assert env.move('beam', (4.0, 2.0, 0.5), math.pi / 2.0)
    (beam,) = env.lamps((0.0, 0.0, 1.0))
    assert beam.position == (4.0, 2.0, 0.5)
    assert beam.direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert not env.move('headlamp', (0.0, 0.0, 0.0), 0.0)


def test_bulb_shines_all_around_at_half_the_intensity_of_a_panel():
    env = Lighting()
    env.spawn('bulb', Light(shape='sphere', positions=[(0.0, 0.0, 1.0)], direction=_DOWN, flux=800.0, color=_WHITE, intrinsic=True))
    bulb = env.lamps((0.0, 0.0, 1.0))[1]
    assert bulb.cutoff == 180.0
    assert bulb.diffuse[0] == pytest.approx(800.0 * _PER_LUMEN / (2.0 * math.tau))
    assert bulb.ambient == (0.0, 0.0, 0.0)


def test_compiled_model_mirrors_the_view_only_when_it_changes():
    store = SceneStore()
    store.ensure_env(0)
    model, data = store.get_model(0), store.get_data(0)
    assert model.nlight == SLOTS
    assert list(model.light_active) == [1] + [0] * (SLOTS - 1)

    env = Lighting()
    eye = (0.0, 0.0, 1.0)
    assert not env.dress(model, data, eye)
    env.spawn('room', _rig(2, 1))
    assert env.dress(model, data, eye)
    assert not env.dress(model, data, eye)
    assert list(model.light_active) == [1, 1] + [0] * (SLOTS - 2)
    assert tuple(model.vis.headlight.ambient) == (0.0, 0.0, 0.0)
    assert int(model.light_type[0]) == int(mujoco.mjtLightType.mjLIGHT_SPOT)
    assert model.light_cutoff[0] == 90.0
    assert sorted(float(x) for x in data.light_xpos[:2, 0]) == pytest.approx([0.0, 2.4])

    env.clear()
    assert env.dress(model, data, eye)
    assert list(model.light_active) == [1] + [0] * (SLOTS - 1)
    assert int(model.light_type[0]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    assert tuple(model.light_diffuse[0]) == pytest.approx(DEFAULT_SUN.diffuse)
    assert tuple(model.vis.headlight.ambient) == pytest.approx(Headlight().ambient)

    store.add_body(0, lambda spec: spec.worldbody.add_body(name='late'))
    store.recompile(0)
    assert env.dress(store.get_model(0), store.get_data(0), eye)


@pytest.fixture
def room() -> tuple[mujoco.Renderer, mujoco.MjData, mujoco.MjvCamera]:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = _SIZE
    spec.visual.global_.offheight = _SIZE
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10.0, 10.0, 0.5], rgba=[0.5, 0.5, 0.5, 1.0])
    spec.worldbody.add_light(pos=[0.0, 0.0, 10.0], dir=[0.0, 0.0, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, _SIZE, _SIZE)
    except Exception as exc:
        pytest.skip(f'no offscreen GL: {exc}')
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.elevation = -89.0
    yield renderer, data, camera
    renderer.close()


def _floor(room: tuple[mujoco.Renderer, mujoco.MjData, mujoco.MjvCamera], env: Lighting) -> float:
    renderer, data, camera = room
    renderer.update_scene(data, camera)
    env.shade(renderer.scene, (0.0, 0.0, 3.0))
    return float(renderer.render()[_SIZE // 2, _SIZE // 2].mean())


def test_floor_under_a_panel_renders_at_its_illuminance_and_dark_when_off(room: tuple[mujoco.Renderer, mujoco.MjData, mujoco.MjvCamera]):
    env = Lighting()
    untouched = _floor(room, env)
    env.spawn('panel', _panel((0.0, 0.0, 2.0)))
    lit = _floor(room, env)
    intensity = 3600.0 * _PER_LUMEN / math.tau
    assert lit == pytest.approx(255.0 * 0.5 * intensity / (1.0 + 2.0**2), abs=3.0)
    assert lit != pytest.approx(untouched, abs=3.0)

    env.set_output('panel', 0.5, [])
    assert _floor(room, env) == pytest.approx(lit / 2.0, abs=3.0)
    env.set_output('panel', 0.0, [])
    assert _floor(room, env) == 0.0
    env.clear()
    assert _floor(room, env) > 200.0


def test_light_services_spawn_set_and_delete_per_env():
    env_id = 41
    light = LightMsg(name='room', shape='rect', cone_deg=180.0, positions=[Point(x=0.0, y=0.0, z=2.6), Point(x=2.4, y=0.0, z=2.6)], direction=Vector3(z=-1.0), flux=3600.0, color=[1.0, 1.0, 1.0], level=1.0, alive=[True, True])
    assert list(_spawn_lights_cb(SpawnLights.Request(env_id=env_id, lights=[light]), SpawnLights.Response()).ret) == [True]
    eye = (0.0, 0.0, 1.0)
    assert len(lighting(env_id).lamps(eye)) == 2
    assert lighting(env_id + 1).lamps(eye) == [DEFAULT_SUN]

    states = [LightState(name='room', level=1.0, alive=[True, False]), LightState(name='hallway', level=1.0)]
    assert list(_set_lights_cb(SetLights.Request(env_id=env_id, states=states), SetLights.Response()).ret) == [True, False]
    assert len(lighting(env_id).lamps(eye)) == 1

    assert list(_delete_lights_cb(DeleteLights.Request(env_id=env_id, names=['room', 'room']), DeleteLights.Response()).ret) == [True, False]
    assert lighting(env_id).lamps(eye) == [DEFAULT_SUN]


_SHADE = (0.85, 0.8, 0.7, 1.0)
_WARM = (1.0, 0.7, 0.4)


def _box_glb(path: Path, extents: tuple[float, float, float], center: tuple[float, float, float], rgba: tuple[float, float, float, float]) -> str:
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    box.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=[int(c * 255) for c in rgba]))
    box.export(str(path))
    return str(path)


@pytest.fixture
def lamps(tmp_path: Path) -> SceneStore:
    store = SceneStore()
    store.ensure_env(0)
    context.set_scene_store(store)
    stand = _box_glb(tmp_path / 'lamp.glb', (0.1, 0.1, 1.0), (0.0, 0.0, 0.5), (0.2, 0.2, 0.2, 1.0))
    shade = _box_glb(tmp_path / 'lamp.glow_0.glb', (0.4, 0.4, 0.3), (0.0, 0.0, 1.15), _SHADE)
    for name, x in (('lamp_a', 0.0), ('lamp_b', 2.0)):
        prim = Prim(name=name, pose=Pose(position=Point(x=x)), scale=Scale(x=1.0, y=1.0, z=1.0), mesh_path=stand, visuals=[PrimVisual(name='glow_Shade', mesh_path=shade)])
        prim.pose.orientation.w = 1.0
        assert _add_prim(0, prim)
    store.recompile(0)
    return store


def _glow_ids(model: mujoco.MjModel, owner: str) -> list[int]:
    return [index for index in range(model.nmat) if model.material(index).name.startswith(f'{glow_material(owner, "Shade")}/')]


def test_each_lamp_glows_with_its_own_light_and_is_its_own_color_while_off(lamps: SceneStore):
    model = lamps.get_model(0)
    (a,), (b,) = _glow_ids(model, 'lamp_a'), _glow_ids(model, 'lamp_b')
    assert tuple(model.mat_rgba[a]) == pytest.approx(_SHADE, abs=0.01)
    lamp_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ('lamp_a', 'lamp_b')]
    for body in lamp_bodies:
        geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
        assert sum(model.geom_group[g] == 0 for g in geoms) == 2
        assert sum(model.geom_group[g] == COLLISION_GROUP for g in geoms) >= 2

    env = Lighting()
    assert not env.glow(model)
    env.spawn('a', Light(shape='sphere', positions=[(0.0, 0.0, 1.1)], direction=(0.0, 0.0, -1.0), flux=800.0, color=_WARM, intrinsic=True, glow=glow_material('lamp_a', 'Shade')))
    env.spawn('b', Light(shape='sphere', positions=[(2.0, 0.0, 1.1)], direction=(0.0, 0.0, -1.0), flux=800.0, color=_WARM, intrinsic=True, level=0.0, glow=glow_material('lamp_b', 'Shade')))
    assert env.glow(model)
    assert not env.glow(model)
    assert (model.mat_emission[a], model.mat_emission[b]) == (1.0, 0.0)
    assert tuple(model.mat_rgba[a]) == pytest.approx((*_WARM, 1.0))
    assert tuple(model.mat_rgba[b]) == pytest.approx(_SHADE, abs=0.01)

    env.set_output('a', 0.3, [])
    assert env.glow(model)
    assert model.mat_emission[a] == pytest.approx(0.3**GLOW_RESPONSE)
    env.set_output('a', 1.0, [False])
    assert env.glow(model)
    assert model.mat_emission[a] == 0.0
    assert tuple(model.mat_rgba[a]) == pytest.approx(_SHADE, abs=0.01)

    env.set_output('b', 1.0, [])
    env.remove('a')
    lamps.add_body(0, lambda spec: spec.worldbody.add_body(name='late'))
    lamps.recompile(0)
    fresh = lamps.get_model(0)
    assert env.dress(fresh, lamps.get_data(0), (0.0, 0.0, 1.0))
    (a,), (b,) = _glow_ids(fresh, 'lamp_a'), _glow_ids(fresh, 'lamp_b')
    assert (fresh.mat_emission[a], fresh.mat_emission[b]) == (0.0, 1.0)


def test_glowing_shade_renders_bright_in_the_dark_and_black_when_off(lamps: SceneStore):
    model, data = lamps.get_model(0), lamps.get_data(0)
    try:
        renderer = mujoco.Renderer(model, _SIZE, _SIZE)
    except Exception as exc:
        pytest.skip(f'no offscreen GL: {exc}')
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 1.15)
    camera.distance = 1.0
    camera.elevation = 0.0
    env = Lighting()
    env.spawn('night', Light(shape='dome', positions=[], direction=(0.0, 0.0, -1.0), flux=400.0, color=(1.0, 1.0, 1.0), level=0.0))
    env.spawn('a', Light(shape='sphere', positions=[(0.0, 0.0, 3.0)], direction=(0.0, 0.0, -1.0), flux=0.0, color=_WARM, intrinsic=True, glow=glow_material('lamp_a', 'Shade')))

    def center() -> np.ndarray:
        env.glow(model)
        renderer.update_scene(data, camera)
        env.shade(renderer.scene, (1.0, 0.0, 1.15))
        return renderer.render()[_SIZE // 2, _SIZE // 2].astype(float)

    lit = center()
    assert lit == pytest.approx([c * 255.0 for c in _WARM], abs=3.0)
    env.set_output('a', 0.0, [])
    assert center().max() == 0.0
    renderer.close()


def test_light_service_names_the_glow_material_of_its_owner():
    env_id = 43
    light = LightMsg(name='lamp_light', shape='sphere', positions=[Point(z=1.1)], direction=Vector3(z=-1.0), flux=800.0, color=[1.0, 0.7, 0.4], level=1.0, intrinsic=True, glow='Shade', owner='env_43/lamp')
    plain = LightMsg(name='room', shape='rect', positions=[Point(z=2.6)], direction=Vector3(z=-1.0), flux=3600.0, color=[1.0, 1.0, 1.0], level=1.0)
    assert list(_spawn_lights_cb(SpawnLights.Request(env_id=env_id, lights=[light, plain]), SpawnLights.Response()).ret) == [True, True]
    assert {name: entry.glow for name, entry in lighting(env_id)._lights.items()} == {'lamp_light': glow_material('env_43/lamp', 'Shade'), 'room': ''}
