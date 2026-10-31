import mujoco
import numpy as np
import pytest

from arena_mujoco.scene import MOVING_BODY_KG, PHYSICS_DT, RejectedBodies, SceneStore


def _box(name: str, pos: tuple[float, float, float], free: bool = False):
    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name=name, pos=list(pos))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
        if free:
            body.add_freejoint().name = f'{name}_free'
        return body

    return build


def test_wire_names_resolve_after_sanitizing(store: SceneStore):
    name = store.add_body(0, _box('env_0/Obstacles/box-1', (1.0, 0.0, 0.1)))
    store.recompile(0)
    assert name == 'env_0_Obstacles_box_1'
    assert store.find(0, 'env_0/Obstacles/box-1')
    assert store.get_body_pose(0, 'env_0/Obstacles/box-1')[0] == pytest.approx((1.0, 0.0, 0.1))


def test_prefix_delete_keeps_other_namespaces(store: SceneStore):
    store.add_body(0, _box('env_0/Walls/a', (1.0, 0.0, 0.1)))
    store.add_body(0, _box('env_0/Walls/b', (2.0, 0.0, 0.1)))
    store.add_body(0, _box('env_0/Obstacles/c', (3.0, 0.0, 0.1)))
    assert store.remove_by_prefix(0, 'env_0/Walls') == 2
    store.recompile(0)
    assert not store.find(0, 'env_0/Walls/a')
    assert store.find(0, 'env_0/Obstacles/c')


def test_empty_prefix_deletes_every_body(store: SceneStore):
    store.add_body(0, _box('env_0/Walls/a', (1.0, 0.0, 0.1)))
    store.add_body(0, _box('env_0/Obstacles/c', (3.0, 0.0, 0.1)))
    assert store.remove_by_prefix(0, '') == 2
    store.recompile(0)
    assert store.get_model(0).nbody == 1


def test_moved_static_body_stays_moved_across_a_recompile(store: SceneStore):
    store.add_body(0, _box('env_0/Obstacles/crate', (1.0, 0.0, 0.1)))
    store.recompile(0)
    assert store.set_body_pose(0, 'env_0/Obstacles/crate', (4.0, 5.0, 0.1), (0.0, 0.0, 0.0, 1.0))
    store.add_body(0, _box('late', (9.0, 9.0, 0.1)))
    store.recompile(0)
    mujoco.mj_forward(store.get_model(0), store.get_data(0))
    pos, quat = store.get_body_pose(0, 'env_0/Obstacles/crate')
    assert pos == pytest.approx((4.0, 5.0, 0.1))
    assert quat == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_model_edits_count_static_moves_but_not_free_body_moves(store: SceneStore):
    store.add_body(0, _box('crate', (1.0, 0.0, 0.1)))
    store.add_body(0, _box('mover', (0.0, 0.0, 1.0), free=True))
    store.recompile(0)
    assert store.model_edits(0) == 0
    store.set_body_pose(0, 'mover', (4.0, 5.0, 0.1), (1.0, 0.0, 0.0, 0.0))
    assert store.model_edits(0) == 0
    store.set_body_pose(0, 'crate', (4.0, 5.0, 0.1), (1.0, 0.0, 0.0, 0.0))
    assert store.model_edits(0) == 1
    assert store.model_edits(7) == 0


def test_recompile_carries_named_free_joint_state(store: SceneStore):
    store.add_body(0, _box('mover', (0.0, 0.0, 1.0), free=True))
    store.recompile(0)
    store.set_body_pose(0, 'mover', (4.0, 5.0, 0.1), (1.0, 0.0, 0.0, 0.0))
    store.step(5)
    mujoco.mj_forward(store.get_model(0), store.get_data(0))
    before = store.get_body_pose(0, 'mover')[0]
    store.add_body(0, _box('late', (9.0, 9.0, 0.1)))
    store.recompile(0)
    assert store.get_body_pose(0, 'mover')[0] == pytest.approx(before, abs=1e-6)


def test_step_advances_one_clock_for_every_env(store: SceneStore):
    store.step(10)
    store.ensure_env(1)
    store.step(15)
    assert store.time == pytest.approx(25 * PHYSICS_DT)
    assert store.get_clock_time(0) == store.get_clock_time(1) == store.time
    assert store.get_data(1).time == pytest.approx(store.time)


def test_mocap_pool_moves_and_recycles_without_recompiling(store: SceneStore):
    handles = store.alloc_mocap(0, 3)
    model = store.get_model(0)
    store.set_mocap_pose(0, handles[1], (2.0, 3.0, 0.85), (1.0, 0.0, 0.0, 0.0))
    store.step(1)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'env0_mocap_1')
    assert tuple(store.get_data(0).xpos[body]) == pytest.approx((2.0, 3.0, 0.85))
    store.free_mocap(0, handles[1])
    store.step(1)
    assert store.get_data(0).xpos[body][2] == pytest.approx(-1000.0)
    assert store.get_model(0) is model


def test_mocap_capsule_blocks_a_ray_at_lidar_height(store: SceneStore):
    handle = store.alloc_mocap(0, 1)[0]
    store.set_mocap_pose(0, handle, (3.0, 0.0, 0.85), (1.0, 0.0, 0.0, 0.0))
    store.step(1)
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(store.get_model(0), store.get_data(0), np.array([0.0, 0.0, 0.3]), np.array([1.0, 0.0, 0.0]), None, 1, -1, geomid)
    assert dist == pytest.approx(2.75, abs=1e-3)


def _flat_box(spec: mujoco.MjSpec) -> mujoco.MjsBody:
    body = spec.worldbody.add_body(name='flat')
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.0])
    return body


def test_uncompilable_additions_are_dropped_and_the_env_keeps_working(store: SceneStore):
    store.add_body(0, _box('kept', (1.0, 0.0, 0.1)))
    store.recompile(0)

    cache = store.asset_cache(0)

    def tinted(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        material = spec.add_material(name='tint')
        cache.add(('rgba', 'tint'), material)
        return _box('tinted', (2.0, 0.0, 0.1))(spec)

    store.add_body(0, tinted)
    store.add_body(0, _flat_box)
    with pytest.raises(RejectedBodies):
        store.recompile(0)

    assert store.find(0, 'kept')
    assert not store.find(0, 'tinted')
    assert not store.find(0, 'flat')
    assert cache.get(('rgba', 'tint')) is None
    assert store.get_model(0).nmat == 0

    store.add_body(0, _box('later', (3.0, 0.0, 0.1)))
    store.recompile(0)
    assert store.get_model(0).nbody == 3


def test_discarding_pending_bodies_also_drops_what_references_them(store: SceneStore):
    def robot(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name='robot')
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
        body.add_joint(name='wheel', type=mujoco.mjtJoint.mjJNT_HINGE)
        body.add_site(name='imu')
        actuator = spec.add_actuator()
        actuator.name = 'drive'
        actuator.target = 'wheel'
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        gyro = spec.add_sensor()
        gyro.name = 'gyro'
        gyro.type = mujoco.mjtSensor.mjSENS_GYRO
        gyro.objtype = mujoco.mjtObj.mjOBJ_SITE
        gyro.objname = 'imu'
        return body

    store.add_body(0, robot)
    store.discard_pending(0)
    model = store.get_model(0)
    assert (model.nbody, model.nu, model.nsensor) == (1, 0, 0)
    assert not store.find(0, 'robot')


def _tread(name: str, pos: tuple[float, float, float]):
    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name=name, pos=list(pos))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.05])
        return body

    return build


def _settled_pos(store: SceneStore, name: str) -> tuple[float, float, float]:
    mujoco.mj_forward(store.get_model(0), store.get_data(0))
    return store.get_body_pose(0, name)[0]


def test_moving_box_travels_at_its_velocity_and_carries_a_box_resting_on_it(store: SceneStore):
    store.add_body(0, _tread('tread', (0.0, 0.0, 1.0)), velocity=(0.5, 0.0, 0.2))
    store.add_body(0, _box('load', (0.0, 0.0, 1.15), free=True))
    store.recompile(0)
    store.step(500)
    assert _settled_pos(store, 'tread') == pytest.approx((0.5, 0.0, 1.2), abs=1e-3)
    assert _settled_pos(store, 'load') == pytest.approx((0.5, 0.0, 1.35), abs=0.03)
    assert store.get_model(0).body_mass[mujoco.mj_name2id(store.get_model(0), mujoco.mjtObj.mjOBJ_BODY, 'tread')] == pytest.approx(MOVING_BODY_KG)


def test_moving_box_keeps_its_drive_across_a_teleport_and_a_recompile_until_removed(store: SceneStore):
    store.add_body(0, _tread('tread', (0.0, 0.0, 1.0)), velocity=(0.5, 0.0, 0.0))
    store.add_body(0, _box('crate', (5.0, 0.0, 0.1)))
    store.recompile(0)
    assert store.set_body_velocity(0, 'tread', (0.0, -1.0, 0.0))
    assert not store.set_body_velocity(0, 'crate', (1.0, 0.0, 0.0))
    assert not store.set_body_velocity(0, 'ghost', (1.0, 0.0, 0.0))
    assert store.set_body_pose(0, 'tread', (2.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0))
    store.add_body(0, _box('late', (9.0, 9.0, 0.1)))
    store.recompile(0)
    store.step(250)
    assert _settled_pos(store, 'tread') == pytest.approx((2.0, -0.5, 1.0), abs=1e-3)
    assert store.remove(0, 'tread')
    store.recompile(0)
    assert not store.set_body_velocity(0, 'tread', (0.5, 0.0, 0.0))
    store.step(10)


def test_comb_and_moving_boxes_pass_through_each_other_but_plain_boxes_keep_default_contacts(store: SceneStore):
    store.add_body(0, _box('plate', (0.0, 0.0, 1.2)), comb=True)
    store.add_body(0, _tread('tread', (0.0, 0.0, 1.0)), velocity=(0.5, 0.0, 0.0))
    store.add_body(0, _box('crate', (5.0, 0.0, 0.1)))
    store.recompile(0)
    model = store.get_model(0)

    def contacts(name: str) -> tuple[int, int]:
        geom = model.body_geomadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
        return int(model.geom_contype[geom]), int(model.geom_conaffinity[geom])

    assert contacts('plate') == (1, 0)
    assert contacts('tread') == (1, 0)
    assert contacts('crate') == (1, 1)
