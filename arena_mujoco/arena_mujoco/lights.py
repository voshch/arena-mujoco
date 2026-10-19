"""World lights of an env, reduced per view to the lights MuJoCo's renderer shades."""

from __future__ import annotations

import math
from collections.abc import Sequence

import attrs
import mujoco

Vec = tuple[float, float, float]

SLOTS = 7
REFERENCE_LUX = 400.0
REFLECTANCE = 0.4
FILL_GAIN = 2.0
RIG_GAIN = 1.3
DOME_GAIN = 0.6
DEFAULT_AMBIENT: Vec = (0.5, 0.5, 0.5)
DEFAULT_DIFFUSE: Vec = (0.4, 0.4, 0.4)
DEFAULT_SPECULAR: Vec = (0.1, 0.1, 0.1)

_DARK: Vec = (0.0, 0.0, 0.0)
_DOWN: Vec = (0.0, 0.0, -1.0)
GLOW_RESPONSE = 0.65

GLOW_SEPARATOR = ':glow_'
_SOFTENING_M2 = 1.0
_DOME_AMBIENT_SHARE = 0.5


@attrs.frozen
class Lamp:
    """One light as the renderer shades it, colors in units of REFERENCE_LUX."""

    position: Vec
    direction: Vec
    diffuse: Vec
    ambient: Vec = _DARK
    specular: Vec = _DARK
    softening: float = _SOFTENING_M2
    cutoff: float = 180.0
    directional: bool = False
    castshadow: bool = False
    flat: bool = False


DEFAULT_SUN = Lamp(position=(0.0, 0.0, 10.0), direction=_DOWN, diffuse=(0.7, 0.7, 0.7), specular=(0.3, 0.3, 0.3), directional=True)


@attrs.frozen
class Headlight:
    """Camera-bound light of a view, its ambient term the share of the domes that lights every surface alike."""

    ambient: Vec = DEFAULT_AMBIENT
    diffuse: Vec = DEFAULT_DIFFUSE
    specular: Vec = DEFAULT_SPECULAR


@attrs.define
class Light:
    """A declared light: its fixtures and the output they give now."""

    shape: str
    positions: list[Vec]
    direction: Vec
    flux: float
    color: Vec
    cone_deg: float = 180.0
    cast_shadows: bool = False
    intrinsic: bool = False
    level: float = 1.0
    alive: list[bool] = attrs.Factory(list)
    rig: bool = False
    glow: str = ''
    yaw: float = 0.0

    @property
    def ambient(self) -> bool:
        return self.shape in ('dome', 'distant')

    @property
    def aim(self) -> Vec:
        """Emission direction, turned by the yaw of the frame a moved light follows."""
        x, y, z = self.direction
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        return _unit((cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y, z))

    def output(self) -> Vec:
        """Color of one live fixture times its flux at the current level, in units of REFERENCE_LUX."""
        gain = max(self.flux, 0.0) * max(self.level, 0.0) / REFERENCE_LUX
        if self.shape == 'dome':
            gain *= DOME_GAIN
        elif self.rig:
            gain *= RIG_GAIN
        return (self.color[0] * gain, self.color[1] * gain, self.color[2] * gain)

    def live(self) -> list[Vec]:
        return [position for index, position in enumerate(self.positions) if index >= len(self.alive) or self.alive[index]]

    def glow_level(self) -> float:
        """Emission of the glow material, zero while the light is off or its fixture dead."""
        return max(self.level, 0.0) ** GLOW_RESPONSE if self.live() else 0.0

    def fill(self, eye: Vec) -> Vec:
        """Light the room of a rig reflects off its surfaces, for an eye above its floor plan, in units of REFERENCE_LUX."""
        if not self.rig or not self.positions:
            return _DARK
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        pitches = [math.dist(a[:2], b[:2]) for i, a in enumerate(self.positions) for b in self.positions[i + 1 :]]
        pitch = min((d for d in pitches if d > 0.0), default=2.0)
        if not (min(xs) - pitch / 2.0 <= eye[0] <= max(xs) + pitch / 2.0 and min(ys) - pitch / 2.0 <= eye[1] <= max(ys) + pitch / 2.0):
            return _DARK
        length = max(xs) - min(xs) + pitch
        width = max(ys) - min(ys) + pitch
        height = self.positions[0][2]
        surface = 2.0 * (length * width + length * height + width * height)
        flux = max(self.flux, 0.0) * max(self.level, 0.0) * len(self.live())
        return _scaled(self.color, FILL_GAIN * REFLECTANCE * flux / (surface * (1.0 - REFLECTANCE)) / REFERENCE_LUX)

    def bounce(self, eye: Vec) -> Lamp | None:
        """The fill as an unattenuated lamp at half height over the rig's center, as the Gazebo backend places it, None outside its floor plan."""
        fill = self.fill(eye)
        if not any(fill):
            return None
        center = (sum(p[0] for p in self.positions) / len(self.positions), sum(p[1] for p in self.positions) / len(self.positions), self.positions[0][2] / 2.0)
        return Lamp(position=center, direction=_DOWN, diffuse=fill, flat=True)


def _unit(vector: Vec) -> Vec:
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    return _DOWN if norm == 0.0 else (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _scaled(vector: Vec, factor: float) -> Vec:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


@attrs.frozen
class _Node:
    """Sources merged into one lamp, split in two for a closer look."""

    center: Vec
    radius: float
    spread: float
    weight: float
    glow: Vec
    lamp: Lamp
    children: tuple[_Node, ...] = ()


def _leaf(light: Light, position: Vec) -> _Node:
    """One fixture: a bulb shines all around, a spot into its cone, any other fixture into the half space it faces."""
    output = light.output()
    if light.shape == 'sphere':
        glow = _scaled(output, 1.0 / (2.0 * math.tau))
        lamp = Lamp(position=position, direction=_DOWN, diffuse=glow)
    elif light.cone_deg < 180.0:
        glow = _scaled(output, 1.0 / math.tau)
        solid_angle = math.tau * (1.0 - math.cos(math.radians(light.cone_deg) / 2.0))
        lamp = Lamp(position=position, direction=light.aim, diffuse=_scaled(output, 1.0 / solid_angle), cutoff=light.cone_deg / 2.0, castshadow=light.cast_shadows)
    else:
        glow = _scaled(output, 1.0 / math.tau)
        lamp = Lamp(position=position, direction=light.aim, diffuse=glow, cutoff=90.0)
    return _Node(center=position, radius=0.0, spread=0.0, weight=sum(glow), glow=glow, lamp=lamp)


def _merge(first: _Node, second: _Node) -> _Node:
    weight = first.weight + second.weight
    share = second.weight / weight
    cx, cy, cz = (a + (b - a) * share for a, b in zip(first.center, second.center, strict=True))
    center = (cx, cy, cz)
    to_first, to_second = math.dist(center, first.center), math.dist(center, second.center)
    spread = (first.weight * (first.spread + to_first**2) + second.weight * (second.spread + to_second**2)) / weight
    gr, gg, gb = (a + b for a, b in zip(first.glow, second.glow, strict=True))
    glow = (gr, gg, gb)
    return _Node(
        center=center,
        radius=max(to_first + first.radius, to_second + second.radius),
        spread=spread,
        weight=weight,
        glow=glow,
        lamp=Lamp(position=center, direction=_DOWN, diffuse=glow, softening=_SOFTENING_M2 + spread, cutoff=max(first.lamp.cutoff, second.lamp.cutoff, 90.0)),
        children=(first, second),
    )


def _grow(leaves: list[_Node]) -> _Node:
    """Binary tree over leaves, halved at the median of their widest axis."""
    if len(leaves) == 1:
        return leaves[0]
    axis = max(range(3), key=lambda i: max(leaf.center[i] for leaf in leaves) - min(leaf.center[i] for leaf in leaves))
    leaves = sorted(leaves, key=lambda leaf: leaf.center[axis])
    half = len(leaves) // 2
    return _merge(_grow(leaves[:half]), _grow(leaves[half:]))


def _cut(root: _Node, eye: Vec, budget: int) -> list[_Node]:
    """At most budget nodes covering root, the ones that look largest from eye split first."""
    picked = [root]
    while len(picked) < budget:
        splittable = [node for node in picked if node.children]
        if not splittable:
            break
        largest = max(splittable, key=lambda node: node.radius / (1.0 + max(math.dist(eye, node.center) - node.radius, 0.0)))
        picked.remove(largest)
        picked.extend(largest.children)
    return picked


class Lighting:
    """Lights of one env and the view of them a camera at a given eye gets."""

    def __init__(self) -> None:
        self._lights: dict[str, Light] = {}
        self._root: _Node | None = None
        self._grown = False
        self._touched = False
        self._dressed: tuple[mujoco.MjModel, Headlight, list[Lamp]] | None = None
        self._glowed: tuple[mujoco.MjModel, tuple] | None = None
        self._glow_bases: dict[int, tuple[float, float, float, float]] = {}

    def _changed(self) -> None:
        self._grown = False
        self._touched = True

    def spawn(self, name: str, light: Light) -> None:
        self._lights[name] = light
        self._changed()

    def remove(self, name: str) -> bool:
        if self._lights.pop(name, None) is None:
            return False
        self._changed()
        return True

    def clear(self) -> None:
        if self._lights:
            self._lights.clear()
            self._changed()

    def set_output(self, name: str, level: float, alive: Sequence[bool]) -> bool:
        light = self._lights.get(name)
        if light is None:
            return False
        light.level = level
        light.alive = list(alive)
        self._changed()
        return True

    def move(self, name: str, position: Vec, yaw: float) -> bool:
        light = self._lights.get(name)
        if light is None or light.ambient:
            return False
        light.positions = [position]
        light.yaw = yaw
        self._changed()
        return True

    @property
    def declared(self) -> bool:
        """Whether a light replaces the default lighting, lights that objects carry do not."""
        return any(not light.intrinsic for light in self._lights.values())

    def headlight(self, eye: Vec) -> Headlight:
        """Default headlight, or once lights are declared only an ambient term: half of each dome."""
        if not self.declared:
            return Headlight()
        ambient = [0.0, 0.0, 0.0]
        for light in self._lights.values():
            if light.shape != 'dome':
                continue
            for channel, value in enumerate(_scaled(light.output(), _DOME_AMBIENT_SHARE)):
                ambient[channel] += value
        return Headlight(ambient=(ambient[0], ambient[1], ambient[2]), diffuse=_DARK, specular=_DARK)

    def _distant(self) -> list[Lamp]:
        """Lamps without a place: the default sun of an env that declares no light, else its suns and the downward half of its domes."""
        if not self.declared:
            return [DEFAULT_SUN]
        lamps: list[Lamp] = []
        for light in self._lights.values():
            output = light.output()
            if not any(output):
                continue
            if light.shape == 'distant':
                lamps.append(Lamp(position=DEFAULT_SUN.position, direction=light.aim, diffuse=output, directional=True, castshadow=light.cast_shadows))
            elif light.shape == 'dome':
                lamps.append(Lamp(position=DEFAULT_SUN.position, direction=_DOWN, diffuse=_scaled(output, 1.0 - _DOME_AMBIENT_SHARE), directional=True))
        return lamps[:SLOTS]

    def _tree(self) -> _Node | None:
        if not self._grown:
            leaves = [_leaf(light, position) for light in self._lights.values() if not light.ambient and any(light.output()) for position in light.live()]
            self._root = _grow(leaves) if leaves else None
            self._grown = True
        return self._root

    def lamps(self, eye: Vec) -> list[Lamp]:
        """The SLOTS lamps a camera at eye is shaded with: distant ones first, then the fill of the brightest rig the eye stands in, the rest merged the coarser the smaller they look."""
        lamps = self._distant()
        bounce = max((lamp for light in self._lights.values() if (lamp := light.bounce(eye)) is not None), key=lambda lamp: sum(lamp.diffuse), default=None)
        if bounce is not None and len(lamps) < SLOTS:
            lamps.append(bounce)
        root = self._tree()
        if root is not None and len(lamps) < SLOTS:
            lamps.extend(node.lamp for node in _cut(root, eye, SLOTS - len(lamps)))
        return lamps

    def shade(self, scene: mujoco.MjvScene, eye: Vec) -> None:
        """Replace the lights of a scene freshly updated from the model, untouched while the env never had a light."""
        if not self._touched:
            return
        first = 0
        if scene.nlight and scene.lights[0].headlight:
            head = self.headlight(eye)
            scene.lights[0].ambient[:] = head.ambient
            scene.lights[0].diffuse[:] = head.diffuse
            scene.lights[0].specular[:] = head.specular
            first = 1
        lamps = self.lamps(eye)
        for index, lamp in enumerate(lamps):
            target = scene.lights[first + index]
            target.pos[:] = lamp.position
            target.dir[:] = lamp.direction
            target.type = _kind(lamp)
            target.texid = -1
            target.attenuation[:] = _attenuation(lamp)
            target.cutoff = lamp.cutoff
            target.exponent = 0.0
            target.ambient[:] = lamp.ambient
            target.diffuse[:] = lamp.diffuse
            target.specular[:] = lamp.specular
            target.headlight = 0
            target.castshadow = int(lamp.castshadow)
        scene.nlight = first + len(lamps)

    def dress(self, model: mujoco.MjModel, data: mujoco.MjData, eye: Vec) -> bool:
        """Write the view from eye into the model's own lights and glow materials, for a viewer that builds its scene itself. True when the model changed."""
        if not self._touched:
            return False
        glowed = self.glow(model)
        head = self.headlight(eye)
        lamps = self.lamps(eye)
        if self._dressed is not None and self._dressed[0] is model and self._dressed[1:] == (head, lamps):
            return glowed
        self._dressed = (model, head, lamps)
        model.vis.headlight.ambient[:] = head.ambient
        model.vis.headlight.diffuse[:] = head.diffuse
        model.vis.headlight.specular[:] = head.specular
        model.light_active[:] = 0
        for index, lamp in enumerate(lamps[: model.nlight]):
            model.light_pos[index] = lamp.position
            model.light_dir[index] = lamp.direction
            data.light_xpos[index] = lamp.position
            data.light_xdir[index] = lamp.direction
            model.light_type[index] = _kind(lamp)
            model.light_attenuation[index] = _attenuation(lamp)
            model.light_cutoff[index] = lamp.cutoff
            model.light_exponent[index] = 0.0
            model.light_ambient[index] = lamp.ambient
            model.light_diffuse[index] = lamp.diffuse
            model.light_specular[index] = lamp.specular
            model.light_castshadow[index] = lamp.castshadow
            model.light_active[index] = 1
        return True

    def glow(self, model: mujoco.MjModel) -> bool:
        """Make the glow materials in model emit the color of their light at its level, back to their own color while it is off. True when the model changed."""
        if not self._touched:
            return False
        state = tuple((light.glow, light.glow_level(), light.color) for light in self._lights.values() if light.glow)
        if self._glowed is not None and self._glowed[0] is model and self._glowed[1] == state:
            return False
        if self._glowed is None or self._glowed[0] is not model:
            self._glow_bases = {index: tuple(model.mat_rgba[index]) for index in range(model.nmat) if GLOW_SEPARATOR in model.material(index).name}
        self._glowed = (model, state)
        lit: dict[int, tuple[float, Vec]] = {}
        for prefix, level, color in state:
            for index in self._glow_bases:
                if model.material(index).name.startswith(f'{prefix}/'):
                    lit[index] = (level, color)
        for index, base in self._glow_bases.items():
            level, color = lit.get(index, (0.0, _DARK))
            model.mat_emission[index] = level
            model.mat_rgba[index] = (*color, base[3]) if level > 0.0 else base
        return True


def glow_material(owner: str, glow: str) -> str:
    """Name prefix of the materials of owner's glow material, one per mesh part below it."""
    return f'{owner}{GLOW_SEPARATOR}{glow}'


def _kind(lamp: Lamp) -> int:
    return int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if lamp.directional else mujoco.mjtLightType.mjLIGHT_SPOT)


def _attenuation(lamp: Lamp) -> Vec:
    return (1.0, 0.0, 0.0) if lamp.directional or lamp.flat else (lamp.softening, 0.0, 1.0)


_lightings: dict[int, Lighting] = {}


def lighting(env_id: int) -> Lighting:
    """The lighting of env_id, empty until a light is spawned into it."""
    return _lightings.setdefault(env_id, Lighting())


__all__ = ['DEFAULT_AMBIENT', 'DEFAULT_DIFFUSE', 'DEFAULT_SPECULAR', 'DEFAULT_SUN', 'DOME_GAIN', 'FILL_GAIN', 'GLOW_RESPONSE', 'GLOW_SEPARATOR', 'REFERENCE_LUX', 'REFLECTANCE', 'RIG_GAIN', 'SLOTS', 'Headlight', 'Lamp', 'Light', 'Lighting', 'glow_material', 'lighting']
