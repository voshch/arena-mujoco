string name
# rect | disk | sphere | dome | distant
string shape
# full emission cone in degrees, 180 => no cone
float32 cone_deg
# one per fixture, empty for dome and distant
geometry_msgs/Point[] positions
geometry_msgs/Vector3 direction
# lumens per fixture, lux for dome and distant
float32 flux
# each channel in 0..1
float32[3] color
bool cast_shadows
# output in 0..1
float32 level
# per fixture, empty => all alive
bool[] alive
# true => the default lighting stays
bool intrinsic
# true => a zone ceiling rig, which adds a room fill
bool rig
# material of owner glowing with the light, empty => none
string glow
# prim that carries the glow material
string owner
