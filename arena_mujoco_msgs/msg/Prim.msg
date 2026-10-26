string name
geometry_msgs/Pose pose
Scale scale
# mesh file in any format trimesh reads, empty => box sized from scale
string mesh_path
# look of a box, empty name and path => the plain default
Material material
# further visuals of a mesh prim, each also a collider
PrimVisual[] visuals
# world-frame velocity of a gravity-free moving box, zero => static
geometry_msgs/Vector3 velocity
# static box moving boxes pass through, an escalator comb plate
bool comb
