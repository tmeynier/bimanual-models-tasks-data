# T2 — Object retrieval

Instruction: “Pick up the cube from inside the cubic container and place it
outside the container.”

Task 2 is the inverse of Task 1. At every reset, the episode seed reproducibly
samples a source container, a cube resting inside it, and an outside destination
on the workspace. Seeds are independent and reproducible; there are no paired
or reflected episodes. Sampling first enforces the geometric object constraints.
Both arms start at the exported rest pose
and hold it for one second with open grippers; the arm closest to the contained
cube then performs the complete task while the other arm stays at rest. Only the
selected arm's actual pose-aware trajectory is solved. If that real trajectory
is unreachable, the complete layout is resampled and tried again, up to 32
times. The controller does not perform a preliminary multi-pose reachability
sweep for either arm, and it caches the accepted layout across its second
backend reset.

Sampling geometry, in metres:

- workspace rectangle: center `(0, 0.1175)`, X length `0.30`, Y width `0.105`;
- source-container center: uniformly within `x in [-0.115, 0.115]` and
  `y in [0.100, 0.135]`, with `z = 0`;
- cube-center offset inside that container: independently uniform within
  `[-0.010, 0.010]` on X and Y, with cube-center `z = 0.045`;
- outside destination: uniformly within `x in [-0.135, 0.135]`
  and `y in [0.080, 0.155]`, with intended cube-center `z = 0.035`;
- destination-to-container clearance: at least `0.030` between footprints.

The accepted distribution is conditioned on the selected arm being able to
execute the complete real trajectory. The container exterior
is `0.070 x 0.070 x 0.035 m`. Its bottom and four walls are all 5 mm thick,
leaving a clear `0.060 x 0.060 m` opening and `0.030 m` usable depth.

For grasping, the working gripper first becomes compact (`0.20 rad`) while the
arm is still at rest. The first arm movement is one IK directly to
`(cube_x, cube_y, cube_z + 0.035)`, above both the cube and container. The compact
hand then descends inside the container to
`(cube_x, cube_y, cube_z + 0.015)`, at the cube's top face. Only there does the
gripper open to `0.40 rad`, which provides approximately 32.5 mm of clearance
between the inner fingertip collision surfaces for the 30 mm cube. It then
descends while open to the inside-container grasp target
`(cube_x, cube_y, cube_z + 0.005)` and closes after completing that IK motion.
After closure, the arm lifts vertically at the same `(cube_x, cube_y)` to a TCP
height of `0.125 m`, clearing the container with an additional 20 mm of vertical
margin. One final Cartesian IK moves the
closed gripper directly to the requested workspace placement target at
`(destination_x, destination_y, 0.075)`, where the fingers open.
This includes 30 mm of release clearance above the nominal cube-on-floor TCP
height, so the cube drops onto the workspace instead of being driven into its
collision surface.

The compact entry command remains above the fully closed value (`0.15 rad`), so
it cannot activate the artificial grasp. The cube pose is locked relative to
the working hand TCP only when the final grasp reaches the fully closed command.
Every lift, transport, and lowering waypoint explicitly retains that command.
The lock is removed as soon as the fingers begin opening, returning the cube to
ordinary MuJoCo contact and gravity.

Complete success requires the released cube to be outside the sampled source
container, resting on the floor, and fully inside the rectangular workspace. The
sampled destination is used to generate a reproducible transport trajectory;
the paper's success condition itself only requires placement outside the
container. After release, both arms interpolate directly back to the exported
rest joint pose with open grippers; this final return is joint-space motion, not
another Cartesian IK.

With `bimanual-task-demo 2 --debug`, the viewer shows a yellow frame at the
cube center, a white frame at the workspace-release TCP target, and a moving
frame with a magenta origin at the working hand's fingertip-midpoint TCP. A cyan
rectangle on the floor shows the exact `0.30 x 0.105 m` workspace boundary
centered at `(0, 0.1175)`. All frames use red X, green Y, and blue Z axes.

The preferred left-hand cube-grasp orientation is the MuJoCo quaternion
`(0.707107, 0, 0, -0.707107)`. The right hand uses its exact reflection,
`(0.707107, 0, 0, 0.707107)`. Because each arm has five positioning joints, the
controller selects the reachable pose nearest this orientation mapping while
retaining its Cartesian position tolerance, exactly as in Task 1.
