# T1 — Basic object placement

Instruction: “Pick up the cube and place it inside the target cubic container.”

The task scene owns one cube and one cubic container. At every reset, the episode
seed independently and reproducibly samples the container first and the cube
second. There are no paired or reflected seeds. Both centers are sampled
uniformly across every position where the object's complete footprint remains
inside the displayed rectangle. A candidate is rejected if the two footprints
do not have 30 mm of clearance.
Both arms begin with a one-second hold at the exported zero-reference pose with
open grippers. The controller
compares the cube's planar distance to both resting grippers, then the nearest arm
performs the task while the other arm remains at rest. Only that selected arm's
actual pose-aware trajectory is solved. If that real trajectory is unreachable,
the controller samples another complete layout and tries again; it does not run
the former preliminary seven-pose reachability sweep for both arms. The accepted
layout is cached and reused after the solver's second backend reset.

The container exterior is `0.070 x 0.070 x 0.035 m`. Its bottom and four walls
are all 5 mm thick, leaving a clear `0.060 x 0.060 m` opening and 30 mm of usable
depth above the bottom panel.

For grasping, the working gripper opens to its `pi/2` joint limit before the
arm moves its live fingertip-midpoint TCP to `(cube_x, cube_y, cube_z + 0.010)`,
10 mm above the sampled cube center. The controller then holds the completed IK
action for 0.5 seconds and closes the gripper to `0.32 rad`. Closing has its own
IK solve at the same Cartesian target, compensating for movement of the live TCP
as the moving finger rotates. The lift starts only after closure. With the elbow limited
to 90 degrees, the lift moves symmetrically inward to `x = +/-0.040 m` while
rising, keeping the closed-gripper TCP within the reachable workspace. Once
closure completes,
the simulation stores the cube pose relative to the working hand TCP and locks
that transform while the working gripper command is fully closed. Every lift,
transport, and lowering waypoint explicitly keeps that command closed. The lock
is removed as soon as the gripper command starts opening, returning the cube to
ordinary MuJoCo contact and gravity. The container delivery and release TCP is
at `z = 0.135 m`.

Sampling geometry, in metres:

- displayed sampling workspace: center `(0, 0.1175)`, X length `0.30`,
  Y width `0.105`;
- cube-center range: `x in [-0.135, 0.135]` and
  `y in [0.080, 0.155]`, with `z = 0.035`;
- container-center range: `x in [-0.115, 0.115]` and
  `y in [0.100, 0.135]`, with `z = 0`;
- cube-to-container clearance: at least `0.030 m` between the initial
  footprints.

Randomization is uniform across these footprint-aware ranges. The final
distribution is necessarily conditioned on the selected arm being able to
execute the actual complete trajectory; unreachable candidates are resampled.

Complete success requires the released cube to remain inside the sampled
container. Object motion remains governed by MuJoCo contact physics.

With `bimanual-task-demo 1 --debug`, the viewer displays a yellow object frame
at the cube center and a white frame at the container-release coordinate. The
cube is initialized with the identity quaternion, so its red X and green Y axes
are parallel to the floor and its blue Z axis is vertical. Because this frame is
attached to the free cube body, it follows any subsequent physical translation
or rotation of the cube.

A moving frame with a magenta origin is the hand tool-center point used by IK.
Its origin is recomputed from the live finger geometry and always lies halfway
between the fixed and moving distal finger tips, including while the gripper is
opening or closing. All frames use red X, green Y, and blue Z axes.

The preferred left-hand cube-grasp orientation is the MuJoCo quaternion
`(0.707107, 0, 0, -0.707107)`. The right hand uses its exact reflection,
`(0.707107, 0, 0, 0.707107)`, so both independently sampled sides request
physically symmetric wrist poses.

Each arm has five positioning joints, so it cannot realize an arbitrary
six-degree-of-freedom pose. For waypoints that do not provide an explicit
quaternion, the controller selects a reachable orientation near the upright
identity frame, records that complete pose as the target, and then minimizes
both its translational and rotational errors.

Task 1 uses the cube-grasp mapping above as its orientation preference. With
only five positioning joints, the arm cannot generally satisfy position plus
all three requested orientation components simultaneously. The controller
therefore selects the reachable pose nearest this mapping while retaining the
Cartesian position tolerance.
