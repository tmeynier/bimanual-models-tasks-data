# T4 — Number-conditioned drawer opening

Instruction: “Open the drawer with the requested number.”

The task owns the 3 x 3 stack of nine shallow drawers from the source robot assembly's
MuJoCo description at `simulation/backends/mujoco/assets/robot.xml`. Task 4 keeps
self-contained copies of that repository's original rack and drawer STL exports
under `assets/`; it does not depend on another repository's absolute filesystem
path or restore the removed warehouse/CNC model.

The original drawer dimensions, body heights, three-row spacing, three-column
spacing, handles, rack shelves, and dividers are retained. The nominal rigid
assembly has its closed drawer-handle center plane at `y = -0.035 m`. At every
reset, seeded randomization translates the complete rack and all nine drawers
independently in world X and Y by values uniformly sampled from `[-0.02, 0.02] m`
and rotates the complete assembly around its vertical Z axis uniformly within
`[-5, 5]` degrees. Body positions and orientations are transformed about the
physical center of the rack, preserving all relative assembly geometry.

Each drawer slide is also initialized independently within `[-0.01, 0.01] m`.
The lower mechanical slide limit is therefore extended from zero to `-0.01 m`;
the upper limit remains `0.08 m`. Negative initialization represents a drawer
10 mm farther closed than its nominal position, while positive initialization
represents a drawer up to 10 mm initially open. The same seed reproduces the
same accepted rack pose and nine drawer offsets.
Numbering follows the source assembly:

Task 4 uses the same strict numerical IK behavior as every other task. An
unreachable randomized pose is resampled while keeping the requested drawer
fixed, using at most 32 actual-trajectory attempts. If none succeeds,
`IKConvergenceError` is raised; debug mode does not substitute an approximate
trajectory or suppress the failure.

```text
3  2  1
6  5  4
9  8  7
```

`seed % 9 + 1` selects the requested drawer. Task 4 starts and finishes in a
task-specific mirrored rest pose with the arm-to-wrist joints turned moderately
inward (`+pi/8` left and `-pi/8` right).

Every drawer, including drawer 8 (seed 7, bottom center), uses the exact live
center of its handle after rack and slide randomization as the Cartesian IK
target. No lateral or forward target offset is applied.

A visible target frame is placed on the selected handle. The working gripper
first compacts to `0.20 rad`, then the chosen hand matches the handle position
while preferring a vertical scissor orientation rotated by the rack's sampled
Z yaw. The final orientation is the
closest one available within the robot's joint limits. Closing is itself a Cartesian IK waypoint:
the arm compensates for motion of the moving fingertip so the midpoint-based TCP
remains on the handle after the fingers close. It then activates a temporary MuJoCo connect
constraint between the fingertip TCP and the selected drawer handle. With the
fingers held closed, the wrist-to-hand actuator is locked at its completed
grasp value while the remaining arm joints solve and execute the pull. The arm
pulls the constrained handle along the rotated rack-local `+Y` direction,
forward from the rack toward the robot. Once fully open, the
fingers open only to `0.40 rad` and the constraint is disabled. The released
hand then retreats an additional 45 mm backward toward the robot, holds clear of
the drawer, and only then returns to the Task 4 rest pose.

The task writes each drawer slide coordinate only during seeded scene
initialization. Subsequent drawer movement comes from the arm pulling the
temporary handle constraint through MuJoCo dynamics. Handle
collision is disabled because the temporary constraint supplies the coupling;
this prevents contact from opening a drawer before attachment or pushing it
closed during the return motion. Complete success requires attach, pull, detach,
the requested drawer at least 35 mm farther open than its sampled initial
position, and every other drawer remaining within 15 mm of its own sampled
initial position.
