# Task-scoped IK simulations

The paper defines seven benchmark tasks. Each task owns one directory containing:

- `README.md`: task instruction, staged subgoals, and success criteria.
- `ik.py`: the task metadata and its task-specific IK controller entry point.
- `scene.xml`: task-local MuJoCo objects and fixtures for implemented tasks.
- `__init__.py`: the public task package boundary.

Shared numerical IK and waypoint execution live in `simulation/tasks/ik.py`, but
task objects, waypoints, state machines, and success predicates remain inside the
owning task directory. An implementation must not silently depend on assets from
a different task.

Task reset never performs a preliminary IK reachability sweep or solves trial
trajectories for both arms. Randomized scenes are sampled once using geometric
constraints, the closest resting arm is selected from the cube position, and
only the actual task trajectory is solved. Tasks whose instructions are
intrinsically bimanual or assign an arm a specific role retain those roles.
Every implemented trajectory ends with both complete arm chains at the exported
rest joint pose and both grippers open. This final motion uses the known rest
joint command directly rather than running another Cartesian IK solve.

Every cube pickup uses the same final grasp convention. The working gripper
remains compact through its approach, opens at a separate pre-grasp pose before
contact, and then descends while open to the final TCP target 10 mm above the
cube center. Fixed floor-cube tasks open 35 mm above the center. Task 2 first
moves directly to that 35 mm pose above the container, descends compact to 15 mm
above the contained cube, and opens there to a 0.40 rad command sized for the
30 mm cube. Every final descent uses Task 1's
mirrored left/right grasp-orientation preference, and finger closure begins only
after that IK waypoint.

Cartesian waypoints are full pose targets. The end-effector origin is the live
fingertip tool-center point, recomputed midway between the two distal finger
tips at every gripper opening. Its position Jacobian is the average of both
fingertip Jacobians. The damped least-squares solver uses this translational
Jacobian together with the hand rotational Jacobian. Since each arm has five
positioning joints, unspecified orientations are resolved to reachable poses
near the upright frame; tasks may instead provide an explicit MuJoCo quaternion
when the requested pose lies on the arm's five-degree-of-freedom pose manifold.

T1–T7 have scenes and numerically solved Cartesian waypoint controllers. In
debug mode, Task 4 inherits a non-colliding cyan rectangle showing the common
workspace boundary centered at `(0, 0.175)`: 50 cm along the left-right X axis
and 30 cm along the front-back Y axis. Tasks 1 through 3 and Tasks 5 through 7 resize that marker to their
30 x 10.5 cm full sampling workspace centered at `(0, 0.1175)`. Every cubic container
uses 5 mm panels, a clear `0.060 x 0.060 m` opening, and 30 mm of usable depth.
Tasks 1 through 3, Task 5, Task 6, and Task 7 use a reversible
hand-relative cube lock only between completed finger closure and the beginning
of release; each cube otherwise remains a MuJoCo free body. Other tasks use
ordinary contact throughout.
The removed legacy warehouse controller is not reused because it described a
different scene and mixed several task definitions in one large state machine.
