# T3 — Color-conditioned sorting

Instruction: “Sort the blue and green cubes into their matching cubic containers.”

The scene owns two 30 mm colored cubes and two matching cubic containers. Every
episode independently randomizes all four placements and the execution order.
There are no paired or mirrored seeds. The accepted layout is cached and reused
after the controller's second backend reset.

The debug workspace is the same `0.30 x 0.105 m` rectangle used by Tasks 1 and
2, centered at `(0, 0.1175)`. Cube centers are sampled across
`x in [-0.135, 0.135]` and `y in [0.080, 0.155] m`. Container centers are
sampled across `x in [-0.115, 0.115]` and `y in [0.100, 0.135] m`. These bounds
keep every complete footprint inside the rectangle. The two initially unlabeled
cube positions and two initially unlabeled container positions are assigned to
blue and green randomly, avoiding a color-order sampling bias. Each
container has a `70 x 70 x 35 mm` exterior, 5 mm bottom and side walls, a
`60 x 60 mm` clear opening, and 30 mm usable depth.

The two container footprints cannot overlap, and neither can the two cube
footprints. Each cube additionally starts with at least 30 mm of edge clearance
from both containers.
After geometric sampling, the arm closest to each cube is selected independently
and only the complete real two-object trajectory is solved. If either sequence
is unreachable, all four positions and the color order are resampled, up to 64
attempts. There is no preliminary reachability sweep for both arms.

The episode randomly chooses blue-first or green-first. Each cube then executes
a complete Task-1-style sequence before the other begins:

1. choose the arm closest to that cube from the exported rest pose;
2. approach 35 mm above the cube with the gripper compact at `0.20 rad`;
3. open to `0.40 rad` (approximately 32.5 mm of inner clearance) before contact
   and descend to the TCP grasp target 10 mm above the cube center with the
   arm's mirrored grasp orientation;
4. close only after the grasp IK has finished, lift, transport, lower, and
   release into the matching container;
5. return both arms to rest before beginning the other color.

Arm selection is independent for each cube. Therefore, the same arm performs
both sequences when both cubes are closest to it; otherwise the two arms work
one after another. Each cube uses the same reversible hand-relative lock as
Task 1 only from completed finger closure until opening begins. Complete success
requires both released cubes to remain in their matching containers, followed
by both arms at the exported rest pose.
