# T5 — Precision cube stacking

Instruction: “Place the green cube stably on top of the blue cube.”

The task scene owns the green source cube and blue base cube. The episode seed
independently and reproducibly samples both positions; there are no fixed-layout
catalogues, paired seeds, or mirrored episodes. The debug workspace is the same
`0.30 x 0.105 m` rectangle used by Tasks 1–3, centered at `(0, 0.1175)`.
Both cube centers are sampled uniformly across `x in [-0.135, 0.135]` and
`y in [0.080, 0.155] m`, keeping each complete 30 mm cube footprint inside the
rectangle. The blue base is sampled first, followed by the green source cube,
with at least 30 mm of edge clearance between them.

The arm closest to the green cube performs the complete task. Only that arm's
real stacking trajectory is solved. If it is unreachable, both positions are
resampled, up to 32 attempts; no preliminary reachability sweep is run for both
arms. The accepted layout is cached across the controller's second backend
reset.

From rest, the working hand solves directly to the grasp target 10 mm above the
green cube center and closes while holding that same TCP target. Once the grasp
lock is active, the hand first lifts vertically by exactly 40 mm at the green
cube's X/Y position. It then moves to a high waypoint directly above the blue
cube, 40 mm above the final placement TCP, before descending vertically to the
release target. Consequently, the final approach is always top-down and never a
low lateral sweep through the blue cube.

At the release target, the fingers open while holding the same Cartesian pose.
The 20 mm release clearance keeps the fingers from striking the base cube, and
the green cube drops vertically onto it. Holding the same target while opening
compensates for the fingertip-midpoint shift. At unlock, the green cube is aligned to the blue cube's
live center at the requested release clearance, leveled, and given zero velocity
before falling under MuJoCo physics. This removes residual IK error and remains
correct if contact moved the base cube slightly during approach. The open hand
then lifts vertically away from the stacked cubes before both arms return to
rest. Complete success requires a recorded grasp, horizontal
alignment, and a final 30 mm vertical support offset after settling.
