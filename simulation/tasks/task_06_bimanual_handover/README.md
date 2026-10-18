# T6 — Bimanual mid-air handover

Instruction: “Pick up the cube from the first cubic container with the closest
arm, transfer it to the opposite arm in mid-air, and place it in the second
container.”

The scene owns both cubic containers and the cube. Their positions are sampled
independently and reproducibly from the episode seed; there are no paired or
mirrored episodes. The debug workspace is the same `0.30 x 0.105 m` rectangle
used by Tasks 1–3 and 5, centered at `(0, 0.1175)`. Both container centers are
sampled across `x in [-0.115, 0.115]` and `y in [0.100, 0.135] m`, keeping their
complete footprints inside the rectangle. The first container is sampled first,
the cube center is offset independently within `[-0.010, 0.010] m` along its X
and Y interior axes, and the second container is then sampled with at least
30 mm of edge clearance from the first.

After sampling, the planar distance from the cube to each hand TCP in the rest
pose is computed. The closest arm becomes the pickup arm and the opposite arm
becomes the receiver and placement arm. The complete trajectory is generated
in that direction, so episodes can be right-to-left or left-to-right without
paired or mirrored seeds. If the selected trajectory is unreachable, all three
positions are resampled, up to 32 attempts; no preliminary two-arm reachability
sweep is performed. The accepted layout is cached across the controller's
second backend reset. Reusing a seed reproduces the same accepted layout and
handover direction.
Both containers have a `70 x 70 x 35 mm` exterior, 5 mm bottom and side walls,
a `60 x 60 mm` clear opening, and 30 mm usable depth.

The pickup procedure matches Task 2. The compact gripper first moves 35 mm above
the cube, descends into the container to 15 mm above it, opens only to the 0.40
rad cube-grasp aperture, descends to the target 5 mm above the cube center,
closes, and lifts vertically clear of the container. The pickup arm alone then
carries the cube to the handover pose while the receiving arm remains at rest.
Once the cube is stationary, the receiving arm moves with a compact gripper to
a pre-grasp target 50 mm toward its own resting side and 10 mm below the pickup
TCP. It opens only to `0.40 rad` at that target, then moves horizontally by 50
mm onto the cube and closes. This vertical separation lets the receiving hand
grasp the lower half of the cube without placing both arm chains at the same
height. The cube lock transfers from the pickup TCP to the receiving TCP while
preserving the cube's live world pose; only then does the pickup gripper open
and retreat. Every pre-grasp and retreat displacement is mirrored according to
the selected direction.
The receiving arm follows Task 1's delivery pattern: move above the second
container, lower to its delivery target, open, retreat outward clear of the
container, and return to rest. Complete success requires the initial pickup,
mid-air transfer, and stable containment in the second container.
