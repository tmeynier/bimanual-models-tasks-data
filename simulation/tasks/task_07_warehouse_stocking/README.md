# T7 — Warehouse stocking

Instruction: “Open the requested drawer, retrieve the cube of the requested
color, place the cube inside, and close the drawer.”

The task owns a copy of the numbered 3 x 3 drawer stack used by Task 4 and two
30 mm cubes, one blue and one green. The requested drawer and requested color
are derived reproducibly from the episode seed. The rack receives the same
Task 4 randomization: X/Y translation in `[-20, 20] mm`, yaw in `[-5, 5] deg`,
and an independent initial offset in `[-10, 10] mm` for every drawer. Both cube
centers are sampled independently over the `300 x 105 mm` cube workspace with
30 mm of edge clearance between their footprints. Seeds are not mirrored.

The visible rack and drawers are the original exported STL assets used by Task
4, at a `0.001` millimetre-to-metre scale. Primitive collision boxes follow the
exported shell: 4 mm drawer bottoms, side walls, and back walls; 10 mm front
walls; and the 2 mm-deep horizontal handle bar.

The trajectory deliberately opens the drawer before picking up the cube:

1. Select the requested drawer and move the closest hand to its live handle.
2. Close the gripper vertically around the handle, activate the temporary
   handle constraint, and pull along the rack-local opening direction. The
   lower row opens 80 mm so its interior is reachable from above; the upper and
   middle rows open 60 mm.
3. Open the gripper, detach the handle, retreat, and return both arms to rest.
4. Select the arm closest to the requested-color cube. Approach and grasp it
   using Task 1's fingertip TCP, mirrored grasp orientation, 10 mm-above-center
   target, and reversible hand-relative cube lock.
5. Lift the cube and move over the exposed drawer opening. The final placement
   IK remains entirely above the drawer: the cube center is 20 mm above the rim
   (leaving its lower face 5 mm clear) and 25 mm from the lateral center toward
   the working arm, while remaining within the clear opening. Open the fingers
   there so gravity drops the cube into the drawer, then retreat upward.
6. Return both arms to rest, grasp the open handle, physically push the drawer
   back to its initial opening, release it, retreat, and finish at rest.

Neither drawer opening nor closing is produced by writing the slide coordinate.
Both motions are caused by the same temporary hand-to-handle equality constraint
used by Task 4. The cube lock exists only between completed grasp closure and
completed release. All other drawer and cube motion remains under MuJoCo
physics. Cube-to-drawer contacts remain enabled throughout the trajectory. A
high slide-friction detent keeps the opened drawer from drifting during the cube
phase and is removed before the closing pull. The detent does not write a drawer
joint position.

Complete success requires the requested drawer to have opened, the requested
cube to have been grasped and released inside it, the requested drawer to finish
closed, and every other drawer to remain at its initial offset.
