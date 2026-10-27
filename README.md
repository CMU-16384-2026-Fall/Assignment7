# Assignment 7

Denavit-Hartenberg parameters and the forward kinematics of the xArm7 in 3D.
The writeup is built from `16-384 Assignment 7/assignment7.tex`; the code is in
`ex1/`.

## Setup

The graded code needs only numpy (and matplotlib for `ex1/sample_path.py`).
The simulator demo needs the course's robot library. Install it from the copy
in this handout; it may be newer than the one you installed for an earlier
assignment:

```bash
cd 16384-robot-lib
conda env update -f environment.yml    # or `conda env create -f environment.yml` on a new machine
conda activate 16384
cd ..
```

## What to write

`ex1/robot.py` holds a general `Robot` class for a serial chain of revolute
joints described by its DH table, and the xArm7's table, `XARM7_DH`. Fill in,
in this order:

- (a) `Robot.dh_transform(a, alpha, d, theta)`: one link's 4x4 transform,
  `H_i^{i-1} = Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)`.
- (b) `Robot.forward_kinematics(thetas)`: every frame, `frames[:, :, i] = H_i^0`
  for `i = 0, ..., n`, with frame 0 the base and the last frame the end
  effector.

## Checking your work

```bash
python local_autograder.py            # every test
python local_autograder.py -p ex1     # just ex1
python local_autograder.py -v         # list every failing case, not just the first few
```

This runs your code on the test cases in `ex1/inK.txt` and compares the results
with `ex1/outK.txt`: `in1.txt` checks `dh_transform` on its own, `in2.txt`
checks `forward_kinematics` on the xArm7 and on random chains of one to seven
joints. Each failure says where in the chain the first mistake is: for
`dh_transform`, which of its four factors (or their order); for
`forward_kinematics`, which step of the chain is the first wrong one, and
usually why (the product order, a missing offset, the wrong row or angle, or a
`dh_transform` result that is itself wrong). Code that crashes, calls
`sys.exit()` or loops forever fails the case with a message saying so, rather
than stopping the check.

Gradescope runs the same tests on randomly perturbed versions of these cases,
with the answers computed fresh each time. Code that is correct in general
scores the same on both; code that only works for these exact numbers does
not.

```bash
cd ex1 && python sample_path.py
```

plots your xArm7's flange path through a sample run, the arm writing RKD from
the Assignment 5 lab, next to the correct one.

## Seeing it on the robot

```bash
python local_autograder.py -p vis1
```

opens the simulated xArm7 in your browser, with a slider per joint. Your
`forward_kinematics` draws all eight of its frames over the arm, and the status
line checks them against the simulator's own model: your end effector against
the flange, and each frame's z axis against the next joint's axis, which is
where the DH convention puts it.

A matplotlib window opens beside it with the same frames as a stick figure in
3D, in the robot's base frame: a line from each frame's origin to the next, the
end effector's axes, and a green cross where the simulator's flange really is.
It follows the sliders, and you can rotate it with the mouse. Closing it
doesn't stop the demo. Press ctrl-c in the terminal to stop. The demo
works before you have written anything: the arm still moves, and the terminal
tells you your FK isn't ready yet.

## Submitting

```bash
python create_submission.py
```

This writes `handin.zip`. Upload it to Gradescope.
