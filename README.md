# Assignment 7

3D homogeneous transforms and the forward kinematics of the xArm7. The writeup
is `16_384_Assignment_7.pdf` (its LaTeX source is in `16-384 Assignment 7.zip`);
the code is in `ex1/`.

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

`ex1/robot.py` holds the 3D transforms, a general `Robot` class for a serial
chain of revolute joints, and the xArm7's joint table, `XARM7_JOINTS`: one row
`[x, y, z, roll, pitch, yaw]` per joint, placing it relative to the frame
before it. Fill in, in this order:

- (a) `rot_x`, `rot_y`, `rot_z`, `translation`: the elementary 4x4 transforms;
  then `transform(xyz, rpy) = Trans(x, y, z) Rot_z(yaw) Rot_y(pitch) Rot_x(roll)`.
- (b) `Robot.forward_kinematics(thetas)`: every frame, `frames[:, :, i] = H_i^0`
  for `i = 0, ..., n`, with frame 0 the base and the last frame the end
  effector. Each joint is its row's fixed offset, then its rotation about z:
  `H_i^0 = H_{i-1}^0 transform(xyz_i, rpy_i) Rot_z(q_i)`.

## Checking your work

```bash
python local_autograder.py            # every test
python local_autograder.py -p ex1     # just ex1
python local_autograder.py -v         # list every failing case, not just the first few
```

This runs your code on the test cases in `ex1/inK.txt` and compares the results
with `ex1/outK.txt`: `in1.txt` checks the part (a) functions one by one,
`in2.txt` checks `forward_kinematics` on the xArm7 and on random chains of one
to seven joints. Each failure says where the first mistake is: for part (a),
which function, and for `transform`, which of its factors (or their order); for
`forward_kinematics`, which step of the chain is the first wrong one, and
usually why (the product order, the rotation before the offset, the wrong row
or angle, or a part (a) function that is itself wrong). Code that crashes, calls
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
the flange, and each frame against the link it belongs to, naming the first one
that is off.

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
