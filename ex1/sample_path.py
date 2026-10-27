"""Plot the xArm7's end-effector path through a sample run, with your FK.  Provided.

    python sample_path.py

The sample run is the xArm7 writing RKD on a sheet of paper (the Assignment 5
lab, planned by its reference solution): 1242 joint-angle samples. Your
forward_kinematics turns each into a flange position, plotted next to the
correct path. If yours spells RKD too, parts (a) and (b) are right.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from robot import XARM7_DH, Robot

HERE = os.path.dirname(os.path.abspath(__file__))


def sample_path():
    data = np.load(os.path.join(HERE, "sample_ground_truth.npz"))
    theta, gt = data["theta"], data["gt_ee"]

    # Calculate the end effector path through the entire run.
    try:
        robot = Robot(XARM7_DH)
        path = np.array([robot.end_effector(q) for q in theta], dtype=float).reshape(-1, 3)
    except Exception as err:
        raise SystemExit(f"Your code isn't ready for this yet: {type(err).__name__}: {err}\n"
                         "It needs parts (a) and (b); run python local_autograder.py "
                         "(from the top folder) to see which.")

    off = np.linalg.norm(path - gt, axis=1)
    print(f"Your flange path is within {off.max() * 1000:.3g} mm of the correct one "
          f"at every sample (median {np.median(off) * 1000:.3g} mm).")

    # Side-by-side subplots so the two paths are easy to compare, viewed from
    # above, as you would see the paper standing in front of the robot.
    fig = plt.figure(figsize=(11, 5.5))
    fig.suptitle("Plot of end effector position over a sample run.")
    lo = gt.min(axis=0) - 0.02
    hi = gt.max(axis=0) + 0.02
    for k, (points, title, style) in enumerate([(path, "Your Kinematics", "k-"),
                                                (gt, "Correct Kinematics", "g-")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot(points[:, 0], points[:, 1], points[:, 2], style, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2] + 0.05)
        ax.set_box_aspect((hi - lo + [0, 0, 0.05]).tolist())
        # Looking down, with the robot at the top of the plot.
        ax.view_init(elev=60, azim=0)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample_path()
