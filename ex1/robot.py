# robot.py

import numpy as np

# The xArm7, joint by joint (provided; do not modify). These are UFACTORY's
# own numbers, as the arm's URDF gives them. One row per joint, in order from
# the base:
#
#     [x, y, z (m), roll, pitch, yaw (rad)]
#
# Row i places joint i's frame relative to the frame before it (the base frame
# for joint 1, link i-1's frame after that): a translation by (x, y, z), then
# a rotation by (roll, pitch, yaw), exactly as transform() below builds it.
# Joint i then turns everything after it about that frame's z axis, by its
# angle q_i. Robot(XARM7_JOINTS) is the xArm7.
XARM7_JOINTS = np.array([
    [0.0, 0.0, 0.267, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0],
    [0.0, -0.293, 0.0, np.pi / 2, 0.0, 0.0],
    [0.0525, 0.0, 0.0, np.pi / 2, 0.0, 0.0],
    [0.0775, -0.3425, 0.0, np.pi / 2, 0.0, 0.0],
    [0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0],
    [0.076, 0.097, 0.0, -np.pi / 2, 0.0, 0.0],
])


# --------------------------------------------------------------------------
# Part (a): 3D homogeneous transforms. Each returns a 4x4 numpy array.
# --------------------------------------------------------------------------

def rot_x(theta):
    """A rotation by theta (rad) about the x axis, as a 4x4 homogeneous transform."""
    H = np.zeros((4, 4))

    # --------------- BEGIN STUDENT SECTION: part (a.1), rotation about x ---------------
    # TODO (a.1): x stays fixed; y and z turn by theta (right-hand rule about x).

    # Your code starts here

    # Your code ends here
    # --------------- END STUDENT SECTION ------------------------------------------------

    return H


def rot_y(theta):
    """A rotation by theta (rad) about the y axis, as a 4x4 homogeneous transform."""
    H = np.zeros((4, 4))

    # --------------- BEGIN STUDENT SECTION: part (a.2), rotation about y ---------------
    # TODO (a.2): y stays fixed; z and x turn by theta. Watch the signs: about
    #   y, positive theta turns z towards x.

    # Your code starts here

    # Your code ends here
    # --------------- END STUDENT SECTION ------------------------------------------------

    return H


def rot_z(theta):
    """A rotation by theta (rad) about the z axis, as a 4x4 homogeneous transform."""
    H = np.zeros((4, 4))

    # --------------- BEGIN STUDENT SECTION: part (a.3), rotation about z ---------------
    # TODO (a.3): z stays fixed; x and y turn by theta.

    # Your code starts here

    # Your code ends here
    # --------------- END STUDENT SECTION ------------------------------------------------

    return H


def translation(x, y, z):
    """A translation by (x, y, z) (m), as a 4x4 homogeneous transform."""
    H = np.zeros((4, 4))

    # --------------- BEGIN STUDENT SECTION: part (a.4), translation ---------------
    # TODO (a.4): no rotation, and the offset in the last column.

    # Your code starts here

    # Your code ends here
    # --------------- END STUDENT SECTION ------------------------------------------------

    return H


def transform(xyz, rpy):
    """
    The transform a URDF writes as <origin xyz="x y z" rpy="roll pitch yaw"/>:
    a translation by xyz, then a roll-pitch-yaw rotation,

        H = Trans(x, y, z) Rot_z(yaw) Rot_y(pitch) Rot_x(roll)

    Parameters:
    xyz: (x, y, z) in metres.
    rpy: (roll, pitch, yaw) in radians.

    Returns:
    np.ndarray: shape (4, 4).
    """
    x, y, z = xyz
    roll, pitch, yaw = rpy
    H = np.zeros((4, 4))

    # --------------- BEGIN STUDENT SECTION: part (a.5), offset and orientation ---------------
    # TODO (a.5): Multiply your four functions above in the order of the
    #   formula: the translation first (leftmost), then yaw, pitch, roll.
    # H = ...

    # Your code starts here

    # Your code ends here
    # --------------- END STUDENT SECTION ------------------------------------------------

    return H


# --------------------------------------------------------------------------
# Part (b): forward kinematics
# --------------------------------------------------------------------------

class Robot:
    """Represents a general fixed-base serial chain of revolute joints in 3D,
    described joint by joint: each joint is a fixed offset from the frame before
    it, then a rotation about its own z axis."""

    def __init__(self, joints):
        """
        Constructor: Initializes a new robot from its joint table.

        Parameters:
        joints (np.ndarray): shape (n, 6), one row [x, y, z, roll, pitch, yaw]
            per joint, as in XARM7_JOINTS above. Robot(XARM7_JOINTS) is the xArm7.
        """
        # Validate inputs (Do not modify this section)
        joints = np.asarray(joints, dtype=float)
        if joints.ndim != 2 or joints.shape[1] != 6:
            raise ValueError(f'Invalid joints: Should be (n, 6), is {joints.shape}.')

        self.joints = joints
        self.dof = joints.shape[0]

    def forward_kinematics(self, thetas):
        """
        Returns the forward kinematic map for every frame: frames[:, :, i] is
        H_i^0, the transform from frame i to the base frame, for
        i = 0, 1, ..., dof. Frame 0 is the base itself (so frames[:, :, 0] is
        the identity), frame i is link i's frame, the one joint i turns, and the
        last one, frames[:, :, -1], is the end effector (on the xArm7, the
        flange).

        Parameters:
        thetas (np.ndarray): 1D array of joint angles q_i, shape (dof,).

        Returns:
        np.ndarray: Frames, shape (4, 4, dof + 1)
        """
        thetas = np.asarray(thetas, dtype=float)
        if thetas.ndim != 1:
            raise ValueError('Expecting a 1D array of joint angles.')

        if thetas.shape[0] != self.dof:
            raise ValueError(f'Invalid number of joints: {thetas.shape[0]} found, expecting {self.dof}')

        # Allocate frames array
        frames = np.zeros((4, 4, self.dof + 1))
        n = self.dof

        # --------------- BEGIN STUDENT SECTION: part (b), forward kinematics in 3D ---------------
        # Calculate frames[:, :, 0] through frames[:, :, n], from the base out.

        # Your code starts here
        # TODO
        # frames[:, :, 0] = ...


        # remember for each link offset first from self.joints,
        # then rotation by thetas

        # frames[:, :, i] = ...

        

        # Your code ends here
        # --------------- END STUDENT SECTION ------------------------------------------------

        return frames

    def fk(self, thetas):
        """Shorthand for returning the forward kinematics."""
        return self.forward_kinematics(thetas)

    def end_effector(self, thetas):
        """
        Returns [x, y, z], the end effector's position in the base frame, for a
        set of joint angles. Its orientation is the rotation block of the last
        frame, self.fk(thetas)[:3, :3, -1].
        """
        # Do not modify this method
        frames = self.fk(thetas)
        return frames[:3, 3, -1].copy()

    def ee(self, thetas):
        """Shorthand for returning the end effector position."""
        return self.end_effector(thetas)
