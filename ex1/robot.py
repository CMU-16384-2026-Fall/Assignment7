# robot.py

import numpy as np

# The xArm7's standard Denavit-Hartenberg parameters, from UFACTORY (provided;
# do not modify). One row per joint, in order from the base:
#
#     [a_i (m), alpha_i (rad), d_i (m), theta_offset_i (rad)]
#
# Joint i's DH angle is theta_i = q_i + theta_offset_i, where q_i is the joint
# angle the arm reports. (On the xArm7 every offset happens to be zero, but a
# general chain need not have that.) Robot(XARM7_DH) is the xArm7.
XARM7_DH = np.array([
    [0.0, -np.pi / 2, 0.267, 0.0],
    [0.0, np.pi / 2, 0.0, 0.0],
    [0.0525, np.pi / 2, 0.293, 0.0],
    [0.0775, np.pi / 2, 0.0, 0.0],
    [0.0, np.pi / 2, 0.3425, 0.0],
    [0.076, -np.pi / 2, 0.0, 0.0],
    [0.0, 0.0, 0.097, 0.0],
])


class Robot:
    """Represents a general fixed-base serial chain of revolute joints, in 3D,
    described by its Denavit-Hartenberg parameters."""

    def __init__(self, dh_parameters):
        """
        Constructor: Initializes a new robot from its DH table.

        Parameters:
        dh_parameters (np.ndarray): shape (n, 4), one row [a, alpha, d, theta_offset]
            per joint, as in XARM7_DH above. Robot(XARM7_DH) is the xArm7.
        """
        # Validate inputs (Do not modify this section)
        dh_parameters = np.asarray(dh_parameters, dtype=float)
        if dh_parameters.ndim != 2 or dh_parameters.shape[1] != 4:
            raise ValueError(f'Invalid dh_parameters: Should be (n, 4), is {dh_parameters.shape}.')

        self.dh_parameters = dh_parameters
        self.dof = dh_parameters.shape[0]

    @staticmethod
    def dh_transform(a, alpha, d, theta):
        """
        Returns the homogeneous transform from frame i to frame i - 1 for one
        link, given its DH parameters:

            H_i^{i-1} = Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)

        Parameters:
        a, alpha, d, theta (float): the link's DH parameters (m, rad, m, rad).
            theta already includes the joint's offset.

        Returns:
        np.ndarray: shape (4, 4).
        """
        H = np.zeros((4, 4))

        # --------------- BEGIN STUDENT SECTION: part (a), the DH transform ---------------
        # Build the four elementary 4x4 transforms, then multiply them in order.
        # (Or write out their product directly: the writeup's Background has it.)
        

        # Your code starts here
        # TODO (a.1): Rot_z(theta), a rotation by theta about z.
        # Rot_z = ...

        # TODO (a.2): Trans_z(d), a translation by d along z.
        # Trans_z = ...

        # TODO (a.3): Trans_x(a), a translation by a along x.
        # Trans_x = ...

        # TODO (a.4): Rot_x(alpha), a rotation by alpha about x.
        # Rot_x = ...

        # TODO (a.5): Multiply them in the order of the formula above.
        # H = ...

        

        # Your code ends here
        # --------------- END STUDENT SECTION ------------------------------------------------

        return H

    def forward_kinematics(self, thetas):
        """
        Returns the forward kinematic map for every DH frame: frames[:, :, i]
        is H_i^0, the transform from frame i to the base frame, for
        i = 0, 1, ..., dof. Frame 0 is the base itself (so frames[:, :, 0] is
        the identity), frame i sits at the end of link i, and the last one,
        frames[:, :, -1], is the end effector (on the xArm7, the flange).

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
        # Fill in frames[:, :, 0] through frames[:, :, n], from the base out.

        # Your code starts here

        # TODO (USE the given DH parameters given at the top of the file
        #  alongside your dh_transform method you implemented to find each frame)
        

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
