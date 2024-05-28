import numpy as np


def rotate_zenith(theta, phi):
    """
    Get the rotation matrix for sending the zenith vector to the given angles.

    The zenith vector is the vector pointing straight up, (0, 0, 1). We
    first rotate by theta around the y-axis, then by phi around the z-axis.

    Parameters
    ----------
    theta : float
        The polar angle in radians.
    phi : float
        The azimuthal angle in radians.

    Returns
    -------
    rmat : np.ndarray
        The rotation matrix.

    """
    # Rotation around y-axis
    r1 = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    # Rotation around z-axis
    r2 = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ]
    )
    return r2 @ r1


def rotmat_to_euler(mat):
    """
    Convert a rotation matrix to Euler angles in the ZYX convention. This is
    sometimes referred to as Tait-Bryan angles X1-Y2-Z3.

    Parameters
    ----------
    mat : np.ndarray
        The rotation matrix.

    Returns
    --------
    eul : tup
        The Euler angles.

    """
    beta = np.arcsin(mat[0, 2])
    alpha = np.arctan2(mat[1, 2] / np.cos(beta), mat[2, 2] / np.cos(beta))
    gamma = np.arctan2(mat[0, 1] / np.cos(beta), mat[0, 0] / np.cos(beta))
    eul = (gamma, beta, alpha)
    return eul
