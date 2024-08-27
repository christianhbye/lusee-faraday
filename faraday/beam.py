from functools import partial
import numpy as np
import healpy as hp
from astropy.io import fits
import croissant as cro

# stokes I, Q, U polarization matrices
PAULI_MATRICES = {
    "I": 1 / 2 * np.eye(2),
    "Q": 1 / 2 * np.array([[1, 0], [0, -1]]),
    "U": 1 / 2 * np.array([[0, 1], [1, 0]]),
}


def rotate_X2Y(beam_X, phi_axis=-1):
    """
    Rotate the X polarization to the Y polarization.

    NOTE: This assumes that the beam is specified on a grid where
    the spacing in phi is 1 degree.
    """
    return np.roll(beam_X, 90, axis=phi_axis)


def rotate_Y2X(beam_Y, phi_axis=-1):
    """
    Rotate the Y polarization to the X polarization.

    NOTE: This assumes that the beam is specified on a grid where
    the spacing in phi is 1 degree.
    """
    return np.roll(beam_Y, -90, axis=phi_axis)


class Beam:
    def __init__(self, beam_X=None, beam_Y=None, frequency=None):
        """
        Initialize the beam object.

        Beam_X and beam_Y are the maps of the X and Y polarization with
        shape (2, nfreq, npix) where the first axis corresponds to the
        theta and phi components of the electric field. The frequency
        axis is the second axis and must be 1 if the beam is achromatic.
        """
        self.beam_X = beam_X
        self.beam_Y = beam_Y
        self.frequency = frequency

    def del_pix(self, pix):
        """
        Delete pixels from the beam by specifying the pixel indices TO KEEP.
        """
        if self.beam_X is not None:
            self.beam_X = self.beam_X[..., pix]
        if self.beam_Y is not None:
            self.beam_Y = self.beam_Y[..., pix]

    def rotate(self, rot):
        """
        Rotate the beam.

        Parameters
        ----------
        rot : hp.Rotator
            Healpy Rotator object.
        """
        if self.beam_X is not None:
            theta_re = np.real(self.beam_X[0])
            theta_im = np.imag(self.beam_X[0])
            phi_re = np.real(self.beam_X[1])
            phi_im = np.imag(self.beam_X[1])
            rot_theta = np.array(
                [
                    rot.rotate_map_alms(theta_re[i])
                    + 1j * rot.rotate_map_alms(theta_im[i])
                    for i in range(len(theta_re))
                ]
            )
            rot_phi = np.array(
                [
                    rot.rotate_map_alms(phi_re[i])
                    + 1j * rot.rotate_map_alms(phi_im[i])
                    for i in range(len(phi_re))
                ]
            )
            self.beam_X = np.array([rot_theta, rot_phi])

        if self.beam_Y is not None:
            theta_re = np.real(self.beam_Y[0])
            theta_im = np.imag(self.beam_Y[0])
            phi_re = np.real(self.beam_Y[1])
            phi_im = np.imag(self.beam_Y[1])
            rot_theta = np.array(
                [
                    rot.rotate_map_alms(theta_re[i])
                    + 1j * rot.rotate_map_alms(theta_im[i])
                    for i in range(len(theta_re))
                ]
            )
            rot_phi = np.array(
                [
                    rot.rotate_map_alms(phi_re[i])
                    + 1j * rot.rotate_map_alms(phi_im[i])
                    for i in range(len(phi_re))
                ]
            )
            self.beam_Y = np.array([rot_theta, rot_phi])

    @property
    def beam_powers(self):
        """
        Get the beam powers for each polarization. This is needed for the
        visibility calculation.

        Returns
        -------
        powers : dict
            Nested dictionary containing the beam powers for each combination
            of X and Y and Stokes I, Q, U. Keys are "XX", "XY", "YY" for the
            first level and "I", "Q", "U" for the second level.

        """
        powers = {"XX": {}, "XY": {}, "YY": {}}
        bx = self.beam_X  # (theta/phi, npix)
        by = self.beam_Y  # (theta/phi, npix)
        ein = partial(np.einsum, "afp, bfp, ab -> fp")
        for stokes in ["I", "Q", "U"]:
            mat = PAULI_MATRICES[stokes]
            powers["XX"][stokes] = ein(bx, bx.conj(), mat)
            powers["XY"][stokes] = ein(bx, by.conj(), mat)
            powers["YY"][stokes] = ein(by, by.conj(), mat)
        return powers


class ShortDipole(Beam):
    def __init__(self, nside, frequency=None):
        self.nside = nside
        theta, phi = hp.pix2ang(
            nside, np.arange(hp.nside2npix(nside)), nest=False, lonlat=False
        )
        horizon = np.where(theta < np.pi / 2, 1, 0)[None]
        # X beam
        E_theta = -np.cos(theta) * np.cos(phi)
        E_phi = np.sin(phi)
        beam_X = np.array([E_theta, E_phi]) * horizon
        # Y beam
        E_theta = -np.cos(theta) * np.sin(phi)
        E_phi = -np.cos(phi)
        beam_Y = np.array([E_theta, E_phi]) * horizon
        # add frequency axis
        beam_X = beam_X[:, None]
        beam_Y = beam_Y[:, None]
        super().__init__(beam_X=beam_X, beam_Y=beam_Y, frequency=frequency)


class LuseeBeam(Beam):
    def __init__(self, path, frequency=30, nside=128):
        """
        Load the LUSEE beam from a FITS file.

        Parameters
        ----------
        path : str
            Path to the FITS file containing the LUSEE beam.
        frequency : float
            Frequency at which to extract the beam in MHz.
        nside : int
            Healpix nside at which to interpolate the beam.

        """
        with fits.open(path) as hdul:
            E_theta = hdul["Etheta_real"].data + 1j * hdul["Etheta_imag"].data
            E_phi = hdul["Ephi_real"].data + 1j * hdul["Ephi_imag"].data
            if frequency is not None:
                ix = np.argwhere(hdul["freq"].data == frequency)[0, 0]
                E_theta = E_theta[ix][None]  # extract freq but keep axis
                E_phi = E_phi[ix][None]
            E_theta = E_theta[:, :, :-1]
            E_phi = E_phi[:, :, :-1]
        # lusee Y has shape (nfreq, 4, ntheta, nphi)
        lusee_Y = np.stack(
            [E_theta.real, E_theta.imag, E_phi.real, E_phi.imag],
            axis=1,
        )
        # add horizon, but cut off one theta pixel
        lusee_Y = np.concatenate(
            (lusee_Y, np.zeros_like(lusee_Y)[:, :, :-1, :]), axis=2
        )

        lusee_X = rotate_Y2X(lusee_Y, phi_axis=-1)

        # convert to healpix
        lusee_X = np.array(
            [cro.healpix.grid2healpix(lX, nside) for lX in lusee_X]
        )
        lusee_Y = np.array(
            [cro.healpix.grid2healpix(lY, nside) for lY in lusee_Y]
        )

        # move frequency axis to 1st axis
        lusee_X = np.swapaxes(lusee_X, 0, 1)
        lusee_Y = np.swapaxes(lusee_Y, 0, 1)

        beam_X = np.array(
            [lusee_X[0] + 1j * lusee_X[1], lusee_X[2] + 1j * lusee_X[3]]
        )
        beam_Y = np.array(
            [lusee_Y[0] + 1j * lusee_Y[1], lusee_Y[2] + 1j * lusee_Y[3]]
        )
        super().__init__(beam_X=beam_X, beam_Y=beam_Y, frequency=frequency)
