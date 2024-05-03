import numpy as np
import healpy as hp
from astropy.io import fits
from croissant.healpix import grid2healpix


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
        self.beam_X = beam_X
        self.beam_Y = beam_Y
        self.frequency = frequency

    def del_pix(self, pix):
        """
        Delete pixels from the beam by specifying the pixel indices TO KEEP.
        """
        if self.beam_X is not None:
            self.beam_X = self.beam_X[:, pix]
        if self.beam_Y is not None:
            self.beam_Y = self.beam_Y[:, pix]


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
        super().__init__(beam_X=beam_X, beam_Y=beam_Y, frequency=frequency)


class LuseeBeam(Beam):
    def __init__(self, path, frequency, nside=128):
        with fits.open(path) as hdul:
            E_theta = hdul["Etheta_real"].data + 1j * hdul["Etheta_imag"].data
            E_phi = hdul["Ephi_real"].data + 1j * hdul["Ephi_imag"].data
            ix = np.argwhere(hdul["freq"].data == frequency)[0, 0]
            E_theta = E_theta[ix]
            E_phi = E_phi[ix]
        lusee_Y = np.array(
            [E_theta.real, E_theta.imag, E_phi.real, E_phi.imag]
        )
        # add horizon
        lusee_Y = np.concatenate(
            (lusee_Y, np.zeros_like(lusee_Y)[:, :-1, :]), axis=1
        )

        lusee_X = rotate_Y2X(lusee_Y, phi_axis=-1)

        # convert to healpix
        lusee_X = grid2healpix(lusee_X, nside)
        lusee_Y = grid2healpix(lusee_Y, nside)

        beam_X = np.array(
            [lusee_X[0] + 1j * lusee_X[1], lusee_X[2] + 1j * lusee_X[3]]
        )
        beam_Y = np.array(
            [lusee_Y[0] + 1j * lusee_Y[1], lusee_Y[2] + 1j * lusee_Y[3]]
        )
        super().__init__(beam_X=beam_X, beam_Y=beam_Y, frequency=frequency)
