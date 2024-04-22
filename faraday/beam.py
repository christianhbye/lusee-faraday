import numpy as np
import healpy as hp


class Beam:
    def __init__(self, beam_X=None, beam_Y=None, frequency=None):
        self.beam_X = beam_X
        self.beam_Y = beam_Y
        self.frequency = frequency

    def rotate_X2Y(self, phi_axis=-1):
        """
        Rotate the X polarization to the Y polarization.

        NOTE: This assumes that the beam is specified on a grid where
        the spacing in phi is 1 degree.
        """
        if self.beam_X is None:
            raise ValueError("No X beam specified.")
        return np.roll(self.beam_X, 90, axis=phi_axis)

    def rotate_Y2X(self, phi_axis=-1):
        """
        Rotate the Y polarization to the X polarization.

        NOTE: This assumes that the beam is specified on a grid where
        the spacing in phi is 1 degree.
        """
        if self.beam_Y is None:
            raise ValueError("No Y beam specified.")
        return np.roll(self.beam_Y, -90, axis=phi_axis)


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
