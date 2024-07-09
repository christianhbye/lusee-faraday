import numpy as np
import healpy as hp


def pol_angle(freq, rm, ref_freq=23e3):
    """
    Compute polarization angle as a function of frequency.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies in MHz.
    rm : float
        Rotation measure in rad/m^2.
    ref_freq : float
        Reference frequency in MHz where the polarization angle is zero.

    """
    c = 299792458  # speed of light in m/s
    # convert frequencies to Hz
    f_Hz = freq * 1e6
    f0_Hz = ref_freq * 1e6
    dlambda_sq = c**2 * (1 / f_Hz**2 - 1 / f0_Hz**2)
    return rm * dlambda_sq


class Sky:
    def __init__(self, stokes=None, freq=None):
        """
        Parameters
        ----------
        stokes : np.ndarray
           Stokes parameters of the sky. Shape (3, npix) or (3, nfreq, npix).
        freq : np.ndarray
           Frequency in MHz.
        """
        self.stokes = stokes
        self.freq = freq

    @property
    def npix(self):
        return self.stokes.shape[-1]

    @property
    def nside(self):
        return hp.npix2nside(self.npix)

    @property
    def sky_angle(self):
        """
        Returning the longitude and latitude of the sky pixels.
        """
        return hp.pix2ang(self.nside, np.arange(self.npix), lonlat=True)

    @property
    def bright_pixels(self):
        """
        Return the pixels that have non-zero Stokes I at the first frequency.
        """
        return self.stokes[0, 0] != 0

    def del_dark_pixels(self):
        """
        Delete the pixels that have no intensity at any frequency. Return the
        pixels that were kept.
        """
        pixels = self.bright_pixels
        self.stokes = self.stokes[:, :, pixels]
        return pixels

    @classmethod
    def zeros(cls, nside=128, freq=30):
        npix = hp.nside2npix(nside)
        freq = np.atleast_1d(freq)
        nf = len(freq)
        return cls(stokes=np.zeros((3, nf, npix)), freq=freq)

    def add_point_source(self, extent=5):
        """
        Add a linearly polarized point source to the sky at zenith.

        Parameters
        ----------
        extent : float
            Extent of the source in degrees.

        """
        if self.stokes is None:
            raise ValueError(
                "Sky must be initialized before adding point sources"
            )

        lon, lat = self.sky_angle
        phi = np.deg2rad(lon)
        src = lat > 90 - extent
        self.stokes[0] += np.where(src, 1, 0)[None]  # stokes I
        self.stokes[1] += np.where(src, -np.cos(2 * phi), 0)[None]  # stokes Q
        self.stokes[2] += np.where(src, np.sin(2 * phi), 0)[None]  # stokes U
