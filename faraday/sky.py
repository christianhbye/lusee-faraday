import numpy as np
import healpy as hp


def coherency(stokes):
    """
    Coherecy matrix of the sky assuming no V.

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters of the sky. Shape (nfreq, 3, npix).

    Returns
    -------
    T : np.ndarray
        Coherency matrix of the sky. Shape (nfreq, 2, 2, npix).
    """
    i = stokes[:, 0]
    q = stokes[:, 1]
    u = stokes[:, 2]
    return 1 / 2 * np.array([[i + q, u], [u, i - q]])


def pol_angle(freq, rm, ref_freq):
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
           Stokes parameters of the sky. Shape (nfreq, 3, npix).
        freq : np.ndarray
           List of frequencies in MHz.
        """
        self.stokes = stokes
        self.freq = freq

    @property
    def coherency(self):
        return coherency(self.stokes)

    @property
    def coherency_rot(self):
        return coherency(self.stokes_rot)

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
        Return the pixels that have some intensity at at least one frequency.
        """
        return np.any(self.stokes[:, 0] != 0, axis=0)

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
        if freq is None:
            nfreq = 1
        else:
            freq = np.atleast_1d(freq)
            nfreq = freq.size
        return cls(stokes=np.zeros((nfreq, 3, npix)), freq=freq)

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
        # currently only doing source at zenith
        mask = lat > 90 - extent
        self.stokes[:, 0, mask] += 1  # stokes I
        # add None to phi for frequency broadcasting
        self.stokes[:, 1, mask] += -np.cos(2 * phi[None, mask])  # stokes Q
        self.stokes[:, 2, mask] += np.sin(2 * phi[None, mask])  # stokes U

    def power_law(self, freqs, beta):
        """
        Scale the nonzero (bright pixels) sky by a power law.

        Parameters
        ----------
        freqs : np.ndarray
            Frequencies in MHz to scale the sky to.
        beta : float
            Power law index.
        """
        if self.stokes.shape[0] != 1:
            raise ValueError(
                "Sky can only be specified at one frequency before scaling"
            )
        if self.freq is None:
            raise ValueError(
                "Sky must have a referency frequency before scaling"
            )
        self.stokes = self.stokes * (freqs[:, None, None] / self.freq) ** beta
        self.freq = freqs

    def apply_faraday(self, rm=100):
        """
        Apply Faraday rotation to the sky.

        Parameters
        ----------
        rm : float
            Rotation measure in rad/m^2.

        """
        q = self.stokes[:, 1]
        u = self.stokes[:, 2]
        p = q + 1j * u
        chi = pol_angle(self.freq, rm, 23e3)  # XXX
        p_rot = p * np.exp(2j * chi[:, None])
        self.stokes_rot = self.stokes.copy()
        self.stokes_rot[:, 1] = np.real(p_rot)
        self.stokes_rot[:, 2] = np.imag(p_rot)
