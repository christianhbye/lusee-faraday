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
        return hp.pix2ang(self.nside, np.arange(self.npix))

    @classmethod
    def zeros(cls, nside, freq=None):
        npix = hp.nside2npix(nside)
        if freq is None:
            nfreq = 1
        else:
            nfreq = len(freq)
        return cls(stokes=np.zeros((nfreq, 3, npix)), frequency=freq)

    def add_point_source(self, alt=90, az=0, extent=5):
        """
        Add a linearly polarized point source to the sky.

        Parameters
        ----------
        alt : float
            Altitude of the source in degrees.
        az : float
            Azimuth of the source in degrees.
        extent : float
            Extent of the source in degrees.

        """
        if self.stokes is None:
            raise ValueError(
                "Sky must be initialized before adding point sources"
            )

        theta, phi = self.sky_angle[1]
        # currently only doing source at zenith
        assert alt == 90
        mask = np.abs(np.degrees(theta) - alt) < extent
        self.stokes[:, 0, mask] += 1  # stokes I
        # add None to phi for frequency broadcasting
        self.stokes[:, 1, mask] += -np.cos(2 * phi[None, mask])  # stokes Q
        self.stokes[:, 2, mask] += np.sin(2 * phi[None, mask])  # stokes U

        # alt = np.radians(alt)
        # az = np.radians(az)
        # extent = np.radians(extent)

        # delta = np.arccos(np.sin(alt) * np.sin(theta) + np.cos(alt) *
        # np.cos(theta) * np.cos(az - phi))
        # mask = delta < extent

        # self.stokes[0, mask] = 1
        # self.stokes[1, mask] = 1

    def power_law(self, freqs, beta):
        """
        Scale the sky by a power law.

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
        self.stokes = self.stokes * (freqs / self.freq) ** beta
        self.freq = freqs

    def apply_faraday(self, rm):
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
        p_rot = p * np.exp(2j * chi)
        self.stokes_rot = self.stokes.copy()
        self.stokes_rot[:, 1] = np.real(p_rot)
        self.stokes_rot[:, 2] = np.imag(p_rot)
