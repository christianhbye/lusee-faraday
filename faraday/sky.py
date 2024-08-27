from astropy.io import fits
from croissant import rotations
import healpy as hp
import numpy as np


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

    @classmethod
    def wmap(cls, nside=64):
        """
        Load the WMAP K-band data and return a Sky object.

        Parameters
        ----------
        nside : int
            Resolution of the map.

        """
        wmap_path = (
            "/home/christian/Documents/research/lusee/faraday/data/"
            "wmap_band_iqumap_r9_9yr_K_v5.fits"
        )
        with fits.open(wmap_path) as hdul:
            d = hdul["Stokes Maps"].data  # in mK XXX
            I_wmap = d["TEMPERATURE"] * 1e-3  # in K
            Q_wmap = d["Q_POLARISATION"] * 1e-3
            U_wmap = d["U_POLARISATION"] * 1e-3
        I_wmap = hp.ud_grade(I_wmap, nside, order_in="NEST", order_out="RING")
        Q_wmap = hp.ud_grade(Q_wmap, nside, order_in="NEST", order_out="RING")
        U_wmap = hp.ud_grade(U_wmap, nside, order_in="NEST", order_out="RING")
        stokes = np.stack([I_wmap, Q_wmap, U_wmap])
        # add freq axis
        stokes = np.expand_dims(stokes, axis=1)
        return cls(stokes=stokes, freq=23e3)

    def gal_to_topo(self, lat, lon, time, moon=True):
        """
        Convert the sky to the topocentric frame on the moon.

        Parameters
        ----------
        lat : float
            Latitude of the observer in degrees.
        lon : float
            Longitude of the observer in degrees.
        time : astropy.time.Time
            Time of observation.
        moon : bool
            If True, the observer is on the moon. Otherwise, the observer is
            on Earth.

        """
        # combine stokes and freq axes to one
        npix = self.stokes.shape[-1]
        stokes = self.stokes.reshape(-1, npix)
        # galactic to mcmf (equatorial)
        if moon:
            frame = "M"  # mcmf
        else:
            frame = "C"  # equatorial
        r_g2m = rotations.Rotator(coord=f"G{frame}")
        stokes_mcmf = r_g2m.rotate_map_alms(stokes, lmax=50)
        # mcmf to topo (XXX equatorial)
        r_m2t = rotations.Rotator(coord=f"{frame}T", loc=(lon, lat), time=time)
        stokes_topo = r_m2t.rotate_map_alms(stokes_mcmf, lmax=50)
        self.stokes = stokes_topo.reshape(self.stokes.shape)
