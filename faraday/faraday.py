from astropy import units as u
from astropy.coordinates import Galactic
import croissant as cro
from healpy import Rotator
from lunarsky import LunarTopo, MoonLocation, Time
import numpy as np


class Simulator:
    def __init__(self, beam, sky, center_freq=30):
        """
        Beam in topo frame, sky in galactic frame.

        beam : faraday.Beam
            beam.beam_X and beam.beam_Y are the beam patterns, they have
            shapes (2, npix) where 2 is for Etheta and Ephi.
        sky : faraday.Sky
            sky.stokes has shape (3, nfreq, npix) where 3 is for I, Q, U.
            sky.coherency has shape (2, 2, nfreq, npix) where the first 2 axes
            is the 2x2 coherency matrix.

        """
        spec_path = (
            "/home/christian/Documents/research/lusee/faraday/data/"
            "zoom_response_4tap.txt"
        )
        self.center_freq = center_freq
        spec = np.loadtxt(spec_path)[500:1500]
        self.offset = spec[:, 0] / 1e3
        spec = spec[:, 1:] / spec[:, 1:].sum(axis=0, keepdims=True)
        self.wide_bin = spec[:, 0]
        self.spec = spec[:, 1:]

        beam.beam_X = beam.beam_X[:, 0]  # killing the freq axis
        beam.beam_Y = beam.beam_Y[:, 0]
        self.beam = beam
        self.sky = sky

        self.norm = 2 / np.sum(np.abs(beam.beam_XY) ** 2, axis=(0, 2))

        self.nfreq = sky.stokes.shape[1]

    def convolve(self):
        """
        Compute the visibility at one time.
        """
        vis = {}
        vis["XX"] = np.einsum(
            "ap, bp, abfp -> fp",
            self.beam.beam_X,
            self.beam.beam_X.conj(),
            self.sky.coherency,
        )
        vis["YY"] = np.einsum(
            "ap, bp, abfp -> fp",
            self.beam.beam_Y,
            self.beam.beam_Y.conj(),
            self.sky.coherency,
        )
        vis["XY"] = np.einsum(
            "ap, bp, abfp -> fp",
            self.beam.beam_X,
            self.beam.beam_Y.conj(),
            self.sky.coherency,
        )
        for k, v in vis.items():
            vis[k] = v * self.norm

        return vis

    def rotate_sky(self, from_frame, to_frame):
        """
        Rotate the sky from one frame to another.

        """
        rm = cro.utils.get_rot_mat(from_frame, to_frame)
        eul = cro.utils.rotmat_to_euler(rm, eulertype="ZYX")
        rot = Rotator(
            rot=eul, coord=None, inv=False, deg=False, eulertype="ZYX"
        )
        rsky = np.empty_like(self.sky.stokes)
        for i in range(self.nfreq):
            rsky[:, i] = rot.rotate_map_alms(self.sky.stokes[:, i], lmax=50)
        self.sky.stokes = rsky

    def run(
        self,
        lat=-23.183,
        lon=182.258,
        t0="2026-02-01T00:00:00",
        dt=12,
        ntimes=56,
    ):
        """
        Run the simulation.

        dt : int
            Time step in hrs.

        """
        self.ntimes = ntimes
        all_vis = {}
        all_vis["XX"] = np.zeros((ntimes, self.nfreq))
        all_vis["YY"] = np.zeros((ntimes, self.nfreq))
        all_vis["XY"] = np.zeros((ntimes, self.nfreq), dtype=complex)
        # rotate sky and RM from galactic to topocentric
        loc = MoonLocation(lon, lat)
        time = Time(t0)
        topo_frame = LunarTopo(location=loc, obstime=time)
        self.rotate_sky(Galactic(), topo_frame)
        for i in range(ntimes):
            if i > 0:  # rotate sky and RM with time
                time = time + dt * u.hour
                topo_frame = LunarTopo(location=loc, obstime=time)
                self.rotate_sky(Galactic(), topo_frame)
            vis = self.convolve()
            for k, v in vis.items():
                all_vis[k][i] = v
        self.vis = all_vis

    def channelize(self, bins="narrow"):
        # narrow: run sim at 1000 freqs (from spec) close to center freq,
        # then dot product with the spec (1000, 64) to get 64 freqs.
        if bins == "narrow":
            for k, v in self.vis.items():
                self.vis[k] = v @ self.spec
        # wide: run sim at 64 * 1000 freqs (one for each bin), then dot
        # each one with the wide bin to get 64 freqs.
        elif bins == "wide":
            for k, v in self.vis.items():
                v = v.reshape(self.ntimes, -1, self.wide_bin.size)
                self.vis[k] = np.sum(v * self.wide_bin[None, None], axis=2)
        else:
            raise ValueError("bins must be 'narrow' or 'wide'.")
            

