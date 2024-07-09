from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from .sky import pol_angle


def vis2stokes(vis_arr):
    """
    Convert simulated visibilities to Stokes parameters.

    Parameters
    -----------
    vis_arr: np.ndarray
        Visibilities in the order V11, V12_real, V12_imag, V22.
        Shape (4, nfreqs).

    Returns
    -------
    stokes : np.ndarray
        The Stokes parameters I, Q, and U (shape (3, nfreq).

    """
    V11, V12_real, V12_imag, V22 = vis_arr

    pI = (V11 + V22) / 2
    pQ = (V11 - V22) / 2
    pU = V12_real
    return np.array([pI, pQ, pU])


def plot_vis(freqs, vis_arr, vis_arr_rot, sharey="row", return_fig=False):
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey=sharey, constrained_layout=True
    )
    axs[0, 0].plot(freqs, vis_arr[0], c="C0", label="V11")
    axs[0, 0].plot(freqs, vis_arr[3], c="C1", label="V22")
    axs[0, 1].plot(freqs, vis_arr_rot[0], c="C0")
    axs[0, 1].plot(freqs, vis_arr_rot[3], c="C1")
    axs[1, 0].plot(freqs, vis_arr[1], label="V12 real")
    axs[1, 0].plot(freqs, vis_arr[2], label="V12 imag")
    axs[1, 1].plot(freqs, vis_arr_rot[1])
    axs[1, 1].plot(freqs, vis_arr_rot[2])
    for ax in axs[:, 0].ravel():
        ax.legend()
    axs[0, 0].set_title("No Faraday")
    axs[0, 1].set_title("With Faraday")
    for ax in axs[1]:
        ax.set_xlabel("Frequency [MHz]")
    if return_fig:
        return fig, axs
    plt.show()


def plot_stokes(
    freqs, stokes_arr, stokes_arr_rot, sharey="row", return_fig=False
):
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey=sharey, constrained_layout=True
    )
    axs[0, 0].plot(freqs, stokes_arr[1], label="Q")
    axs[0, 1].plot(freqs, stokes_arr_rot[1], label="Q")
    axs[0, 0].plot(freqs, stokes_arr[2], label="U")
    axs[0, 1].plot(freqs, stokes_arr_rot[2], label="U")
    axs[1, 0].plot(
        freqs,
        (stokes_arr[1] ** 2 + stokes_arr[2] ** 2) / stokes_arr[0] ** 2,
        label="(Q^2+U^2)/I^2",
    )
    axs[1, 1].plot(
        freqs,
        (stokes_arr_rot[1] ** 2 + stokes_arr_rot[2] ** 2)
        / stokes_arr_rot[0] ** 2,
    )
    axs[0, 0].legend()
    axs[1, 0].legend()
    axs[0, 0].set_title("No Faraday")
    axs[0, 1].set_title("With Faraday")
    for ax in axs[1]:
        ax.set_xlabel("Frequency [MHz]")
    if return_fig:
        return fig, axs
    plt.show()


class Simulator:
    ref_freq = 23e3  # 23 GHz, reference frequency for pol angle = 0

    def __init__(self, beam, sky, center_freq=30, full_map=False):
        """
        Simulator class.

        Parameters
        ----------
        beam : Beam
            Beam object.
        sky : Sky
            Sky object.
        center_freq : float
            Center frequency in MHz.
        full_map : bool
            If true, using a full sky map specified at many pixels and
            frequencies. Otherwise, using a single source scaled by a power
            law.

        Sky and beam can be provided at one frequency or at multiple
        frequencies. If provided at multiple frequencies, the beam and sky
        objects should have the same number of frequency channels. If at one
        frequency, the beam is achromatic and the sky is scaled by a power law
        from the reference frequency (23 GHz).

        """
        path = (
            "/home/christian/Documents/research/lusee/faraday/data/"
            "zoom_response_4tap.txt"
        )
        self.center_freq = center_freq
        spec = np.loadtxt(path)
        self.offset = spec[:, 0] / 1e3  # spacing in MHz
        spec = spec[:, 1:] / spec[:, 1:].sum(axis=0, keepdims=True)
        self.wide_bin = spec[:, 0]  # shape (2000,)
        self.spec = spec[:, 1:]

        _beam = deepcopy(beam)
        _sky = deepcopy(sky)

        if not sky or not beam:
            return

        self.norm = 2 / np.sum(np.abs(_beam.beam_X) ** 2, axis=(0, 2))
        if not full_map:
            _npix = _sky.npix
            pix = _sky.del_dark_pixels()
            self.source_frac = _sky.npix / _npix  # fraction of pixels in src
            _beam.del_pix(pix)

        self.nfreq = sky.stokes.shape[1]

        self.beam = _beam
        self.sky = _sky

    def compute_vis(self):
        """
        Compute visibilities for each polarization.

        The visibilities are stored in a nested dictionary containing the
        visibilities for each polarization. First level keys are "XX", "XY",
        and "YY". The second level keys are "I", "Q", "U", "UQ", and "QU",
        where the last two denote mode-mixing due to Faraday rotation.
        """
        vis = {}
        beam_powers = self.beam.beam_powers
        for pair in beam_powers:  # XX, XY, YY
            vis[pair] = {}
            for i, stokes in enumerate(["I", "Q", "U"]):
                b = beam_powers[pair][stokes]
                s = self.sky.stokes[i]
                # sum over pixels
                vis[pair][stokes] = np.sum(b * s, axis=1) * self.norm
            # Faraday rotation terms mixes Q and U
            bQ = beam_powers[pair]["Q"]
            bU = beam_powers[pair]["U"]
            sQ = self.sky.stokes[1]
            sU = self.sky.stokes[2]
            vis[pair]["QU"] = np.sum(bU * sQ, axis=1) * self.norm
            vis[pair]["UQ"] = np.sum(bQ * sU, axis=1) * self.norm

        self._vis_components = vis

    def channelize(self, vis_arr, bins="narrow"):
        """
        Channelize using narrow or wide frequency bins.

        Parameters
        ----------
        vis_arr : np.ndarray
            Visibilities in the order V11, V12_real, V12_imag, V22.
            Shape (4, nfreqs).
        """
        if bins == "narrow":
            return vis_arr @ self.spec
        else:  # wide
            vis_arr = vis_arr.reshape(4, -1, self.wide_bin.size)
            return np.sum(vis_arr * self.wide_bin[None, None], axis=2)

    def run(self, channelize="narrow", RM=100, src_frac=None):
        """
        Run the simulation.

        Parameters
        ----------
        channelize : str
            If "narrow", channelize using narrow frequency bins.
            If "wide", channelize using wide frequency bins.
            If None, do not channelize.
        RM : float
            Rotation measure in rad/m^2.
        src_frac : float
            Fraction of pixels in the source. If provided, the visibilities
            are scaled by the inverse of this fraction to account for the
            fact that the source is not the entire sky.

        """
        # first do computation at a reference frequency
        # where the polarization angle is zero (23 GHz XXX)
        self.compute_vis()
        vis_arr = np.zeros((3, self.nfreq), dtype=complex)
        for i, pair in enumerate(["XX", "XY", "YY"]):
            for stokes in ["I", "Q", "U"]:
                vis_arr[i] += self._vis_components[pair][stokes]

        if channelize == "narrow":
            # this has shape equal to spec (2000)
            sim_freq = self.offset + self.center_freq
            # these are the ouput channels after  convolution (64 chans)
            off_max = self.offset[self.spec.argmax(axis=0)]
            self.freq = self.center_freq + off_max
        else:
            self.freq = np.linspace(0, 25 / 1e3, 64) + self.center_freq
            if channelize == "wide":
                sim_freq = self.freq[:, None] + self.offset[None, :]
                sim_freq = sim_freq.flatten()
            else:  # not channelizing
                sim_freq = self.freq

        # the frequency dependence is just a power law if no Faraday rotation
        if self.nfreq == 1:
            pl_factor = (sim_freq / self.ref_freq) ** (-2.5)
            vis_arr = pl_factor[None, :] * vis_arr

        # for faraday rotation, we get UQ and QU terms
        chi = pol_angle(sim_freq, RM, ref_freq=self.ref_freq)
        vis_arr_rot = np.zeros_like(vis_arr)
        for i, pair in enumerate(["XX", "XY", "YY"]):
            vis_arr_rot[i] += self._vis_components[pair]["I"]
            vis_arr_rot[i] += (
                np.cos(2 * chi) * self._vis_components[pair]["Q"]
                - np.sin(2 * chi) * self._vis_components[pair]["UQ"]
            )
            vis_arr_rot[i] += (
                np.sin(2 * chi) * self._vis_components[pair]["QU"]
                + np.cos(2 * chi) * self._vis_components[pair]["U"]
            )
        if self.nfreq == 1:
            vis_arr_rot = pl_factor[None, :] * vis_arr_rot

        # _vis are raw visibilities, vis are channelized
        self._vis = np.array(
            [vis_arr[0], np.real(vis_arr[1]), np.imag(vis_arr[1]), vis_arr[2]]
        )
        self._vis_rot = np.array(
            [
                vis_arr_rot[0],
                np.real(vis_arr_rot[1]),
                np.imag(vis_arr_rot[1]),
                vis_arr_rot[2],
            ]
        )
        if src_frac:
            self._vis /= src_frac
            self._vis_rot /= src_frac
        if channelize:
            self.vis = self.channelize(self._vis, bins=channelize)
            self.vis_rot = self.channelize(self._vis_rot, bins=channelize)
        else:
            self.vis = self._vis
            self.vis_rot = self._vis_rot
        self.stokes = vis2stokes(self.vis)
        self.stokes_rot = vis2stokes(self.vis_rot)
