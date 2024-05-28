import numpy as np
import matplotlib.pyplot as plt


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


def plot_vis(freqs, vis_arr, vis_arr_rot):
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey="row", constrained_layout=True
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
    plt.show()


def plot_stokes(freqs, stokes_arr, stokes_arr_rot):
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey="row", constrained_layout=True
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
    plt.show()


class Simulator:
    center_freq = 30  # MHz

    def __init__(self, beam, sky):
        path = (
            "/home/christian/Documents/research/lusee/faraday/data/"
            "zoom_response_4tap.txt"
        )
        spec = np.loadtxt(path)
        self.offset = spec[:, 0] / 1e3  # spacing in MHz
        spec = spec[:, 1:] / spec[:, 1:].sum(axis=0, keepdims=True)
        self.wide_bin = spec[:, 0]
        self.spec = spec[:, 1:]

        if sky and beam:
            self.norm = 2 / np.sum(np.abs(beam.beam_X) ** 2)  # XXX
            pix = sky.del_dark_pixels()
            beam.del_pix(pix)
            self.beam = beam
            self.sky = sky

    def compute_vis(self, faraday=True):
        """
        Compute visibilities.
        """
        # a,b : E_theta/E_phi, p : pixel axis, ... : frequency axis/axes
        ein = "ap, bp, ab...p"
        if faraday:
            T = self.sky.coherency_rot
        else:
            T = self.sky.coherency
        bX = self.beam.beam_X
        bY = self.beam.beam_Y
        V11 = np.einsum(ein, bX, bX.conj(), T)
        V22 = np.einsum(ein, bY, bY.conj(), T)
        V12 = np.einsum(ein, bX, bY.conj(), T)
        return np.real([V11, V12.real, V12.imag, V22]) * self.norm

    def channelize(self, vis_arr):
        """
        Channelize using narrow frequency bins.

        Parameters
        ----------
        vis_arr : np.ndarray
            Visibilities in the order V11, V12_real, V12_imag, V22.
            Shape (4, nfreqs).
        """
        return vis_arr @ self.spec

    def channelize_wide(self, freqs, vis_arr):
        raise NotImplementedError

    def run(self, channelize="narrow"):
        """
        Run the simulation.

        Parameters
        ----------
        channelize : str
            If "narrow", channelize using narrow frequency bins.
            If "wide", channelize using wide frequency bins.
            If None, do not channelize.

        """
        if channelize == "narrow":
            # this has shape equal to spec (2000)
            sim_freq = self.offset + self.center_freq

            # these are the ouput channels after  convolution (64 chans)
            off_max = self.offset[self.spec.argmax(axis=0)]
            self.freq = self.center_freq + off_max
        elif channelize == "wide":
            pass
        else:
            sim_freq = np.linspace(0, 25 / 1e3, 64) + self.center_freq
            self.freq = sim_freq

        self.sky.power_law(sim_freq, -2.5)
        self.sky.apply_faraday()
        self.vis = self.compute_vis(faraday=False)
        self.vis_rot = self.compute_vis(faraday=True)
        if channelize:
            self.vis = self.channelize(self.vis)
            self.vis_rot = self.channelize(self.vis_rot)
        self.stokes = vis2stokes(self.vis)
        self.stokes_rot = vis2stokes(self.vis_rot)
