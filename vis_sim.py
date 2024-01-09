import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

DTYPE = np.float64

def interp_freqs(nchans, bin_width, fmin, res_factor=100):
    """
    Compute the frequencies to interpolate spectrometer response to.
    """
    chans = np.arange(nchans) * bin_width + fmin
    freqs = np.arange(-2*res_factor, (nchans+1)*res_factor+1)
    freqs = freqs * bin_width/res_factor
    off = freqs[:(4*res_factor+1)] - freqs[2*res_factor]
    freqs += fmin
    return off, freqs

def pl_sky(I, Q, U, freqs, beta=-2.5):
    pl = np.array((freqs / freqs.min()) ** beta, dtype=DTYPE)
    sky = np.array([I, Q, U], dtype=DTYPE)
    for _ in range(pl.ndim):  # add necessary axes to sky for broadcasting
        sky = np.expand_dims(sky, axis=1)
    sky = sky * pl[None, ..., None]
    return sky


def pol_angle(freqs, RM, ref_freq):
    c = 299792458  # m/s
    dlambda_sq = c**2 * (1 / freqs**2 - 1 / ref_freq**2)
    for _ in range(freqs.ndim):
        RM = np.expand_dims(RM, axis=0)
    chi = RM * dlambda_sq[..., None]
    return chi.astype(DTYPE)


def coherency(I, Q, U, faraday=False, chi=None):
    if faraday:
        P = Q + 1j * U
        P_rot = P * np.exp(2j * chi)
        Q_rot = P_rot.real
        U_rot = P_rot.imag
        T_rot = 1 / 2 * np.array([[I + Q_rot, U_rot], [U_rot, I - Q_rot]])
        return Q_rot.astype(DTYPE), U_rot.astype(DTYPE), T_rot.astype(DTYPE)
    else:
        T = 1 / 2 * np.array([[I + Q, U], [U, I - Q]])
        return T.astype(DTYPE)


def vis2stokes(vis_arr):
    """
    Convert simulated visibilities to Stokes parameters.

    Parameters
    -----------
    vis_arr: np.ndarray
        Visibilities in the order V11, V12_real, V12_imag, V22.
        Shape (4, nfreqs, nchans).

    Returns
    -------
    stokes : np.ndarray
        The Stokes parameters I, Q, and U (no V for now).

    """
    V11, V12_real, V12_imag, V22 = vis_arr

    pI = (V11 + V22) / 2
    pQ = (V11 - V22) / 2
    pU = V12_real
    stokes = np.array([pI, pQ, pU])
    return stokes.astype(DTYPE)


def plot_vis(freqs, vis_arr, vis_arr_rot):
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey="row", constrained_layout=True
    )
    axs[0, 0].plot(freqs, vis_arr[0], c="C0", label="V11")
    axs[0, 0].plot(freqs, vis_arr[3], ls="--", c="C1", label="V22")
    axs[0, 1].plot(freqs, vis_arr_rot[0], c="C0")
    axs[0, 1].plot(freqs, vis_arr_rot[3], c="C1", ls="--")
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
    def __init__(
        self,
        freqs,
        stokes,
        RM,
        beam,
        healpix=True,
        beta=-2.5,
        spec=None,
        **kwargs,
    ):
        self.freqs = freqs
        I, Q, U = stokes
        npix = I.size
        self.I, self.Q, self.U = pl_sky(I, Q, U, self.freqs, beta=beta)
        self.RM = RM
        self.chi = pol_angle(
            self.freqs, RM, 23e9
        )  # 23 GHz is the WMAP frequency
        self.T = coherency(self.I, self.Q, self.U, faraday=False)
        self.Q_rot, self.U_rot, self.T_rot = coherency(
            self.I, self.Q, self.U, faraday=True, chi=self.chi
        )
        self.beam1, self.beam2 = beam
        self.spec = spec

        if healpix:
            self.dOmega = np.full(npix, hp.nside2pixarea(kwargs["nside"]))
        else:
            self.dOmega = kwargs["dOmega"]

    def vis(self, faraday=True, use_spec=True):
        """
        Compute visibilities.
        """
        # a,b : E_theta/E_phi, p : pixel axis, ... : frequency axis/axes
        ein = "ap, bp, ab...p, p"
        norm = 2 / np.sum(np.abs(self.beam1) ** 2 * self.dOmega)
        # norm = 1
        if faraday:
            coherency_mat = self.T_rot
        else:
            coherency_mat = self.T
        V11 = np.einsum(
            ein, self.beam1, self.beam1.conj(), coherency_mat, self.dOmega
        )
        V22 = np.einsum(
            ein, self.beam2, self.beam2.conj(), coherency_mat, self.dOmega
        )
        V12 = np.einsum(
            ein, self.beam1, self.beam2.conj(), coherency_mat, self.dOmega
        )
        v_arr = np.real([V11, V12.real, V12.imag, V22]) * norm
        if use_spec:
            v_arr = np.sum(v_arr[:, :, None] * self.spec[None], axis=1)
        return v_arr
