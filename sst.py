import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
from numpy import quantile

def sst(x, Fs, hlength=None, hop=1, n=None, hf=np.inf, lf=0, ths=1):
% Computes the synchrosqueezing transform of the signal x.
% INPUT
%    x          :  Signal (x should be a column vector).
%    Fs         :  Sampling rate of x.
%    hlength    :  Window length (in samples).
%    hop        :  Calculate the fft every hop samples, starting at 1.
%    n          :  Number of pixels in the frequency axis.
%    lf         :  Crop output to display only frequencies larger than lf.
%    hf         :  Crop output to display only frequencies less than hf.
%    ths        :  Fraction of values to reassign. 
% OUTPUT
%    sst        :  The SST of the signal x. 
%    tfr        :  The STFT of the signal x.
%    frequency  :  The frequency axis for output.
% Adapted from the MATLAB implementation by John Malik (jrvmalik) into Python by Shrikant Chand (shrikant.chand@duke.edu) and Po-Ying Chen in 04.2024.

    if hlength is None:
        raise ValueError("Select a window length.")
    if Fs is None:
        raise ValueError("Select a sampling rate.")

    # Window bandwidth
    sigma = 0.15

    # Do reassignment
    squeeze_flag = True

    # Organize input
    x = np.array(x).flatten()
    if np.any(np.isnan(x)):
        x = interp1d(np.where(~np.isnan(x))[0], x[~np.isnan(x)], kind='pchip', fill_value="extrapolate")(np.arange(len(x)))

    # Time (samples)
    NN = len(x)
    t = np.arange(1, NN + 1, hop)
    tcol = len(t)

    # N-point fft
    n = int(n)
    n = n + 1 - n % 2
    N = 2 * (n - 1)
    N = int(N)

    # Ensure window length is odd
    hlength = hlength + 1 - hlength % 2
    Lh = (hlength - 1) // 2

    # Gaussian window and its derivative
    ex = np.linspace(-0.5, 0.5, hlength)
    h = np.exp(-ex**2 / (2 * sigma**2))
    dh = -ex / sigma**2 * h

    # Perform convolution
    tfr = np.zeros((N, tcol))
    tfr2 = np.zeros((N, tcol))

    for icol in range(tcol):
        ti=t[icol]
        tau = np.arange(-min(n - 1, Lh, ti - 1), min(n - 1, Lh, NN - ti) + 1)
        indices = np.remainder(N + tau, N)
        rSig = x[ti - 1 + tau]
        tfr[indices, icol] = rSig * h[Lh + tau]
        tfr2[indices, icol] = rSig * dh[Lh + tau]

    # Fourier transform
    tfr = fft(tfr, axis=0)
    tfr = tfr[:n, :]

    # Restrict to non-negative frequency axis
    frequency = Fs / 2 * np.linspace(0, 1, n)

    if squeeze_flag:
        tfr2 = fft(tfr2, axis=0)
        tfr2 = tfr2[:n, :]

    # Crop output
    u = frequency <= hf
    tfr = tfr[u, :]
    frequency = frequency[u]

    if not squeeze_flag:
        return tfr, tfr, frequency

    # Crop
    tfr2 = tfr2[u, :]

    # Reassignment threshold 
    ths = quantile(np.abs(tfr), (1 - ths),axis=0)

    # Omega operator
    neta = len(frequency)
    omega = -np.inf * np.ones((neta, tcol))
    ups = np.abs(tfr) > ths
    omega[ups] = np.round(N / hlength * np.imag(tfr2[ups] / tfr[ups] / (2 * np.pi)))

    # Mapped out of range
    index = np.tile(np.arange(1, neta + 1).reshape(-1, 1), (1, tcol))
    omega = index - omega
    id = (omega < 1) | (omega > neta) | ~ups
    omega[id] = index[id]
    sst = tfr
    sst[id] = 0

    # Reassignment
    id = omega + neta * np.arange(tcol)
    id_flat = np.transpose(id.ravel())
    sst_flat = np.transpose(sst.ravel())

    # Need real part and imaginary part also
    temp_real = np.bincount(id_flat.astype(int), np.real(sst_flat), minlength=tcol * neta)
    temp_imag = np.bincount(id_flat.astype(int), np.imag(sst_flat), minlength=tcol * neta)

    # Reshape the result to match the desired shape [neta, tcol]
    sst = np.zeros((neta,tcol),dtype=np.complex_)


    if temp_real.size == tcol*neta:
        sst = sst + np.transpose(np.reshape(temp_real[:], (tcol, neta)))
        sst = sst + np.transpose(np.reshape(temp_imag[:], (tcol, neta)))*(0 + 1j)
    else:
        sst = sst + np.transpose(np.reshape(temp_real[:-1], (tcol, neta)))
        sst = sst + np.transpose(np.reshape(temp_imag[:-1], (tcol, neta)))*(0 + 1j)

    # Crop
    # tfr[frequency < lf, :] = 0
    # sst[frequency < lf, :] = 0
    frequency = frequency[frequency >= lf]

    return sst, tfr, frequency
