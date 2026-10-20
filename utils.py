from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft, next_fast_len, rfftfreq
import os
from dataclasses import dataclass, field
import json

plt.style.use('default')

# ---------- CONSTANTS ---------- # ? maybe move these into the ExperimentalConstants dataclass?

fs = 100000                 # data collection rate
dt = 1/fs         
kBT = 4.11e-21              # room temp (J)

# ---------- UTILITY FUNCTIONS ----------        
def openfile(filename):   
    """Open files."""
    fid = open(filename, 'rb')
    dim = np.fromfile(fid, dtype='>d') # big-endian (google it lol) decimal
    fid.close()
    return dim

def calcpowerspec2(data):
    """Calculate power spectrum."""
    f, Pxx_den = signal.welch(data, fs,  nperseg=int(len(data)), window='hamming')
    return f, Pxx_den

def autocorrFFT(r):
    """Used by the program that calculates the meansquared displacement."""
    N = r.shape[0]
    fsize = next_fast_len(2 * N) # optimze the FFT length with zero padding
    F = rfft(r, n=fsize, axis=0)
    PSD = F * F.conjugate()
    res = irfft(PSD, n=fsize, axis=0)[:N].real # autocorrelation in convention B
    n = (N - np.arange(N))[:, None] # divide by (N-m), broadcast over columns
    return res / n # autocorrelation in convention A

def msd_fft(x, y):
    """Calculate the mean squared displacement."""
    # Organize the data into a 2-column array where each row contains an x and y position
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.column_stack((x, y))
    N = r.shape[0]

    # calculate the MSD using the FFT
    D = np.square(r).sum(axis=1)
    S2 = autocorrFFT(r).sum(axis=1) # sum x_autocorr + y_autocorr

    Dsum = D.sum()
    cumD = np.cumsum(D)
    m = np.arange(N)

    C1 = np.zeros(N)
    C1[1:] = cumD[:-1]

    C2 = Dsum - cumD[N - m - 1]
    
    Q = 2 * Dsum - C1 - C2
    S1 = Q / (N - m)

    msdval = S1 - 2 * S2
    tau = dt * np.arange(1, N + 1)
    return tau, msdval

def calcmsd2(x, y, analysislength, msdlength, silent=False):
    """Calculate the two dimensional mean squared displacement."""
    # Select data to analyze MSD and fourier transform
    xdata = x[0:int(analysislength/dt)]
    ydata = y[0:int(analysislength/dt)]

    # sos = signal.butter(10, 1, 'hp', fs=50000, output='sos')
    # xdata = signal.sosfilt(sos, xdata)
    # ydata = signal.sosfilt(sos, ydata)

    tau, msd = msd_fft(xdata, ydata)
    tautemp = tau
    msdtemp = msd
    tau = tautemp[0:int(msdlength/dt)]
    msd = msdtemp[0:int(msdlength/dt)]

    if not silent:
        plt.figure()
        plt.loglog(tau[1:int(msdlength/(dt))], msd[1:int(msdlength/(dt))], 'b.', label = 'Bead in the Optical Trap')
        plt.xlabel('tau (s)')
        plt.ylabel('MSD (m^2)')
        plt.title('calcmsd2 Plot: MSD vs tau')
        plt.figure()
        msddata = np.array(msd[1:int(msdlength/dt)]) / (1e18)
        N = len(msddata)
        T = 1/fs
        yf = rfft(msddata)
        xf = rfftfreq(N, T)[:N//2]

        print(len(xf))
        print(len(yf))
        plt.semilogy(xf[100:3000], 2.0/N * np.abs(yf[100:3000]))
        plt.xlabel('frequency (Hz)')
        plt.title('Fourier transform of the MSD')
        plt.show()
        
    return tau, msd

def software_demod(bead, fg, f):
    """Compute bead and FG phase/amplitude via software demodulation.
    
    Parameters
    ----------
    bead : np.ndarray
        Bead position data (meters)
    fg : np.ndarray
        Function generator position data (meters)
    f : np.ndarray
        Function generator frequency (Hz)

    Returns
    -------
    phi : float
        Phase difference between the bead and FG in radians
    trap_amp : float
        Amplitude of the trap in meters
    bead_amp : float
        Amplitude of the bead in meters
    """
    t = np.arange(len(bead)) * dt
    bead = bead - np.mean(bead)
    fg = fg - np.mean(fg)

    cos_ref = np.cos(2 * np.pi * f * t)
    sin_ref = np.sin(2 * np.pi * f * t)

    Xb, Yb = np.mean(bead * cos_ref), np.mean(bead * sin_ref)
    Xf, Yf = np.mean(fg   * cos_ref), np.mean(fg   * sin_ref)

    return np.arctan2(Yb*Xf - Xb*Yf, Xb*Xf + Yb*Yf), 2 * np.sqrt(Xf**2 + Yf**2), 2 * np.sqrt(Xb**2 + Yb**2)

def load_data(fh:str, starttime, endtime, return_power_spectrum:bool=False, chunk_dur=1.0) -> tuple[np.array, np.array, np.array, np.array]:
    """Load the data from file, concatenate it and return it as a tuple of arrays. Optionally return the power spectrum of the data as well.

    Parameters
    ----------
    fh: str
        File handle
    starttime : float
        Start time of data (in seconds)
    endtime : float
        End time of data (in seconds)
    return_power_spectrum: bool = False 
        If True, return the power spectrum of the data
    chunk_dur: float = 1.0
        Duration of chunk size (in seconds) for the power spectrum

    Returns
    -------
    ch1: np.ndarray
        First data stream channel
    cb2: np.ndarray
        Second data stream channel
    f : np.ndarray
        Frequencies for the power spectrum
    powerspecx: np.ndarray
        Power spectrum of x position data
    powerspecy: np.ndarray
        Power spectrum of y position data
    """
    
    file_sort = lambda s: int(s.split('_')[-1]) # slow?

    channel1 = []
    channel2 = []

    fhs = sorted(os.listdir(fh), key=file_sort)
    fhs = fhs[:-1] # remove the last file, as this causes errors

    count = 0
    for i, sh in enumerate(fhs): # this is problematic, because it takes every data file; let's include a selection based on the time
        dim = openfile(fh+sh)
        ch1 = dim[0::2]
        ch2 = dim[1::2]
        for i in range(len(ch1)):
            channel1.append(ch1[i])
            channel2.append(ch2[i])

    t = np.arange(len(channel1))*dt
    idxs = (t > starttime) & (t < endtime)
    channel1 = np.asarray(channel1)[idxs]
    channel2 = np.asarray(channel2)[idxs]

    if return_power_spectrum:
        powerspecx = np.zeros(int(chunk_dur * fs//2 + 1))
        powerspecy = np.zeros(int(chunk_dur * fs//2 + 1))
        
        for k in range(0, (len(channel1)), int(chunk_dur * fs)):
            if len(channel1[k:int(k + chunk_dur * fs)]) == chunk_dur * fs:
                dataxchunk = channel1[k:int(k + chunk_dur * fs)]
                dataychunk = channel2[k:int(k + chunk_dur * fs)]
                f, Pxx_den = calcpowerspec2(dataxchunk)
                powerspecx += Pxx_den
                _, Pyy_den = calcpowerspec2(dataychunk)
                powerspecy += Pyy_den
                count += 1
        powerspecx = powerspecx/count
        powerspecy = powerspecy/count

        return channel1, channel2, f, powerspecx, powerspecy

    return channel1, channel2