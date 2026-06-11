from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import scipy

from dataclasses import dataclass, field

plt.style.use('default')

# ---------- CONSTANTS ----------

fs = 50000                  # data collection rate
dt = 1/fs         
kBT = 4.11e-21              # room temp (J)

# viscosity presets Pa s
PRESET_ETA = {
    "water": 0.001,
    "bsa":   0.002548,
    "iodixinol20": 0.001856,
}

# helper class to store analysis parameters
@dataclass
class CalibrationParams:
    starttime: int = 0
    endtime: int = 120
    startfreqx: int = 20
    endfreqx: int = 5000
    startfreqy: int = 20
    endfreqy: int = 5000
    fg_starttime: int = 0
    fg_endtime: int = 200
    chunk_dur: float = 1.0

@dataclass
class AnalysisParams:
    calib: CalibrationParams = field(default_factory=CalibrationParams)
    data_starttime: int = 0
    data_endtime: int = 300
    alphaincrement: float = 0.001

# ---------- UTILITY FUNCTIONS ----------

class ExperimentalConstants:
    
    eta = None
    D = None            # diffusion constant
    beta = None
    r = None            # bead radius (m)
    
    @classmethod
    def init(cls, arg='iodixinol20', bead_dia=2.0e-6):
        """Initialize constants; call once per experiment."""
        if isinstance(arg, str):
            if arg not in PRESET_ETA:
                raise ValueError(f'Preset \'{arg}\' not recognized. Choose from {list(PRESET_ETA)} or custom float')
            cls.eta = PRESET_ETA[arg]
        elif isinstance(arg, float):
            cls.eta = arg
        else:
            raise ValueError(f'Input \'{arg}\' not recognized.')
        
        cls.r = bead_dia / 2
        cls.D = kBT /(6 * np.pi * cls.eta * cls.r)
        cls.beta = 6 * np.pi * cls.eta * cls.r
        
def rw(data, window_size):
  df = pd.DataFrame({'value': data})
  rollingwindow = np.array((df['value'].rolling(window=window_size).mean()))
  #ewma_result = df.ewm(span=window_size, adjust=False).mean()
  data = np.array(rollingwindow)
  return data

def openfile(filename):   # open files
        fid = open(filename, 'rb')
        dim = np.fromfile(fid, dtype='>d') # big-endian (google it lol) decimal
        fid.close()
        return dim

def calcpowerspec2(data):    # Calculate power spectrum
    f, Pxx_den = signal.welch(data, fs,  nperseg=int(len(data)), window='hamming')
    return f, Pxx_den

def func(x, a, b):    # fitting function for the PSD
    return a**2*(ExperimentalConstants.D/((np.pi)**2)) / ( (b/(2*np.pi*ExperimentalConstants.beta))**2 + x**2) # from eqn 10 Berg-Sorensen et al. 2004
       # a - Sensitivity - units of V/m
       # b - Stiffness - N/m

def autocorrFFT(x):     # used by the program that calculates the meansquared displacement
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A

def msd_fft(x, y):      # calculate the mean squared displacement
    # Organize the data into a 2-column array where each row contains an x and y position
    r = np.ones((int(len(x)), 2))
    for i in range(0,int(len(x))):
        values = [x[i], y[i]]
        r[i] =  values

    # calculate the MSD using the FFT
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    msdval = S1-2*S2
    tau = np.multiply(dt, [i for i in range(1, len(msdval)+1)])
    msdvalues = pd.DataFrame({'tau': tau, 'msd': msdval})
    return msdvalues

def calcmsd2(x, y, analysislength, msdlength, silent=False):
    # Select data to analyze MSD and fourier transform
    xdata = x[0:int(analysislength/dt)]
    ydata = y[0:int(analysislength/dt)]

    # sos = signal.butter(10, 1, 'hp', fs=50000, output='sos')
    # xdata = signal.sosfilt(sos, xdata)
    # ydata = signal.sosfilt(sos, ydata)

    msdvalues = msd_fft(xdata, ydata)
    tautemp = msdvalues.tau
    msdtemp = msdvalues.msd
    msdvalues.tau = tautemp[0:int(msdlength/dt)]
    msdvalues.msd = msdtemp[0:int(msdlength/dt)]

    if not silent:
        plt.figure()
        plt.loglog(msdvalues.tau[1:int(msdlength/(dt))], msdvalues.msd[1:int(msdlength/(dt))], 'b.', label = 'Bead in the Optical Trap')
        plt.xlabel('tau (s)')
        plt.ylabel('MSD (m^2)')
        plt.title('calcmsd2 Plot: MSD vs tau')
        plt.figure()
        msddata = np.array(msdvalues.msd[1:int(msdlength/dt)]) / (1e18)
        N = len(msddata)
        T = 1/fs
        yf = scipy.fft.fft(msddata)
        xf = scipy.fft.fftfreq(N, T)[:N//2]

        print(len(xf))
        print(len(yf))
        plt.semilogy(xf[100:3000], 2.0/N * np.abs(yf[100:3000]))
        plt.xlabel('frequency (Hz)')
        plt.title('Fourier transform of the MSD')
        plt.show()
        
    return msdvalues

def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed