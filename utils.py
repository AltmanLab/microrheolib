from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import scipy
from dataclasses import dataclass

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
class AnalysisParams:
    calib_starttime: int = 0
    calib_endtime: int = 120
    fgcalib_starttime: int = 0
    fgcalib_endtime: int = 200
    calib_startfreq: int = 20
    calib_endfreq: int = 5000
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
    tau = np.multiply(dt, [i for i in range(0, len(msdval))])
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