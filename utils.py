from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import scipy.fft
import os

plt.style.use('default')

# ---------- CONSTANTS ---------- # ? maybe move these into the ExperimentalConstants dataclass?

fs = 50000                  # data collection rate
dt = 1/fs         
kBT = 4.11e-21              # room temp (J)

# ---------- UTILITY FUNCTIONS ----------        
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
    return tau, msdval

def calcmsd2(x, y, analysislength, msdlength, silent=False):
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
        yf = scipy.fft.fft(msddata)
        xf = scipy.fft.fftfreq(N, T)[:N//2]

        print(len(xf))
        print(len(yf))
        plt.semilogy(xf[100:3000], 2.0/N * np.abs(yf[100:3000]))
        plt.xlabel('frequency (Hz)')
        plt.title('Fourier transform of the MSD')
        plt.show()
        
    return tau, msd

def load_data(fh:str, starttime, endtime, return_power_spectrum:bool=False, chunk_dur=1.0) -> tuple[np.array, np.array, np.array, np.array]:
    """Load the data from file, concatenate it and return it as a tuple of arrays
    Optionally return the power spectrum of the data as well

    Parameters
    ----------
    fh: str
        File handle
    return_power_spectrum: bool = False 
        whether to return the power spectrum of the data or not
    chunk_dur: float = 1.0
        duration of chunk size in seconds for power spectrum

    Returns
    -------
    longdatax: np.array
        x position data
    longdatay: np.array
        y position data
    longdatasum: np.array
        sum signal
    longdatafg: np.array
        function generator data
    powerspecx: np.array
        power spectrum of x position data
    powerspecy: np.array
        power spectrum of y position data
    """
    
    file_sort = lambda s: int(s.split('_')[-1]) # slow?

    longdatax = []
    longdatay = []
    longdatasum = []
    longdatafg = []

    fhs = sorted(os.listdir(fh), key=file_sort)
    fhs = fhs[:-1] # remove the last file, as this causes errors
    count = 0
    for i, sh in enumerate(fhs):    # this is problematic, because it takes every data file; let's include a selection based on the time
        dim = openfile(fh+sh)
        datax = dim[0::4]
        datay = dim[2::4]
        datafg = dim[3::4]
        datasum = dim[1::4]
        for i in range(len(datax)):
            longdatax.append(datax[i])
            longdatay.append(datay[i])
            longdatasum.append(datasum[i])
            longdatafg.append(datafg[i])

    longt = np.arange(len(longdatax))*dt
    indexcut = (longt > starttime) * (longt < endtime)
    longdatax = np.array(longdatax)[indexcut]
    longdatay = np.array(longdatay)[indexcut]
    longdatasum = np.array(longdatasum)[indexcut]
    longdatafg = np.array(longdatafg)[indexcut]

    if return_power_spectrum:
        powerspecx = np.zeros(int(chunk_dur*fs//2+1))
        powerspecy = np.zeros(int(chunk_dur*fs//2+1))
        
        for k in range(0, (len(longdatax)), int(chunk_dur*fs)):
            if len(longdatax[k:int(k+chunk_dur*fs)]) == chunk_dur*fs:
                dataxchunk = longdatax[k:int(k+chunk_dur*fs)]
                dataychunk = longdatay[k:int(k+chunk_dur*fs)]
                fx, Pxx_den = calcpowerspec2(dataxchunk)
                powerspecx += Pxx_den
                fy, Pyy_den = calcpowerspec2(dataychunk)
                powerspecy += Pyy_den
                count += 1
        powerspecx = powerspecx/count
        powerspecy = powerspecy/count

        return np.array(longdatax), np.array(longdatay), np.array(longdatasum), np.array(longdatafg), fx, powerspecx, powerspecy

    return np.array(longdatax), np.array(longdatay), np.array(longdatasum), np.array(longdatafg)

def load_lockin_data(fh:str, folders=None) -> dict:
    """Load the data recorded from the lock-in and return it as a dictionary structured as,
    {
        [folder1_name]: {
            'x':  [x_data...],
            'y':  [y_data...],
            'r':  [r_data...],
            'fg': [fg_data...],
        },
        
        [folder2_name]:
        ...
    }
    """
    
    data_dict = {}
    
    file_sort = lambda s: int(s.split('_')[-1])
    
    if folders is None:
        folders = [f for f in os.listdir(fh) if os.path.isdir(os.path.join(fh, f))]
        
    for folder in folders:
        folder_path = os.path.join(fh, folder)
        x, y, bead_y, fg = [], [], [], []
        
        files = sorted(os.listdir(folder_path), key=file_sort)[:-1]
        
        if not files: continue # skip empty folders

        for file in files:
            dim = openfile(os.path.join(folder_path, file))
            x.extend(dim[1::4])
            y.extend(dim[2::4])
            bead_y.extend(dim[3::4])
            fg.extend(dim[0::4])

        data_dict[folder] = {
            'x':  np.array(x),
            'y':  np.array(y),
            'bead_y':  np.array(bead_y),
            'fg': np.array(fg)
            }

    return data_dict