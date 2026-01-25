from microrheo.utils import *

import os
from scipy.optimize import curve_fit

class OTDataset:
    '''Parent class for passive and active datasets, ideally should handle the calibration & the function generator calibration'''
    def __init__(self, name, fh, analysisparams, calibration_params=None):   # doesn't matter what you set calib_starttime and calib_endtime to if you have the calibration already
        '''
        Args:
        name: display name
        fh: file handle for the dataset !!NO TAILING SLASH!! ex: 'My Drive/Altman Lab Articles and Protocols/UPLOADED DATA/name/date/bead1'
        optional:
        calibration_params: tuple of floats, (Sx, Sy, kx, ky, trapx, trapy) -> calibration parameters for the trap
        '''
        self.name = name
        self.fh = fh
        if calibration_params is None or not calibration_params:
        # if no calibration parameters are given, calibrate the trap from the brownian data
            self.Sx, self.Sy, self.kx, self.ky, self.trapx, self.trapy = self._brownian_analysis(analysisparams, visualize=True)
            print(f'[{self.Sx}, {self.Sy}, {self.kx}, {self.ky}, {self.trapx}, {self.trapy}]')
            
        else:
            self.Sx, self.Sy, self.kx, self.ky, self.trapx, self.trapy = calibration_params        
            print('Calibration parameters:')
            print(' Sx:   ', self.Sx)
            print(' Sy:   ', self.Sy)
            print(' kx:   ', self.kx)
            print(' ky:   ', self.ky)
            print(' trapx:', self.trapx)
            print(' trapy:', self.trapy)
            print('')

    def free_memory(self):
        '''
        Free up memory when done with the analysis
        WARNING: this will break any references to the raw data
        '''
        self.x = None
        self.y = None
        self.sumsignal = None
        self.fg = None

    def _load_data(self, fh:str, starttime, endtime, return_power_spectrum:bool=False, chunk_dur=1.0) -> tuple[np.array, np.array, np.array, np.array]:

        # I had to add sections that allow us to choose a chunk of the data (this is the starttime and endtime)

        '''Load the data from file, concatenate it and return it as a tuple of arrays
        Optionally return the power spectrum of the data as well

        Args:
        fh: file handle
        return_power_spectrum: bool = False, whether to return the power spectrum of the data or not
        chunk_dur: float = 1.0, duration of chunk size in seconds for power spectrum

        Returns:
        longdatax: np.array, x position data
        longdatay: np.array, y position data
        longdatasum: np.array, sum signal
        longdatafg: np.array, function generator data
        optional:
        powerspecx: np.array, power spectrum of x position data
        powerspecy: np.array, power spectrum of y position data
        '''
        
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

    def _load_lockin_data(self, fh:str, folders=None) -> dict:
        '''Load the data recorded from the lock-in and return it as a dictionary structured as,
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
        '''
        
        data_dict = {}
        
        file_sort = lambda s: int(s.split('_')[-1])
        
        if folders is None:
            folders = [f for f in os.listdir(fh) if os.path.isdir(os.path.join(fh, f))]
            
        for folder in folders:
            folder_path = os.path.join(fh, folder)
            x, y, r, fg = [], [], [], []
            
            files = sorted(os.listdir(folder_path), key=file_sort)[:-1]
            
            if not files: continue # skip empty folders

            for file in files:
                dim = openfile(os.path.join(folder_path, file))
                x.extend(dim[1::4])
                y.extend(dim[2::4])
                r.extend(dim[3::4] / self.Sy * np.sqrt(2)) # RMS * sqrt(2) = peak
                fg.extend(dim[0::4])

            data_dict[folder] = {
                'x':  np.array(x),
                'y':  np.array(y),
                'r':  np.array(r),
                'fg': np.array(fg)
                }

        return data_dict

    def _brownian_analysis(self, analysisparams, visualize=False):
        '''Given the file handle in self, run load_data for brownian data, and return:
        Sx, Sy: (V/m) -> mapping from voltage to position
        kx, ky: (N/m) -> spring constant of the trap for x and y
        trapx, trapy: (m) -> position of the center of the trap
        '''
        fh = self.fh[:self.fh.rfind('/')+1]
        ls_dir = os.listdir(fh)
        if 'brownian' in ls_dir:
            fh += 'brownian/'
        elif 'Brownian' in ls_dir:
            fh += 'Brownian/'
        else:
            raise ValueError('No brownian data found in the folder')

        xb, yb, sumsignalb, fg, freqs, powerspecx, powerspecy = self._load_data(fh, analysisparams.calib.starttime, analysisparams.calib.endtime,
                                                                                return_power_spectrum=True, chunk_dur=analysisparams.calib.chunk_dur)
                
        # x, loop forward and backwards for efficiency
        initialx, finalx = 0, 0 # indicies for x fit
        for i, freq in enumerate(freqs):
            if freq >= analysisparams.calib.startfreqx:
                initialx = i
                break
        
        for i, freq in reversed(list(enumerate(freqs))):
            if freq <= analysisparams.calib.endfreqx:
                finalx = i
                break
            
        # y, loop forward and backwards for efficiency
        initialy, finaly = 0, 0 # indicies for y fit
        for i, freq in enumerate(freqs):
            if freq >= analysisparams.calib.startfreqy:
                initialy = i
                break
        
        for i, freq in reversed(list(enumerate(freqs))):
            if freq <= analysisparams.calib.endfreqy:
                finaly = i
                break

        # Fit the power spectrum to get the trap stiffness
        # x
        poptx, pcov = curve_fit(func, freqs[initialx:finalx], powerspecx[initialx:finalx], p0=[1e5, 1e-3])
        alphax = abs(poptx[1])

        # y
        popty, pcov = curve_fit(func, freqs[initialy:finaly], powerspecy[initialy:finaly], p0=[1e5, 1e-3])
        alphay = abs(popty[1])
        
        if visualize:
            fig = plt.figure(figsize=(10, 10), tight_layout=True)
            fig.suptitle('OT Calibration')
            xplot = fig.add_subplot(221)
            xplot.set_title('Power Spectrum of X')
            xplot.loglog(freqs, powerspecx, 'k.')
            xplot.loglog(freqs[initialx:finalx], func(freqs[initialx:finalx], *poptx), 'r-')
            xplot.set_xlabel('Frequency [Hz]')
            xplot.set_ylabel('Power Spectrum [$V^2$/Hz]')

            yplot = fig.add_subplot(222)
            yplot.set_title('Power Spectrum of Y')
            yplot.loglog(freqs, powerspecy, 'k.')
            yplot.loglog(freqs[initialy:finaly], func(freqs[initialy:finaly], *popty), 'r-')
            yplot.set_xlabel('Frequency [Hz]')
            yplot.set_ylabel('Power Spectrum [$V^2$/Hz]')

            fig2=plt.figure(figsize=(10, 5))
            fig2.suptitle('Calibration Data')
            t = np.arange(len(xb))*dt
            xpos=fig2.add_subplot(311)
            xpos.plot(t,xb,'k')
            xpos.set_ylabel('x [V]')
            ypos=fig2.add_subplot(312)
            ypos.plot(t,yb,'k')
            ypos.set_ylabel('y [V]')
            ypos.set_xlabel('Time [s]')
            sumsignal=fig2.add_subplot(313)
            sumsignal.plot(t,sumsignalb,'k')
            sumsignal.set_ylabel('sum [V]')
            sumsignal.set_xlabel('Time [s]')
            plt.plot()

        Sx=poptx[0]
        Sy=popty[0]
        return Sx, Sy, alphax, alphay, np.mean(xb)/Sx, np.mean(yb)/Sy


    def see_calibration(self):
        '''See the calibration of the trap for the dataset'''
        pass

    def calibrate_from_data(self, x, y, fg, poly_fit=(1, (-1000, -200)), lin_range=[-200*1e-9,150*1e-9], show_plots=('lin_fit',)):
        '''Core function generator calibration logic
        
        Args:
        x: RAW x voltages
        y: RAW y voltages
        fg: RAW function generator voltages
        poly_fit: degree of polynomial to fit, default is 1 (linear fit), (range of bead positions to fit)
        lin_range: range of y values to fit the linear region, default is [-200nm, 150nm]
        show_plots: list of plots to show, default has only the 'lin_fit'
        
        Returns:
        a: linear fit parameter a
        b: linear fit parameter b
        coeffs: polynomial fit coefficients
        new_y: maybe
        '''
        plots = {'position': False, 'signal_jumps': False, 'lin_fit': False}
        for plot_name in show_plots:
            plots[plot_name] = True
        
        # normalize position data into meters
        x /= self.Sx
        y /= self.Sy
        t = np.arange(len(x)) * dt
        
        # plot raw position vs fg
        if plots['position']:
            fig, axs = plt.subplots(2, 1, figsize=(10, 5), tight_layout=True)
            fig.suptitle('Bead Position vs Function Generator')
            axs[0].plot(t, y*1e9, 'k')
            axs[0].set_ylabel('Position (nm)')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_title('Bead Position (nm)')
            
            axs[1].plot(t, fg, 'k')
            axs[1].set_title('Function Generator Signal (V)')
            axs[1].set_ylabel('Signal (V)')
            
        # plot the function generator signal jumps
        abs_diff = np.abs(np.diff(fg))
        jump_y_idxs = abs_diff > (np.mean(abs_diff) * 100)
        
        if plots['signal_jumps']:
            fig, axs = plt.subplots(2, 1, figsize=(10, 5), tight_layout=True)
            fig.suptitle('Function Generator Signal Jumps')
            axs[0].plot(t[1:], fg[1:], 'k')
            axs[0].plot(t[1:][jump_y_idxs], fg[1:][jump_y_idxs], 'r.')
            axs[0].set_ylabel('Signal (V)')
            axs[0].set_title('Highlighted Jumps in Signal')
            plt.xlabel('Time (s)')

            axs[1].plot(t[1:], abs_diff, 'k,')
            axs[1].set_ylabel('Signal Difference (V)')
            axs[1].set_title('Absolute Differences')
            
        # reduce signal jumps to a single point i.e. throw out everything but the end of the jump
        # also convert from boolean indexing to integer indexing
        idx = []
        prev: bool = jump_y_idxs[0]
        for i, current in enumerate(jump_y_idxs):
            if not current and prev: # last point was a jump, and this is not a jump
                idx.append(i-1) # marking the end of jumps for exclusive indexing
            prev = current
            
        # compile the standard deviation of the signal in the stable regions
        stdfg = []
        for i in range(len(idx) - 1):
            mask = (t[idx[i]] < t) & (t < t[idx[i+1]])
            stdfg.append(np.std(np.abs(fg[mask]))) # why do we abs value first?
        stdfg = np.array(stdfg)
        threshold = 1.1 * np.mean(stdfg) # setting the threshold to the mean of the std of the signal

        # get the avg fg and y in each stable region
        avg_y, avg_fg = [], []
        for i in range(len(stdfg) - 1):
            if stdfg[i] < threshold: # ensure the fg is stable (i.e. we're not in a jump)
                mask = (t[idx[i] + int(fs/2 * .75)] < t) * (t < t[idx[i] + int(fs/2 * .95)]) #getting the last 0.2 seconds of the signal
                avg_fg.append(np.mean(fg[mask]))
                avg_y.append(np.mean(y[mask]))

        avg_y, avg_fg = np.array(avg_y), np.array(avg_fg)
        
        # fit the avg fg to the avg y in our linear region
        upper, lower = lin_range[1] + self.trapy, lin_range[0] + self.trapy
        inrange = (avg_y > lower) & (avg_y < upper)
        a, b = np.polyfit(avg_fg[inrange], 1e9 * avg_y[inrange], 1)
        
        if plots['lin_fit']:
            plt.figure()
            plt.plot(avg_fg, 1e9 * avg_y, 'k.')
            plt.plot(avg_fg[inrange], 1e9 * avg_y[inrange], 'r.', alpha=0.25)
            plt.plot(avg_fg[inrange], a * avg_fg[inrange] + b, 'g*')
            plt.plot(0, 1e9 * self.trapy, 'pb', label='Trap Center')
            plt.legend()
            
        coeffs, new_y = None, None
        
        # optional higher degree fitting
        if poly_fit[0] > 1:
            bead_pos = a * avg_fg + b
            detector_V = avg_y * self.Sy
            
            mask = (bead_pos > poly_fit[1][0]) & (bead_pos < poly_fit[1][1])
            coeffs = np.polyfit(detector_V[mask], bead_pos[mask], poly_fit[0])
        
            plt.figure()
            plt.plot(detector_V, bead_pos, 'k.')
            plt.plot(detector_V[mask], bead_pos[mask], 'gs', alpha=0.25)
            plt.plot(detector_V, np.polyval(coeffs, detector_V), 'r.')
            plt.xlabel('Detector Voltage (V)')
            plt.ylabel('Bead Position (nm)')     
            
        return a, b, coeffs
               
    def calibrate_function_gen(self, poly_fit=(1, (-1000, -200)), lin_range=[-200*1e-9,150*1e-9], show_plots=['lin_fit']):
        '''Calibrate the function generator
        Args:
        poly_fit: degree of polynomial to fit, default is 1 (linear fit), (range of bead positions to fit)
        lin_range: range of y values to fit the linear region, default is [-200nm, 150nm]
        show_plots: list of plots to show, default has only the 'lin_fit'

        Returns:
        None, but sets self.a and self.b to the calibration parameters, which is needed for active analysis
        '''

        # Load the data
        fh = self.fh[0:len(self.fh)-1]
        fh = fh[:fh.rfind('/')+1]
        ls_dir = os.listdir(fh)
        if 'functiong' in ls_dir:
            fh += 'functiong/'
        elif 'FunctionG' in ls_dir:
            fh += 'FunctionG/'
        else:
            raise ValueError('No function generator data found in the folder')
        x, y, sumsignal, fg = self._load_data(fh, self.analysisparams.calib.fg_starttime, self.analysisparams.calib.fg_endtime)
        
        a, b, coeffs = self.calibrate_from_data(x, y, fg, poly_fit=poly_fit, lin_range=lin_range, show_plots=show_plots)
        
        self.a, self.b = a, b
        
        if poly_fit[0] > 0: # attempting a higher order (entire detector) fit
            try:
                # self.yBACKUP = self.y
                self.y = 1e-9 * np.polyval(coeffs, self.y * self.Sy)
            except AttributeError: # self.y does not exist (most likely a lockin dataset)
                self.rBACKUP = {}
                for folder, data in self.data_dict.items():
                    # self.rBACKUP[folder] = data['r']
                    data['r'] = 1e-9 * np.polyval(coeffs, data['r'] * self.Sy)
        
    def reset_y(self):
        '''Reset the y data to the original data'''
        pass
        # self.y = self.yBACKUP
        # self.yBACKUP = None
        
    def reset_r(self):
        '''Reset all the r data to the original data'''
        pass
        # for folder, data in self.data_dict.items():
        #     data['r'] = self.rBACKUP[folder]
        #     self.rBACKUP = None