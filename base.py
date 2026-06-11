from microrheo.utils import *

import os
from scipy.optimize import curve_fit
from scipy.interpolate import LSQUnivariateSpline

class OTDataset:
    '''Parent class for passive and active datasets, ideally should handle the calibration & the function generator calibration'''
    def __init__(self, name, fh, analysisparams, calibration_params=None):   # doesn't matter what you set calib_starttime and calib_endtime to if you have the calibration already
        """
        Parameters
        ----------
        name : str
            Human readable display name
        fh : str
            Relative file handle of the passive dataset i.e. 2600608/bead1p
        aalysisparams : dict
            Analysis configuration
        params : optional
            If provided, skips computing the Brownian power spectrum calibration and uses these parameters instead
        """
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

    def _load_lockin_data(self, fh:str, folders=None) -> dict:
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
        """Given the file handle in self, run load_data for brownian data, and return:
        Sx, Sy: (V/m) -> mapping from voltage to position
        kx, ky: (N/m) -> spring constant of the trap for x and y
        trapx, trapy: (m) -> position of the center of the trap
        """
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

    def fit_ed_spline(self, x, y, y_int, k=5, center_width=0.75):
        """Fits y = y_int + m*x + spline(x)

        Spline is fit to residuals and will vanish near x = 0 to keep the center linear.

        Parameters
        ----------
        x : np.ndarray
            Array of x-axis values
        y : np.ndarray
            Array of y-axis values
        y_int : float
            Y intercept to fit through
        knots : integer
            Number of knots on each side of of the curve to give the spline (default 5)
        center_width : float
            Width on the x-axis to fit the central linear relationship (radius)
        """

        mask = np.abs(x) < center_width # center region to estimate slope from

        m, _ = np.polyfit(x[mask], y[mask] - y_int, 1)

        resid = y - (y_int + m*x)

        # knots only outside center
        knots = np.r_[
            x.min() * 0.8 : -center_width : k * 1j, # imag number acts as step size; shorthand for np.linspace
            center_width : x.max() * 0.8 : k * 1j
        ]

        spline = LSQUnivariateSpline(x, resid, knots, k=3)

        # ? optionally we could subtract this offset to make the spline hit the trap center
        # offset = spline(0.0)

        def f(x):
            return y_int + m*x + spline(x)
        
        return f


    def calibrate_from_data(self, x, y, fg, poly_fit=(0, (-1000, -200)), lin_range=[-200e-9, 150e-9], show_plots=('lin_fit',)):
        """Calibrate detector position and function generator position from FunctionG data.
        
        Parameters
        --------
        x: ndarray 
            RAW x voltages
        y: ndarray
            RAW y voltages
        fg: ndarray
            RAW function generator voltages
        poly_fit: tuple
            (knots, (lower_bound_nm, upper_bound_nm))
            
            knots is the amount of knots per side desired on the spline fit.
            knots = 0 disables nonlinear (entire detector) fitting
        lin_range: list
            range of y values to fit the linear trap calibration region, in meters
        show_plots: iterable
            may include:
                'position': plots raw position and function generator data
                'signal_jumps': plots the jumps in the function generator signal
                'lin_fit': plots the linear fit of the function generator calibration (default)

        Returns
        --------
        a: float
            linear fit parameter a
        b: float
            linear fit parameter b
        coeffs: array
            polynomial fit coefficients
        """

        plots = set(show_plots) # position, signal_jumps, lin_fit

        # normalize position data into meters
        x = np.asarray(x) / self.Sx
        y = np.asarray(y) / self.Sy
        fg = np.asarray(fg)

        t = np.arange(len(x)) * dt
        
        # plot raw position vs fg
        if 'position' in plots:
            fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, tight_layout=True)
            fig.suptitle('Bead Position vs Function Generator')
            axs[0].plot(t, y * 1e9, 'k')
            axs[0].set_ylabel('Position (nm)')
            axs[0].set_title('Bead Position')
            
            axs[1].plot(t, fg, 'k')
            axs[1].set_ylabel('FG Voltage (V)')
            axs[1].set_title('Function Generator')
            axs[1].set_xlabel('Time (s)')
            plt.show()
            
        # find plateau boundaries
        abs_diff = np.abs(np.diff(fg))
        threshold = np.mean(abs_diff) + 3*np.std(abs_diff)

        jumpy_idxs = np.where(abs_diff > threshold)[0]
        jumpy_idxs = jumpy_idxs[np.insert(np.diff(jumpy_idxs) > 1, 0, True)] # remove consecutive indicies

        offset_start = int(fs * .75) # start at 0.75 seconds
        offset_stop = int(fs * .95) # end at 0.95 seconds

        if 'signal_jumps' in plots:
            fig, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True, tight_layout=True)
            fig.suptitle('Function Generator Signal Jumps')
            axs[0].plot(t[1:], 1e9 * y[1:], 'k.')
            [axs[0].plot(t[1:][start+offset_start:start+offset_stop], 1e9 * y[1:][start+offset_start:start+offset_stop], 'gs', alpha=0.25) for start in jumpy_idxs[:-1]]
            axs[0].plot(t[1:][jumpy_idxs], 1e9 * y[1:][jumpy_idxs], 'r.', ms=8)
            axs[0].set_ylabel('Bead Position (nm)')
            axs[0].set_title('Detected Jumps')

            axs[1].plot(t[1:], fg[1:], 'k')
            axs[1].plot(t[1:][jumpy_idxs], fg[1:][jumpy_idxs], 'r.', ms=8)
            axs[1].set_ylabel('FG Voltage (V)')

            axs[2].plot(t[1:], abs_diff, 'k.', ms=8)
            axs[2].axhline(threshold, color='g', label='Threshold')
            axs[2].set_ylabel('Signal Difference (V)')
            axs[2].set_xlabel('Time (s)')
            axs[2].legend()
            axs[2].set_title('Absolute Differences')
            plt.show()
        
        # compute average fg and bead position from each stable region
        avg_fg = []
        avg_y = []
        std_fg = []

        for start in jumpy_idxs[:-1]:
            if start + offset_stop >= len(fg):
                continue

            std_fg.append(np.std(np.abs(fg[start + offset_start : start + offset_stop])))
            avg_fg.append(np.mean(fg[start + offset_start : start + offset_stop]))
            avg_y.append(np.mean(y[start + offset_start : start + offset_stop]))

        avg_fg = np.array(avg_fg)
        avg_y = np.array(avg_y)
        std_fg = np.array(std_fg)

        # ensure the fg was stable in each region (i.e. not in the middle of a jump)
        stable_mask = std_fg < (1.1 * np.mean(std_fg))

        avg_fg = avg_fg[stable_mask]
        avg_y = avg_y[stable_mask]

        # linear trap calibration (convert from volts to meters)
        upper, lower = lin_range[1] + self.trapy, lin_range[0] + self.trapy
        linear_region = (avg_y > lower) & (avg_y < upper)

        if np.sum(linear_region) < 2:
            raise ValueError(f' {np.sum(linear_region)} points is not enough points in the linear region for calibration. Please adjust lin_range or check the data.')

        a, b = np.polyfit(avg_fg[linear_region], 1e9 * avg_y[linear_region], deg=1)
        
        if 'lin_fit' in plots:
            plt.figure()
            plt.plot(avg_fg, 1e9 * avg_y, 'k.', label='Calibration Points')
            plt.plot(avg_fg[linear_region], 1e9 * avg_y[linear_region], 'go', alpha=0.25, label='Linear Region')

            fg_fit = np.linspace(min(avg_fg), max(avg_fg), 100)
            plt.plot(fg_fit, a * fg_fit + b, 'r-', label='Linear Fit')
            plt.plot(0, 1e9 * self.trapy, 'pb', label='Trap Center')
            plt.xlabel('FG Voltage (V)')
            plt.ylabel('Bead Position (nm)')
            plt.legend()
            plt.title('Linear Fit Calibration')
            plt.show()
                    
        # optional higher degree fitting (entire detector)
        if poly_fit[0] > 0:
            fit_min, fit_max = poly_fit[1]

            bead_pos_nm = a * avg_fg + b
            detector_v = avg_y * self.Sy
            x_offset = np.mean(detector_v)

            detector_v -= x_offset
            
            mask = (bead_pos_nm > fit_min) & (bead_pos_nm < fit_max)

            sort_idx = np.argsort(detector_v[mask])
            X = detector_v[mask][sort_idx]
            Y = bead_pos_nm[mask][sort_idx]

            dx = np.diff(X)
            if np.any(dx < 0):
                plt.figure()
                plt.plot(detector_v, bead_pos_nm, 'k.')
                plt.plot(detector_v[mask], bead_pos_nm[mask], 'go', alpha=0.25)
                violating_idxs = np.where(dx < 0)[0]
                plt.plot(X[violating_idxs], Y[violating_idxs], 'ro', alpha=0.75)
                plt.xlabel('Detector Voltage (V)')
                plt.ylabel('Bead Position (nm)')
                plt.title(f'Polynomial Fit')
                plt.show()

                print("Selected poly fit region is not monotonic in detector voltage.")

            X, uniq_idx = np.unique(X, return_index=True)
            Y = Y[uniq_idx]

            ed_fit_func = self.fit_ed_spline(X, Y, 1e9 * self.trapy, k=poly_fit[0])
        
            plt.figure()
            plt.plot(detector_v, bead_pos_nm, 'k.')
            plt.plot(detector_v[mask], bead_pos_nm[mask], 'go', alpha=0.25)
            plt.plot(0, 1e9 * self.trapy, 'pb', label='Trap Center')

            xfit = np.linspace(X.min(), X.max(), 1000)
            plt.plot(xfit, ed_fit_func(xfit), 'r-', lw=2)

            plt.xlabel('Detector Voltage (V)')
            plt.ylabel('Bead Position (nm)')
            plt.title(f'Spline Fit (Knots = {poly_fit[0]})')
            plt.grid(True)
            plt.legend()
            plt.show()
            
        return a, b, ed_fit_func, x_offset
               
    def calibrate_function_gen(self, poly_fit=(0, (-1000, -200)), lin_range=(-200,150), show_plots=['lin_fit']):
        '''Calibrate the function generator

        Parameters
        --------
        poly_fit: tuple
            (knots, (lower_bound_nm, upper_bound_nm))
            
            knots the amount of knots per side desired on the spline fit.
            knots = 0 disables nonlinear (entire detector) fitting
        lin_range: tuple
            range of y values to fit the linear trap calibration region, in nanometers
        show_plots: iterable
            may include:
                'position': plots raw position and function generator data
                'signal_jumps': plots the jumps in the function generator signal
                'lin_fit': plots the linear fit of the function generator calibration (default)

        Returns
        -------
            None, but sets self.a and self.b to the calibration parameters, which is needed for the analysis
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
        
        a, b, self.ed_fit_func, self.x_offset = self.calibrate_from_data(x, y, fg, poly_fit=poly_fit, lin_range=lin_range, show_plots=show_plots)
        
        self.a, self.b = a, b
        
        if poly_fit[0] > 0: # attempting a higher order (entire detector) fit
            try:
                # self.yBACKUP = self.y
                self.y = 1e-9 * self.ed_fit_func((self.y * self.Sy) - self.x_offset)

            except AttributeError: # self.y does not exist (most likely a lockin dataset)
                self.rBACKUP = {}
                for folder, data in self.data_dict.items():
                    # self.rBACKUP[folder] = data['r']
                    data['r'] = 1e-9 * self.ed_fit_func((data['r'] * self.Sy))
        
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