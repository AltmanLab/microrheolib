from microrheo.base import OTDataset, SerializedDataset
from microrheo.utils import *

from itertools import cycle
from scipy.optimize import curve_fit
from scipy.interpolate import LSQUnivariateSpline
from datetime import datetime
import glob

# viscosity presets (Pa s)
PRESET_ETA = {
    "water": 0.001,
    "bsa": 0.002548,
    "iodixinol20": 0.001856,
}

@dataclass
class ExperimentalConstants:
    """Experimental constants."""
    eta: float                          # viscosity
    r: float                            # bead radius (m)
    D: float = field(init=False)        # diffusion constant
    beta: float = field(init=False)     # derived
    rho: float = 2000                   # density of bead (kg/m^3)
    
    def __post_init__(self):
        self.D = kBT /(6 * np.pi * self.eta * self.r)
        self.beta = 6 * np.pi * self.eta * self.r

    @classmethod
    def from_preset(cls, preset='iodixinol20', bead_dia=1.97e-6):
        if preset not in PRESET_ETA:
            raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESET_ETA)}")
        return cls(eta=PRESET_ETA[preset], r=bead_dia / 2)

@dataclass
class ExperimentalConfig:
    """Experimental (mostly calibration) parameters."""
    constants: ExperimentalConstants
    starttime: float = 0
    endtime: float = 120
    startfreqx: int = 20
    endfreqx: int = 5000
    startfreqy: int = 20
    endfreqy: int = 5000
    fg_starttime: float = 0
    fg_endtime: float = 200
    chunk_dur: float = 1.0

@dataclass
class TrapCalibration:
    """Reulting configurations from Brownian and FunctionG calibrations."""
    Sx: float
    Sy: float
    kx: float
    ky: float
    trapx: float
    trapy: float
    t_offset: float = 0.95 / fs # time offset between the FG and bead signals due to sampling delay

    # Function gen calibration (populated after calibrate())
    a: float = field(init=False)
    b: float = field(init=False)
    ed_fit_func: callable = field(init=False)
    v_offset: float = field(init=False)

    def __str__(self) -> str:
        return f"Brownian calibration parameters:\n Sx:    {self.Sx: }\n Sy:    {self.Sy: }\n kx:    {self.kx: }\n ky:    {self.ky: }\n trapx: {self.trapx: }\n trapy: {self.trapy: }\n"

class Datasets(dict):
    """Dictionary wrapper to optionally allow dot notation (e.g. '.passive2')."""
    def __getattr__(self, key):
        try: 
            return self[key]
        except KeyError:
            raise AttributeError(f"Experiment has no dataset named '{key}'")
        
    def __setattr__(self, key, value):
        self[key] = value

class Experiment:
    def __init__(self, fh: str, config: ExperimentalConfig, brownian_name: str | None=None, brownian_params: list | None=None):
        """Initialize global experiment variables and perform the Brownian calibration.
        
        Parameters
        ----------
        fh : str
            Relative path to the folder containing the data for the entire experiment
        config : ExperimentalConfig
            Parameters needed for the experiment wide calibration steps
        brownian_name : str
            Optional folder name for the Brownian data
        brownian_params : list
            Optional Brownian calibration parameters. If provided, it skips computing the Brownian calibration
        """
        self.fh = fh.removesuffix('/')
        self.config = config
        self.brownian_params = brownian_params
        self.datasets: Datasets[str, OTDataset] = Datasets({})

        if not os.path.isdir(fh):
            raise ValueError(f"No folder found at '{fh}', make sure you spelled it correctly.")
        
        # if no calibration parameters are given, calibrate the trap from the brownian data
        if brownian_params is None or not brownian_params:
            Sx, Sy, kx, ky, trapx, trapy = self._brownian_analysis(folder_name=brownian_name, show_plots=True)
            print(f'[{Sx}, {Sy}, {kx}, {ky}, {trapx}, {trapy}]')
        else:
            Sx, Sy, kx, ky, trapx, trapy = brownian_params        
            
        self.calib = TrapCalibration(Sx, Sy, kx, ky, trapx, trapy)
        print(self.calib)

    def add(self, cls: OTDataset, name: str, folder_name: str, **kwargs) -> OTDataset:
        """Add a new rheology analysis object to this experiment.
        
        Parameters
        ----------
        cls : OTDataset
            Analysis object class (ActiveAnalysis or PassiveAnalysis)
        name : str
            User display name and internal lookup name. If a valid python variable name, can be used with the dot notation (e.g. '.passive2')
        folder_name : str
            Name of folder containing the data
        **kwargs
            Aditional keyword arguments to be passed to the class constructor

        Returns
        -------
        OTDataset
        """        
        ds = cls(self.fh + '/' + folder_name.removesuffix('/'), self.config.constants, self.calib, **kwargs)
        self.datasets[name] = ds
        return ds

    def fit_ed_spline(self, x, y, y_int, k=5, center_width=0.65):
        """Fit spline to capture the nonlinear QPD detector response curve.
        
        Fits `y = y_int + m*x + spline(x)`.

        Spline is fit to residuals and will vanish near x = 0 to keep the center linear.

        Parameters
        ----------
        x : np.ndarray
            Array of x-axis values
        y : np.ndarray
            Array of y-axis values
        y_int : float
            Y intercept to fit through
        k : integer
            Number of knots on each side of of the curve to give the spline (default 5)
        center_width : float
            Half-width on the x-axis to not put any knots to keep the center linear

        Returns
        -------
        f : callable
            The entire detector fitting function
        x_knots : list
            X coordinate of the knots
        y_knots : list
            Y coordinate of the knots
        """

        m = 1e-9 / self.calib.Sy # slope of the linear region
        resid = y - (y_int + m*x)

        # knots only outside center (evenly spaced by arc length)
        dx = np.diff(x)
        dy = np.diff(y)

        arc = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
        left_arc = arc[x < -center_width]
        right_arc = arc[x > center_width]

        left_knots_arc = np.linspace(
            left_arc.min(),
            left_arc.max(),
            k+1
        )[1:]   # remove outside endpoint

        right_knots_arc = np.linspace(
            right_arc.min(),
            right_arc.max(),
            k + 1
        )[:-1]   # remove outside endpoint

        # Convert arc-length locations back to x locations
        left_knots = np.interp(left_knots_arc, arc, x)
        right_knots = np.interp(right_knots_arc, arc, x)

        knots = np.concatenate([left_knots, right_knots])

        spline = LSQUnivariateSpline(x, resid, knots, k=3)

        def f(x):
            return y_int + m*x + spline(x)
        
        x_knots = spline.get_knots()
        return f, x_knots, f(x_knots)

    def calibrate(self, folder_name=None, lin_range:tuple[float, float]=(-100, 100), spline_fit:tuple[int, tuple[float, float]]=(2, (-500, 500)), center_width=0.65, show_plots=()):
        """Function generator calibration and entire detector fitting. Sets the linear fit parameters self.a and self.b for converting the function generator signal into distance. Also computes the entire detector fit function (usage shown below).
        
        Parameters
        ----------
        folder_name : str
            Optional name of the FunctionG data. If 'None' defaults to 'FunctionG'.
        lin_range : tuple(min_nm, max_nm)
            Range of values to fit the linear trap calibration region, in nanometers relative to the trap center.
        spline_fit : tuple(knots, (lower_bound_nm, upper_bound_nm))
            Range of values to fit the entire detector to, in nanometers relative to the trap center.
            knots is the amount of knots per side desired on the spline fit.
            knots = 0 disables nonlinear (entire detector) fitting
        show_plots: iterable
            may include:
            - 'position': plots raw position and function generator data
            - 'signal_jumps': plots the jumps in the function generator signal

        For Passive/Active:
        ----
            ```y = ed_fit_func(raw_y_volts - v_offset)
            ```
        
        """
        # Load the data
        fh = self.fh
        if folder_name is None: # default folder name
            folder_name = 'FunctionG'

        folder = next(
            (name for name in os.listdir(fh)
             if os.path.isdir(os.path.join(fh, name))
             and name.lower() == folder_name.lower()),
             None,
        )

        if folder is None:
            raise ValueError('No FunctionG data found in the folder')
        
        fh = os.path.join(fh, folder) + '/'

        fg, y = load_data(fh, self.config.fg_starttime, self.config.fg_endtime)

        plots = set(show_plots) # position, signal_jumps
        
        # normalize position data into meters
        y = np.asarray(y) / self.calib.Sy
        fg = np.asarray(fg)

        t = np.arange(len(y)) * dt

        # plot raw position vs fg
        if 'position' in plots:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, tight_layout=True)
            fig.suptitle('Bead Position vs Function Generator')
            ax1.plot(t, y * 1e9, 'k')
            ax1.set_ylabel('Position (nm)')
            ax1.set_title('Bead Position')
            
            ax2.plot(t, fg, 'k')
            ax2.set_ylabel('FG Voltage (V)')
            ax2.set_title('Function Generator')
            ax2.set_xlabel('Time (s)')
            plt.show()
        
        # find plateau boundaries (i.e. where the bead has leveled off after a FG jump)
        abs_diff = np.abs(np.diff(fg))
        threshold = np.mean(abs_diff) + 5*np.std(abs_diff)

        jumpy_idxs = np.where(abs_diff > threshold)[0]
        jumpy_idxs = jumpy_idxs[np.insert(np.diff(jumpy_idxs) > 1, 0, True)] # remove consecutive indicies

        offset_start = int(fs * .75) # start at 0.75 seconds
        offset_stop = int(fs * .95) # end at 0.95 seconds

        if 'signal_jumps' in plots:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5), sharex=True, tight_layout=True)
            fig.suptitle('Function Generator Signal Jumps')
            ax1.plot(t[1:], 1e9 * y[1:], 'k.')
            [ax1.plot(t[1:][start+offset_start:start+offset_stop], 1e9 * y[1:][start+offset_start:start+offset_stop], 'gs', alpha=0.25) for start in jumpy_idxs[:-1]]
            ax1.plot(t[1:][jumpy_idxs], 1e9 * y[1:][jumpy_idxs], 'r.', ms=8)
            ax1.set_ylabel('Bead Position (nm)')
            ax1.set_title('Detected Jumps')

            ax2.plot(t[1:], fg[1:], 'k')
            ax2.plot(t[1:][jumpy_idxs], fg[1:][jumpy_idxs], 'r.', ms=8)
            ax2.set_ylabel('FG Voltage (V)')

            ax3.plot(t[1:], abs_diff, 'k.', ms=8)
            ax3.axhline(threshold, color='g', label='Threshold')
            ax3.set_ylabel('Signal Difference (V)')
            ax3.set_xlabel('Time (s)')
            ax3.legend()
            ax3.set_title('Absolute Differences')
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

        avg_fg = np.asarray(avg_fg)
        avg_y = np.asarray(avg_y)
        std_fg = np.asarray(std_fg)

        # ensure the fg was stable in each region (i.e. not in the middle of a jump)
        stable_mask = std_fg < (3 * np.mean(std_fg))

        avg_fg = avg_fg[stable_mask]
        avg_y = avg_y[stable_mask]

        # remove any points that are within 20nm of the center of the trap; these must be caused by noise
        mask = np.abs(avg_y - self.calib.trapy) > 20e-9

        avg_y = avg_y[mask]
        avg_fg = avg_fg[mask]

        # linear trap calibration (convert from volts to meters)
        upper, lower = 1e-9 * lin_range[1] + self.calib.trapy, 1e-9 * lin_range[0] + self.calib.trapy
        linear_region = (avg_y > lower) & (avg_y < upper)

        if np.sum(linear_region) < 2:
            raise ValueError(f' {np.sum(linear_region)} points is not enough points in the linear region for calibration. Please adjust lin_range or check the data.')

        self.calib.a, self.calib.b = np.polyfit(avg_fg[linear_region], avg_y[linear_region], deg=1) 

        # optional higher degree fitting (entire detector)
        if spline_fit[0] > 0:
            fit_min, fit_max = spline_fit[1]
            fit_min = 1e-9 * fit_min + self.calib.trapy
            fit_max = 1e-9 * fit_max + self.calib.trapy

            bead_pos = self.calib.a * avg_fg + self.calib.b
            detector_v = avg_y * self.calib.Sy
            self.calib.v_offset = np.mean(detector_v) # center the x-axis to simplify fitting

            detector_v -= self.calib.v_offset

            mask = (bead_pos > fit_min) & (bead_pos < fit_max)

            sort_idx = np.argsort(detector_v[mask])
            X = detector_v[mask][sort_idx]
            Y = bead_pos[mask][sort_idx]

            dx = np.diff(X)
            if np.any(dx < 0):
                plt.figure()
                plt.plot(detector_v, 1e9*bead_pos - 1e9*self.calib.trapy, 'k.')
                plt.plot(detector_v[mask], 1e9*bead_pos[mask] - 1e9*self.calib.trapy, 'go', alpha=0.25)
                violating_idxs = np.where(dx < 0)[0]
                plt.plot(X[violating_idxs], Y[violating_idxs] - 1e9*self.calib.trapy, 'ro', alpha=0.75)
                plt.xlabel('Detector Voltage (V)')
                plt.ylabel('Bead Position (nm)')
                plt.title(f'Polynomial Fit')
                plt.show()

                print("Selected poly fit region is not monotonic in detector voltage.")

            X, uniq_idx = np.unique(X, return_index=True)
            Y = Y[uniq_idx]

            idxs = np.argsort(X)[5:-5] # remove first and last 5 points to clean up noise from edges
            X = X[idxs]
            Y = Y[idxs]
            self.calib.ed_fit_func, x_knots, y_knots = self.fit_ed_spline(X, Y, self.calib.trapy, k=spline_fit[0], center_width=center_width)
        
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
            ax1.plot(avg_fg, 1e9 * avg_y - 1e9*self.calib.trapy, 'k.', label='Calibration Points')
            ax1.plot(avg_fg[linear_region], 1e9 * avg_y[linear_region] - 1e9*self.calib.trapy, 'go', alpha=0.25, label='Linear Region')

            fg_fit = np.linspace(min(avg_fg), max(avg_fg), 100)
            ax1.plot(fg_fit, 1e9 * (self.calib.a * fg_fit + self.calib.b) - 1e9*self.calib.trapy, 'r-', label='Linear Fit')
            ax1.plot(0, 0, 'pb', label='Trap Center')
            ax1.set_xlabel('FG Voltage (V)')
            ax1.set_ylabel('Bead Position (nm)')
            ax1.legend()
            ax1.set_title('Linear Fit Calibration')

            ax2.plot(detector_v + self.calib.v_offset, 1e9 * bead_pos - 1e9*self.calib.trapy, 'k.')
            ax2.plot(detector_v[mask] + self.calib.v_offset, 1e9 * bead_pos[mask] - 1e9*self.calib.trapy, 'go', alpha=0.25)
            ax2.plot(self.calib.trapy * self.calib.Sy, 0, 'pb', label='Trap Center')

            xfit = np.linspace(X.min(), X.max(), 1000) + self.calib.v_offset
            ax2.plot(xfit, 1e9 * self.calib.ed_fit_func(xfit - self.calib.v_offset) - 1e9*self.calib.trapy, 'r-', lw=2)
            ax2.plot(x_knots + self.calib.v_offset, 1e9 * y_knots - 1e9*self.calib.trapy, 'r.')
            ax2.plot(xfit, xfit * 1e9 / self.calib.Sy - 1e9*self.calib.trapy, color='gray', linestyle='--', label='Linear Fit')

            ax2.set_xlabel('Detector Voltage (V)')
            ax2.set_ylabel('Bead Position (nm)')
            ax2.set_title(f'Spline Fit (Knots = {spline_fit[0]})')
            ax2.grid(True)
            ax2.legend()
            plt.tight_layout()
            plt.show()

    def _brownian_analysis(self, folder_name=None, show_plots=False):
        """Compute the brownian analysis.

        Parameters
        ----------
        folder_name : str
            Optional folder name of Brownian data. If 'None' defaults to 'Brownian'
        show_plots : bool
            If True, plots the power spectrum and calibration data

        Returns
        -------
        Sx, Sy: (V/m)
            Mapping from voltage to position
        kx, ky: (N/m) 
            Spring constant of the trap for x and y
        trapx, trapy: (m) 
            Position of the center of the trap
        """

        fh = self.fh
        if folder_name is None:
            folder_name = 'Brownian'

        folder = next(
            (name for name in os.listdir(fh)
             if os.path.isdir(os.path.join(fh, name))
             and name.lower() == folder_name.lower()),
             None,
        )

        if folder is None:
            raise ValueError('No brownian data found in the folder')
        
        fh = os.path.join(fh, folder) + '/'

        xb, yb, freqs, powerspecx, powerspecy = load_data(fh, self.config.starttime, self.config.endtime,
                                                            return_power_spectrum=True, chunk_dur=self.config.chunk_dur)
                
        # x, use binary search for efficiency
        initialx = np.searchsorted(freqs, self.config.startfreqx, side='left')
        finalx = np.searchsorted(freqs, self.config.endfreqx, side='right') - 1

        # y, use binary search for efficiency
        initialy = np.searchsorted(freqs, self.config.startfreqy, side='left')
        finaly = np.searchsorted(freqs, self.config.endfreqy, side='right') - 1

        def psd_func(x, a, b):    # fitting function for the PSD
            return a**2*(self.config.constants.D/((np.pi)**2)) / ( (b/(2*np.pi*self.config.constants.beta))**2 + x**2) # from eqn 10 Berg-Sorensen et al. 2004
            # a - Sensitivity - units of V/m
            # b - Stiffness - N/m

        # Fit the power spectrum to get the trap stiffness
        # x
        poptx, _ = curve_fit(psd_func, freqs[initialx:finalx], powerspecx[initialx:finalx], p0=[1e5, 1e-3])
        alphax = abs(poptx[1])

        # y
        popty, _ = curve_fit(psd_func, freqs[initialy:finaly], powerspecy[initialy:finaly], p0=[1e5, 1e-3])
        alphay = abs(popty[1])

        Sx = poptx[0]
        Sy = popty[0]
        
        if show_plots:
            fig, axs1 = plt.subplots(1, 2, figsize=(10, 6), sharey=True, tight_layout=True)
            axs1[0].loglog(freqs, powerspecx, 'k.')
            axs1[0].loglog(freqs[initialx:finalx], psd_func(freqs[initialx:finalx], *poptx), 'r-')
            axs1[0].set_xlabel('Frequency [Hz]')
            axs1[0].set_ylabel('Power Spectrum [$V^2$/Hz]')
            axs1[0].set_title('Power Spectrum of X')

            axs1[1].loglog(freqs, powerspecy, 'k.')
            axs1[1].loglog(freqs[initialy:finaly], psd_func(freqs[initialy:finaly], *popty), 'r-')
            axs1[1].set_xlabel('Frequency [Hz]')
            axs1[1].set_title('Power Spectrum of Y')
            fig.suptitle('Brownian Calibration')

            fig2, axs2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True, tight_layout=True)
            t = np.arange(len(xb)) * dt
            axs2[0].plot(t, 1e9 * xb / Sx,'k')
            axs2[0].set_ylabel('x [nm]')

            axs2[1].plot(t , 1e9 * yb / Sy,'k')
            axs2[1].set_ylabel('y [nm]')
            axs2[1].set_xlabel('Time [s]')
            fig2.suptitle('Calibration Data')

        return Sx, Sy, alphax, alphay, np.mean(xb) / Sx, np.mean(yb) / Sy
    
    def compute_fischer_correction(self, active_name:str, passive_name:str, freq_range:tuple[float, float] = (1e2, 1e3), show_plots=False) -> float:
        """Compute a Fischer et al. method correction on both datasets.
        
        Parameters
        ----------
        active_name : str
            Active microrheology analysis object name to use for the correction
        passive_name : str
            Passive microrheology analysis object name to use for the correction
        freq_range : tuple(low_hz, high_hz)
            Frequency range trusted for the mean correction
        show_plots : bool
            If True, show plots of Fischer et al. derived stiffnesses

        Returns
        -------
        float
            Fischer correction
        """
        active = self.datasets[active_name]
        passive = self.datasets[passive_name]
        results = active.results

        demod = [
            software_demod(y, fg, freq)
            for y, fg, freq in zip(results['bead_y'], results['fg'], results['fg_freqs'])
        ]
        phi, trap_amp, bead_amp = map(np.asarray, zip(*demod)) # convert to numpy arrays

        _, yb = load_data(passive.fh, passive.starttime, passive.endtime)
        freqs, Pyy = signal.welch(yb, fs, nperseg=fs, noverlap=0, window='hann', scaling='density')

        P = np.interp(results['fg_freqs'], freqs, Pyy) / 2 / self.calib.Sy**2
        R_L = bead_amp / (2 * np.pi * results['fg_freqs'] * trap_amp) * np.sin(phi)
        ks = 2 * kBT / P * R_L

        mask = (results['fg_freqs'] >= freq_range[0]) & (results['fg_freqs'] <= freq_range[1])
        correction = np.mean(ks[mask]) / self.calib.ky

        if show_plots:
            plt.figure(tight_layout=True)
            plt.semilogx(results['fg_freqs'], ks, '.', color='gray', alpha=0.4)
            plt.semilogx(results['fg_freqs'][mask], ks[mask], 'o', color='tab:blue', label='Used')
            plt.axhline(self.calib.ky, color='tab:red', label='PSD')
            plt.axhline(np.mean(ks[mask]), color='tab:blue', ls='--', label='Fischer mean')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Trap Stiffness [N/m]')
            plt.title(f"'{active_name}' vs '{passive_name}' - Correction: {correction:.4f}")
            plt.legend()
            plt.show()

        active.fischer_correction = correction
        passive.fischer_correction = correction

        return correction

    @staticmethod
    def _is_passive_dataset(ds) -> bool:
        """Return if a dataset is or was an instance of PassiveAnalysis."""
        from microrheo.passive import PassiveAnalysis
        return isinstance(ds, PassiveAnalysis) or bool(getattr(ds, 'is_passive', False))
    
    @staticmethod
    def _filter_by(items: dict, values : list[str] | None, key_fn) -> dict:
        """Generic include/exclude-by-value filter for name or sample_type selection. 

        key_fn(key, value) should return the thing to be compared against 'values'. 
        Prefix a value with '-' to exlucde instead of include. Only include or exclude at once.
        """
        if values is None: # default: return all
            return dict(items)
        if np.any([v.startswith('-') for v in values]):
            excluded = {v[1:] for v in values if v.startswith('-')}
            return {k: v for k, v in items.items() if key_fn(k, v) not in excluded}
        return {k: v for k, v in items.items() if key_fn(k, v) in set(values)}

    def _select_datasets(self, datasets: list[str] | None=None, sample_types: list[str] | None=None) -> dict:
        """Include/exclude datasets by name and/or sample type. 
        
        Each filter uses the same '-' include/exclude syntax and applies independently. (e.g. datasets=['-passive2'], sample_type=['actin'] keeps every actin dataset except passive2).

        Only include or exclude at one time; default is to include everything.
        """
        selected = self._filter_by(self.datasets, datasets, key_fn=lambda k, v: k)
        selected = self._filter_by(selected, sample_types, key_fn=lambda k, v: getattr(v, 'sample_type', None))
        return selected

    def plot_moduli(self, datasets: list[str] | None=None, sample_types: list[str] | None=None, title: str | None=None, region: tuple[float, float] | None=None, corrected: bool=False, average: bool=False, show_std: bool=True, std_err: bool=False, show_individual: bool=True, combine_axes: bool=False, axes=None) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plot G' and G'' for all datasets (or a subset) in the experiment.
        
        Parameters
        ----------
        datasets : list[str] | None
            A list of datasets to include or exclude by name. To exclude datasets use the '-' prefix. Only include or exclude at one time, mixing will throw an error. Default is to plot everything.
        sample_types : list[str] | None
            Include/exclude by sample_type (e.g. ['actin'] or ['-water']) using the same syntax as 'datasets'. Combines with 'datasets' as an AND filter. Default is everything.
        title : str | None
            A custom title for the plot
        region : tuple[low_hz, high_hz] | None
            Optional frequency range for the plot. If 'None', use full range
        corrected : bool
            Whether or not to use the Fischer et al. method for stiffness correction
        average: bool
            If True, aditionally compute and plot the mean G' and G" active and passive curves
        show_std : bool
            If True (and average=True), show +/- 1 standard deviation as a filled region. Ignored if average=False
        std_err : bool
            If True (and average=True and show_std=True), show +/- 1 standard error as a filled region instead of the standard deviation.
        show_individual : bool
            If False (and average=True), hide the individual dataset traces and show only the averaged curves. Ignored if average=False
        combine_axes : bool
            If True, plot both G' and G" on a single axis instead of two side by side
        axes: plt.axes | None
            Custom axes to plot the datasets on

        Returns
        -------
        fig, ax1, ax2
        """
        if combine_axes:
            if axes is not None:
                ax1 = axes
                fig = ax1.get_figure()
            else:
                fig, ax1 = plt.subplots(figsize=(9, 6))
            ax2 = ax1
        else:
            if axes is not None:
                ax1, ax2 = axes
                fig = ax1.get_figure()
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        selected = self._select_datasets(datasets, sample_types)

        colors = [
            "#0072B2",
            "#009E73",
            "#56B4E9",
            "#E69F00",
            "#CC79A7",
            "#D55E00",
            "#000000",
        ]

        passive_group = {}
        active_group = {}

        show_individual = show_individual if average else True
        
        for (name, ds), color in zip(selected.items(), cycle(colors)):
            is_passive = self._is_passive_dataset(ds)
            style_passive = dict(marker='None', linestyle='-',    color=color, alpha=0.8)
            style_active  = dict(marker='.',    linestyle='None', color=color, alpha=0.8)
            style = style_passive if is_passive else style_active

            if corrected:
                G1, G2 = ds.corrected_moduli()
            else:
                G1, G2 = ds.G1, ds.G2

            omega = ds.omega_list
            if region is not None:
                mask = (omega >= 2 * np.pi * region[0]) & (omega <= 2 * np.pi * region[1])
                omega, G1, G2 = omega[mask], G1[mask], G2[mask]

            (passive_group if is_passive else active_group)[name] = (omega, G1, G2)

            if show_individual:
                ax1.loglog(omega, G1, **style, label=f"{name} G'")
                ax2.loglog(omega, G2, **style, label=f'{name} G"')

        if average:
            self._plot_group_average(ax1, ax2, passive_group, label='Passive', n_points=150, show_std=show_std, std_err=std_err, marker=None, linestyle='-', lw=1.5, alpha=0.9)
            self._plot_group_average(ax1, ax2, active_group,  label='Active',  n_points=35, show_std=show_std, std_err=std_err, marker='o', linestyle='None', ms=4, alpha=0.9)

        for ax in (ax1, ax2):
            ax.set_xlabel(r'Angular Frequency $\omega$ [rad/s]', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5)
            ax.legend()

        if combine_axes:
            ax1.set_ylabel('Viscoelastic Moduli G\', G\" [Pa]', fontsize=12)
        else:
            ax1.set_ylabel('Viscoelastic Moduli G\', G\" [Pa]', fontsize=12)
            ax1.set_title("G'")
            ax2.set_title('G"')

        fig.suptitle(title or f'{self.fh.rsplit('/', 1)[-1]} — Viscoelastic Shear Moduli G\', G"', fontsize=16)
        fig.tight_layout()

        return fig, ax1, ax2
    
    def _plot_group_average(self, ax1, ax2, group: dict, label: str, n_points: int=150, show_std: bool=True, std_err: bool=False, **style):
        """Interpolate G' and G" onto a common log-spaced grid and plot the mean (+/- 1 std in logspace) for a group of datasets.
        
        Parameters
        ----------
        ax1 : Axes
            First matplotlib axes
        ax2 : Axes
            Second matplotlib axes
        group : dict
            Mapping of name -> omega_list, G1, G2 arrays
        label : str
            Label for plot legend
        n_points : int
            Number of points to resample the data onto a common log-spaced grid
        show_std : bool
            If True, plot +/- 1 standard deviation as a filled region
        **style
            Extra style keyword arguments for the matplotlib curves
        """

        if len(group) == 0:
            return
        
        omega_arrays = [np.asarray(o) for o, _, _ in group.values()]
        lo = max(o.min() for o in omega_arrays)
        hi = min(o.max() for o in omega_arrays)

        omega_common = np.geomspace(lo, hi, n_points) # resample all data to a common log-spaced grid to ensure points matchup in frequency
        log_omega_common = np.log10(omega_common)

        for ax, idx, col in ((ax1, 1, '#4BB3FD'), (ax2, 2, '#00487C')):
            curves = []
            for omega, G1, G2 in group.values():
                omega = np.asarray(omega)
                G = np.asarray(G1 if idx == 1 else G2)

                # keep only positive moduli
                valid = G > 0
                if np.count_nonzero(valid) < 2:
                    print('Not enough positive points to interpolate.')
                    continue
                omega = omega[valid]
                G = G[valid]

                order = np.argsort(omega)
                curves.append(
                    np.interp(log_omega_common, np.log10(omega[order]), G[order], left=np.nan, right=np.nan)
                )
            curves = np.vstack(curves)
            mean = np.nanmean(curves, axis=0)

            modulus_label = f"{label} G'" if idx == 1 else f'{label} G"'
            ax.loglog(omega_common, mean, color=col, **style, label=modulus_label)

            if show_std and curves.shape[0] > 1:
                std = np.nanstd(curves, axis=0, ddof=1)
                if std_err: std /= np.sqrt(curves.shape[0])
                ax.fill_between(
                    omega_common, mean - std, mean + std,
                    color=col, alpha=0.15,
                )

    # ---------- SERIALIZE/LOAD JSON DATA ----------

    def serialize(self, filepath: str, datasets: list[str] | None=None, sample_types: list[str] | None=None, **metadata) -> dict:
        """Serialize omega, G', G", and extra metadata for selected datasets to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Where to write the file to. Parent directories are created if needed.
        datasets : list[str] | None
            Include/exclude datasets by name, same syntax as plot_moduli.
        sample_types : list[str] | None
            Include/exclude by sample_type (e.g. ['actin'] or ['-water']) using the same syntax as 'datasets'.
        **metadata
            Extra experiment level metadata
        """
        selected = self._select_datasets(datasets, sample_types)

        payload = {
            'metadata': {
                'name': self.fh.rsplit('/', 1)[-1],
                'exported_at': datetime.now().isoformat(timespec='seconds'),
                **metadata,
            },
            'datasets': {
                name: {
                    'is_passive': self._is_passive_dataset(ds),
                    'sample_type': ds.sample_type,
                    **(
                        {'trap_offset': ds.trap_offset(ds.fischer_correction).tolist()}
                        if self._is_passive_dataset(ds)
                        else {'fischer_correction': ds.fischer_correction}
                    ),
                    'omega': np.asarray(ds.omega_list).tolist(),
                    'G1': np.asarray(ds.G1).tolist(),
                    'G2': np.asarray(ds.G2).tolist(),
                }
                for name, ds in selected.items()
            },
        }

        if not filepath.lower().endswith('.json'):
            filepath += '.json'

        if parent := os.path.dirname(filepath):
            os.makedirs(parent, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)

        return payload
    
    def load(self, filepath: str, recursive: bool=False, datasets: list[str] | None=None, sample_types: list[str] | None=None) -> list[str]:
        """Load data from disk written by serialize() into this Experiment's dataset registry.
        
        Parameters
        ----------
        filepath : str
            A literal path, or a glob pattern containing '*', '?', or '[' to a JSON file written by serialize().
        recursive : bool
            If True, '**' in a glob pattern matches any number of subdirectories. Ignored for literal (non-pattern) paths.
        datasets : list[str] | None
            Include/exclude by name from file.
        sample_types : list[str] | None
            Include/exclude by sample type from file.

        Returns
        -------
        list[str]
            Names added to self.datasets, across all matched file(s).
        """
        if any(c in filepath for c in '*?['):
            pattern = filepath if filepath.lower().endswith('.json') else filepath + '.json'
            matches = sorted(glob.glob(pattern, recursive=recursive))
            if not matches:
                raise FileNotFoundError(f"No files matched pattern '{pattern}'.")
        else:
            matches = [filepath if filepath.lower().endswith('.json') else filepath + '.json']

        added = []
        for fp in matches:
            added += self._load_single(fp, datasets=datasets, sample_types=sample_types)

    def _load_single(self, filepath:str, datasets: list[str] | None=None, sample_types: list[str] | None=None) -> list[str]:
        """Load one already-resolved JSON file into self.datasets."""
        with open(filepath) as f:
            payload = json.load(f)

        entries = self._filter_by(payload['datasets'], datasets, key_fn=lambda k, v: k)
        entries = self._filter_by(entries, sample_types, key_fn=lambda k, v: v.get('sample_type'))

        added = []
        for name, entry in entries.items():
            key = f'{payload['metadata']['name']}_{name}'
            if key in self.datasets:
                raise ValueError(f"Dataset '{key}' already exists.")
            self.datasets[key] = SerializedDataset(
                omega_list=np.array(entry['omega']),
                G1=np.array(entry['G1']),
                G2=np.array(entry['G2']),
                is_passive=entry['is_passive'],
                sample_type=entry.get('sample_type'),
                fischer_correction=entry.get('fischer_correction'),
                trap_offset=(
                    np.array(entry['trap_offset'])
                    if entry.get('trap_offset') is not None else None
                ),
            )
            added.append(key)

        return added
    
    @classmethod
    def from_serialized(cls, filepath: str, recursive: bool=False, datasets: list[str] | None=None, sample_types: list[str] | None=None) -> Experiment:
        """Alternative constructor: build a new Experiment purely from serialized moduli data. 
        
        Use load() to merge in more files. self.config and self.calib are None; only plotting works.

        Examples::

            exp = Experiment.from_serialized('data1.json')
            exp.load('data2.json')
            exp.plot_moduli(average=True)

            exp = Experiment.from_serialized('data/**/*_cell.json', recursive=True)
        """
        self = cls.__new__(cls)
        self.config = self.calib = self.brownian_params = None
        self.datasets = Datasets({})
        self.fh = os.path.splitext(os.path.basename(filepath))[0]
        self.load(filepath, recursive=recursive, datasets=datasets, sample_types=sample_types)
        return self