from microrheo.base import OTDataset
from microrheo.utils import *

from itertools import cycle
from scipy.optimize import curve_fit
from scipy.interpolate import LSQUnivariateSpline
from dataclasses import dataclass, field

# viscosity presets Pa s
PRESET_ETA = {
    "water": 0.001,
    "bsa":   0.002548,
    "iodixinol20": 0.001856,
}

# helper class to store analysis parameters
@dataclass
class ExperimentalConstants:
    eta: float                          # viscosity
    r: float                            # bead radius (m)
    D: float = field(init=False)        # diffusion constant
    beta: float = field(init=False)     # derived
    
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
    """Reulting configurations from Brownian and FunctionG calibrations"""
    Sx: float
    Sy: float
    kx: float
    ky: float
    trapx: float
    trapy: float

    # Function gen calibration (populated after calibrate_function_gen)
    a: float = field(init=False)
    b: float = field(init=False)
    ed_fit_func: callable = field(init=False)
    v_offset: float = field(init=False)

    def __str__(self) -> str:
        return f"Brownian calibration parameters:\n Sx:    {self.Sx: }\n Sy:    {self.Sy: }\n kx:    {self.kx: }\n ky:    {self.ky: }\n trapx: {self.trapx: }\n trapy: {self.trapy: }\n"

class Datasets(dict): # wrapper around a dict to utilize the . notation
    def __getattr__(self, key):
        try: 
            return self[key]
        except KeyError:
            raise AttributeError(f"Experiment has no dataset named '{key}'")
        
    def __setattr__(self, key, value):
        self[key] = value

class Experiment:
    def __init__(self, fh: str, config: ExperimentalConfig, brownian_params: list | None=None):
        """Initialize global experiment variables and perform the Brownian calibration.
        
        Parameters
        ----------
        fh : str
            Relative path to the folder containing the data for the entire experiment
        calib : CalibrationParams
            Parameters needed for the experiment wide calibration steps
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
            Sx, Sy, kx, ky, trapx, trapy = self._brownian_analysis(show_plots=True)
            print(f'[{Sx}, {Sy}, {kx}, {ky}, {trapx}, {trapy}]')
            
        else:
            Sx, Sy, kx, ky, trapx, trapy = brownian_params        
            
        self.calib = TrapCalibration(Sx, Sy, kx, ky, trapx, trapy)
        print(self.calib)

    def add(self, cls, name: str, folder_name: str, **kwargs) -> OTDataset:
        """Add a new rheology analysis object to this experiment
        
        Parameters
        ----------
        cls
        name
        folder_name

        Returns
        -------
        OTDataset
        """
        if not self.calib.a:
            raise RuntimeError("Call calibrate() before adding datasets.")
        
        ds = cls(self.fh + '/' + folder_name.removesuffix('/'), self.config.constants, self.calib, **kwargs)
        self.datasets[name] = ds
        return ds

    def fit_ed_spline(self, x, y, y_int, k=5, center_width=0.65):
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

        Returns
        -------
        f : callable
            The entire detector fitting function
        x_knots : list
            X coordinate of the knots
        y_knots : list
            Y coordinate of the knots
        """

        mask = np.abs(x) < center_width # center region to estimate slope from

        m, _ = np.polyfit(x[mask], y[mask] - y_int, 1)

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

        # ? optionally we could subtract this offset to make the spline hit the trap center
        # offset = spline(0.0)

        def f(x):
            return y_int + m*x + spline(x)
        
        x_knots = spline.get_knots()
        return f, x_knots, f(x_knots)

    def calibrate(self, folder_name=None, lin_range:tuple[float, float]=(-200, 200), spline_fit:tuple[int, tuple[float, float]]=(2, (-40, 400)), center_width=0.65, show_plots=()):
        """Function generator calibration and entire detector fitting. Sets the linear fit parameters self.a and self.b for converting the function generator signal into distance. Also computes the entire detector fit function (usage shown below).
        
        Parameters
        ----------
        folder_name : str
            Optional name of the FunctionG data. If 'None' defaults to 'FunctionG'.
        lin_range : tuple(min_nm, max_nm)
            Range of values to fit the linear trap calibration region, in nanometers
        spline_fit : tuple(knots, (lower_bound_nm, upper_bound_nm))
            
            knots is the amount of knots per side desired on the spline fit.
            knots = 0 disables nonlinear (entire detector) fitting
        show_plots: iterable
            may include:
            - 'position': plots raw position and function generator data
            - 'signal_jumps': plots the jumps in the function generator signal

        For Passive/BeadXY:
        ----
            ```y = 1e-9 * ed_fit_func(raw_y_volts - v_offset)
            ```
        
        For Active/Lockin:
        ---
            ```for data in self.data_dict.values():
                data['beady'] = 1e-9 * self.ed_fit_func((data['beady']))
            ```
        """
        # Load the data
        fh = self.fh
        if folder_name is None:
            folder_name = 'functiong'

        folder = next(
            (name for name in os.listdir(fh)
             if os.path.isdir(os.path.join(fh, name))
             and name.lower() == folder_name),
             None,
        )

        if folder is None:
            raise ValueError('No FunctionG data found in the folder')
        
        fh = os.path.join(fh, folder) + '/'

        x, y, sumsignal, fg = load_data(fh, self.config.fg_starttime, self.config.fg_endtime)

        plots = set(show_plots) # position, signal_jumps
        
        # normalize position data into meters
        x = np.asarray(x) / self.calib.Sx
        y = np.asarray(y) / self.calib.Sy
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
        
        # find plateau boundaries (i.e. where the bead has leveled off after a FG jump)
        abs_diff = np.abs(np.diff(fg))
        threshold = np.mean(abs_diff) + 5*np.std(abs_diff)

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
        stable_mask = std_fg < (3 * np.mean(std_fg))

        avg_fg = avg_fg[stable_mask]
        avg_y = avg_y[stable_mask]

        # remove any points that are within 20nm of the center of the trap; these must be caused by noise
        mask = np.abs(avg_y - self.calib.trapy) > 20e-9

        avg_y = avg_y[mask]
        avg_fg = avg_fg[mask]

        # linear trap calibration (convert from volts to meters)
        upper, lower = 1e-9*lin_range[1] + self.calib.trapy, 1e-9*lin_range[0] + self.calib.trapy
        linear_region = (avg_y > lower) & (avg_y < upper)

        if np.sum(linear_region) < 2:
            raise ValueError(f' {np.sum(linear_region)} points is not enough points in the linear region for calibration. Please adjust lin_range or check the data.')

        self.calib.a, self.calib.b = np.polyfit(avg_fg[linear_region], 1e9 * avg_y[linear_region], deg=1) # TODO: does this need to be done in nanometers? I feel like it is more internally consistent to work in SI units always. Also should we rename 'a' and 'b' to be more... descriptive
                            
        # optional higher degree fitting (entire detector)
        if spline_fit[0] > 0:
            fit_min, fit_max = spline_fit[1]

            bead_pos_nm = self.calib.a * avg_fg + self.calib.b
            detector_v = avg_y * self.calib.Sy
            self.calib.v_offset = np.mean(detector_v)

            detector_v -= self.calib.v_offset
      
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


            idxs = np.argsort(X)[5:-5] # remove first and last 5 points to clean up noise from edges
            X = X[idxs]
            Y = Y[idxs]
            self.calib.ed_fit_func, x_knots, y_knots = self.fit_ed_spline(X, Y, 1e9 * self.calib.trapy, k=spline_fit[0], center_width=center_width)
        
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
            ax1.plot(avg_fg, 1e9 * avg_y, 'k.', label='Calibration Points')
            ax1.plot(avg_fg[linear_region], 1e9 * avg_y[linear_region], 'go', alpha=0.25, label='Linear Region')

            fg_fit = np.linspace(min(avg_fg), max(avg_fg), 100)
            ax1.plot(fg_fit, self.calib.a * fg_fit + self.calib.b, 'r-', label='Linear Fit')
            ax1.plot(0, 1e9 * self.calib.trapy, 'pb', label='Trap Center')
            ax1.set_xlabel('FG Voltage (V)')
            ax1.set_ylabel('Bead Position (nm)')
            ax1.legend()
            ax1.set_title('Linear Fit Calibration')

            ax2.plot(detector_v, bead_pos_nm, 'k.')
            ax2.plot(detector_v[mask], bead_pos_nm[mask], 'go', alpha=0.25)
            ax2.plot(0, 1e9 * self.calib.trapy, 'pb', label='Trap Center')

            xfit = np.linspace(X.min(), X.max(), 1000)
            ax2.plot(xfit, self.calib.ed_fit_func(xfit), 'r-', lw=2)
            ax2.plot(x_knots, y_knots, 'r.')

            ax2.set_xlabel('Detector Voltage (V)')
            ax2.set_ylabel('Bead Position (nm)')
            ax2.set_title(f'Spline Fit (Knots = {spline_fit[0]})')
            ax2.grid(True)
            ax2.legend()
            plt.tight_layout()
            plt.show()

    def _brownian_analysis(self, folder_name=None, show_plots=False):
        """Brownian analysis for an experiment

        Parameters
        ----------
        folder_name : str
            Optional folder name of Brownian data. If 'None' defaults to 'Brownian'.

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
            folder_name = 'brownian'

        folder = next(
            (name for name in os.listdir(fh)
             if os.path.isdir(os.path.join(fh, name))
             and name.lower() == folder_name),
             None,
        )

        if folder is None:
            raise ValueError('No brownian data found in the folder')
        
        fh = os.path.join(fh, folder) + '/'

        xb, yb, sumsignalb, fg, freqs, powerspecx, powerspecy = load_data(fh, self.config.starttime, self.config.endtime,
                                                                                return_power_spectrum=True, chunk_dur=self.config.chunk_dur)
                
        # x, loop forward and backwards for efficiency
        initialx, finalx = 0, 0 # indicies for x fit
        for i, freq in enumerate(freqs):
            if freq >= self.config.startfreqx:
                initialx = i
                break
        
        for i, freq in reversed(list(enumerate(freqs))):
            if freq <= self.config.endfreqx:
                finalx = i
                break
            
        # y, loop forward and backwards for efficiency
        initialy, finaly = 0, 0 # indicies for y fit
        for i, freq in enumerate(freqs):
            if freq >= self.config.startfreqy:
                initialy = i
                break
        
        for i, freq in reversed(list(enumerate(freqs))):
            if freq <= self.config.endfreqy:
                finaly = i
                break

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
        
        if show_plots:
            fig = plt.figure(figsize=(10, 10), tight_layout=True)
            fig.suptitle('OT Calibration')
            xplot = fig.add_subplot(221)
            xplot.set_title('Power Spectrum of X')
            xplot.loglog(freqs, powerspecx, 'k.')
            xplot.loglog(freqs[initialx:finalx], psd_func(freqs[initialx:finalx], *poptx), 'r-')
            xplot.set_xlabel('Frequency [Hz]')
            xplot.set_ylabel('Power Spectrum [$V^2$/Hz]')

            yplot = fig.add_subplot(222)
            yplot.set_title('Power Spectrum of Y')
            yplot.loglog(freqs, powerspecy, 'k.')
            yplot.loglog(freqs[initialy:finaly], psd_func(freqs[initialy:finaly], *popty), 'r-')
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

        Sx = poptx[0]
        Sy = popty[0]
        return Sx, Sy, alphax, alphay, np.mean(xb)/Sx, np.mean(yb)/Sy
    
    def plot_moduli(self, datasets: list[str] | None=None, title=None, axes=None) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plot G' and G'' for all datasets in the experiment.
        
        Parameters
        ----------
        datasets : list[str] | None
            A list of datasets to include or exclude by name. To exclude datasets use the '-' prefix. Only include or exclude at one time, mixing will throw an error. Default is to plot everything.
        title : str | None
            A custom title for the plot.
        axes: plt.axes | None
            Custom axes to plot the datasets on.
        """
        if axes is not None:
            ax1, ax2 = axes
            fig = ax1.get_figure()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        if datasets is None:
            selected = self.datasets
        elif np.any([name.startswith('-') for name in datasets]):
            excluded = {name[1:] for name in datasets if name.startswith('-')}
            selected = {
                k: v for k, v in self.datasets.items() if k not in excluded
            }
        else:
            selected = {
                k: self.datasets[k] for k in datasets
                if k in self.datasets
            }

        from microrheo.passive import PassiveAnalysis

        colors = [
            "#0072B2",
            "#009E73",
            "#56B4E9",
            "#E69F00",
            "#CC79A7",
            "#D55E00",
            "#000000",
        ]
        
        for (name, ds), color in zip(selected.items(), cycle(colors)):
            style_passive = dict(marker='None', linestyle='-',  color=color, alpha=0.8)
            style_active  = dict(marker='.',    linestyle='None', color=color, alpha=0.8)

            style = style_passive if isinstance(ds, PassiveAnalysis) else style_active
            sort_idx = np.argsort(ds.omega_list)[:-1] # drop the 50000 Hz point

            ax1.loglog(ds.omega_list[sort_idx], ds.G1[sort_idx], **style, label=f"{name} G'")
            ax2.loglog(ds.omega_list[sort_idx], ds.G2[sort_idx], **style, label=f'{name} G"')

        for ax in (ax1, ax2):
            ax.set_xlabel(r'Angular Frequency $\omega$ [rad/s]', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5)
            ax.legend()

        ax1.set_ylabel('Viscoelastic Moduli G\', G\" [Pa]', fontsize=12)
        ax1.set_title("G'")
        ax2.set_title('G"')
        fig.suptitle(title or f'{self.fh.rsplit('/', 1)[-1]} — Viscoelastic Shear Moduli G\', G"', fontsize=16)
        fig.tight_layout()

        return fig, ax1, ax2