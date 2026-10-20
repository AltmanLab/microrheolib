from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.signal import savgol_filter

class PassiveAnalysis(OTDataset):
    """Process passive microrheology datasets.

    Typical workflow::

        passive = PassiveAnalysis(name, fh, starttime, endtime, alphaincrement)
        passive.compute_msd()
        passive.compute_alpha()
        passive.compute_moduli()
        passive.plot_g1g2()
    """

    def __init__(self, fh, constants, calib, starttime=0, endtime=185, sample_type='water'):
        """Initialize a passive microrheology analysis object.

        Parameters
        ---------
        fh : str
            Relative file handle of the passive dataset i.e. 260608/bead1p
        constants : ExperimentalConstants
            Experimental constants
        calib : TrapCalibration
            Brownian and FunctionG calibration parameters
        starttime : float
            Start time of data (in seconds)
        endtime : float
            End time of data (in seconds)
        sample_type : str
            Sample type (e.g., 'water', 'actin', 'cell', etc.)
        """
        super().__init__(fh, constants, calib, starttime, endtime, sample_type)

        # Load the data and convert to meters
        self.fh = fh + ('' if fh.endswith('/') else '/')
        self.x, self.y = load_data(self.fh, starttime, endtime)
        self.t = np.arange(len(self.x)) * dt
        self.x /= self.calib.Sx

        if not hasattr(self.calib, 'a'): # if 'a' doesn't exist the entire detector hasn't been calibrated; use Sy instead
            self.y /= self.calib.Sy
        else:
            self.y = self.calib.ed_fit_func(self.y - self.calib.v_offset)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), tight_layout=True, sharex=True)
        fig.suptitle('Data to Analyze')
        ax1.plot(self.t, 1e9 * self.x, 'k')
        ax1.set_ylabel('x [nm]')
        ax2.plot(self.t, 1e9 * self.y, 'k')
        ax2.set_ylabel('y [nm]')
        ax2.set_xlabel('Time [s]')
        plt.show()
        
    def compute_msd(self, max_tau=20, analysis_length=20, show_plots=False):
        '''Compute msd and tau arrays.
        
        Parameters
        ----------
        max_tau: float
            Max value for tau (seconds)
        analysis_length: float
            Duration in seconds to trim the MSD to
        show_plots : bool

        Returns
        -------
        tau : np.ndarray
            timescale
        msd : np.ndarray
            mean squared displacement
        '''
        
        tau, msd = calcmsd2(self.x, self.y, self.t[-1], max_tau, not show_plots)
        msd = msd[:int(analysis_length*fs)]
        tau = tau[:int(analysis_length*fs)]

        mask = np.isfinite(tau) & np.isfinite(msd) & (tau > 0) & (msd > 0)
        tau, msd = tau[mask], msd[mask]

        self.tau, self.msd = tau, msd
        return self.tau, self.msd
    
    def compute_alpha(self, n_points=500, window=151, polyorder=5, show_plots=False, water_theory=False):
        """Compute log-log MSD slope (alpha) using log-spacing binning and spline fitting
        
        Parameters
        ---------
        n_points : int
            Number of log-spaced points to resample the data
        window : int
            Window size (in samples) for the savgol filter; must be an odd number
        polyorder : int
            Polynomial order for the savgol filter.
        show_plots : bool
            If True, plot the resulting MSD and alpha curves
        water_theory : bool
            If True (and show_plots=True) add the water theory curves to the MSD and alpha plots
        """

        if window % 2 == 0: window += 1 # ensure window size is odd

        logtau, logmsd = np.log10(self.tau[1:]), np.log10(self.msd[1:]) # skip first data point
        
        # remove any nans/infs
        mask = np.isfinite(logtau) & np.isfinite(logmsd)
        logtau, logmsd = logtau[mask], logmsd[mask]

        # sort
        sort_idxs = np.argsort(logtau)
        logtau, logmsd = logtau[sort_idxs], logmsd[sort_idxs]
        
        # resamble the points to equally space them in logspace
        logtau_rs = np.linspace(logtau.min(), logtau.max(), n_points)
        logmsd_rs = np.interp(logtau_rs, logtau, logmsd) # linear interp in log-log space

        # remove any potentally overlapping points (must be monotonic)
        logtau_rs, uniq_idxs = np.unique(logtau_rs, return_index=True)
        logmsd_rs = logmsd_rs[uniq_idxs]

        # remove any nans/infs
        mask = np.isfinite(logtau_rs) & np.isfinite(logmsd_rs)
        logtau_rs, logmsd_rs = logtau_rs[mask], logmsd_rs[mask]
        delta = logtau_rs[1] - logtau_rs[0]

        logmsd_smooth = savgol_filter(logmsd_rs, window, polyorder, mode='interp')
        alpha_list = savgol_filter(logmsd_rs, window, polyorder, deriv=1, delta=delta, mode='interp')

        omega_list = 1 / (10**logtau_rs)
        msd_list = 10**(logmsd_smooth)

        if show_plots:
            is_negative = alpha_list < 0

            # use np.diff to identify True-False and False-True transitions
            transitions = np.diff(is_negative.astype(int))
            start_idxs = np.where(transitions == 1)[0] + 1
            end_idxs = np.where(transitions == -1)[0]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
            for start, end in zip(start_idxs, end_idxs):
                ax1.axvspan(logtau_rs[start], logtau_rs[end], color='r', alpha=0.25, label='Negative Derivative' if start == start_idxs[0] else '')

            ax1.plot(logtau_rs, logmsd_rs, 'mo', ms=3, label=f'Resampled Data (n={n_points})')
            ax1.plot(logtau_rs, logmsd_smooth, 'b-', label='Gaussian Smoothed')
            ax1.plot(logtau, logmsd, 'k*', ms=5, alpha=0.01, label='Raw')

            if water_theory:
                gamma = 6 * np.pi * self.constants.r * self.constants.eta
                f_c = (self.calib.kx + self.calib.ky) / 2 / (2 * np.pi * gamma)
                msd_theory = 2 * (self.constants.D / (np.pi * f_c)) * (1 - np.exp(-10**logtau_rs * 2 * np.pi * f_c))

                plateau_equipartition = 4 * kBT / ((self.calib.kx + self.calib.ky) / 2)

                alpha_theory = np.gradient(np.log10(msd_theory), delta)

                ax1.plot(logtau_rs, np.log10(msd_theory), 'g--', label='Theoretical MSD')
                ax1.axhline(np.log10(plateau_equipartition), color='k', lw=0.8, ls='--', label='Equipartition Plateau')

            ax1.legend()
            ax1.set_title('LogLog MSD')
            ax1.set_xlabel('log(tau)')
            ax1.set_ylabel('log(MSD)')
            
            for start, end in zip(start_idxs, end_idxs):
                ax2.axvspan(logtau_rs[start], logtau_rs[end], color='r', alpha=0.25, label='Negative Derivative' if start == start_idxs[0] else '')
            ax2.axhline(0, color='k', lw=0.8, ls='--')
            ax2.axhline(1, color='b', lw=0.8, ls='--')
            ax2.plot(logtau_rs, alpha_list, 'r-', label='Smoothed')
            if water_theory:
                ax2.plot(logtau_rs, alpha_theory, 'g--', label='Theoretical')
            ax2.legend()
            plt.text(0.9 * logtau_rs.max(), 1, 'diffusive', color='b', ha='right', va='bottom')
            ax2.set_title(r'LogLog MSD Derivative ($\alpha$)')
            ax2.set_xlabel('log(tau)')
            ax2.set_ylabel(r'$\alpha = $ d log(MSD) / d log(tau)')
            plt.show()
        
        self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list = logtau_rs, logmsd_smooth, alpha_list, omega_list, msd_list
        
        return self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list

    def trap_offset(self, stiffness_correction=None, inertia=True):
        ktrap = (self.calib.kx + self.calib.ky) / 2 # average trap stiffness

        if stiffness_correction is not None:
            ktrap *= stiffness_correction
        if inertia:
            mass = 4/3 * np.pi * self.constants.r**3 * self.constants.rho
            ktrap -= self.omega_list**2 * mass

        return ktrap/(6*np.pi*self.constants.r)

    def compute_moduli(self, free_memory=True):
        '''Compute G' and G'' moduli'''
        
        d = 2 # dimensionality of the system
        
        gamma = (0.457 * ( 1 + self.alpha_list)**2) - (1.36 * (1 + self.alpha_list)) + 1.9 # approx value from Mason 2000 p.373 below eqn. 8
        G_star = np.abs((2 * d * kBT) / (6 * np.pi * self.constants.r * self.msd_list * gamma))
        G1 = G_star * np.cos(self.alpha_list * np.pi / 2)
        G2 = G_star * np.sin(self.alpha_list * np.pi / 2)

        if free_memory:
            self.logtau_list, self.logmsd_list, self.alpha_list, self.msd_list = None, None, None, None
            self.x, self.y, self.sumsignal, self.fg = None, None, None, None

        self.G1 = np.array(G1)
        self.G2 = np.array(G2)
        self.omega_list = np.array(self.omega_list)
        
        return self.omega_list, self.G1, self.G2

    def corrected_moduli(self):
        """Return G' and G" with the stiffness correction from Fischer et al."""
        return self.G1 - self.trap_offset(self.fischer_correction), self.G2

    def plot_msd(self):
        plt.figure()
        plt.plot(np.log10(self.tau), np.log10(self.msd), 'k.')
        plt.plot(self.logtau_list, self.logmsd_list, 'r.')
        plt.xlabel('log(tau)')
        plt.ylabel('log(MSD)')
        plt.title('Log-Log plot of MSD vs tau')
        plt.grid()
        plt.show()
        
    def plot_params(self):
        plt.figure()
        plt.suptitle('Passive Analysis Parameters')
        plt.subplot(2,1,1)
        plt.plot(self.logtau_list, self.logmsd_list, 'ko')
        plt.ylabel('log(msd)')

        plt.subplot(2,1,2)
        plt.plot(self.logtau_list, self.alpha_list, 'ko')
        plt.xlabel('log(tau)')
        plt.ylabel('alpha')
        plt.show()

    def plot_g1g2(self):
        plt.figure()
        plt.loglog(self.omega_list, self.G1 - self.trap_offset(), 'k.', label='G\'')
        plt.loglog(self.omega_list, self.G2, 'r.', label='G\'\'')
        plt.xlabel('Angular Frequency (Radians)', )
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()
        plt.title('Viscoelastic Shear Moduli G\', G\"')
        plt.tight_layout()
        plt.show()