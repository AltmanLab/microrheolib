from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.ndimage import gaussian_filter1d

class PassiveAnalysis(OTDataset):
    """Process passive microrheology datasets.

    Typical workflow::

        passive = PassiveAnalysis(name, fh, starttime, endtime, alphaincrement)
        passive.compute_msd()
        passive.compute_alpha()
        passive.compute_moduli()
        passive.plot_g1g2()
    """

    def __init__(self, fh, constants, calib, starttime=0, endtime=185):
        """Initialize a passive microrheology analysis object

        Parameters
        ---------
        fh : str
            Relative file handle of the passive dataset i.e. 2600608/bead1p
        aalysisparams : dict
            Analysis configuration
        params : optional
            If provided, skips computing the Brownian power spectrum calibration and uses these parameters instead
        """
        super().__init__(fh, constants, calib)

        # Load the data, all in meters
        self.fh = fh +'/'
        self.x, self.y, self.sumsignal, self.fg = load_data(self.fh, starttime, endtime)
        self.x /= self.calib.Sx
        self.y = 1e-9 * self.calib.ed_fit_func(self.y - self.calib.v_offset)
        self.t = np.arange(len(self.x)) * dt   # starts at zero

        fig=plt.figure(figsize=(10, 5))
        fig.suptitle('Data to Analyze')
        xpos=fig.add_subplot(211)
        xpos.plot(self.t, self.x*1e9,'k')
        xpos.set_ylabel('x[nm]')
        ypos=fig.add_subplot(212)
        ypos.plot(self.t, self.y*1e9,'k')
        ypos.set_ylabel('y[nm]')
        ypos.set_xlabel('Time [s]')
        plt.plot()
        
    def compute_msd(self, max_tau=20, analysis_length=20, show_plots=False):
        '''Compute msd and tau arrays
        
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
    
    def compute_alpha(self, n_points=500, sigma=25, show_plots=True):
        """Compute log-log MSD slope (alpha) using log-spacing binning and spline fitting
        
        Parameters
        ---------
        n_points : int
            Number of log-spaced points to resample the data at
        sigma : int
            Smoothing parameter passed to the gaussian filter. Nominally around 5% of `n_points`
        """
                
        logtau, logmsd = np.log10(self.tau), np.log10(self.msd)
        
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
        n_pad = 4 * sigma # add enough padding to cover the filter's edges
        n_local = max(10, len(logtau_rs) // 10) # use the first 10% of the data to capture the local slope at the edge

        left_slope = np.polyfit(logtau_rs[:n_local], logmsd_rs[:n_local], 1)[0]
        right_slope = np.polyfit(logtau_rs[-n_local:], logmsd_rs[-n_local:], 1)[0]

        pad_steps_l = np.arange(-n_pad, 0) * delta
        pad_logmsd_l = logmsd_rs[0] + left_slope * pad_steps_l

        pad_steps_r = np.arange(1, n_pad + 1) * delta
        pad_logmsd_r = logmsd_rs[-1] + right_slope * pad_steps_r

        logmsd_padded = np.concatenate([pad_logmsd_l, logmsd_rs, pad_logmsd_r])
        logmsd_smooth = gaussian_filter1d(logmsd_padded, sigma)
        logmsd_smooth = logmsd_smooth[n_pad:-n_pad]

        alpha_list = np.gradient(logmsd_smooth, delta)

        alpha_list = gaussian_filter1d(alpha_list, sigma)

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

            ax1.plot(logtau, logmsd, 'k.', ms=1, alpha=0.5, label='Raw')
            ax1.plot(logtau_rs, logmsd_rs, 'mo', ms=3, label=f'Resampled Data (n={n_points})')
            ax1.plot(logtau_rs, logmsd_smooth, 'r-', label='Gaussian Smoothed')

            ax1.legend()
            ax1.set_title('LogLog MSD')
            ax1.set_xlabel('log(tau)')
            ax1.set_ylabel('log(MSD)')
            
            for start, end in zip(start_idxs, end_idxs):
                ax2.axvspan(logtau_rs[start], logtau_rs[end], color='r', alpha=0.25, label='Negative Derivative' if start == start_idxs[0] else '')
            ax2.plot(logtau_rs, alpha_list, 'r-', label='Spline')
            ax2.axhline(0, color='k', lw=0.8, ls='--')
            ax2.axhline(1, color='b', lw=0.8, ls='--')
            plt.text(0.9 * logtau_rs.max(), 1, 'diffusive', color='b', ha='right', va='bottom')
            ax2.set_title(r'LogLog MSD Derivative ($\alpha$)')
            ax2.set_xlabel('log(tau)')
            ax2.set_ylabel(r'$\alpha = $ d log(MSD) / d log(tau)')
            plt.show()
        
        self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list = logtau_rs, logmsd_smooth, alpha_list, omega_list, msd_list
        
        return self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list

    def compute_moduli(self, free_memory=True, trap=True):
        '''Compute G' and G'' moduli'''
        
        d = 2 # dimensionality of the system

        ktrap = (self.calib.kx + self.calib.ky)/2 # average trap stiffness
        offset = ktrap/(6*np.pi*self.constants.r)
        
        gamma = (0.457 * ( 1 + self.alpha_list)**2) - (1.36 * (1 + self.alpha_list)) + 1.9 #approx value from Mason 2000 p.373 below eqn. 8
        G_star = np.abs((2 * d * kBT) / (6 * np.pi * self.constants.r * self.msd_list * gamma))
        G1 = G_star * np.cos(self.alpha_list * np.pi / 2)
        G2 = G_star * np.sin(self.alpha_list * np.pi / 2)

        if trap: # if the trap was on, we need to subract its forces on the bead
            G1 -= offset
            
        if free_memory:
            self.logtau_list, self.logmsd_list, self.alpha_list, self.msd_list = None, None, None, None
            self.x, self.y, self.sumsignal, self.fg = None, None, None, None

        self.G1 = np.array(G1)
        self.G2 = np.array(G2)
        self.omega_list = np.array(self.omega_list)
        
        return self.omega_list, self.G1, self.G2

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
        plt.loglog(self.omega_list, self.G1, 'k.', label='G\'')
        plt.loglog(self.omega_list, self.G2, 'r.', label='G\'\'')
        plt.xlabel('Angular Frequency (Radians)', )
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()
        plt.title('Viscoelastic Shear Moduli G\', G\"')
        plt.tight_layout()
        plt.show()