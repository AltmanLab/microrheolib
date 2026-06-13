from typing_extensions import deprecated
from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.signal import savgol_filter
from numpy.polynomial import Polynomial
from scipy.interpolate import UnivariateSpline

class PassiveAnalysis(OTDataset):
    """Process passive microrheology datasets.

    Typical workflow::

        passive = PassiveAnalysis(name, fh, starttime, endtime, alphaincrement)
        passive.compute_msd()
        passive.compute_alpha()
        passive.compute_moduli()
        passive.plot_g1g2()
    """

    def __init__(self, fh, constants, calib, starttime=0, endtime=185, alphaincrement=0.1):
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
        self.alphaincrement = alphaincrement

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
        
    def compute_msd(self, max_tau=20, analsis_length=20, show_plots=False):
        '''Compute msd and tau arrays
        Args:
        max_tau: max value for tau (seconds)
        analysis_length: ???
        returns: tau, msd
        '''
        
        msd_df:pd.DataFrame = calcmsd2(self.x, self.y, self.t[-1], max_tau, not show_plots)
        msd = msd_df.msd[:int(analsis_length*fs)].to_numpy()
        tau = msd_df.tau[:int(analsis_length*fs)].to_numpy()
        
        # mask = np.isnan(tau) | np.isnan(msd) # remove nan values
        # return tau[~mask], msd[~mask]
        self.tau, self.msd = tau, msd
        return self.tau, self.msd
    
    def compute_alpha(self, degree=10, show_plots=True):
        """Compute log-log MSD slope (alpha) by fitting a higher order polynomial"""
                
        logtau, logmsd = np.log10(self.tau), np.log10(self.msd) # skip first data point
        
        # remove any nans/infs
        mask = np.isfinite(logtau) & np.isfinite(logmsd)
        logtau, logmsd = logtau[mask], logmsd[mask]
                
        window_size = 19 # must be odd
        poly_deg = 2
        # logmsd_smoothed = savgol_filter(logmsd[10:], window_size, polyorder=poly_deg) # skip first 10 datapoints due to sparcity of data

        logmsd_smoothed = non_uniform_savgol(logtau, logmsd, window_size, poly_deg)
        # logmsd_smoothed = np.concatenate((logmsd[:10], logmsd_smoothed)) # remerge with firt 10 datapoints of the raw data

        poly_fit = Polynomial.fit(logtau, logmsd_smoothed, deg=degree)
        poly_deriv = poly_fit.deriv(m=1)
        
        logtau_list = np.arange(logtau.min(), logtau.max(), N)
        omega_list = 1 / (10**logtau_list)
        logmsd_list = poly_fit(logtau_list)
        msd_list = 10**(logmsd_list)
        alpha_list = poly_deriv(logtau_list)

        if show_plots:
            is_negative = alpha_list < 0

            # use np.diff to identify True-False and False-True transitions
            transitions = np.diff(is_negative.astype(int))
            start_idxs = np.where(transitions == 1)[0] + 1
            end_idxs = np.where(transitions == -1)[0]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), tight_layout=True)
            # plt.plot(logtau_bins, logmsd_bins, 'mo', label="Binned Data")
            for start, end in zip(start_idxs, end_idxs):
                plt.axvspan(logtau_list[start], logtau_list[end], color='r', alpha=0.25, label='Negative Derivative' if start == start_idxs[0] else '')

            ax1.plot(logtau, logmsd, 'k.', label="Raw")
            ax1.plot(logtau, logmsd_smoothed, 'b--', label="Smoothed")
            ax1.plot(logtau_list, logmsd_list, 'r-', label="Fit")

            ax1.legend()
            ax1.set_title('LogLog MSD')
            ax1.set_xlabel('log(tau)')
            ax1.set_ylabel('log(MSD)')
            
            ax2.plot(logtau_list, alpha_list, 'r-', label='Spline')

            for start, end in zip(start_idxs, end_idxs):
                ax2.axvspan(logtau_list[start], logtau_list[end], color='r', alpha=0.25, label='Negative Derivative' if start == start_idxs[0] else '')
            ax2.set_title('LogLog MSD Derivative')
            ax2.set_xlabel('log(tau)')
            ax2.set_ylabel('log(MSD)\'')
            plt.show()
        
        self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list = logtau_list, logmsd_list, alpha_list, omega_list, msd_list
        
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