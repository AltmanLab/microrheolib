from typing_extensions import deprecated
from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.signal import savgol_filter
from numpy.polynomial import Polynomial
from scipy.interpolate import UnivariateSpline

class PassiveAnalysis(OTDataset):
    """Process passive microrheology datasets.

    Typical workflow::

        passive = PassiveAnalysis(name, fh, analysisparams)
        passive.compute_msd()
        passive.compute_alpha(show_plots=True, degree=10)
        passive.compute_moduli(trap=True, free_memory=False)
        passive.plot_g1g2()
    """

    def __init__(self, name, fh, analysisparams, params=None):
        """Initialize a passive microrheology analysis object

        Parameters
        ---------
        name : str
            Human readable display name
        fh : str
            Relative file handle of the passive dataset i.e. 2600608/bead1p
        aalysisparams : dict
            Analysis configuration
        params : optional
            If provided, skips computing the Brownian power spectrum calibration and uses these parameters instead
        """
        super().__init__(name, fh, analysisparams, params)

        self.analysisparams = analysisparams

        # Load the data, all in meters
        self.fh = fh +'/'
        self.x, self.y, self.sumsignal, self.fg = self._load_data(self.fh, analysisparams.data_starttime, analysisparams.data_endtime)
        self.x /= self.Sx
        self.y /= self.Sy
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
        
        self.free_memory()

        # mask = np.isnan(tau) | np.isnan(msd) # remove nan values
        # return tau[~mask], msd[~mask]
        self.tau, self.msd = tau, msd
        return self.tau, self.msd
    
    def compute_alpha(self, degree=20, show_plots=False):
        '''Compute log-log MSD slope (alpha) by fitting a higher order polynomial, and other stuff ???'''
                
        logtau, logmsd = np.log10(self.tau[1:]), np.log10(self.msd[1:]) # skip first data point
        N = self.analysisparams.alphaincrement
        
        # remove any nans
        logtau = logtau[~np.isnan(logtau)]
        logmsd = logmsd[~np.isnan(logmsd)]
                
        window_size = 19 # must be odd
        poly_deg = 2
        # logmsd_smoothed = savgol_filter(logmsd[10:], window_size, polyorder=poly_deg) # skip first 10 datapoints due to sparcity of data

        logmsd_smoothed = non_uniform_savgol(logtau, logmsd, window_size, poly_deg) # skip first 10 datapoints due to sparcity of data
        # logmsd_smoothed = np.concatenate((logmsd[:10], logmsd_smoothed)) # remerge with firt 10 datapoints of the raw data
        
        poly_fit = Polynomial.fit(logtau, logmsd_smoothed, deg=degree)
        poly_deriv = poly_fit.deriv(m=1)

        # try it with a spline instead of a polynomial fit
        # first, bin the data into logtau bins of size N, and take the mean logmsd in each bin
        # logtau_bins = np.logspace(np.log10(logtau.min()), np.log10(logtau.max()), 512011) # logspace bins of size N
        # logmsd_bins = np.zeros_like(logtau_bins)
        # for i in range(len(logtau_bins)-1):
        #     idx = (logtau >= logtau_bins[i]) & (logtau < logtau_bins[i+1])
        #     logmsd_bins[i] = np.mean(logmsd[idx])
        # logtau_bins = logtau_bins[:-1] + N/2 # convert to bin centers
        # logmsd_bins = logmsd_bins[:-1] # remove last bin which is empty
        # spline_fit = UnivariateSpline(logtau_bins, logmsd_bins, s=0.3)
        # spline_deriv = spline_fit.derivative(n=1)
        
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

            plt.figure()
            # plt.plot(logtau_bins, logmsd_bins, 'mo', label="Binned Data")
            for start, end in zip(start_idxs, end_idxs):
                plt.axvspan(logtau_list[start], logtau_list[end], color='r', alpha=0.25, label='Negative Derivative' if start == start_idxs[0] else '')

            plt.plot(logtau, logmsd, 'k.', label="Raw")
            plt.plot(logtau, logmsd_smoothed, 'b--', label="Smoothed")
            plt.plot(logtau_list, logmsd_list, 'r-', label="Fit")

            # plt.plot(logtau_list, spline_fit(logtau_list), 'g-', label="Spline Fit")
            plt.legend()
            plt.title('LogLog MSD')
            plt.xlabel('log(tau)')
            plt.ylabel('log(MSD)')
            plt.show()
            
            plt.figure()
            plt.plot(logtau_list, alpha_list, color='tab:red', linestyle='solid')
            for start, end in zip(start_idxs, end_idxs):
                plt.axvspan(logtau_list[start], logtau_list[end], color='r', alpha=0.25, label='Negative Derivative' if start == start_idxs[0] else '')
            plt.title('LogLog MSD Derivative')
            plt.xlabel('log(tau)')
            plt.ylabel('log(MSD)\'')
            plt.show()
        
        self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list = logtau_list, logmsd_list, alpha_list, omega_list, msd_list
        
        return self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list

    def compute_moduli(self, free_memory=False, trap=False):
        '''Compute G' and G'' moduli'''
        
        d = 2 # dimensionality of the system

        ktrap = (self.kx + self.ky)/2 # average trap stiffness
        offset = ktrap/(6*np.pi*ExperimentalConstants.r)
        
        # G1, G2 = [], []
        # for i in range(0, len(self.omega_list)):
        #     gamma = (0.457*(1+self.alpha_list[i])**2) - (1.36*(1+self.alpha_list[i])) + 1.9 #approx value from Mason 2000 p.373 below eqn. 8
        #     Gstar = np.abs((2*d*K_B*T)/(6*np.pi*ExperimentalConstants.r*self.msd_list[i]*gamma))
        #     G1temp = Gstar*np.cos(self.alpha_list[i]*np.pi/2)
        #     G2temp = Gstar*np.sin(self.alpha_list[i]*np.pi/2)
        #     if trap: #if the trap was on, we need to subract its forces on the bead
        #         G1temp -= offset
        #     G1.append(G1temp)
        #     G2.append(G2temp)

        gamma = (0.457 * ( 1 + self.alpha_list)**2) - (1.36 * (1 + self.alpha_list)) + 1.9 #approx value from Mason 2000 p.373 below eqn. 8
        G_star = np.abs((2 * d * kBT) / (6 * np.pi * ExperimentalConstants.r * self.msd_list * gamma))
        G1 = G_star * np.cos(self.alpha_list * np.pi / 2)
        G2 = G_star * np.sin(self.alpha_list * np.pi / 2)
        if trap: # if the trap was on, we need to subract its forces on the bead
            G1 -= offset
            
        if free_memory:
            self.logtau_list, self.logmsd_list, self.alpha_list, self.msd_list = None, None, None, None

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

    def plot_g1g2(self):
        plt.figure()
        plt.loglog(self.omega_list, self.G1, 'k.', label='G\'')
        plt.loglog(self.omega_list, self.G2, 'r.', label='G\'\'')
        plt.xlabel('Angular Frequency (Radians)', )
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()

    @deprecated('Use the seperated functions instead!')
    def final_analysisold(self, analysisparams, trap=False, plots=['G1G2']):
        '''Run the final analysis on the passive data, resulting in self.G1, self.G2
        Args:
        trap: bool, indicate if the trap was on or off
        plots: list of plots to show, options:
            'msd': plot the log-log plot of MSD vs tau
            'params': plot the passive analysis parameters
            'G1G2': plot the G1 and G2 values
        '''

        x,y,t = self.x, self.y, self.t

        avg_msd = np.zeros(int(0.1*fs))
        msd_df:pd.DataFrame = calcmsd2(x, y, t[-1], 20) #max tau of 20 seconds
        if plots == []:
            plt.close('all')
        alpha_list = []
        omega_list = []
        logtau_list = []
        logmsd_list = []
        tau_list = []
        msd_list = []
        analsis_length = 20 #???  -- perhaps make this variable?

        def linear_alpha(x, m, b):
            return m*x + b

        N = analysisparams.alphaincrement # iteration steps (typically 0.2)

        msd = msd_df.msd[:int(analsis_length*fs)]
        tau = msd_df.tau[:int(analsis_length*fs)]
        logmsd = np.log10(msd)
        logtau = np.log10(tau)

        if 'msd' in plots:
            plt.figure()
            plt.plot(logtau, logmsd, 'k')
            plt.xlabel('log(tau)')
            plt.ylabel('log(MSD)')
            plt.title('Log-Log plot of MSD vs tau')
            plt.grid()

        for i in np.arange(-5, 5-N, N):
            idx = (i < logtau) * (logtau < i+N)

            logmsd_list.append(np.mean(logmsd[idx]))
            #logtau_list.append(np.mean(logtau[idx]))
            logtau_list.append(((2*i+N)/2))
            tau_list.append(np.mean(tau[idx]))
            msd_list.append(np.mean(msd[idx]))

            omega_list.append(1/(10**logtau_list[-1]))

            if len(logtau_list) > 1:
                slope = (logmsd_list[-1] - logmsd_list[-2]) / (logtau_list[-1] - logtau_list[-2])
            else:
                slope = float("NaN")

            alpha_list.append(slope)

        if 'params' in plots:
            plt.figure()
            plt.suptitle('Passive Analysis Parameters')
            plt.subplot(2,1,1)
            plt.plot(logtau_list, logmsd_list, 'ko')
            plt.ylabel('log(msd)')

            plt.subplot(2,1,2)
            plt.plot(logtau_list, alpha_list, 'ko')
            plt.xlabel('log(tau)')
            plt.ylabel('alpha')

        d = 2 # dimensionality of the system
        T = 293.15 # room temperature in K
        K_B = 1.38064852e-23 # Boltzmann constant in J/K
        R = (2e-6)/2 # radius of the bead in m

        ktrap = (self.kx + self.ky)/2 # average trap stiffness
        offset = ktrap/(6*np.pi*R)

        G1 = []
        G2 = []

        for i in range(1,len(omega_list)):
            gamma = (0.457*(1+alpha_list[i])**2) - (1.36*(1+alpha_list[i])) + 1.9 #approx value from Mason 2000 p.373 below eqn. 8
            Gstar = abs((2*d*K_B*T)/(6*np.pi*R*msd_list[i]*gamma))
            G1temp = Gstar*np.cos(alpha_list[i]*np.pi/2)
            G2temp = Gstar*np.sin(alpha_list[i]*np.pi/2)
            if trap: #if the trap was on, we need to subract its forces on the bead
                G1temp -= offset
            G1.append(G1temp)
            G2.append(G2temp)
        G1 = np.array(G1)
        G2 = np.array(G2)
        omega_list = np.array(omega_list[1:])

        self.G1 = G1
        self.G2 = G2
        self.omegalist = omega_list

        if 'G1G2' in plots:
            plt.figure()
            plt.loglog(omega_list, G1, 'k*', label='G\'')
            plt.loglog(omega_list, G2, 'r*', label='G\'\'')
            plt.xlabel('Angular Frequency (Radians)', )
            plt.ylabel('Viscoelastic Moduli (Pa)')
            plt.legend()