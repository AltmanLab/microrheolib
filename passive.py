from typing_extensions import deprecated
from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.interpolate import interp1d

class PassiveAnalysis(OTDataset): # inheritance, with OTDataset = parent, PassiveAnalysis = child
    '''Class for processing passive OT datasets'''
    def __init__(self, name, fh, analysisparams, params=None):
        '''fh: file handle for where brownian folder can be found, subhandle: relative file handle of the passive dataset i.e. bead1p/'''
        super().__init__(name, fh, analysisparams, params)

        self.analysisparams = analysisparams

        # Load the data, all in meters
        self.fh = fh +'/'
        self.x, self.y, self.sumsignal, self.fg = self._load_data(self.fh, analysisparams.data_starttime, analysisparams.data_endtime)
        self.x/=self.Sx
        self.y/=self.Sy
        self.t = np.arange(len(self.x))*dt   # starts at zero

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
        '''Compute log-log slope (alpha) by fitting a higher order polynomial, and other stuff ???'''
                
        log_tau, log_msd = np.log10(self.tau), np.log10(self.msd)
        N = self.analysisparams.alphaincrement
        
        logmsd_list, logtau_list, tau_list, msd_list, omega_list, alpha_list = [], [], [], [], [],[]

        for i in np.arange(-5, 5 - N, N): # step through tau in increments of N
          idx = (i < log_tau) & (log_tau < i + N)
          logmsd_list.append(np.mean(log_msd[idx]))
          logtau_list.append(((2 * i + N) / 2))
          tau_list.append(np.mean(self.tau[idx]))
          msd_list.append(np.mean(self.msd[idx]))
          omega_list.append(1 / (10**logtau_list[-1]))

        # clean NaNs with interpolation
        index = ~np.isnan(msd_list)
        logtau_list_filtered = np.array(logtau_list)[index]
        logmsd_list_filtered = np.array(logmsd_list)[index]
        omega_list_filtered = np.array(omega_list)[index]

        #? there is likely no reason to be doing any interpolating to fill in the alpha steps
        # index = (logtau_list > np.min(logtau_list_temp))*(logtau_list < np.max(logtau_list_temp))
        # logtau_list = np.array(logtau_list)[index]
        # f_interp = interp1d(logtau_list_temp, logmsd_list_temp, kind='linear') # or 'cubic'
        # logmsd_list= f_interp(logtau_list)
        # omega_list = np.array(omega_list)[index]
        # msd_list = np.array(msd_list)[index]

        # fit polynomial to MSD curve
        X = logtau_list_filtered# - logtau_list_filtered[0]
        
        coef = np.polyfit(X, logmsd_list_filtered, deg=degree)
        poly_func = np.poly1d(coef)
        derivative_func = np.polyder(poly_func)
        
        # calculate smoothed data
        logmsd_list = np.polyval(coef, X)
        msd_list = 10**logmsd_list
        alpha_list = [derivative_func(x) for x in X]
        
        if show_plots:
            plt.figure()
            plt.plot(X, logmsd_list_filtered, 'k.', label='Raw')
            plt.plot(X, logmsd_list, color='tab:red', linestyle='solid', label='Fit')
            plt.title('LogLog MSD')
            plt.xlabel('log(tau)')
            plt.ylabel('log(MSD)')
            plt.legend()
            plt.show()
        
        self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list = X, logmsd_list, alpha_list, omega_list_filtered, msd_list
        
        return self.logtau_list, self.logmsd_list, self.alpha_list, self.omega_list, self.msd_list

    def compute_moduli(self, free_memory=False, trap=False):
        '''Compute G' and G'' moduli'''
        
        d = 2 # dimensionality of the system
        T = 293.15 # room temperature in K
        K_B = 1.38064852e-23 # Boltzmann constant in J/K

        ktrap = (self.kx + self.ky)/2 # average trap stiffness
        offset = ktrap/(6*np.pi*ExperimentalConstants.r)
        
        G1, G2 = [], []
        for i in range(1, len(self.omega_list)):
            gamma = (0.457*(1+self.alpha_list[i])**2) - (1.36*(1+self.alpha_list[i])) + 1.9 #approx value from Mason 2000 p.373 below eqn. 8
            Gstar = abs((2*d*K_B*T)/(6*np.pi*ExperimentalConstants.r*self.msd_list[i]*gamma))
            G1temp = Gstar*np.cos(self.alpha_list[i]*np.pi/2)
            G2temp = Gstar*np.sin(self.alpha_list[i]*np.pi/2)
            if trap: #if the trap was on, we need to subract its forces on the bead
                G1temp -= offset
            G1.append(G1temp)
            G2.append(G2temp)
            
        if free_memory:
            self.logtau_list, self.logmsd_list, self.alpha_list, self.msd_list = None, None, None, None

        self.G1 = np.array(G1)
        self.G2 = np.array(G2)
        self.omega_list = np.array(self.omega_list[1:])
        
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