from typing_extensions import deprecated
from microrheo.base import OTDataset
from microrheo.utils import *

class ActiveAnalysis(OTDataset):
    '''Class for processing active OT datasets
    
    General order:
        compute_frequency_changes: calculates frequencies and identifies the frequency regions
        compute_chunk_results: calculates y and fg amplitudes, as well as the phase difference for each frequency region
        compute_moduli: calculates G' and G" from the resulting data
        plot_g1g2: plot final G' and G" graph
    '''
    def __init__(self, fh, analysisparams, params=None):
        '''fh: file handle of the active dataset i.e. bead1 !!!NO TAILING SLASH!!! '''
        super().__init__(fh, analysisparams, params)
        
        self.analysisparams = analysisparams

        # Load the data
        self.fh = fh +'/'
        self.x, self.y, self.sumsignal, self.fg = self._load_data(self.fh, analysisparams.data_starttime, analysisparams.data_endtime)
        self.x /= self.Sx
        self.y /= self.Sy
        self.t = np.arange(len(self.x))*dt

    def compute_freq_changes(self, fg:np.ndarray, show_plots=False):
        dominant_freqs = []
        for i in range(0, len(fg) - fs, fs): # second chunks
            fg_chunk = fg[i:i+fs]
            
            fft = scipy.fft.fft(fg_chunk)
            N = len(fg_chunk)
            n = np.arange(N)
            T = N / fs
            freq = n / T
            index = (freq > 0.1) * (freq < 20000)
            freq = freq[index]
            fft = np.abs(fft[index])
            top_freq = freq[fft == np.max(fft)]
            dominant_freqs.append(top_freq[0])
            
        dominant_freqs = np.array(dominant_freqs)
        freq_changes = np.where(np.diff(dominant_freqs) > 0.5)[0]
        freq_changes = np.append(freq_changes, len(dominant_freqs) - 1) # add last index
        
        if show_plots:
            plt.figure()
            plt.plot(dominant_freqs, label='Dominant Frequencies')
            [plt.text(i, dominant_freqs[i] + 100, f"{dominant_freqs[i]:.0f}") for i in freq_changes]
            plt.title("Dominant Frequency Per Chunk")
            plt.xlabel("Chunk Index")
            plt.ylabel("Frequency (Hz)")
            plt.show()
            
        return dominant_freqs, freq_changes

    def compute_chunk_results(self, dominant_freqs, freq_changes, show_plots=False):
        results = []
        for i in range(len(freq_changes) - 1):
            result = self.analyze_freq_chunk(i, dominant_freqs, freq_changes, show_plots)
            if result:
                results.append(result)
                
        return results
    
    # TODO: why does this silently skip over chunks that fail to fit
    def analyze_freq_chunk(self, i, dominant_freqs, freq_changes, show_plots):
        if i == 0:
            f = dominant_freqs[0]
            start = 0
            end = freq_changes[0] * fs
        else:
            f = dominant_freqs[freq_changes[i]]
            start = (freq_changes[i - 1] + 1) * fs + (fs // 2) #catching tailing ends of frequencies by throwing out starting half second of data
            end = (freq_changes[i]) * fs

        fg_chunk = self.fg[start:end]
        y_chunk = self.y[start:end]
        t_chunk = np.arange(len(fg_chunk)) * dt

        if len(fg_chunk) == 0:
            print(f'Empty signal chunk for {start/fs} - {end/fs}s')
            return None

        fgfit = fg_chunk - np.mean(fg_chunk) # in V
        OTdata = self.a * fg_chunk + self.b # in Nm
        
        def sinefit(t, A, C, D, omega):
            return np.abs(A) * np.sin(omega*t - C) + D

        def sinefit2(freqoscil):
            return lambda t, A, C, D: np.abs(A) * np.sin(freqoscil*t - C) + D

        try:
            params, _ = scipy.optimize.curve_fit(
                sinefit, t_chunk, fgfit,
                p0=[np.std(fgfit)*np.sqrt(2), 0, 0, 2*np.pi*f]
            )
        except RuntimeError:
            print(f'[Freq Chunk {f:.2f} Hz] first curve fit failed')
            return None
        
        # if the fit is bad, try alternate
        if np.abs(np.std(fgfit)*np.sqrt(2)/params[0]) > 2:
            try: 
                params, _ = scipy.optimize.curve_fit(
                    sinefit, t_chunk, fgfit,
                    p0=[np.std(fgfit)*np.sqrt(2), np.pi, 0, 2*np.pi*f]
                )
            except RuntimeError:
                print(f'[Freq Chunk {f:.2f} Hz] second curve fit failed')
                return None
            
        fg_freq = params[3] / (2*np.pi)
            
        # check fit quality
        if np.mean(np.abs(fg_chunk - sinefit(t_chunk, *params))) > 0.002:
            print(f'[Freq Chunk {f:.2f} Hz] skipping chunk; bad quality')
            return None
        
        if show_plots:
            plt.figure(tight_layout=True)
            plt.subplot(411)
            plt.plot(t_chunk, fg_chunk, 'k')
            plt.plot(t_chunk, sinefit(t_chunk, *params), 'r-')
            plt.title(f'{i}th Frequency: {fg_freq:.2f} Hz')
            
        # --- force and y processing ---
        Fcurve = -(self.ky / 1e-12 * 1e-9) * (((1e9*y_chunk - np.mean(1e9*y_chunk)) - (OTdata - np.mean(OTdata)))) # pN
        oscil_freq = fg_freq
        Noscil = 5
        avg_len = int(Noscil / oscil_freq / dt)
        times = dt * np.arange(avg_len)
                
        # fit y (is this even neccessary?)
        try:
            y_params_raw, _ = scipy.optimize.curve_fit(
                sinefit2(2*np.pi*oscil_freq), t_chunk, y_chunk,
                p0=[np.std(y_chunk) * np.sqrt(2), 0, 0]
            )
        except RuntimeError:
            return None
        
        y_phase_raw = y_params_raw[1]
        
        if show_plots:
            plt.subplot(412).set_title('Y Data Averaging (black is average)')

        # average y data and fit for amplitude
        avg_y = np.zeros(avg_len)
        count = 0
        for j in range(0, len(y_chunk), avg_len):
            segment = y_chunk[j:j + avg_len]
            if len(segment) == avg_len:
                avg_y += segment
                count += 1
                if show_plots:
                    plt.plot(times, segment, ',')
                    
        avg_y /= count
        
        y_params_avg, _ = scipy.optimize.curve_fit(
                sinefit2(2*np.pi*oscil_freq), times, avg_y, 
                p0=[np.std(y_chunk)*np.sqrt(2), 0, 0]
            )
        y_phase = y_params_avg[1]
        y_amp = y_params_avg[0]

        if show_plots:
            plt.plot(times, avg_y, 'k-')
            
        # --- average Fcurve and fit ---
        if show_plots:
            plt.subplot(413).set_title('Force Averaging (black is average)')

        avg_F = np.zeros(avg_len)
        count = 0
        for j in range(0, len(Fcurve), avg_len):
            segment = Fcurve[j:j + avg_len]
            if len(segment) == avg_len:
                avg_F += segment
                count += 1
                if show_plots:
                    plt.plot(times, segment, ',')
                    
        avg_F /= count
        
        f_params_avg, _ = scipy.optimize.curve_fit(
            sinefit2(2 * np.pi * oscil_freq), times, avg_F,
            p0=[np.std(Fcurve) * np.sqrt(2), 0, 0]
        )
        f_phase = f_params_avg[1]
        f_amp = f_params_avg[0]

        if show_plots:
            plt.plot(times, avg_F, 'k-')
            
        # phase difference
        delta_phase = y_phase - f_phase
        if delta_phase < 0:
            delta_phase += 2*np.pi
            
        if show_plots:
            plt.subplot(414).set_title('Bead distance from trap')
            plt.plot(t_chunk, (1e9 * y_chunk - np.mean(1e9 * y_chunk)) - (OTdata - np.mean(OTdata)), 'C7,')
                
        return {
            'fg_freq': fg_freq,
            'ynot': y_amp,
            'fnot': f_amp * 1e-12, # convert to SI units
            'delta': delta_phase
        }
        
    def compute_moduli(self, results:dict):
        freq_list = np.array([r['fg_freq'] for r in results])
        ynot = np.array([r['ynot'] for r in results])
        fnot = np.array([r['fnot'] for r in results])
        delta = np.array([r['delta'] for r in results])

        G1 = np.abs(fnot / (6 * np.pi * ExperimentalConstants.r * ynot)) * np.cos(delta)
        G2 = np.abs(fnot / (6 * np.pi * ExperimentalConstants.r * ynot)) * np.sin(delta)
        
        self.G1 = G1
        self.G2 = G2
        self.omega_list = 2*np.pi * freq_list 

    def plot_g1g2(self):
        plt.figure()
        plt.loglog(self.omega_list, self.G1, 'k.', label='G\'')
        plt.loglog(self.omega_list, self.G2, 'r.', label='G\'\'')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()

    @deprecated('Use the seperated functions instead!')
    def final_analysis(self, analysisparams, show_freq_plot=True):
        '''Run the final analysis on the active data, resulting in self.G1, self.G2'''
        y,fg,t = self.y, self.fg, self.t #just for ease
        # Calculate the power spectrum of the data
        frequences = []
        for i in range(0,len(fg)-fs,fs): #second chunks
            fgchunk = fg[i:i+fs]
            tchunk = t[i:i+fs]
            Y = scipy.fft.fft(fgchunk)
            N = len(fgchunk)
            n = np.arange(N)
            T = N/fs
            freq = n/T
            index = (freq > 0.1) * (freq < 20000)
            freq = freq[index]
            Y = np.abs(Y[index])
            top_freq = freq[Y == np.max(Y)]
            frequences.append(top_freq[0])

        frequences = np.array(frequences)
        changeinfreq = (np.diff(frequences) > 0.5)
        freq_changes = np.arange(len(changeinfreq))[changeinfreq]

        print('Frequency changes:', freq_changes)
        def sinefit(t, A, C, D, omega):
            return np.abs(A) * np.sin(omega*t - C)+D
        def sinefit2(freqoscil):
            def func(t,*p):
                return np.abs(p[0]) * np.sin(freqoscil*t - p[1])+p[2]
            return func

        freqlist = []
        fnot = []
        ynot = []
        delta = []

        for i in range(len(freq_changes)-1):
            if i == 0:
                f = 1
                start = 0
                end = freq_changes[0]*fs
            else:
                f = frequences[freq_changes[i]]
                start = (freq_changes[i-1]+1)*fs + (fs // 2) #catching tailing ends of frequencies by throwing out starting half second of data
                end = (freq_changes[i])*fs

            T = 1/f #period
            y = self.y[start:end]
            fg = self.fg[start:end]
            t = np.arange(len(fg))*dt

            if len(fg) == 0:
                print(f'function generator signal is empty for {start/fs} - {end/fs} seconds segment')
                continue

            fgfit = fg - np.mean(fg) # in V
            OTdata = self.a * fg + self.b # in nm

            # Fit the fg data to a sine
            params, params_cov = scipy.optimize.curve_fit(sinefit, t, fgfit, p0=[np.std(fgfit)*np.sqrt(2), 0, 0, 2*np.pi*f])
            #print('For frequency ', params[3]/(2*np.pi), ' the mean difference is ', np.mean(np.abs(fg-sinefit(t, *params))))
            if np.abs(np.std(fgfit)*np.sqrt(2)/params[0]) > 2:
                params, params_cov = scipy.optimize.curve_fit(sinefit, t, fgfit, p0=[np.std(fgfit)*np.sqrt(2), np.pi, 0, 2*np.pi*f])
                #print('TRIED AGAIN')
                #print('*** For frequency ', params[3]/(2*np.pi), ' the mean difference is ', np.mean(np.abs(fg-sinefit(t, *params))))
            if np.mean(np.abs(fg-sinefit(t, *params))) < 0.002:
              phase = params[1]
              fg_freq = params[3]/(2*np.pi)
              freqlist.append(fg_freq)

              if show_freq_plot:
                  plt.figure(tight_layout=True)
                  plt.subplot(411)
                  plt.plot(t, fg, 'k')
                  plt.plot(t, sinefit(t, *params), 'r-')
                  plt.title(f'{i}th Frequency: {fg_freq} Hz')

              Fcurve = -(self.ky / 1e-12 * 1e-9) * (((1e9*y-np.mean(1e9*y)) - (OTdata-np.mean(OTdata)))) # pN

              # Fit the y data to a sine
              Noscil = 5
              oscil_freq = fg_freq
              params, params_cov = scipy.optimize.curve_fit(sinefit2(2*np.pi*oscil_freq), t, y, p0=[np.std(y)*np.sqrt(2), 0, 0])
              params = np.append(params, 2*np.pi*oscil_freq)
              phase = params[1]

              if show_freq_plot:
                  plt.subplot(412).set_title('Y Data Averaging (black is average)')

              avg_y = np.zeros(int(Noscil/oscil_freq/dt))
              times = dt*np.arange(len(avg_y))
              count = 0
              for j in range(0, len(y), avg_y.shape[0]):
                  if len(y[j:j+int(Noscil/oscil_freq/dt)]) == len(avg_y):
                      count += 1
                      avg_y += y[j:j+int(Noscil/oscil_freq/dt)]
                      if show_freq_plot:
                          plt.plot(times, y[j:j+int((Noscil/oscil_freq)/dt)], ',')
              avg_y /= count

              if show_freq_plot:
                  plt.plot(times, avg_y, 'k-')
                  
              params, params_cov = scipy.optimize.curve_fit(sinefit2(2*np.pi*oscil_freq), times, avg_y, p0=[np.std(y)*np.sqrt(2), 0, 0])
              params = np.append(params, 2*np.pi*oscil_freq)
              yphase = params[1]
              ynot.append(params[0])

              # Fit the forces
              params, params_covariance = scipy.optimize.curve_fit(sinefit2(2*np.pi*oscil_freq), t, Fcurve, p0=[np.std(Fcurve)*np.sqrt(2), 0, 0])
              params = np.append(params, 2*np.pi*oscil_freq)

              if show_freq_plot:
                  plt.subplot(413).set_title('Force Averaging (black is average)')

              avg_F = np.zeros(int(Noscil/oscil_freq/dt))
              times = dt*np.arange(len(avg_F))
              count = 0
              for j in range(0, len(Fcurve), avg_F.shape[0]):
                  if len(Fcurve[j:j+int(Noscil/oscil_freq/dt)]) == len(avg_F):
                      count += 1
                      avg_F += Fcurve[j:j+int(Noscil/oscil_freq/dt)]
                      if show_freq_plot:
                          plt.plot(times, Fcurve[j:j+int((Noscil/oscil_freq)/dt)], ',')
              avg_F /= count

              if show_freq_plot:
                  plt.plot(times, avg_F, 'k-')
              params, params_cov = scipy.optimize.curve_fit(sinefit2(2*np.pi*oscil_freq), times, avg_F, p0=[np.std(Fcurve)*np.sqrt(2), 0, 0])
              params = np.append(params, 2*np.pi*oscil_freq)

              deltatemp = yphase - params[1]
              if deltatemp < 0:
                  deltatemp += 2*np.pi
              delta.append(deltatemp)
              fnot.append(params[0]*1e-12)

              if show_freq_plot:
                  plt.subplot(414).set_title('Bead distance from trap')
                  plt.plot(t, (1e9*y-np.mean(1e9*y)) - (OTdata-np.mean(OTdata)), 'C7,')



        freqlist = np.array(freqlist)
        print(freqlist)
        ynot = np.array(ynot)
        fnot = np.array(fnot)
        delta = np.array(delta)

        # Calculate the G'(G1) and G''(G2)
        G1 = np.abs(fnot/(6*np.pi*1e-6*ynot))*np.cos(delta)
        G2 = np.abs(fnot/(6*np.pi*1e-6*ynot))*np.sin(delta)
        self.G1 = G1
        self.G2 = G2
        self.freqlist = freqlist
        plt.figure()
        plt.loglog(freqlist, G1, 'k*', label='G\'')
        plt.loglog(freqlist, G2, 'r*', label='G\'\'')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()