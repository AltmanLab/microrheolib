from numpy import ndarray
from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.signal import ShortTimeFFT, windows

class LockinAnalysis(OTDataset):
    '''Class for processing active OT datasets using the lock-in amplifier
    
    General order:
        analyze_all_trials
        compute_moduli
        plot_g1g2
    '''

    def __init__(self, name, fh, analysisparams, params=None):
        '''fh: file handle of the active lock-in dataset i.e. bead1 !!!NO TAILING SLASH!!! '''
        super().__init__(name, fh, analysisparams, params)
        
        self.analysisparams = analysisparams

        # Load the data
        self.fh = fh +'/'
        self.data_dict = self._load_lockin_data(fh)
        
    def analyze_chunk(self, data:dict, fg_freq, start_index:int, end_index:int, show_plots=False, debug_mode=False) -> dict:
        """Analyze one frequency chunk and create the 'result' object correlating to that frequency chunk"""
        fg_chunk = self.a * data['fg'][start_index:end_index] + self.b
        
        skip_percent = 0.75
        skip_index = int((end_index - fs//2 - start_index) // (1/skip_percent))
        x_chunk = np.mean(data['x'][start_index + skip_index:end_index - fs//2]) # only keep the second half of the data
        y_chunk = np.mean(data['y'][start_index + skip_index:end_index - fs//2]) # this gives time for the lock-in to equilibrate
        r_chunk = np.mean(data['r'][start_index + skip_index:end_index - fs//2])
        
        if show_plots:
            fig, axs = plt.subplots(4, 1, figsize=(12, 5), tight_layout=True)
            fig.suptitle(f'Frequency Chunk: {fg_freq} Hz')
            axs[0].plot(data['x'][start_index:end_index])
            axs[0].set_title('X Data (Lock-in)')
            axs[0].axvline(skip_index, color='r', linestyle='--')
            axs[0].axvline(len(data['x'][start_index:end_index]) - fs // 2, color='r', linestyle='--')
            axs[1].plot(data['y'][start_index:end_index])
            axs[1].set_title('Y Data (Lock-in)')
            axs[1].axvline(skip_index, color='r', linestyle='--')
            axs[1].axvline(len(data['y'][start_index:end_index]) - fs // 2, color='r', linestyle='--')
            axs[2].plot(data['r'][start_index:end_index])
            axs[2].set_title('R Data (Lock-in)')
            axs[2].axvline(skip_index, color='r', linestyle='--')
            axs[2].axvline(len(data['r'][start_index:end_index]) - fs // 2, color='r', linestyle='--')
            fig.show()

        # F_trap = -ky*y_trap - ky*y_bead

        y_trap = np.sqrt(2) * np.std(fg_chunk - np.mean(fg_chunk)) # sqrt(2) * std(zero mean fg)
        A = self.ky * y_trap * 1e-9
        
        B = -self.ky * r_chunk 
        phi = np.arctan2(y_chunk, x_chunk)
        
        C = np.sqrt(A**2 + B**2 + 2 * A * B * np.cos(phi))
        phi_new = np.arctan2(B * np.sin(phi), A + B * np.cos(phi))

        delta_phi = phi - phi_new
                
        result = {
            'fg_freq': None, # gets filled in later
            'fnot': C,
            'ynot': r_chunk,
            'delta': delta_phi,
        }
        
        if debug_mode:
            result.update({'phi': phi, 'x': x_chunk, 'y': y_chunk})
        
        return result
    
    def analyze_trial(self, data:dict, show_plots=False, debug_mode=False) -> dict:
        '''Calculate the 'results' object for one set of data (e.g. one trial)'''
        fg = data['fg']
        freqs, change_idxs = self.compute_freq_changes_sfft(fg, show_plots=show_plots)

        results = []
        for i in range(len(change_idxs) - 1):
            chunk_len = fs // 4 # divide by hop length
            start_idx = change_idxs[i] * chunk_len
            end_idx = change_idxs[i + 1] * chunk_len
            
            freq_idx = (change_idxs[:-1] + change_idxs[1:] - 1) // 2
                        
            result = self.analyze_chunk(data, freqs[freq_idx[i]], start_idx, end_idx, show_plots=show_plots, debug_mode=debug_mode)
            result['fg_freq'] = freqs[freq_idx[i]]
            results.append(result)
            
        return results
        
    def analyze_all_trials(self, data:dict, show_plots=False, debug_mode=False) -> dict:
        '''Calculate the 'results' object for all of the trials'''
        results_all = []
        for trial in data.keys():
            trial_results = self.analyze_trial(data[trial], show_plots=show_plots, debug_mode=debug_mode)
            results_all.extend(trial_results)
            
        return results_all
    
    def compute_freq_changes_sfft(self, fg:np.ndarray, show_plots=False):
        """SFFT style dominant frequency identifier"""
        T = 0.5 # window_duration (seconds)
        N = len(fg) # number of samples
        
        window_len = int(fs * T)
        hop_len = int(window_len * 0.5) # overlap window 50%
        
        window = windows.hann(window_len, sym=False)
        
        SFT = ShortTimeFFT(window, hop=hop_len, fs=fs, mfft=4*window_len) # TODO mfft parameter?
        fft = SFT.stft(fg)
        mag = np.abs(fft)

        if show_plots:
            plt.figure()
            t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
            plt.title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Hann Window)")
            plt.xlabel(f'Time $t$ in seconds ({SFT.p_num(N)}, $\Delta T$ = {SFT.delta_t:.2f} s)')
            plt.ylabel(f'Freq. $f$ in Hz ({SFT.f_pts} bins, $\Delta f = {SFT.delta_f} Hz$)')
            plt.xlim(t_lo, t_hi)
            plt.ylim(1, 6000)
            
            im = plt.imshow(mag, origin='lower', aspect='auto', extent=SFT.extent(N), cmap='viridis')
            plt.colorbar(im, label='Magnitude')
            
            plt.tight_layout()
            plt.show()
        
        dominant_bins = np.argmax(mag, axis=0)
        dom_freqs = SFT.f[dominant_bins] # dominant freq of bin i
                
        freq_tol = 1 # Hz
        
        freq_changes = np.where(np.abs(np.diff(dom_freqs)) > freq_tol)[0] + 1
        boundaries = np.concatenate(([0], freq_changes, [len(dom_freqs)])) # add first and last index

                        
        if show_plots:
            plt.figure()
            plt.plot(dom_freqs, marker='.', label='Dominant Frequencies')
            mid_idx = (boundaries[:-1] + boundaries[1:] - 1) // 2
            [plt.text(i, dom_freqs[i], f"{dom_freqs[i]:.0f}") for i in mid_idx]
            plt.title("Dominant Frequency Per Chunk")
            plt.xlabel("Chunk Index")
            plt.ylabel("Frequency (Hz)")
            plt.show()
        
        return dom_freqs, boundaries
            
    def compute_moduli(self, results:dict, free_memory=False, debug_mode=False):
        freq_list = np.array([r['fg_freq'] for r in results])
        ynot = np.array([r['ynot'] for r in results])
        fnot = np.array([r['fnot'] for r in results])
        delta = np.array([r['delta'] for r in results])
        
        if debug_mode:
            phi = np.array([r['phi'] for r in results])
            x = np.array([r['x'] for r in results])
            y = np.array([r['y'] for r in results])
            
            r_calc = np.concatenate([np.sqrt(data['x']**2 + data['y']**2) for data in self.data_dict.values()])
            r_filt = np.sqrt(x**2 + y**2)
                
        G1 = np.abs(fnot / (6 * np.pi * ExperimentalConstants.r * ynot)) * np.cos(delta)
        G2 = np.abs(fnot / (6 * np.pi * ExperimentalConstants.r * ynot)) * np.sin(delta)
               
        if debug_mode:             
            plt.figure()
            plt.plot(freq_list, x, '.', label='X')
            plt.plot(freq_list, y, 'g.', label='Y')
            plt.plot(freq_list, ynot * self.Sy / np.sqrt(2), 'r.', label='R')
            plt.axhline(0, color='r', linestyle='--')
            plt.ylabel('Lockin (V)')
            plt.xlabel('Freq (Hz)')
            plt.xscale('log')
            plt.legend()
            plt.title('X, Y, R Data')
            plt.show()
            
            plt.figure()
            plt.plot(freq_list, delta, 'r.', label='delta')
            plt.plot(freq_list, phi-delta, 'g.', label='phi_new')
            plt.plot(freq_list, phi, '.', label='phi')
            plt.xlabel('freq (Hz)')
            plt.ylabel('delta phi (rad)')
            plt.axhline(np.pi/2, 0 , 1)
            plt.axhline(np.pi, 0 , 1)
            plt.xscale('log')
            plt.legend()
            plt.title('Deltas')
            plt.show()
            
            plt.figure(figsize=(12.8, 4.8))
            plt.subplot(1, 2, 1)
            r = np.concatenate([data['r'] for data in self.data_dict.values()])
            plt.plot(r * self.Sy / np.sqrt(2), '.', alpha=0.7, label='Lockin R')
            plt.plot(r_calc, 'r.', alpha=0.7, label='Calc R')
            plt.title('Raw R Comparison')
            plt.subplot(1, 2, 2)
            plt.plot(np.abs(r_calc - (r * self.Sy / np.sqrt(2))), 'r.')
            plt.title('Absolute Error')
            plt.show()
            
            plt.figure(figsize=(12.8, 4.8))
            plt.subplot(1, 2, 1)
            plt.plot(freq_list, ynot * self.Sy / np.sqrt(2), 'o', alpha=0.7, label='Lockin R')
            plt.plot(freq_list, r_filt, 'go', alpha=0.7, label='Filtered R')
            plt.xscale('log')
            plt.legend()
            plt.title('Filtered R Comparison')
            plt.subplot(1, 2, 2)
            plt.plot(freq_list, np.abs(r_filt - (ynot * self.Sy / np.sqrt(2))), 'r.')
            plt.xscale('log')
            plt.title('Absolute Error')
            plt.show()

        if free_memory:
            self.data_dict = None
        
        self.G1 = G1
        self.G2 = G2
        self.omega_list = 2*np.pi * freq_list

    def plot_g1g2(self):
        plt.figure()
        plt.loglog(self.omega_list, self.G1, 'k.', label='G\'')
        plt.loglog(self.omega_list, self.G2, 'r.', label='G\'\'')
        plt.xlabel('Angular Frequency (Radians)')
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()