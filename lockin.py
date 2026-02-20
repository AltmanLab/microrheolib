from numpy import ndarray
from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.signal import ShortTimeFFT, windows

class LockinAnalysis(OTDataset):
    """Class for processing active OT datasets using the lock-in amplifier.

    Typical workflow:
        analyze_all_trials()
        compute_moduli()
        plot_g1g2()
    """

    def __init__(self, name, fh, analysisparams, params=None):
        '''fh: file handle of the active lock-in dataset i.e. bead1 !!!NO TAILING SLASH!!! '''
        super().__init__(name, fh, analysisparams, params)
        
        self.analysisparams = analysisparams

        # Load the data
        self.fh = fh +'/'
        self.data_dict = self._load_lockin_data(fh)
            
    def analyze_trial(self, data:dict, show_plots=False, debug_mode=False) -> dict:
        """Analyze a single trial and return results"""
        
        dom_freqs, change_idxs = self.compute_freq_changes_sfft(data['fg'], show_plots=show_plots)

        chunk_info = self._segment_chunks(data, dom_freqs, change_idxs)
             
        means = self._compute_chunk_means(data, chunk_info)
        
        means['fg_freq'] = chunk_info['fg_freq']
                
        return means
        
    def analyze_all_trials(self, data:dict, show_plots=False, debug_mode=False) -> dict:
        '''Calculate the 'results' object for all of the trials'''
        
        trial_info = [
            self.analyze_trial(data[trial], show_plots=show_plots, debug_mode=debug_mode) 
            for trial in data
        ]
        
        # Concatenate dict-of-arrays into one dict
        keys = trial_info[0].keys()
        
        trials = {
            k: np.concatenate([res[k] for res in trial_info])
            for k in keys
        }
        
        # normalize X and Y to zero at highest freq; recompute R
        idx_ref = np.argmax(trials['fg_freq'])
        
        x_ref = trials['mean_x'][idx_ref]
        y_ref = trials['mean_y'][idx_ref]
        
        # to avoid a divide by zero error
        trials['mean_x'] -= x_ref - 0.0001
        trials['mean_y'] -= y_ref - 0.0001
        
        mean_r = np.sqrt(trials['mean_x'] ** 2 + trials['mean_y'] ** 2) / self.Sy * np.sqrt(2)
        
        trials['mean_r'] = mean_r
        
        results = self._compute_force_phase(trials, debug_mode=debug_mode)
                    
        # if show_plots:
        #     fig, axs = plt.subplots(4, 1, figsize=(12, 5), tight_layout=True)
        #     fig.suptitle(f'Frequency Chunk: {fg_freq} Hz')
        #     axs[0].plot(data['x'][start_idx:end_idx])
        #     axs[0].set_title('X Data (Lock-in)')
        #     axs[0].axvline(skip_idx, color='r', linestyle='--')
        #     axs[0].axvline(len(data['x'][start_idx:end_idx]) - fs // 2, color='r', linestyle='--')
        #     axs[1].plot(data['y'][start_idx:end_idx])
        #     axs[1].set_title('Y Data (Lock-in)')
        #     axs[1].axvline(skip_idx, color='r', linestyle='--')
        #     axs[1].axvline(len(data['y'][start_idx:end_idx]) - fs // 2, color='r', linestyle='--')
        #     axs[2].plot(data['r'][start_idx:end_idx])
        #     axs[2].set_title('R Data (Lock-in)')
        #     axs[2].axvline(skip_idx, color='r', linestyle='--')
        #     axs[2].axvline(len(data['r'][start_idx:end_idx]) - fs // 2, color='r', linestyle='--')
        #     fig.show()

        return results
    
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
        freq_list = results["fg_freq"]
        ynot = results["ynot"]
        fnot = results["fnot"]
        delta = results["delta"]
        
        if debug_mode:
            phi = results['phi']
            x = results['x']
            y = results['y']
            
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
            
            # plt.figure(figsize=(12.8, 4.8))
            # plt.subplot(1, 2, 1)
            # r = np.concatenate([data['r'] for data in self.data_dict.values()])
            # plt.plot(r * self.Sy / np.sqrt(2), '.', alpha=0.7, label='Lockin R')
            # plt.plot(r_calc, 'r.', alpha=0.7, label='Calc R')
            # plt.title('Raw R Comparison')
            # plt.subplot(1, 2, 2)
            # plt.plot(np.abs(r_calc - (r * self.Sy / np.sqrt(2))), 'r.')
            # plt.title('Absolute Error')
            # plt.show()
            
            # plt.figure(figsize=(12.8, 4.8))
            # plt.subplot(1, 2, 1)
            # plt.plot(freq_list, ynot * self.Sy / np.sqrt(2), 'o', alpha=0.7, label='Lockin R')
            # plt.plot(freq_list, r_filt, 'go', alpha=0.7, label='Filtered R')
            # plt.xscale('log')
            # plt.legend()
            # plt.title('Filtered R Comparison')
            # plt.subplot(1, 2, 2)
            # plt.plot(freq_list, np.abs(r_filt - (ynot * self.Sy / np.sqrt(2))), 'r.')
            # plt.xscale('log')
            # plt.title('Absolute Error')
            # plt.show()

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
        
    def _segment_chunks(self, data, dom_freqs, change_idxs):
        """Create chunk ids and valid mask after equilibration masking."""
        
        chunk_len = fs // 4 # divide by hop length
        n_samples = len(data['fg'])
        n_chunks = len(change_idxs) - 1
        
        chunk_id = np.zeros(n_samples, dtype=int)
        
        for i in range(n_chunks):
            start_idx = change_idxs[i] * chunk_len
            end_idx = change_idxs[i + 1] * chunk_len
            chunk_id[start_idx:end_idx] = i
            
        
        # remove equilibration portion
        skip_percent = 0.75
        valid_mask = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_chunks):
            start_idx = change_idxs[i] * chunk_len
            end_idx = change_idxs[i + 1] * chunk_len
            
            skip_idx = int((end_idx - fs // 2 - start_idx) // (1 / skip_percent))
            valid_start = start_idx + skip_idx
            valid_end = end_idx - fs // 2
            
            valid_mask[valid_start:valid_end] = True
            
        mid_idx = (change_idxs[:-1] + change_idxs[1:] - 1) // 2
        fg_freq = dom_freqs[mid_idx]
        
        return {
            'chunk_id': chunk_id,
            'valid_mask': valid_mask,
            'n_chunks': n_chunks,
            'fg_freq': fg_freq
        }


    def _compute_chunk_means(self, data, chunk_info):
        """Compute chunk average X, Y, R, and fg statistics per trial."""
                
        cid = chunk_info["chunk_id"][chunk_info["valid_mask"]]
        n_chunks = chunk_info["n_chunks"]

        counts = np.bincount(cid, minlength=n_chunks)
                                        
        mean_x = np.bincount(cid, data["x"][chunk_info["valid_mask"]], minlength=n_chunks) / counts
        mean_y = np.bincount(cid, data["y"][chunk_info["valid_mask"]], minlength=n_chunks) / counts
        mean_r = np.bincount(cid, data["r"][chunk_info["valid_mask"]], minlength=n_chunks) / counts
                        
        fg_valid = data["fg"][chunk_info["valid_mask"]]
        mean_fg = np.bincount(cid, weights=fg_valid, minlength=n_chunks) / counts
        mean_fg2 = np.bincount(cid, weights=fg_valid**2, minlength=n_chunks) / counts - mean_fg ** 2

        std_fg = np.sqrt(mean_fg2 - mean_fg ** 2)

        return {
            "mean_x": mean_x,
            "mean_y": mean_y,
            "mean_r": mean_r,
            "std_fg": std_fg,
        }
        
    def _compute_force_phase(self, data, debug_mode=False):
        """Compute force magnitue and phase dependent components."""
        
        y_trap = self.a * np.sqrt(2) * data['std_fg'] + self.b
        A = self.ky * y_trap
        
        B = -self.ky * data['mean_r']
        phi = np.arctan2(data['mean_y'], data['mean_x'])
        
        C = np.sqrt(A**2 + B**2 + 2 * A * B * np.cos(phi))
        phi_new = np.arctan2(B * np.sin(phi), A + B * np.cos(phi))

        delta_phi = phi - phi_new
                
        results = {
            'fg_freq': data['fg_freq'],
            'fnot': C,
            'ynot': data['mean_r'],
            'delta': delta_phi,
        }
        
        if debug_mode:
            results.update({'phi': phi, 'x': data['mean_x'], 'y': data['mean_y']})
            
            plt.figure()
            plt.plot(A, '.', label='trap')
            plt.plot(B, '.', label='bead')
            plt.legend()
            plt.show()
            
        return results