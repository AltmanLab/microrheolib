from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.signal import ShortTimeFFT, windows

class LockinAnalysis(OTDataset):
    """Process active microrheology datasets aquired with a lock-in amplifier.
        
    Typical workflow::

        active = LockinAnalysis(name, fh, analysisparams)
        active.analyze_all_trials()
        active.compute_moduli()
        active.plot_g1g2()
    """

    def __init__(self, fh, constants, calib):
        """
        Parameters
        ----------
        fh : str
            Relative path to the active lock-in dataset directory
        calib : TrapCalibration
            Brownian and FunctionG calibration parameters
        """
        super().__init__(fh, constants, calib)
        
        # Load the data
        self.fh = fh +'/'
        self.data_dict = load_lockin_data(fh)

        for data in self.data_dict.values():
                data['bead_y'] = 1e-9 * self.calib.ed_fit_func(data['bead_y'] - self.calib.v_offset)
            
    def analyze_trial(self, data:dict, show_plots=False, debug_mode=False) -> dict:
        """Segment a signal trail by frequency and return per-chunk statistics"""
        
        dom_freqs, change_idxs = self.compute_freq_changes_sfft(data['fg'], show_plots=show_plots)

        chunk_info = self._segment_chunks(data, dom_freqs, change_idxs)
             
        means = self._compute_chunk_means(data, chunk_info, debug_mode)
        
        means['fg_freq'] = chunk_info['fg_freq']

        means['fg_freq'][means['fg_freq'] == 0] = 50000 # manually set zero to the largest value since it isn't computed correctly by the SFFT
                
        return means
        
    def analyze_all_trials(self, show_plots=False, debug_mode=False) -> dict:
        """Run 'analyze_trial' over every trial and assemble the rheological results"""
        
        if show_plots:
            fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))

        trial_outputs = []
        for ax, (trial_name, trial_data) in zip(axs, self.data_dict.items()):
            dom_freqs, change_idxs = self.compute_freq_changes_sfft(trial_data['fg'], show_plots=show_plots)

            chunk_info = self._segment_chunks(trial_data, dom_freqs, change_idxs)

            means = self._compute_chunk_means(trial_data, chunk_info, debug_mode)
        
            means['fg_freq'] = chunk_info['fg_freq']

            means['fg_freq'][means['fg_freq'] == 0] = 50000 # manually set zero to the largest value since it isn't computed correctly by the SFFT
            trial_outputs.append(means)

            if show_plots:
                ax.plot(dom_freqs, marker='.', label='Dominant Frequencies')
                mid_idx = (change_idxs[:-1] + change_idxs[1:] - 1) // 2
                [ax.text(i, dom_freqs[i]*1.01, f"{dom_freqs[i]:.0f}", ha='center', va='bottom') for i in mid_idx]
                ax.set_title(f"{trial_name}")
        
        if show_plots:
            fig.suptitle('Dominant Frequency Per Chunk')
            fig.supxlabel("Chunk Index")
            axs[0].set_ylabel("Frequency (Hz)")
            plt.tight_layout()
            plt.show()
        
        # Concatenate dict-of-arrays into one dict
        keys = trial_outputs[0].keys()
        
        trials = {
            k: np.concatenate([res[k] for res in trial_outputs])
            if k != 'bead_y' and k != 'fg'
            else [b for res in trial_outputs for b in res[k]]
            for k in keys
        }
        
        # Zero X and Y at the highest drive frequency
        # At the highest frequency the bead response is negligible, 
        # so any non-zero mean is an instrumental offset that should be removed
        idx_ref = np.argmax(trials['fg_freq'])
        
        x_ref = trials['mean_x'][idx_ref]
        y_ref = trials['mean_y'][idx_ref]

        trials['mean_x'] -= x_ref
        trials['mean_y'] -= y_ref
        
        # R magnitude must be recomputed from the zeroed X and Y values
        mean_r = np.sqrt(trials['mean_x'] ** 2 + trials['mean_y'] ** 2)
        trials['mean_r'] = 1e-9 * self.calib.ed_fit_func(mean_r)

        if show_plots:
            plt.figure()
            plt.plot(trials['fg_freq'], trials['mean_x'], 'b.', label='X')
            plt.plot(trials['fg_freq'], trials['mean_y'], 'g.', label='Y')
            plt.plot(trials['fg_freq'], mean_r, 'r.', label=r'$\sqrt{X^2+Y^2}$')
            plt.axhline(0, color='r', linestyle='--')
            plt.text(0.9*trials['fg_freq'].max(), np.median(mean_r)*1.05, f'X Offset: {x_ref: .2e}', ha='right', va='bottom', color='k', alpha=0.8)
            plt.text(0.9*trials['fg_freq'].max(), np.median(mean_r)*0.995, f'Y Offset: {y_ref: .2e}', ha='right', va='top', color='k', alpha=0.8)
            plt.ylabel('Lockin (V)')
            plt.xlabel('Freq (Hz)')
            plt.xscale('log')
            plt.legend()
            plt.title('X, Y, R Data')
            plt.show()
        
        self.results = self._compute_force_phase(trials, debug_mode=debug_mode)
                    
        # if show_plots:
        #     for trial_idx, (trial_key, trial) in enumerate(self.data_dict.items()):
        #         raw = trial
        #         info = chunk_infos[trial_idx]

        #         chunk_id = info['chunk_id']
        #         valid = info['valid_mask']

        #         for i in range(info['n_chunks']):
        #             t = np.arange(len(raw['x']))
        #             mask = (chunk_id == i)
        #             valid_region = t[valid & mask]

        #             if len(valid_region) == 0: 
        #                 continue

        #             fig, axs = plt.subplots(3, 1, figsize=(12, 8), tight_layout=True, sharex=True)
        #             fig.suptitle(f'Frequency Chunk: {info['fg_freq'][i]} Hz')
                    
        #             axs[0].plot(t[mask], raw['x'][mask], '.', alpha=0.5)
        #             axs[0].axvspan(valid_region[0], valid_region[-1], color='r', alpha=0.25)
        #             axs[1].plot(t[mask], raw['y'][mask], '.', alpha=0.5)
        #             axs[1].axvspan(valid_region[0], valid_region[-1], color='r', alpha=0.25)
        #             axs[2].plot(t[mask], raw['r'][mask], '.', alpha=0.5)
        #             axs[2].axvspan(valid_region[0], valid_region[-1], color='r', alpha=0.25)

        #             axs[0].set_title("Raw X (Lock-In)")
        #             axs[1].set_title("Raw Y (Lock-In)")
        #             axs[2].set_title("Raw R (Lock-In)")

        #             for ax in axs:
        #                 ax.grid(True)

        #             plt.show()
    
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

        # if show_plots:
        #     plt.figure()
        #     t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
        #     plt.title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Hann Window)")
        #     plt.xlabel(rf'Time $t$ in seconds ({SFT.p_num(N)}, $\Delta T$ = {SFT.delta_t:.2f} s)')
        #     plt.ylabel(rf'Freq. $f$ in Hz ({SFT.f_pts} bins, $\Delta f = {SFT.delta_f} Hz$)')
        #     plt.xlim(t_lo, t_hi)
        #     plt.ylim(1, 6000)
            
        #     im = plt.imshow(mag, origin='lower', aspect='auto', extent=SFT.extent(N), cmap='viridis')
        #     plt.colorbar(im, label='Magnitude')
            
        #     plt.tight_layout()
        #     plt.show()
        
        dominant_bins = np.argmax(mag, axis=0)
        dom_freqs = SFT.f[dominant_bins] # dominant freq of bin i
                
        freq_tol = 1 # Hz
        
        freq_changes = np.where(np.abs(np.diff(dom_freqs)) > freq_tol)[0] + 1
        boundaries = np.concatenate(([0], freq_changes, [len(dom_freqs)])) # add first and last index        

        chunk_sizes = np.diff(boundaries)
        keep = chunk_sizes > 1 # only keep freq chunks with more then one sample to remove mid transition values
        boundaries = np.concatenate(([boundaries[0]], boundaries[1:][keep]))

        merged_boundaries = [boundaries[0]] # merge neighboring chunks with the same frequency
        for i in range(1, len(boundaries) - 1):
            prev_start, prev_end = merged_boundaries[-1], boundaries[i]
            curr_end = boundaries[i + 1]

            # frequency of previous chunk
            prev_chunk_mid = (prev_start + prev_end) // 2
            prev_freq = dom_freqs[prev_chunk_mid]

            # representative frequency of current chunk
            curr_chunk_mid = (prev_end + curr_end) // 2
            curr_freq = dom_freqs[curr_chunk_mid]

            # if same frequency → merge (do NOT append boundary)
            if np.abs(prev_freq - curr_freq) <= freq_tol:
                continue

            # otherwise start a new chunk
            merged_boundaries.append(prev_end)

        # include the final boundary
        merged_boundaries.append(boundaries[-1])
        merged_boundaries = np.array(merged_boundaries)
                                
        return dom_freqs, merged_boundaries
            
    def compute_moduli(self, free_memory=True, debug_mode=False):
        """Compute the shear moduli G' and G" from the active microrheology results."""
        freq_list = self.results["fg_freq"]
        ynot = self.results["ynot"]
        fnot = self.results["fnot"]
        delta = self.results["delta"]
        
        if debug_mode:
            phi = self.results['phi']
            x = self.results['x']
            y = self.results['y']            
                
        G1 = np.abs(fnot / (6 * np.pi * self.constants.r * ynot)) * np.cos(delta)
        G2 = np.abs(fnot / (6 * np.pi * self.constants.r * ynot)) * np.sin(delta)
               
        if debug_mode:                         
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

        if free_memory:
            self.data_dict, self.chunk_info = None, None
        
        self.G1 = G1
        self.G2 = G2
        self.omega_list = 2*np.pi * freq_list
        
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
        skip_percent = 0.75 # discard the first 75% of each chunk
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

    def _compute_chunk_means(self, data, chunk_info, debug_mode):
        """Compute chunk average X, Y, R, and fg statistics per trial."""
                
        mask = chunk_info['valid_mask']
        cid = chunk_info['chunk_id'][mask]
        n_chunks = chunk_info['n_chunks']

        counts = np.bincount(cid, minlength=n_chunks)
                                        
        mean_x = np.bincount(cid, data['x'][mask], minlength=n_chunks) / counts
        mean_y = np.bincount(cid, data['y'][mask], minlength=n_chunks) / counts
        
        split_idxs = np.flatnonzero(np.diff(cid)) + 1 # locates where the chunk id's change (ex. 1 -> 2)
        bead_y = np.split(data['bead_y'][mask], split_idxs)

        fg_valid = data['fg'][mask]
        mean_fg = np.bincount(cid, weights=fg_valid, minlength=n_chunks) / counts
        mean_fg2 = np.bincount(cid, weights=fg_valid**2, minlength=n_chunks) / counts - mean_fg ** 2

        fg = np.split(data['fg'][mask], split_idxs)

        std_fg = np.sqrt(mean_fg2)

        result = {
            'mean_x': mean_x,
            'mean_y': mean_y,
            'bead_y': bead_y,
            'std_fg': std_fg,
        }

        if debug_mode:
            result['fg'] = fg

        return result
    
    def _software_phi(self, bead, fg, f):
        """Bead phase relative to FG via software demodulation"""
        t = np.arange(len(bead)) * dt
        bead = bead - np.mean(bead)
        fg = fg - np.mean(fg)

        cos_ref = np.cos(2 * np.pi * f * t)
        sin_ref = np.sin(2 * np.pi * f * t)

        Xb, Yb = np.mean(bead * cos_ref), np.mean(bead * sin_ref)
        Xf, Yf = np.mean(fg   * cos_ref), np.mean(fg   * sin_ref)

        return np.arctan2(Yb*Xf - Xb*Yf, Xb*Xf + Yb*Yf)

    def _compute_force_phase(self, data, debug_mode=False):
        """Compute force magnitue and phase dependent components"""
        
        y_trap = self.calib.a * np.sqrt(2) * data['std_fg'] + self.calib.b # ! a and b are in nm
        A = self.calib.ky * y_trap * 1e-9
        
        B = -self.calib.ky * data['mean_r']

        phi = np.array([
            self._software_phi(data['bead_y'][i], data['fg'][i], data['fg_freq'][i])
            for i in range(len(data['fg_freq']))
        ])
        
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
            results.update({'phi': phi, 'x': data['mean_x'], 'y': data['mean_y'], 'bead_y': data['bead_y'], 'fg': data['fg']})
                        
        return results
    
    def plot_g1g2(self):
        """Create the shear moduli plot."""
        plt.figure()
        plt.loglog(self.omega_list, self.G1, 'k.', label='G\'')
        plt.loglog(self.omega_list, self.G2, 'r.', label='G\'\'')
        plt.xlabel('Angular Frequency (Radians)')
        plt.ylabel('Viscoelastic Moduli (Pa)')
        plt.legend()
        plt.title('Viscoelastic Shear Moduli G\', G\"')
        plt.tight_layout()
        plt.show()
