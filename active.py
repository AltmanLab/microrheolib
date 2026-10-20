from microrheo.base import OTDataset
from microrheo.utils import *

from scipy.signal import ShortTimeFFT, windows, savgol_filter

class ActiveAnalysis(OTDataset):
    """Process active microrheology datasets.
        
    Typical workflow::

        active = ActiveAnalysis(name, fh, analysisparams, starttime, endtime)
        active.segment_chunks()
        active.compute_force_phase()
        active.compute_moduli()
        active.plot_g1g2()
    """

    def __init__(self, fh, constants, calib, starttime=0, endtime=185, sample_type='water'):
        """
        Parameters
        ----------
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
        
        if not hasattr(self.calib, 'a'):
            raise RuntimeError("Call calibrate() before adding an active dataset.")

        # Load the data and convert to meters
        self.fh = fh + ('' if fh.endswith('/') else '/')
        self.fg, self.y = load_data(self.fh, starttime, endtime)
        self.fg = self.calib.a * self.fg + self.calib.b
        self.y = self.calib.ed_fit_func(self.y - self.calib.v_offset)
        self.t = np.arange(len(self.y)) * dt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), tight_layout=True, sharex=True)
        fig.suptitle('Data to Analyze')
        ax1.plot(self.t, 1e9 * self.fg, 'k')
        ax1.set_ylabel('fg [nm]')
        ax2.plot(self.t, 1e9 * self.y, 'k')
        ax2.set_ylabel('y [nm]')
        ax2.set_xlabel('Time [s]')
        plt.show()
                        
    def _compute_freq_changes_sfft(self, fg:np.ndarray, debug_mode=False):
        """SFFT style dominant frequency identifier."""
        # TODO: provide an easier way to change STFT window parameters
        T = 1.5 # window_duration (seconds)
        N = len(fg) # number of samples
        
        window_len = int(fs * T)
        self.sfft_hop_len = int(window_len * 0.25) # overlap window 75%
        
        window = windows.hann(window_len, sym=False)

        fg -= np.mean(fg) # remove DC offset
        
        SFT = ShortTimeFFT(window, hop=self.sfft_hop_len, fs=fs, mfft=8*window_len)
        fft = SFT.stft(fg)
        mag = np.abs(fft)

        if debug_mode:
            plt.figure()
            t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
            plt.title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Hann Window)")
            plt.xlabel(rf'Time $t$ in seconds ({SFT.p_num(N)}, $\Delta T$ = {SFT.delta_t:.2f} s)')
            plt.ylabel(rf'Freq. $f$ in Hz ({SFT.f_pts} bins, $\Delta f = {SFT.delta_f} Hz$)')
            plt.xlim(t_lo, t_hi)
            plt.ylim(1, 1e4)
            
            im = plt.imshow(mag, origin='lower', aspect='auto', extent=SFT.extent(N), cmap='viridis')
            plt.colorbar(im, label='Magnitude')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.show()
        
        min_bin = 1 # skip DC bin
        dominant_bins = np.argmax(mag[min_bin:], axis=0) + min_bin

        # clips edge bins to avoid out-of-bounds, where argmax is unreliable anyway
        lo = np.clip(dominant_bins - 1, 0, mag.shape[0] - 1)
        hi = np.clip(dominant_bins + 1, 0, mag.shape[0] - 1)
        alpha, beta, gamma = mag[lo, np.arange(mag.shape[1])], mag[dominant_bins, np.arange(mag.shape[1])], mag[hi, np.arange(mag.shape[1])]
        denom = (alpha - 2*beta + gamma)
        offset = np.where(np.abs(denom) > 1e-12, 0.5 * (alpha - gamma) / denom, 0.0)
        dom_freqs = SFT.f[dominant_bins] + offset * SFT.delta_f # dominant freq of bin i
                
        freq_tol = max(0.5 * SFT.delta_f, 0.05) # Hz
        
        freq_changes = np.where(np.abs(np.diff(dom_freqs)) > freq_tol)[0] + 1
        boundaries = np.concatenate(([0], freq_changes, [len(dom_freqs)])) # add first and last index        

        chunk_sizes = np.diff(boundaries)
        keep = chunk_sizes > 2 # only keep freq chunks with more then 2 samples to remove mid transition values
        boundaries = np.concatenate(([boundaries[0]], boundaries[1:][keep]))

        chunk_freqs = np.array([
            np.median(dom_freqs[boundaries[i]:boundaries[i+1]])
            for i in range(len(boundaries)-1)
        ])

        merged_boundaries = [boundaries[0]]

        for i in range(1, len(chunk_freqs)):
            if abs(chunk_freqs[i] - chunk_freqs[i-1]) > freq_tol:
                merged_boundaries.append(boundaries[i])

        merged_boundaries.append(boundaries[-1])
        merged_boundaries = np.array(merged_boundaries)
                                        
        return dom_freqs, merged_boundaries
            
    def compute_moduli(self, snr_threshold_db:float = 10, min_consecutive_chunks:int = 1, free_memory=True, show_plots=False):
        """Compute the shear moduli G' and G" from the active microrheology results.
        
        Parameters
        ----------
        snr_threshold_db : float
            Minimum SNR (in dB) for a chunk to be included in the moduli
        min_consecutive_chunks : int
            Minimum number of consecutive chunks with SNR above threshold to be included in the moduli
        free_memory : bool
            If True, free memory by deleting the data_dict and chunk_info after computing moduli
        show_plots : bool
            If True, show plots of the phases"""
        
        freq_list = np.asarray(self.results['fg_freqs'])
        y0 = self.results['y0']
        f0 = self.results['f0']
        delta = self.results['delta']
        phi = self.results['phi']

        n = len(freq_list)
        cuttoff_idx = n

        snr = self.results['bead_snr']
        below = snr < snr_threshold_db

        for i in range(n - min_consecutive_chunks + 1):
            if np.all(below[i:i + min_consecutive_chunks]):
                cuttoff_idx = i
                break

        if cuttoff_idx < n:
            freq_list = freq_list[:cuttoff_idx]
            y0 = y0[:cuttoff_idx]
            f0 = f0[:cuttoff_idx]
            delta = delta[:cuttoff_idx]
            phi = phi[:cuttoff_idx]

        plt.figure()
        plt.semilogx(freq_list, snr, 'k.', label='Bead SNR')
        plt.axhline(snr_threshold_db, color='tab:orange', linestyle='--', label='SNR Threshold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('SNR (dB)')
        plt.title('Bead SNR vs Frequency')
        plt.show()
                               
        if show_plots:                         
            plt.figure()
            plt.semilogx(freq_list, delta, 'r.', label='Bead-Force Phase')
            plt.semilogx(freq_list, phi-delta, 'g.', label='Force-FG Phase')
            plt.semilogx(freq_list, phi, '.', label='Bead-FG Phase')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase (rad)')
            plt.axhline(np.pi/2, color='tab:blue')
            plt.text(freq_list.min(), np.pi/2+0.01, r'$\pi/2$', ha='left', va='bottom', color='tab:blue')
            plt.axhline(-np.pi/2, color='tab:blue')
            plt.text(freq_list.min(), -np.pi/2+0.01, r'$-\pi/2$', ha='left', va='bottom', color='tab:blue')

            plt.axvline(freq_list[cuttoff_idx-1], color='tab:orange', linestyle='--', label='SNR Cutoff')
            plt.text(freq_list[cuttoff_idx-1], plt.ylim()[1], f'SNR Cutoff\n{freq_list[cuttoff_idx-1]:.2f} Hz', ha='left', va='top', color='tab:orange')

            plt.legend()
            plt.title('Phases')
            plt.show()

        G1 = np.abs(f0 / (6 * np.pi * self.constants.r * y0)) * np.cos(delta)
        G2 = np.abs(f0 / (6 * np.pi * self.constants.r * y0)) * np.sin(delta)

        if free_memory:
            self.data_dict, self.chunk_info = None, None
        
        self.G1 = G1
        self.G2 = G2
        self.omega_list = 2*np.pi * freq_list

    def corrected_moduli(self):
        """Return G' and G" with the stiffness correction from Fischer et al."""
        return self.G1 * self.fischer_correction, self.G2 * self.fischer_correction
        
    def segment_chunks(self, show_plots, debug_mode):
        """Create chunk ids and valid mask after equilibration masking. Sets self.chunked_fg, self.chunked_y, self.chunked_t, and self.fg_freq."""
        
        dom_freqs, change_idxs = self._compute_freq_changes_sfft(self.fg)

        chunk_len = self.sfft_hop_len # divide by hop length
        n_samples = len(self.fg)
        n_chunks = len(change_idxs) - 1
        
        chunk_id = np.zeros(n_samples, dtype=int)
        
        for i in range(n_chunks):
            start_idx = change_idxs[i] * chunk_len
            end_idx = change_idxs[i + 1] * chunk_len
            chunk_id[start_idx:end_idx] = i
        
        # remove equilibration portion
        skip = 0.5 # seconds to skip after each frequency change
        valid_mask = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_chunks):
            start_idx = change_idxs[i] * chunk_len
            end_idx = change_idxs[i + 1] * chunk_len
            
            valid_start = start_idx + int(skip * fs)
            valid_end = end_idx - int(skip * fs)

            if valid_start >= valid_end:
                print(f"Warning: Chunk {i} has no valid region after equilibration masking. Skipping this chunk.")
                continue
            
            valid_mask[valid_start:valid_end] = True
            
        mid_idx = (change_idxs[:-1] + change_idxs[1:] - 1) // 2
        detected_freqs = dom_freqs[mid_idx]

        if show_plots:
            plt.figure(figsize=(8, 4.8))
            plt.semilogy(dom_freqs, marker='.')
            mid_idx = (change_idxs[:-1] + change_idxs[1:] - 1) // 2
            [plt.text(i, dom_freqs[i]*1.01, f"{dom_freqs[i]:.3g}", ha='center', va='bottom') for i in mid_idx]
            plt.title(f"Dominant Frequency Per Chunk")
            plt.xlabel("Chunk Index")
            plt.ylabel("Frequency (Hz)")
            plt.tight_layout()
            plt.show()


        # commanded_freqs = np.concat(
        #                     [np.geomspace(1, 5, 10)[:-1],
        #                     np.geomspace(5, 1e4, 41)]
        #                 )

        # nearest_idx = np.abs(detected_freqs[:, None] - commanded_freqs[None, :]).argmin(axis=1)
        self.fg_freqs = detected_freqs # commanded_freqs[nearest_idx]

        # mismatch = np.abs(self.fg_freqs - detected_freqs) / self.fg_freqs
        # bad = np.where(mismatch > 0.05)[0]  # >5% off is worth a look
        # if len(bad):
        #     print(f"Warning: {len(bad)} chunk(s) detected far from any commanded frequency: "
        #             f"chunk idx {bad}, detected {detected_freqs[bad]}, matched {self.fg_freqs[bad]}")

        split_idxs = np.flatnonzero(np.diff(chunk_id[valid_mask])) + 1 # locates where the chunk id's change (e.g. 1 -> 2)
        self.chunked_fg = np.split(self.fg[valid_mask], split_idxs)
        self.chunked_y = np.split(self.y[valid_mask], split_idxs)
        self.chunked_t = np.split(self.t[valid_mask], split_idxs)

        # assert len(self.chunked_fg) == len(self.fg_freqs), \
        #     f"Mismatch: {len(self.chunked_fg)} chunks vs {len(self.fg_freqs)} frequencies"
        
        # trim each chunk to an integer number of periods
        trimmed_fg, trimmed_y, trimmed_t = [], [], []
        for i in range(len(self.fg_freqs)):
            fg_chunk, y_chunk, t_chunk = self.chunked_fg[i], self.chunked_y[i], self.chunked_t[i]
            period = 1 / self.fg_freqs[i]
            n_periods = int(len(fg_chunk) / (period * fs))
            n_samples = int(n_periods * period * fs)
            trimmed_fg.append(fg_chunk[:n_samples])
            trimmed_y.append(y_chunk[:n_samples])
            trimmed_t.append(t_chunk[:n_samples])

        self.chunked_fg = trimmed_fg
        self.chunked_y = trimmed_y
        self.chunked_t = trimmed_t

        if debug_mode:
            for i in range(n_chunks):
                mask = chunk_id == i
                valid_region = self.chunked_t[i]

                if len(valid_region) == 0: 
                    print(f'Skipping chunk {i} due to no valid region after equilibration masking.')
                    continue

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), tight_layout=True, sharex=True)
                fig.suptitle(f'Frequency Chunk: {self.fg_freqs[i]:.6f} Hz')
                
                ax1.plot(self.t[mask], self.fg[mask], color='tab:blue')
                ax1.axvspan(valid_region[0], valid_region[-1], facecolor='gray', linewidth=2, edgecolor='black', alpha=0.25)
                ax2.plot(self.t[mask], self.y[mask], color='tab:blue')
                ax2.axvspan(valid_region[0], valid_region[-1], facecolor='gray', linewidth=2, edgecolor='black', alpha=0.25)

                ax1.set_title("Function Generator Signal")
                ax2.set_title("Bead Y Signal")
                ax2.set_xlabel("Time (s)")

                for ax in [ax1, ax2]:
                    ax.grid(True)

                plt.show()

    def _compute_chunk_snr(self, y_chunk, freq, exclude_harmonics=3):
        """Estimate the SNR (dB) of a chunk of data at a given frequency.
        
        Parameters
        ----------
        y_chunk : np.ndarray
            Bead signal for this frequency chunk (already trimmed to an integer number of periods)
        freq : float
            Drive frequency of the chunk (Hz)
        exclude_harmonics : int
            Number of harmonics of 'freq' to exclude from the noise estimate (default=3)

        Returns
        -------
        snr_db : float
            Estimated SNR in decibels (dB). Returns np.nan if the chunk is too short to compute a reliable SNR
        """
        n = len(y_chunk)
        if n < 8: 
            return np.nan # Not enough data to compute SNR

        y = y_chunk - np.mean(y_chunk) # remove DC offset
        win = windows.hann(n, sym=False)
        Y = rfft(y * win)
        freqs = rfftfreq(n, 1/fs)
        power = np.abs(Y)**2

        sig_bin = np.argmin(np.abs(freqs - freq))
        sig_power = power[sig_bin]

        noise_mask = np.ones_like(power, dtype=bool)
        for h in range(1, exclude_harmonics + 1):
            h_bin = np.argmin(np.abs(freqs - h * freq))
            lo, hi = max(0, h_bin - 1), min(len(power), h_bin + 2) # 3 bin region around the harmonic
            noise_mask[lo:hi] = False

        noise_candidates = power[noise_mask]
        if noise_candidates.size == 0:
            return np.nan # No noise candidates left after excluding harmonics
        noise_power = np.median(noise_candidates)
        if noise_power <= 0:
            return np.inf

        plt.figure()
        plt.semilogx(freqs, power, label='Power Spectrum')
        plt.vlines([freq * h for h in range(1, exclude_harmonics + 1)], 0, np.max(power), color='r', linestyle='--', alpha=0.5, label='Excluded Harmonics')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title(f'Power Spectrum for {freq} Hz Chunk')
        plt.legend()
        plt.show()

        return 10 * np.log10(sig_power / noise_power)

    def compute_force_phase(self, debug_mode=False):
        """Compute force magnitue and phase dependent components."""
                
        demod = [
            software_demod(y, fg, freq)
            for y, fg, freq in zip(self.chunked_y, self.chunked_fg, self.fg_freqs)
        ]
        phi, trap_amp, bead_amp = map(np.asarray, zip(*demod)) # convert to numpy arrays

        phi -= 2 * np.pi * self.fg_freqs * self.calib.t_offset # correct for a time offset between the FG and bead signals

        A = self.calib.ky * trap_amp
        B = -self.calib.ky * bead_amp
        C = np.sqrt(A**2 + B**2 + 2 * A * B * np.cos(phi))
        phi_new = np.arctan2(B * np.sin(phi), A + B * np.cos(phi))

        delta_phi = phi - phi_new

        # per_chunk bead SNR
        bead_snr = np.array([
            self._compute_chunk_snr(y, freq)
            for y, freq in zip(self.chunked_y, self.fg_freqs)
        ])
                
        results = {
            'fg_freqs': self.fg_freqs,
            'f0': C,
            'y0': bead_amp,
            'delta': delta_phi,
            'phi': phi,
            'bead_snr': bead_snr,
            'bead_y': self.chunked_y, 
            'fg': self.chunked_fg
        }
                                
        self.results = results
    
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