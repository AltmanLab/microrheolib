from microrheo.base import OTDataset
from microrheo.utils import *

class LockinAnalysis(OTDataset):
    '''Class for processing active OT datasets using the lock-in amplifier
    
    General order:
        analyze_all_trials
        compute_moduli
        plot_g1g2
    '''

    def __init__(self, fh, analysisparams, params=None):
        '''fh: file handle of the active lock-in dataset i.e. bead1 !!!NO TAILING SLASH!!! '''
        super().__init__(fh, analysisparams, params)
        
        self.analysisparams = analysisparams

        # Load the data
        self.fh = fh +'/'
        self.data_dict = self._load_lockin_data(fh)
        
    def analyze_chunk(self, data:dict, fg_freq, start_index:int, end_index:int) -> dict:
        """Analyze one frequency chunk and create the 'result' object correlating to that frequency chunk"""
        fg_chunk = self.a * data['fg'][start_index:end_index] + self.b
        
        skip_percent = 0.75
        skip_index = int((end_index - start_index) // (1/skip_percent))
        x_chunk = data['x'][start_index + skip_index:end_index - fs//2] # only keep the second half of the data
        y_chunk = data['y'][start_index + skip_index:end_index - fs//2] # this gives time for the lock-in to equilibrate
        r_chunk = data['r'][start_index + skip_index:end_index - fs//2]
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 5), tight_layout=True)
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

        y_trap = np.sqrt(2) * np.std(fg_chunk - np.mean(fg_chunk)) # sqrt(2) * std(zero mean fg)
        A = self.ky * y_trap * 1e-9
        
        B = -self.ky * np.mean(r_chunk) 
        phi = np.arctan2(np.mean(y_chunk), np.mean(x_chunk))
        
        C = np.sqrt(A**2 + B**2 + 2 * A * B * np.cos(phi))
        phi_new = np.arctan2(B * np.sin(phi), A + B * np.cos(phi))

        delta_phi = phi - phi_new
        
        # optionally phase wrap the negative values 
        # if delta_phi < 0:
        #     delta_phi += 2 * np.pi

        result = {
            'fg_freq': None, # gets filled in later
            'fnot': C,
            'ynot': np.mean(r_chunk),
            'delta': delta_phi
        }
        return result
    
    def analyze_trial(self, data:dict) -> dict:
        '''Calculate the 'results' object for one set of data (e.g. one trial)'''
        fg = data['fg']
        freqs, change_idxs = self.compute_freq_changes(fg, show_plots=False)

        results = []
        for i in range(len(change_idxs) - 1):
            start = change_idxs[i] * fs
            end = change_idxs[i + 1] * fs
            
            result = self.analyze_chunk(data, freqs[change_idxs[i]], start, end)
            result['fg_freq'] = freqs[change_idxs[i]]
            results.append(result)

        return results
        
    def analyze_all_trials(self, data:dict) -> dict:
        '''Calculate the 'results' object for all of the trials'''
        results_all = []
        for trial in data.keys():
            trial_results = self.analyze_trial(data[trial])
            results_all.extend(trial_results)
            
        return results_all
    
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
        freq_changes = np.where(np.diff(dominant_freqs) > 0.5)[0] + 1
        freq_changes = np.insert(freq_changes, 0, 0) # add first index
        # freq_changes = np.append(freq_changes, len(dominant_freqs) - 1) # add last index
        
        if show_plots:
            plt.figure()
            plt.plot(dominant_freqs, label='Dominant Frequencies')
            [plt.text(i, dominant_freqs[i], f"{dominant_freqs[i]:.0f}") for i in freq_changes]
            plt.title("Dominant Frequency Per Chunk")
            plt.xlabel("Chunk Index")
            plt.ylabel("Frequency (Hz)")
            plt.show()
            
        return dominant_freqs, freq_changes
    
    def compute_moduli(self, results:dict, free_memory=False):
        freq_list = np.array([r['fg_freq'] for r in results])
        ynot = np.array([r['ynot'] for r in results])
        fnot = np.array([r['fnot'] for r in results])
        delta = np.array([r['delta'] for r in results])
                
        G1 = np.abs(fnot / (6 * np.pi * ExperimentalConstants.r * ynot)) * np.cos(delta)
        G2 = np.abs(fnot / (6 * np.pi * ExperimentalConstants.r * ynot)) * np.sin(delta)
        
        if free_memory:
            self.data_dict = None
        
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