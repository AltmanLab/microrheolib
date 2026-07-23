from microrheo.utils import *

class OTDataset:
    """Parent class for passive and active datasets"""
    def __init__(self, fh, constants, calib, starttime, endtime, sample_type):   # doesn't matter what you set calib_starttime and calib_endtime to if you have the calibration already
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
        self.fh = fh
        self.calib = calib
        self.constants = constants
        self.starttime = starttime
        self.endtime = endtime
        self.sample_type = sample_type
        self.fischer_correction = None

@dataclass
class SerializedDataset:
    """Stand in for a live OTDataset when moduli data is loaded in from disk rather then computed this session.
    
    Correcting data differs for passive and active:
     - active: G * fischer_correction
     - passive: G1 - trap_offset(fischer_correction)
    """
    omega_list: np.ndarray
    G1: np.ndarray
    G2: np.ndarray
    is_passive: bool
    sample_type: str
    fischer_correction: float | None = None # active only
    trap_offset: np.ndarray | None = None # passive only

    def corrected_moduli(self):
        if self.is_passive:
            if self.trap_offset is None:
                raise ValueError('No Fischer correction was recorded for this dataset.')
            return self.G1 - self.trap_offset, self.G2
        else:
            if self.fischer_correction is None:
                raise ValueError('No Fischer correction was recorded for this dataset.')
            return self.G1 * self.fischer_correction, self.G2 * self.fischer_correction