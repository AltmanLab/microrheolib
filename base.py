from microrheo.utils import *

class OTDataset:
    """Parent class for passive and active datasets"""
    def __init__(self, fh, constants, calib):   # doesn't matter what you set calib_starttime and calib_endtime to if you have the calibration already
        """
        Parameters
        ----------
        fh : str
            Relative path to the active lock-in dataset directory (e.x. passive1)
        constants : ExperimentalConstants
            Experiment wide constants 
        calib : TrapCalibration
            Brownian and FunctionG calibration parameters
        """
        self.fh = fh
        self.calib = calib
        self.constants = constants