# MicrorheoLib
  
Active and passive microrheology analysis backend library for optical trapping data.
  
## Overview

`microrheo` turns raw optical trapping recordings into viscoelastic shear moduli, G' (storage/elastic) and G" (loss/viscous), using two techniques:

- Passive microrheology (`PassiveAnalysis`): uses the bead's thermal motion. Computes the mean squared displacement (MSD) and applies the Mason (2000) generalized Stokes-Einstein relation.
- Active microrheology (`ActiveAnalysis`): analysis a frequency-swept function generator (FG) signal and extracts the shear moduli from the force amplitude and phase response at each frequency.

Both share a common trap calibration and are managed together through `Experiment`, so results within an experiment (and others) can be compared and plotted side by side. This way active and passive measurements of the same bead can be compared and analyzed.

## Installation
```bash
git clone https://github.com/AltmanLab/microrheolib.git
cd microrheolib
pip install -r requirements.txt
```
Requires `numpy`, `scipy`, and `matplotlib`.

## Key Concepts
- **Trap calibration** (`TrapCalibration`): converts raw detector voltages into physical bead displacements (meters). Combines a Brownian calibration (trap stiffness/detector sensitivity from the thermal motion power spectrum) with a FG calibration (a linear fit near the trap center, plus a spline fit for the nonlinear edge response of the detector).
- **Fischer correction**: computes the local trap stiffness by combining active and passive measurements following the Fischer et al. (2010) procedure. Used to correct both sets of moduli.

See the docstrings in `experiment.py`, `active.py`, and `passive.py` for more details.

## Quick Start
```python
from microrheo import *

# --- 1. sample & experiment constants -------------------------------------
# `from_preset` looks up viscosity for a known sample type; bead_dia is the
# bead diameter in meters. Presets: 'water', 'bsa', 'iodixionol20'.
# For a sample that isn't a preset, build ExperimentalConstants(eta=..., r=...)
# directly instead.
constants = ExperimentalConstants.from_preset('water', bead_dia=1.05e-6)

# ExperimentalConfig controls the Brownian calibration window/frequency
# ranges and the FunctionG calibration window. Adjust these if your
# calibration data was recorded over a different time range, or if the
# fitted frequency band should be narrower/wider than the defaults:
config = ExperimentalConfig(
	constants=constants,
	starttime=0, endtime=120, # Brownian data window (s)
	startfreqx=20, endfreqx=3000, # PSD fit range for x (Hz)
	startfreqy=20, endfreqy=3000, # PSD fit range for y (Hz)
	fg_starttime=0, fg_endtime=405, # FunctionG data window (s)
	chunk_dur=1.0  # PSD chunk side for Brownian fit (s)
)

# --- 2. load & calibrate the experiment -------------------------------------
# `fh` is the path to the experiment's top-level folder (e.g. 260722). This step
# runs the Brownian calibration automatically (looks for a 'Brownian' subfolder by
# default; pass brownian_name='...' if named differently). If calibration parameters
# already exist from a previous run (or to speed up computation) skip re-fitting by
# passing brownian_params=[Sx, Sy, kx, ky, trapx, trapy] instead, which are printed
# properly formatted automatically after being computed.
exp = Experiment('path/to/experiment_folder', config)

# Function generator + nonlinear 'entire detector' calibration. This step looks for
# a folder named 'FunctionG' by default; pass folder_name='...' to manually specify.
# Manually adjust lin_range (min_nm, max_nm) to fit the linear region of the detector
# and spline_fit (num_knots, (min_nm, max_nm)) for the nonlinear region, before
# doubling back. Both window sizes are set relative to the trap center.
exp.calibrate(lin_range=(-100, 100), spline_fit=(3, (-550, 550)), center_width=0.2)

# --- 3. add and process a passive dataset -------------------------------------
# `name` is how you'll refer to this dataset later (e.g. in plot_moduli).
# `folder_name` is the subfolder containing this dataset's raw files.
# `starttime/endtime` defines the data window in seconds.
# `sample_type` allows for filtering by sample type (e.g. 'water', 'actin', 'cell')
exp.add(
	PassiveAnalysis, name='passive', folder_name='Passive',
	starttime=0, endtime=185, sample_type='cell'
)
exp.datasets.passive.compute_msd()

# window/polyorder control the log-log smoothing used to extract
# alpha (the MSD slope). Increase `window` for noisier data. Use
# show_plots=True to visually confirm the fit. `water_theory=True`
# overlays the theoretical water curve for comparison.
exp.datasets.passive.compute_alpha(window=151, polyorder=5, show_plots=True)

# If ram useage becomes a problem, `free_memory=True` will remove
# all references to the raw data once the shear moduli have been
# computed.
exp.datasets.passive.compute_moduli(free_memory=False)
  
# --- 4. add and process an active dataset -------------------------------------
exp.add(
	ActiveAnalysis, name='active', folder_name='Active',
	starttime=0, endtime=185, sample_type='cell'
)
  
# `show_plots=True` plots the detected dominant frequency per chunk (useful to
# confirm the frequency sweep was segmented correctly).
exp.datasets.active.segment_chunks(show_plots=True)
exp.datasets.active.compute_force_phase()
exp.datasets.active.compute_moduli(free_memory=False)
  
# --- 5. fischer correction & plot -------------------------------------
# frequency range is the frequency band (Hz) trusted for computing the mean
# stiffness correction. Low frequencies may show biological processes and high
# frequencies may suffer from aliasing. Use `show_plots=True` to help you identify
# a range where both methods are reliable.
exp.compute_fischer_correction('active', 'passive', freq_range=(1e2, 1e3), show_plots=True)
  
# plot_moduli() options worth knowing:
# datasets/sample_types: include/exclude by name or sample_type, e.g.
#    datasets=['-passive1'] plots everything except passive1;
# 	 sample_types=['cell'] plots only cell data.
#    Both combine as an AND filter.
# corrected: apply the Fischer correction before plotting
# average: overlay the mean curve across matching datasets
# show_std: shade +/- 1 std around the average (needs average=True)
# show_individual: disable the individual dataset curves (needs average=True)
# combine_axes: plot G' and G" on one shared axes instead of two
# region: (low_hz, high_hz) to zoom into a frequency band
exp.plot_moduli(sample_types=('cell',),
	corrected=True, average=True, show_std=True,
	region=(1, 1e5),
	show_individual=False, combine_axes=True
)
  
# --- 6. save results for later -------------------------------------
exp.serialize('results/my_experiment.json')

# Reload later, e.g. to compare against a different experiment
#    exp2 = Experiment.from_serialized('results/my_experiment.json')
#    exp2.load('results/other_experiment.json', prefix='other')
#    exp2.plot_moduli(average=True)
```