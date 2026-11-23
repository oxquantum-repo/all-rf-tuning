# Automated All-RF Tuning for Spin Qubit Readout and Control

Code accompanying "Automated All-RF Tuning for Spin Qubit Readout and Control" by Carlsson and Saez-Mollejo et al.

The routine starts from a double quantum dot device and autonomously:

1. virtualises the charge sensor,
2. records a large charge stability diagram and locates the interdot transitions with a convolutional neural network (CNN),
3. for each interdot transition, virtualises the barrier gates, explores the barrier-gate space and scores each configuration for Pauli spin blockade (PSB),
4. checks for singlet-triplet (ST) oscillations, and, once a qubit is found, maps the operating region with gate, magnetic field and detuning sweeps.

All measurements are saved as QCoDeS datasets (see [Data](#data)).

## Repository structure

```
Measurement_Notebook.ipynb    Main notebook: instrument set-up and the full autonomous routine
requirements.txt              Pinned Python dependencies
models/
    architectures.py          PyTorch CNN architectures (InterdotCNN, LineCNN, AngleCNN)
    Interdot_CNN_model.pth    Trained weights: interdot classification
    Line_CNN_model.pth        Trained weights: transition line reconstruction
    Angle_CNN_model.pth       Trained weights: transition line angles
scripts/
    auto_tuning.py            Interdot finder, clustering, metastable-region masks, scan ranges and pulse lengths
    helpers.py                Geometry, line parametrisation and metastable mask helpers
    score_function.py         PSB score (maximum-likelihood Gaussian fits) and ST oscillation analysis
opx_files/
    configuration.py          Quantum Machines OPX configuration (elements, pulses, waveforms)
    macros.py                 QUA macros for RF reflectometry and DC readout
    measurement_helpers.py    Unit conversions, scan helpers and runtime configuration updates
```

## Hardware

The notebook was written for and run on the following set-up:

- Quantum Machines OPX (QOP >= 2.2, accessed via `qop_ip` and `cluster_name` in `opx_files/configuration.py`) for RF reflectometry readout and fast gate pulses.
- FastDuck DAC (connected on `COM3`) for the DC gate voltages.
- Two American Magnetics AMI430 magnet power supplies (out-of-plane and in-plane fields).

Gate names used throughout the code and stored in the databases: plungers `PL`, `PM`, barriers `BL`, `B12`, `B23`, and the sensor gates `SL`, `SP`, `SR`. Virtual gates are prefixed with `v` (e.g. `vPL`, `vB12`).

To use the routine on a different set-up, adjust the instrument addresses and gate mapping in the notebook, the network parameters and output ports in `opx_files/configuration.py`, and the gate voltages and limits in the notebook cells.

## Installation

The code was developed with Python 3.11.

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `jupyterlab==4.2.5`, which pip reports as conflicting with `qm-qua==1.1.7` (they require different `httpx` versions). If the installation fails for this reason, remove the `jupyterlab` line, install the remaining requirements, and install a Jupyter frontend separately.

Two packages are lab-specific and not available on PyPI:

- the FastDuck QCoDeS driver (`qcodes.instrument_drivers.IST_devices.FastDuck`),
- `SpinQubits`, which provides `VirtualGates` for the virtual gate matrices.

If you do not have access to these packages, replace them with the equivalent drivers for your own DAC and virtual gate implementation.

## Running the notebook

The results of the paper were acquired by running the cells of `Measurement_Notebook.ipynb` in order. The notebook must be opened from this directory, since the imports of `scripts`, `models` and `opx_files` are relative to it.

| Section | What it does |
| --- | --- |
| Import instruments | Connects to the DAC, OPX and magnets and creates the QCoDeS station |
| Create database | Initialises the QCoDeS database |
| Functions | Defines the saving helpers and all functions of the routine |
| Sensor Virtualization | Measures the sensor cross-capacitances and builds the sensor virtual gate matrix |
| Large Charge Stability Diagram with Labelled Interdots | Records a large stability diagram and finds the interdot transitions with the CNNs. To analyse an existing measurement, set the database path and run id in `load_by_id(...)` |
| Full Run of Routine | Runs the autonomous routine over all detected interdot transitions, saving one database per interdot (`autonomous_run_interdot_<n>.db`) |

Code relevant to running each model can be found in the cells of the notebook:

- Interdot CNN: the cell under "Large Charge Stability Diagram with Labelled Interdots", which calls `run_interdot_finder` (`scripts/auto_tuning.py`) with `models/Interdot_CNN_model.pth`.
- Line and Angle CNNs: the functions `get_interdot_location` and `get_PSB_points` in the "Functions" section, which call `get_metastable_vertices` and `mask_metastable_regions` (`scripts/auto_tuning.py`) to locate the metastable regions of each interdot transition.

## Data

The data for the autonomous run reported in the paper is accessible at [doi:10.15479/AT:ISTA:19885](https://doi.org/10.15479/AT:ISTA:19885). The data repository contains:

- `AllDataAutomation`: the QCoDeS databases of the full autonomous run, one per interdot transition.
- `ChangeNsamplesSameIDT`: data pertaining to Supplementary Section IV of the paper; repeated runs on the same interdot transitions with different numbers of exploration samples.

The databases can be opened with QCoDeS, for example:

```python
from qcodes.dataset import initialise_or_create_database_at, load_by_id

initialise_or_create_database_at("DatasetsPaper/AllDataAutomation/10721_S47_20250220_RUN20_interdot_0.db")
dataset = load_by_id(1)
```

## Contact

For questions and bug reporting, contact cornelius.carlsson@spc.ox.ac.uk or jaime.saezmollejo@ist.ac.at.
