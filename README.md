
# blurred_sampling

Utilities and experiments for *bridge-sampled* time-dependent variational Monte Carlo (t-VMC) using NetKet.

This repo contains:

- A TDVP driver (`TDVPSchmittBridge`) implementing Schmitt-style SNR-based regularization and a simple “bridge” proposal kernel to improve sampling stability during real-time evolution.
- A TFIM quench experiment script and plotting notebooks.

## Repository layout

- `src/`
	- `schmitt_tdvp_bridge.py`: TDVP driver with SNR/SVD regularization and bridge sampling.
	- `callbacks.py`: NetKet callbacks for logging/monitoring and (optional) infidelity.
	- `logger.py`: small checkpointing logger used by the experiment.
	- `tdvp_utils.py`: monitoring helpers.
- `experiments/`
	- `tfim_quench_experiment.py`: main runnable TFIM quench experiment.
	- `plot_tfim_quench.ipynb` and other notebooks: analysis/plots.
	- `data/`: (optional) output folder.
- `rbm_qsim/core/`
	- Compatibility package so the existing notebooks/scripts (which do `sys.path.append("../rbm_qsim")` and then import `core.*`) run unchanged.

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

Notes:

- `requirements.txt` pins `jax==0.7.2` (CPU install). If you want GPU acceleration, install the appropriate JAX build for your CUDA setup following the official JAX installation guide, then install the remaining requirements.

## Quickstart: run the TFIM quench experiment

From the repo root:

```bash
python experiments/tfim_quench_experiment.py \
	--experiment_name test \
	--L 3 \
	--alpha 1 \
	--seed 100 \
	--n_samples_sr 2048 \
	--n_samples_tvmc 4096 \
	--hc_multiplier 1.0 \
	--T 2.0 \
	--n_save_times 20
```

What it does:

1. Builds a 2D TFIM Hamiltonian on an `L×L` periodic square lattice.
2. Finds a ground state for an initial Hamiltonian (via SR/VMC).
3. Runs real-time TDVP dynamics with an adaptive integrator (NetKet `RK45`).
4. Logs diagnostics (energy stats, residual, SNR summaries, ESS, etc.) and periodically checkpoints parameters.

## Output / checkpointing

The experiment uses `src/logger.py` to write checkpoint logs and parameters under:

- `./data/...` by default (configurable with `--data_prepend`).

Rerunning the same command with the same configuration will attempt to restore and continue from the existing log files.

## Notebooks

The notebooks in `experiments/` assume a sibling folder `rbm_qsim/` is on the Python path and import `core.*`.
This repo includes `rbm_qsim/core/` as a thin re-export of the implementations in `src/`, so the notebooks/scripts should work without modification.

If you want to run notebooks:

```bash
python -m pip install jupyter
jupyter lab
```

Then open any notebook in `experiments/`.

## Development / tests

```bash
pytest -q
```

## Direct imports

If you prefer not to use the compatibility `core` package, you can import directly from `src`:

```python
from src.schmitt_tdvp_bridge import TDVPSchmittBridge
```

