
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

## Notebooks

The notebooks in `experiments/` assume a sibling folder `rbm_qsim/` is on the Python path and import `core.*`.
This repo includes `rbm_qsim/core/` as a thin re-export of the implementations in `src/`, so the notebooks/scripts should work without modification.

If you want to run notebooks:

```bash
python -m pip install jupyter
jupyter lab
```


