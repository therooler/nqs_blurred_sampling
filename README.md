
# Code for "Removing nodal and support-mismatch pathologies in Variational Monte Carlo via blurred sampling"

Utilities and experiments for *blurred-sampled* time-dependent variational Monte Carlo (t-VMC) using NetKet for the experiments in https://arxiv.org/abs/2603.18148

This repo contains:

- Code in folder `/src/` for performing t-VMC in [Netket](https://github.com/netket/netket). 
- We provide two TDVP driver (`TDVPSchmittBlur` and `TDVPSchmittRandomizedBlur`) that implement Schmitt-style SNR-based regularization and a simple blur proposal kernel to improve sampling stability during real-time evolution.
- Code in folder `/paper/` to reproduce the figures in arxiv:xxxxxxx.
- We provide the data to reproduce all the experiments in `/paper/data`

**IMPORTANT**: The blurred sample kernel `blurred_sample` in `/src/tdvp_utils.py` is tailored to Ising Hamiltonians considered in the paper. A more general kernel is provided with `blurred_sample_general`, which calculates the number of physical connections provided by Netket's `get_conn_padded`. Both will give the correct dynamics for any Hamiltonian, but the latter ensures the value of `q` can actually be interpreted as "probability of moving to a physical off-diagonal term".

All experiments should run in a reasonable time, except for figure 6(c), which requires multiple GPUs for several hours and [JAXMg](https://github.com/flatironinstitute/jaxmg).

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

To install the GPU compatible version of this repo use

```bash
    python -m pip install jax[cuda12]==0.8.1 jaxmg[cuda12]==0.0.7
```

## Notebooks

If you want to run the notebooks:

```bash
python -m pip install jupyter
jupyter lab
```

