
# Blurred Sampling

Utilities and experiments for *bridge-sampled* time-dependent variational Monte Carlo (t-VMC) using NetKet.

This repo contains:

- Code in folder `/src/` for performing t-VMC in Netket. 
- We provide two TDVP driver (`TDVPSchmittBlur` and `TDVPSchmittRandomizedBlur`) that implement Schmitt-style SNR-based regularization and a simple blur proposal kernel to improve sampling stability during real-time evolution.
- Code in folder `/paper/` to reproduce the figures in arxiv:xxxxxxx.
- We provide the data to reproduce all the experiments in `/paper/data`

All experiments should run in a reasonable time, except for figure 6(c), which requires multiple GPUs for several hours.

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

## Notebooks

If you want to run the notebooks:

```bash
python -m pip install jupyter
jupyter lab
```

