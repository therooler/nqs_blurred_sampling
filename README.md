
# Blurred Sampling

Utilities and experiments for *bridge-sampled* time-dependent variational Monte Carlo (t-VMC) using NetKet.

This repo contains:

- A TDVP driver (`TDVPSchmittBridge`) implementing Schmitt-style SNR-based regularization and a simple “bridge” proposal kernel to improve sampling stability during real-time evolution.
- A TFIM quench experiment script and plotting notebooks.

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

To install the infidelity optimization:
```bash
pip install git+https://github.com/NeuralQXLab/ptvmc-systematic-study
```

## Notebooks

If you want to run notebooks:

```bash
python -m pip install jupyter
jupyter lab
```


