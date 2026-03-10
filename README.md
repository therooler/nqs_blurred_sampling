
# Blurred Sampling

Utilities and experiments for *blurred sampling* time-dependent variational Monte Carlo (t-VMC) using NetKet.

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

## Notebooks

If you want to run notebooks:

```bash
python -m pip install jupyter
jupyter lab
```


