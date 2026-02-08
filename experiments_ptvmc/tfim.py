import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import qutip as qt
import numpy as np
import ptvmc

from functools import partial

import advanced_drivers as advd
from advanced_drivers._src.callbacks.autodiagshift import PI_controller_diagshift

from matplotlib import pyplot as plt

import optax

import jax
import jax.numpy as jnp

import netket as nk
from netket.optimizer.solver import cholesky
from netket.operator.spin import sigmax, sigmaz

from callbacks import (
    get_acceptance_rate_callback,
    get_parameter_save_callback,
)
from ptvmc._src.callbacks.logapply import DynamicLogApply as LogApply
from metropolis import LocalDoubleFlipRule
from logger import Logger


def get_model(alpha):
    ma = nk.models.RBM(
        alpha=alpha,
        param_dtype=complex,
    )
    return ptvmc.nn.DiagonalWrapper(ma, param_dtype=complex)


fields_to_track = (
    ("t", "values"),
    ("dt", "values"),
    ("Generator", "Mean"),
    ("Generator", "Variance"),
    ("parity", "Mean"),
    ("parity", "Variance"),
    ("r_squared", "values"),
    # Umbrella/bridge monitoring fields
    ("ess_bridge", "values"),
    ("snr_min", "values"),
    ("snr_10p", "values"),
    ("snr_med", "values"),
    ("snrF_min", "values"),
    ("snrF_med", "values"),
    ("q_bridge", "values"),
    # Per-step SNRs from OVar
    ("snr", "values"),
    ("snr_F", "values"),
    ("acceptance_rate", "values"),
)


def get_vstate(N, n_samples, alpha):
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)

    seed = 300
    model = get_model(alpha)
    sampler = nk.sampler.MetropolisSampler(
        hilbert, LocalDoubleFlipRule(), n_chains=n_samples
    )
    vstate = nk.vqs.MCState(
        sampler=sampler, model=model, n_samples=n_samples, seed=seed, sampler_seed=seed
    )

    # zero everything
    pars = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), vstate.parameters)

    W = pars["network"]["Dense"]["kernel"]
    a = pars["network"]["visible_bias"]
    b = pars["network"]["Dense"]["bias"]
    n = hilbert.size
    # b + W x = i pi/4 (1 - sum_i x_i), x_i in \{+1,-1\}
    #   -> even x: b + W x = i(k+1) pi /2 ->  cosh(b + W x)= +-1
    #   -> odd x: b + W x = i k pi /2 -> cosh(b + W x)= 0
    W = W.at[:, 0].set(-1j * (np.pi / 4))
    b = b.at[0].set(1j * (np.pi / 4) * n)
    # Repeat to get rid of sign since now
    # psi(x) =  cosh(b_0 + sum_i W_0i x) * cosh(b_1 + sum_i W_1i x)
    #   -> even x: 1
    #   -> odd x: 0
    W = W.at[:, 1].set(-1j * (np.pi / 4))
    b = b.at[1].set(1j * (np.pi / 4) * n)
    # Unit 3 left as zero (neutral)
    pars["network"]["Dense"]["kernel"] = W
    pars["network"]["Dense"]["bias"] = (
        b + (1 + 1j) * jax.random.uniform(jax.random.key(100), b.shape) * 1e-4
    )
    pars["network"]["visible_bias"] = (
        jnp.zeros_like(pars["network"]["visible_bias"])
        + (1 + 1j) * jax.random.uniform(jax.random.key(100), a.shape) * 1e-4 * N
    )

    # visible_bias stays zero
    vstate.parameters = pars
    return vstate


def main():

    # 2D Lattice
    N = 4
    alpha = 4
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)

    # Hilbert space of spins on the graph
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

    # Ising spin hamiltonian
    J = 1
    h = 1.0
    generator = sum([J * sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges()])
    generator += sum([h * sigmax(hi, i) for i in g.nodes()])
    n_samples_tvmc = 2**12

    vs = get_vstate(N, n_samples_tvmc, alpha=alpha)

    # Exact simulation
    dt = 0.025
    T = 2.0
    save_times = np.linspace(0.0, T, 40)
    save_path =  f"./data/TFIM_{N}_{alpha}_parity/ptvmc_{n_samples_tvmc}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = Logger(path=save_path, fields=fields_to_track, save_every=5)
    callbacks = []

    sigma_z = nk.operator.PauliStringsJax(hi, "Z" * N, 1.0)

    def measure_parity(step, log, driver):
        log["parity"] = driver.state.expect(sigma_z)
        return True

    callbacks.append(measure_parity)
    acceptance_rate_callback = get_acceptance_rate_callback()
    callbacks.append(acceptance_rate_callback)
    # tdvp_monitor_callback = get_umbrella_monitor_callback(save_times, save_path)
    # callbacks.append(tdvp_monitor_callback)
    parameter_save_callback = get_parameter_save_callback(save_times, logger)
    callbacks.append(parameter_save_callback)

    # Define compression algorithm
    compression_alg = ptvmc.compression.InfidelityCompression(
        driver_class=advd.driver.InfidelityOptimizerNG,
        build_parameters={
            "diag_shift": 1e-5,
            "optimizer": optax.inject_hyperparams(optax.sgd)(learning_rate=0.05),
            "linear_solver_fn": cholesky,
            "proj_reg": None,
            "momentum": None,
            "chunk_size_bwd": None,
            "collect_quadratic_model": False,
            "use_ntk": False,
            "on_the_fly": False,
            "cv_coeff": -0.5,
            "resample_fraction": None,
            "estimator": "cmc",
        },
        run_parameters={
            "n_iter": 100,
            "callback":None,
        },
    )

    # Discretization scheme used to approximate the time-evolution operator
    solver = ptvmc.solver.SLPE3()

    # Define the PTVMC driver
    integration_params = ptvmc.IntegrationParameters(
        dt=dt,
    )
    generator = -1j * generator
    driver = ptvmc.PTVMCDriver(
        generator,
        0.0,
        solver=solver,
        integration_params=integration_params,
        compression_algorithm=compression_alg,
        variational_state=vs,
    )

    driver.run(
        T=T,
        out=logger,
        obs_in_fullsum=False,
        callback=callbacks,
        save_path=os.path.join(_SCRIPT_DIR, "states/"),
        save_every=1,
    )


if __name__ == "__main__":
    main()
