import os
import matplotlib.pyplot as plt
from matplotlib import colors

import sys

sys.path.append("../src")

import jax
import jax.numpy as jnp

import netket as nk
import numpy as np

from netket.operator.spin import sigmax, sigmaz
from metropolis import LocalDoubleFlipRule
from netket.experimental.dynamics import RK45
from callbacks import (
    get_acceptance_rate_callback,
    get_umbrella_monitor_callback,
    get_parameter_save_callback,
)
from logger import Logger

# from schmitt_tdvp_bridge_jaxmg import TDVPSchmittBridgeJAXMg as DynamicsDriver
from schmitt_tdvp_bridge import TDVPSchmittBridge as DynamicsDriver

import argparse
import numpy as np


def main(N, n_samples_tvmc):

    alpha = 8
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)

    def get_model():
        return nk.models.RBM(
            alpha=alpha,
            param_dtype=complex,
        )

    def get_vstate(n_samples):
        seed = 300
        model = get_model()
        sampler = nk.sampler.MetropolisSampler(
            hilbert, LocalDoubleFlipRule(), n_chains=n_samples
        )
        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=n_samples,
            seed=seed,
            sampler_seed=seed,
        )

        # zero everything
        pars = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), vstate.parameters)

        W = pars["Dense"]["kernel"]
        a = pars["visible_bias"]
        b = pars["Dense"]["bias"]
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
        pars["Dense"]["kernel"] = W
        pars["Dense"]["bias"] = (
            b + (1 + 1j) * jax.random.uniform(jax.random.key(100), b.shape) * 1e-4
        )
        pars["visible_bias"] = (
            jnp.zeros_like(pars["visible_bias"])
            + (1 + 1j) * jax.random.uniform(jax.random.key(100), a.shape) * 1e-4 * N
        )

        # visible_bias stays zero
        vstate.parameters = pars
        return vstate

    vstate = get_vstate(2**10)
    sigma_z = nk.operator.PauliStringsJax(hilbert, "Z" * N, 1.0)
    Pz = nk.operator.PauliStringsJax(hilbert, "Z" * N, -1.0)
    graph = nk.graph.Chain(N, pbc=True)
    Hxx = sum([sigmax(hilbert, i) @ sigmax(hilbert, j) for i, j in graph.edges()])
    stab_hamiltonian = -Hxx + Pz
    parity_expect = vstate.expect(sigma_z)
    stab_energy = vstate.expect(stab_hamiltonian)
    print(f"parity: {parity_expect}")
    print(f"parity: {stab_energy}")
    graph = nk.graph.Chain(N, pbc=True)
    # hamiltonian = sum([sigmax(hilbert, i) for i in graph.nodes()])
    # hamiltonian += sum([sigmaz(hilbert, i) @ sigmaz(hilbert, j) for i, j in graph.edges()])
    hamiltonian = nk.operator.IsingJax(hilbert=hilbert, graph=graph, h=-1.0, J=1.0)

    print(vstate.expect(sigma_z))
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

    def measure_parity(step, log, driver):
        log["parity"] = driver.state.expect(sigma_z)
        return True

    T = 2.0
    save_times = np.linspace(0.0, T, 40)
    exp_name = f"bridge_{n_samples_tvmc}"
    # Make sure we always start with the same state in notebook

    save_path = f"./data/TFIM_{N}_{alpha}_parity/{exp_name}/"

    logger = Logger(path=save_path, fields=fields_to_track)
    if logger.restore():
        if logger.done:
            print("Data exists, skipping...")
            return
        else:
            t0 = logger["t"]["values"][-1]
            dt = logger["dt"]["values"][-1]
    else:
        t0 = 0.0
        dt = 1e-5
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vstate = get_vstate(n_samples_tvmc)
    # Thermalize
    for i in range(1000):
        vstate.sample()

    callbacks = []
    callbacks.append(measure_parity)
    acceptance_rate_callback = get_acceptance_rate_callback()
    callbacks.append(acceptance_rate_callback)
    tdvp_monitor_callback = get_umbrella_monitor_callback(save_times, save_path)
    callbacks.append(tdvp_monitor_callback)
    parameter_save_callback = get_parameter_save_callback(save_times, logger)
    callbacks.append(parameter_save_callback)

    integrator = RK45(dt, adaptive=True, rtol=1e-4, dt_limits=(1e-5, 1e-2))
    tvmc_kwargs = {}
    driver = DynamicsDriver(
        hamiltonian,
        vstate,
        integrator,
        t0=t0,
        q=0.9,
        snr_atol=2,
        rcond=1e-14,
        rcond_smooth=1e-10,
        **tvmc_kwargs,
    )

    driver.run(
        T,
        out=logger,
        callback=callbacks,
        show_progress=True,
        timeit=True,
    )
    logger.flush(vstate, done=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=20, type=int)
    parser.add_argument("--power", default=10, type=int)
    args = parser.parse_args()
    main(int(args.N), 2 ** int(args.power))
