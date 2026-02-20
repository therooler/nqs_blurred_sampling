import os
import matplotlib.pyplot as plt
from matplotlib import colors

import sys

sys.path.append("../src")

import netket as nk

from netket.experimental.dynamics import RK45, Heun

from callbacks import (
    get_tdvp_monitor_callback,
    get_umbrella_monitor_callback,
)

from logger import Logger

import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from functools import partial
from flax import serialization
from schmitt_tdvp_bridge import TDVPSchmittBridge
from schmitt_tdvp import TDVPSchmitt
from schmitt_tdvp_randomized_bridge import TDVPSchmittRandomizedBridge
import argparse


def main(q):

    Lx = 8
    Ly = 2
    N = Lx * Ly
    A_p = 0.15

    hilbert = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=0)

    graph = nk.graph.Grid((Lx, Ly), pbc=(True, False))
    hamiltonian = nk.operator.Heisenberg(hilbert, graph, J=1.0, sign_rule=True)

    def pulse(t, A_p, omega_p, sigma_p, t_p):
        return (
            A_p * jnp.sin(omega_p * t) * jnp.exp(-((t - t_p) ** 2) / (2 * sigma_p**2))
        )

    pulse_partial = partial(pulse, A_p=A_p, omega_p=8.0, sigma_p=0.4, t_p=0.987)

    t = jnp.linspace(0, 4, 100)
    plt.plot(t, partial(pulse, A_p=0.20, omega_p=8.0, sigma_p=0.4, t_p=0.987)(t))
    plt.show()

    def get_vstate(n_samples):
        sampler = nk.sampler.MetropolisExchange(
            hilbert, graph=graph, n_chains=n_samples
        )
        # model = nk.models.RBM(alpha=10, param_dtype=complex)
        model = nk.models.RBMSymm(
            symmetries=graph.point_group(), alpha=10, param_dtype=complex
        )
        return nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=n_samples,
            seed=100,
            sampler_seed=100,
        )

    def get_vstate_parameters(n_samples):
        vstate = get_vstate(n_samples)

        # Thermalize
        for i in range(100):
            vstate.sample(n_samples=n_samples)
        e0 = nk.exact.lanczos_ed(hamiltonian)
        print(e0)
        gs = nk.driver.VMC_SR(
            hamiltonian,
            optimizer=nk.optimizer.Sgd(0.001),
            variational_state=vstate,
            diag_shift=1e-4,
        )
        gs.run(10000, callback=lambda s, l, d: d._loss_stats.mean > e0 + 1e-3)
        return vstate.parameters

    vstate = get_vstate(2**12)
    print(vstate.n_parameters)
    if not os.path.exists("./data" + f"/rbmsymm_params_heisenberg_{Lx}_{Ly}.mpack"):
        parameters = get_vstate_parameters(2**12)
        binary_data = serialization.to_bytes(parameters)
        with open(
            "./data" + f"/rbmsymm_params_heisenberg_{Lx}_{Ly}.mpack", "wb"
        ) as outfile:
            outfile.write(binary_data)
        print("saved parameters")
    else:
        with open(
            "./data" + f"/rbmsymm_params_heisenberg_{Lx}_{Ly}.mpack", "rb"
        ) as infile:
            binary_data = infile.read()
            parameters = serialization.from_bytes(vstate.parameters, binary_data)
        print("loaded parameters")
    vstate.parameters = parameters.copy()

    e0 = nk.exact.lanczos_ed(hamiltonian)
    estimated_e = vstate.expect(hamiltonian)
    print(f"GS energy {e0}")
    print(f"variational energy {estimated_e}")

    x_bonds = []
    for y in range(Ly):
        for x in range(Lx - 1):
            i = y * Lx + x
            j = y * Lx + (x + 1)
            x_bonds.append((i, j))
    x_bonds_graph = nk.graph.Graph(edges=x_bonds)
    H = graph.to_networkx()
    G = x_bonds_graph.to_networkx()
    import networkx as nx

    fig, (axs0, axs1) = plt.subplots(1, 2)
    nx.draw(G, node_color="orange", ax=axs0, with_labels=True)
    nx.draw(H, ax=axs1, with_labels=True)
    heisenberg_x = nk.operator.Heisenberg(hilbert, x_bonds_graph, J=1.0)
    quench_hamiltonian = lambda t: hamiltonian + pulse_partial(t) * heisenberg_x
    s_correlator = nk.operator.Heisenberg(hilbert, x_bonds_graph, J=1.0 / N)

    fields_to_track = (
        ("t", "values"),
        ("dt", "values"),
        ("Generator", "Mean"),
        ("Generator", "Variance"),
        ("s_corr", "Mean"),
        ("s_corr", "Variance"),
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
    )

    def measure_corr(step, log, driver):
        log["s_corr"] = driver.state.expect(s_correlator)
        return True

    T = 10
    n_samples_tvmc = 2**13
    save_times = np.linspace(0.0, T, 20)
    exp_name = f"bridge_{n_samples_tvmc}_Ap_{A_p:1.2f}_q_{q:1.2f}"
    # Make sure we always start with the same state in notebook

    save_path = f"./data/HEISENBERG_LADDER_{Lx}_{Ly}/{exp_name}/"

    logger = Logger(path=save_path, fields=fields_to_track)
    vstate = get_vstate(n_samples_tvmc)
    if logger.restore():
        if logger.done:
            print("Data exists, skipping...")
            return
        else:
            t0 = logger["t"]["values"][-1]
            dt = logger["dt"]["values"][-1]
            logger.restore_state(vstate)
    else:
        t0 = 0.0
        dt = 1e-3
        vstate.parameters = parameters.copy()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(100):
        vstate.sample()
    callbacks = []
    callbacks.append(measure_corr)
    if q == 0:
        tdvp_monitor_callback = get_tdvp_monitor_callback(save_times, save_path)
    else:
        tdvp_monitor_callback = get_umbrella_monitor_callback(save_times, save_path)
    callbacks.append(tdvp_monitor_callback)

    # integrator = RK45(dt, adaptive=False, rtol=1e-6, dt_limits=(1e-5, 1e-2))
    integrator = Heun(dt)
    tvmc_kwargs = {}
    if q == 0:
        driver = TDVPSchmitt(
            quench_hamiltonian,
            vstate,
            integrator,
            t0=t0,
            holomorphic=False,
            snr_atol=2,
            rcond=1e-9,
            rcond_smooth=1e-10,
            **tvmc_kwargs,
        )
    else:
        driver = TDVPSchmittBridge(
            quench_hamiltonian,
            vstate,
            integrator,
            t0=t0,
            q=q,
            holomorphic=False,
            snr_atol=2,
            rcond=1e-9,
            rcond_smooth=1e-10,
            **tvmc_kwargs,
        )
        # driver = TDVPSchmittRandomizedBridge(
        #     hamiltonian,
        #     vstate,
        #     integrator,
        #     t0=0,
        #     flip_prob=q,
        #     holomorphic=False,
        #     snr_atol=2,
        #     rcond=1e-7,
        #     rcond_smooth=1e-10,
        #     **tvmc_kwargs,
        # )
    driver.run(
        T - t0,
        out=logger,
        callback=callbacks,
        show_progress=True,
        timeit=True,
    )

    logger.flush(vstate, done=True)
    # fit_bridge(0.0, 2**12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", default=0.5, type=float)
    args = parser.parse_args()
    main(float(args.q))
