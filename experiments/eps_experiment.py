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
from netket.experimental.dynamics import RK45, Heun
from callbacks import (
    get_acceptance_rate_callback,
    get_umbrella_monitor_callback,
    get_tdvp_monitor_callback,
    get_parameter_save_callback,
)
from logger import Logger
from flax import serialization

# from schmitt_tdvp_bridge_jaxmg import TDVPSchmittBridgeJAXMg as DynamicsDriver
from schmitt_tdvp_bridge import TDVPSchmittBridge
from schmitt_tdvp import TDVPSchmitt
from dressed_rbm import DressedRBM

import argparse
import numpy as np

import flax.linen as nn


def main(
    N,
    n_samples_tvmc,
    q1,
    q2,
    T=0.5,
    dt_=0.01,
    ansatz="gaussian",
    chunk_size=1024,
    alpha=1,
    correlation = 0.0
):
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)
    if ansatz == "gaussian":
        from gaussian_state import make_epsilon_model

        def get_vstate(n_samples):
            sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_samples)
            model = make_epsilon_model([-1] * N)
            vstate = nk.vqs.MCState(
                sampler=sampler,
                model=model,
                n_samples=n_samples,
                seed=100,
                sampler_seed=100,
                chunk_size=chunk_size,
            )
            vstate.sampler_state = vstate.sampler_state.replace(
                σ=jnp.array([[-1] * N] * n_samples, dtype=jnp.int8)
            )

            for i in range(1):
                vstate.sample(n_samples=n_samples)
            return vstate

    elif ansatz == "rbm":

        def get_vstate(n_samples):
            sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_samples)
            graph = nk.graph.Chain(N, pbc=True)
            rbm_model = nk.models.RBMSymm(
                alpha=alpha,
                param_dtype=complex,
                # use_visible_bias=True,
                hidden_bias_init=nn.initializers.normal(1e-3),
                visible_bias_init=nn.initializers.normal(1e-3),
                kernel_init=nn.initializers.normal(1e-3),
                symmetries=graph.translation_group(),
            )
            model = DressedRBM(rbm=rbm_model, amp_init=1e-5 * 2.**(-N/2.), correlation=correlation)
            vstate = nk.vqs.MCState(
                sampler=sampler,
                model=model,
                n_samples=n_samples,
                seed=100,
                sampler_seed=100,
                chunk_size=chunk_size,
            )

            vstate.sampler_state = vstate.sampler_state.replace(
                σ=jnp.array([[-1] * N] * n_samples, dtype=jnp.int8)
            )
            for i in range(10):
                vstate.sample(n_samples=n_samples)
            print(f"Number of parameters: {vstate.n_parameters}")
            return vstate

    else:
        raise ValueError("Unknown ansatz")

    graph = nk.graph.Chain(N, pbc=True)
    hamiltonian = nk.operator.IsingJax(hilbert=hilbert, graph=graph, h=1.0, J=-1.0)

    for i in range(N):
        string = ["I"] * N
        string[i] = "Z"
        if i == 0:
            sigma_zs = nk.operator.PauliStringsJax(hilbert, "".join(string), 1.0 / N)
        else:
            sigma_zs += nk.operator.PauliStringsJax(hilbert, "".join(string), 1.0 / N)
    fields_to_track = (
        ("t", "values"),
        ("dt", "values"),
        ("Generator", "Mean"),
        ("Generator", "Variance"),
        ("sigma_z", "Mean"),
        ("sigma_z", "Variance"),
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

    def measure_sigma_z(step, log, driver):
        log["sigma_z"] = driver.state.expect(sigma_zs)
        return True

    from schmitt_tdvp_randomized_bridge import TDVPSchmittRandomizedBridge
    from schmitt_tdvp_bridge import TDVPSchmittBridge
    from schmitt_tdvp import TDVPSchmitt

    save_times = np.linspace(0.0, T, 20)
    exp_name = f"bridge_{n_samples_tvmc}_{ansatz}_q1_{q1:.2f}_q2_{q2:.2f}_T_{T:.2f}_alpha_{alpha}_correlation_{correlation:.2f}"
    # Make sure we always start with the same state in notebook

    save_path = f"./data/TFIM_EPS_{N}/{exp_name}/"

    logger = Logger(path=save_path, fields=fields_to_track, save_every=5)
    if logger.restore():
        if logger.done:
            print("Data exists, skipping...")
            return
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vstate = get_vstate(n_samples_tvmc)
    print(vstate.expect(sigma_zs))
    callbacks = []
    callbacks.append(measure_sigma_z)
    if q1 + q2 > 1e-12:
        tdvp_monitor_callback = get_umbrella_monitor_callback(save_times, save_path)
    else:
        tdvp_monitor_callback = get_tdvp_monitor_callback(save_times, save_path)
    callbacks.append(tdvp_monitor_callback)

    # integrator = RK45(1e-3, adaptive=True, rtol=1e-4, dt_limits=(1e-4, 1e-2))
    integrator = Heun(dt_)
    tvmc_kwargs = {}
    if np.abs(q1 + q2) < 1e-12:
        print(
            "Using standard TDVP without bridge sampling since q1 and q2 are both close to 0."
        )
        driver = TDVPSchmitt(
            hamiltonian,
            vstate,
            integrator,
            t0=0,
            holomorphic=False,
            snr_atol=2,
            rcond=1e-14,
            rcond_smooth=1e-10,
            **tvmc_kwargs,
        )
    else:
        if q2 < 1e-12:
            print("Using TDVP with bridge sampling since q2 is close to 0.")
            driver = TDVPSchmittBridge(
                hamiltonian,
                vstate,
                integrator,
                t0=0,
                q=q1,
                holomorphic=False,
                snr_atol=2,
                rcond=1e-14,
                rcond_smooth=1e-10,
                **tvmc_kwargs,
                distributed_eigh=False,
            )
        else:
            print("Using TDVP with randomized bridge sampling")
            driver = TDVPSchmittRandomizedBridge(
                hamiltonian,
                vstate,
                integrator,
                t0=0,
                q1=q1,
                q2=q2,
                flip_prob = 1/N,
                holomorphic=False,
                snr_atol=2,
                rcond=1e-14,
                rcond_smooth=1e-10,
                **tvmc_kwargs,
                distributed_eigh=False,
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
    parser.add_argument("--q1", default=0.25, type=float)
    parser.add_argument("--q2", default=0.25, type=float)
    parser.add_argument("--T", default=0.5, type=float)
    parser.add_argument("--ansatz", default="gaussian", type=str)
    parser.add_argument("--chunk_size", default=1024, type=int)
    parser.add_argument("--dt", default=0.01, type=float)
    parser.add_argument("--alpha", default=1, type=int)
    parser.add_argument("--correlation", default=0.0, type=float)
    args = parser.parse_args()
    main(
        int(args.N),
        2 ** int(args.power),
        float(args.q1),
        float(args.q2),
        float(args.T),
        dt_=float(args.dt),
        ansatz=args.ansatz,
        chunk_size=int(args.chunk_size),
        alpha=int(args.alpha),
        correlation=float(args.correlation)
    )
