import os
import matplotlib.pyplot as plt
from matplotlib import colors

import sys

sys.path.append("../src")

import jax
import jax.numpy as jnp

import warnings
warnings.filterwarnings("error", category=jnp.ComplexWarning)

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

# from schmitt_tdvp_bridge_jaxmg import TDVPSchmittBridgeJAXMg as DynamicsDriver
from schmitt_tdvp_bridge import TDVPSchmittBridge
from schmitt_tdvp import TDVPSchmitt

import argparse
import numpy as np
from flax import serialization

from gaussian_state import GaussianState


def main(N, n_samples_tvmc, driver_type, q, h, T, dt_, chunk_size):
    print(N, n_samples_tvmc, driver_type, q, h)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)

    def get_model(dtype):
        return GaussianState(param_dtype=dtype)

    def get_vstates(n_samples, sampling_dtype=jnp.float64):
        seed = 300
        model = get_model(jnp.float64)
        sampler = nk.sampler.MetropolisSampler(
            hilbert, LocalDoubleFlipRule(), n_chains=n_samples
        )
        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=n_samples,
            seed=seed,
            sampler_seed=seed,
            n_discard_per_chain=0,
            chunk_size=chunk_size
        )
        model_sampling = get_model(sampling_dtype)
        sampler_sampling = nk.sampler.MetropolisSampler(
            hilbert, LocalDoubleFlipRule(), n_chains=n_samples, dtype=sampling_dtype
        )
        vstate_sampling = nk.vqs.MCState(
            sampler=sampler_sampling,
            model=model_sampling,
            n_samples=n_samples,
            seed=seed,
            sampler_seed=seed,
            n_discard_per_chain=0,
        )

        return vstate, vstate_sampling

    vstate, _ = get_vstates(2**10)
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
    print(f"h={h:1.3f}")
    hamiltonian = nk.operator.IsingJax(hilbert=hilbert, graph=graph, h=-h, J=1.0)

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

    save_times = np.linspace(0.0, T, 40)
    if driver_type == "bridge":
        exp_name = f"bridge_{n_samples_tvmc}_h_{h:1.3f}_q_{q:1.3f}_T_{T:1.3f}"
    elif driver_type == "vanilla":
        exp_name = f"vanilla_{n_samples_tvmc}_h_{h:1.3f}_T_{T:1.3f}"
    else:
        raise NotImplementedError
    # Make sure we always start with the same state in notebook

    save_path = f"./data/TFIM_PFAFF_{N}_parity/{exp_name}/"

    logger = Logger(path=save_path, fields=fields_to_track, save_every=5)
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

    vstate, vstate_sampling = get_vstates(n_samples_tvmc, jnp.float32)
    print(f"Number of parameters: {vstate.n_parameters}")
    if 1 and driver_type == "vanilla":
        with open(
            f"./data/TFIM_PFAFF_{N}_parity/bridge_{2**13}_{0.5:1.2f}/"
            + f"log_params_1.mpack",
            "rb",
        ) as infile:
            binary_data = infile.read()
            vstate.variables = serialization.from_bytes(vstate.variables, binary_data)
            vstate.variables = jax.tree.map(lambda x: jnp.array(x), vstate.variables)
        t0 = 0.05
    # Thermalize
    for i in range(1):
        vstate.sample()
        vstate_sampling.sample()

    callbacks = []
    callbacks.append(measure_parity)
    acceptance_rate_callback = get_acceptance_rate_callback()
    callbacks.append(acceptance_rate_callback)
    if driver_type == "bridge":
        tdvp_monitor_callback = get_umbrella_monitor_callback(save_times, save_path)
    elif driver_type == "vanilla":
        tdvp_monitor_callback = get_tdvp_monitor_callback(save_times, save_path)
    callbacks.append(tdvp_monitor_callback)
    parameter_save_callback = get_parameter_save_callback(save_times, logger)
    callbacks.append(parameter_save_callback)

    # integrator = RK45(dt, adaptive=True, rtol=1e-4, dt_limits=(1e-5, 1e-2))
    dt = dt_
    integrator = Heun(dt)
    tvmc_kwargs = {}

    if driver_type == "bridge":
        dynamics = TDVPSchmittBridge(
            hamiltonian,
            vstate,
            integrator,
            t0=t0,
            q=q,
            snr_atol=2,
            rcond=1e-14,
            rcond_smooth=1e-10,
            sampling_state=vstate_sampling,
            distributed_eigh=True,
            **tvmc_kwargs,
        )
    elif driver_type == "vanilla":
        dynamics = TDVPSchmitt(
            hamiltonian,
            vstate,
            integrator,
            t0=t0,
            snr_atol=2,
            rcond=1e-14,
            rcond_smooth=1e-10,
            distributed_eigh=True,
            **tvmc_kwargs,
        )
    else:
        raise NotImplementedError

    dynamics.run(
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
    parser.add_argument("--driver_type", default="vanilla", type=str)
    parser.add_argument("--q", default=0.5, type=float)
    parser.add_argument("--h", default=1.0, type=float)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--T", default=0.5, type=float)
    parser.add_argument("--dt", default=1e-3, type=float)
    args = parser.parse_args()
    print(args.chunk_size)
    main(
        int(args.N),
        2 ** int(args.power),
        args.driver_type,
        float(args.q),
        float(args.h),
        float(args.T),
        float(args.dt),
        args.chunk_size
    )
