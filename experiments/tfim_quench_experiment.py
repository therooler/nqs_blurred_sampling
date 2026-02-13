import argparse
import sys
from pathlib import Path
from typing import Tuple
import ast

# Ensure `rbm_qsim/` is importable regardless of current working directory.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str((_THIS_DIR / ".." / "src").resolve()))
import netket as nk
import numpy as np

from netket.operator.spin import sigmax, sigmaz
from netket.experimental.dynamics import RK45
import optax

from callbacks import (
    get_parameter_save_callback,
    get_umbrella_monitor_callback,
    get_acceptance_rate_callback,
)
from metropolis import LocalZ2FlipRule
from logger import Logger

from schmitt_tdvp_bridge import TDVPSchmittBridge
from models import CNN

fields_to_track = (
    ("t", "values"),
    ("dt", "values"),
    ("Generator", "Mean"),
    ("Generator", "Variance"),
    ("r_squared", "values"),
    ("R_hat", "values"),
    # SNR monitoring fields
    ("snr_min", "values"),
    ("snr_10p", "values"),
    ("snr_med", "values"),
    ("snrF_min", "values"),
    ("snrF_med", "values"),
    ("snr", "values"),
    ("snr_F", "values"),
    # Bridge
    ("ess_bridge", "values"),
    ("q_bridge", "values"),
    ("mx", "Mean"),
    ("mx", "Variance"),
    ("acceptance_rate", "values"),
)


def get_observable_callback(obs: nk.operator.AbstractOperator, name: str):
    def measure_obs(step, log, driver):
        log[name] = driver.state.expect(obs)
        return True

    return measure_obs


def get_save_path(config: dict, *, create: bool = True) -> Tuple[str, str]:
    required = [
        "experiment_name",
        "L",
        "seed",
        "kernel_size",
        "channels",
        "n_samples_sr",
        "n_samples_tvmc",
        "hc_multiplier",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys for save_path(): {missing}")

    root = Path(config.get("data_prepend", "./data")).expanduser()
    channel_str = "(" + "_".join(map(str, config["channels"])) + ")"
    parts_tvmc = [
        "TFIM_QUENCH",
        str(config["experiment_name"]),
        f"L_{int(config['L'])}",
        f"CNN_K_{int(config['kernel_size'])}_C_{channel_str}",
        f"seed_{int(config['seed'])}",
        f"Ns_SR_{int(config['n_samples_sr'])}",
        f"hc_mult_{float(config['hc_multiplier']):1.3f}",
        f"T_{float(config['T']):1.3f}",
        f"Ns_TVMC_{int(config['n_samples_tvmc'])}",
    ]
    parts_sr = parts_tvmc[:6]
    out_dir_sr = root.joinpath(*parts_sr)
    out_dir_tvmc = root.joinpath(*parts_tvmc)
    if create:
        out_dir_tvmc.mkdir(parents=True, exist_ok=True)
    return str(out_dir_sr) + "/", str(out_dir_tvmc) + "/"


def main(config, return_save_paths: bool = False, return_logger: bool = False):
    L = int(config["L"])
    seed = int(config["seed"])
    n_samples_sr = int(config.get("n_samples_sr", 2048))
    n_samples_tvmc = int(config["n_samples_tvmc"])
    hc_multiplier = float(config.get("hc_multiplier", 1.0))
    kernel_size = int(config.get("kernel_size", 2))
    channels = tuple(config.get("channels", (2,)))

    assert (
        kernel_size <= L
    ), f"kernel size must be smaller than lattice size, received K={kernel_size} for L = {L}"
    if return_save_paths:
        return get_save_path(config, create=False)
    else:
        save_path_sr, save_path_tvmc = get_save_path(config)

    if return_logger:
        logger_tvmc = Logger(path=save_path_tvmc, fields=fields_to_track)
        if logger_tvmc.restore():
            return logger_tvmc
        else:
            raise ValueError(f"No logger found at {save_path_tvmc}")
    # 2D Lattice
    graph = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
    # Hilbert space of spins on the graph
    hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, inverted_ordering=False)
    # Ising coeffs
    J = 1
    hc = 3.044 * J
    h = hc * hc_multiplier
    # Hamiltonian
    ha = sum([-J * sigmaz(hi, i) @ sigmaz(hi, j) for i, j in graph.edges()])
    ha += sum([-h * sigmax(hi, i) for i in graph.nodes()])
    mx = sum([sigmax(hi, i) for i in graph.nodes()]) / graph.n_nodes
    hamiltonian = ha.to_jax_operator()
    mx = mx.to_jax_operator()
    # Times
    T = float(config.get("T", 2.0))
    n_save_times = int(config.get("n_save_times", 20))
    save_times = np.linspace(0.0, T, n_save_times)
    # Monte Carlo Sampling
    sampler = nk.sampler.MetropolisSampler(hi, LocalZ2FlipRule(), n_chains=n_samples_sr)
    model = CNN(
        lattice=graph,
        kernel_size=(kernel_size**2,),
        channels=channels,
        param_dtype=complex,
    )
    # Variational State
    vstate = nk.vqs.MCState(
        sampler, model, n_samples=n_samples_sr, seed=seed, sampler_seed=seed, chunk_size=2**13
    )
    print("Number of parameters: ", vstate.n_parameters)

    # perturbation = 1e-3
    ha_gs = sum([-sigmax(hi, i) for i in graph.nodes()])
    ha_gs = ha_gs.to_jax_operator()
    # ha_gs = nk.operator.IsingJax(hilbert=hi, graph=graph, h=-1., J=-10.)
    logger_sr = Logger(
        path=save_path_sr, fields=(("Generator", "Mean"), ("Generator", "Variance"))
    )
    # If we haven't done ground state yet get it.
    logger_sr.restore()
    if logger_sr.done:
        print("Restoring SR State")
        assert logger_sr.restore_state(vstate), "Failed to restore state."
    else:
        print("Finding ground state")
        init_driver = nk.driver.VMC_SR(
            hamiltonian=ha_gs,
            variational_state=vstate,
            optimizer=optax.sgd(0.05),
            diag_shift=1e-4,
            use_ntk=False,
            on_the_fly=False,
        )
        init_driver.run(n_iter=150, out=logger_sr)
        logger_sr.flush(vstate, done=True)

    logger_tvmc = Logger(path=save_path_tvmc, fields=fields_to_track)
    restored = logger_tvmc.restore()
    dt_min_lim = 5e-5
    if restored:
        t0 = logger_tvmc["t"]["values"][-1]
        dt = logger_tvmc["dt"]["values"][-1]
    else:
        t0 = 0.0
        dt = dt_min_lim
    print(f"Starting from {t0}")
    if not logger_tvmc.done:
        vstate.n_samples = n_samples_tvmc
        vstate.sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_samples_tvmc)
        # Thermalize
        for i in range(100):
            vstate.sample()
        # Callbacks for dynamics
        callbacks = []
        acceptance_rate_callback = get_acceptance_rate_callback()
        callbacks.append(acceptance_rate_callback)
        callbacks.append(get_observable_callback(mx, "mx"))
        callbacks.append(get_umbrella_monitor_callback(save_times, save_path_tvmc))
        callbacks.append(get_parameter_save_callback(save_times, logger_tvmc))

        integrator = RK45(dt, adaptive=True, rtol=1e-4, dt_limits=(dt_min_lim, 1e-2))

        driver = TDVPSchmittBridge(
            hamiltonian,
            vstate,
            integrator,
            t0=t0,
            q=float(config.get("q0", 0.5)),
            holomorphic=False,
            snr_atol=2.0,
            rcond=1e-14,
            rcond_smooth=1e-9,
        )

        driver.run(
            T,
            out=logger_tvmc,
            callback=callbacks,
            show_progress=True,
            timeit=True,
        )
        logger_tvmc.flush(vstate, done=True)
    print("Dynamics done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", default=3, type=int)
    parser.add_argument("--hc_multiplier", default=1.0, type=float)
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--q", default=0.5, type=float)
    parser.add_argument("--n_samples_sr", default=2**12, type=int)
    parser.add_argument("--kernel_size", default=2, type=int)
    parser.add_argument("--channels", default="(5, 4, 3)", type=lambda x: tuple(ast.literal_eval(x)))
    parser.add_argument("--power", default=12, type=int)
    parser.add_argument("--experiment_name", default="Feb11", type=str)
    parser.add_argument("--data_prepend", default="./data", type=str)
    parser.add_argument("--T", default=4.0, type=float)
    parser.add_argument("--n_save_times", default=40, type=int)
    args = parser.parse_args()

    config = {
        "experiment_name": args.experiment_name,
        "data_prepend": args.data_prepend,
        "L": int(args.L),
        "seed": int(args.seed),
        "n_samples_sr": int(args.n_samples_sr),
        "n_samples_tvmc": 2 ** int(args.power),
        "hc_multiplier": float(args.hc_multiplier),
        "T": float(args.T),
        "n_save_times": int(args.n_save_times),
        "q": float(args.q),
        "kernel_size": int(args.kernel_size),
        "channels": args.channels,
    }

    main(config)
