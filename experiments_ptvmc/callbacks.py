import time
from functools import partial

import jax.numpy as jnp
import numpy as np

from netket.optimizer.qgt import QGTJacobianDense
import netket as nk
import jax


def early_stopping_variance(step, log, driver):
    return not driver.energy.variance < 1e-4


def get_energy_callback(certification_hamiltonian, frequency):
    def energy_callback(step, log, driver):
        if step % frequency == 0:
            log["Energy"] = driver.state.expect(certification_hamiltonian, frequency)
        return True

    return energy_callback


def get_time_per_step_callback():
    def time_per_step_callback(step, log, driver):
        log["real_time"] = time.time()
        return True

    return time_per_step_callback


def get_cg_nsteps_callback():
    def time_per_step_callback(step, log, driver):
        try:
            log["cg_steps"] = driver.nsteps_cg
        except AttributeError:
            log["cg_steps"] = 0
        return True

    return time_per_step_callback


def get_qgt_callback_sr(frequency=10):
    qgt_constructor = partial(QGTJacobianDense, holomorphic=False)

    def qgt_callback(step, log, driver):
        if step % frequency == 0:
            rank = jnp.linalg.matrix_rank(qgt_constructor(driver.state).to_dense())
            ### calculate rank of qgt ###
            log["qgt_rank"] = rank
        return True

    return qgt_callback


def get_certification_estimator_callback(save_times, hamiltonian):
    save_times_tracked = save_times.copy()

    def eval_calculator(step, log, _driver):
        hit = np.isclose(step, save_times_tracked, atol=_driver.dt)
        if np.any(hit):
            log["H_cert"] = _driver.state.expect(hamiltonian)
            log["t_cert"] = step
            idx = np.where(np.isclose(step, save_times_tracked, atol=_driver.dt))[0]
            save_times_tracked[idx] = -1
            print(f"Certification Energy at t: {step:1.5f} = ", log["H_cert"])
        return True

    return eval_calculator


def get_certification_estimator_callback_fidelity(save_every, hamiltonian):

    def eval_calculator(step, log, _driver):
        if step % save_every == 0:
            log["H_cert"] = _driver.state.expect(hamiltonian)
            log["t_cert"] = float(_driver.t)
            print(f"Certification Energy at t: {_driver.t:1.4f} = ", log["H_cert"])
        return True

    return eval_calculator


def nystrom_approximate(A, size, l):
    """https://arxiv.org/pdf/1706.05736"""
    Omega = jax.random.normal(jax.random.key(100), (size, l))
    Omega = jnp.linalg.qr(Omega)[0]
    Y = jax.vmap(lambda v: A @ v, in_axes=1, out_axes=1)(Omega)
    # Y = jax.lax.map(lambda v: A @ v, Omega.T, batch_size=1).T
    eps = 10 ** (-jnp.finfo(Y.dtype).precision)
    nu = eps * jnp.linalg.norm(Y, "fro")
    Ynu = Y + nu * Omega
    C = jnp.linalg.cholesky(Omega.T @ Ynu, upper=False)
    B = jax.scipy.linalg.solve_triangular(C, Ynu.T, check_finite=True, lower=True)
    U, Eta, _ = jnp.linalg.svd(B.T, full_matrices=False)
    Lambda = jnp.maximum(jnp.zeros_like(Eta, dtype=Eta.dtype), Eta**2 - nu)
    return U, Lambda


def pinv(A, B, rtol=1e-14):
    Σ, U = jnp.linalg.eigh(A)
    # Discard eigenvalues below numerical precision
    Σ_inv = jnp.where(jnp.abs(Σ / Σ[-1]) > rtol, jnp.reciprocal(Σ), 0.0)
    return U @ (jnp.diag(Σ_inv) @ (U.conj().T @ B)), Σ


def get_complex_structure_callback(save_times, save_path, starting_idx=0):
    save_times_tracked = save_times.copy()

    def complex_structure_callback(step, log, driver):
        hit = np.isclose(step, save_times_tracked, atol=driver.dt)
        if np.any(hit):
            g = nk.optimizer.qgt.QGTJacobianDense(
                driver.state, mode="complex"
            ).to_dense()
            omega = nk.optimizer.qgt.QGTJacobianDense(
                driver.state, mode="imag"
            ).to_dense()
            idx = np.where(np.isclose(step, save_times_tracked, atol=driver.dt))[0]
            J, spectrum = pinv(g, omega, rtol=1e-10)
            print(f"Saving complex structure {idx[0] + starting_idx}...")
            np.save(f"{save_path}/J_{idx[0] + starting_idx}.npy", -J)
            np.save(f"{save_path}/g_ev_{idx[0] + starting_idx}.npy", spectrum)
            print("Completed!")
            save_times_tracked[idx] = -1
        return True

    return complex_structure_callback


def get_symplectic_rank_callback(save_times, rtol: float = 1e-8, atol: float = 1e-12):
    save_times_tracked = save_times.copy()

    def symplectic_rank_callback(step, log, driver):
        # hit = np.isclose(step, save_times_tracked, atol=driver.dt)
        # if np.any(hit):
        omega = nk.optimizer.qgt.QGTJacobianDense(
            driver.state, mode="imag"
        ).to_dense()
        singular_values = jnp.linalg.svd(omega, compute_uv=False)
        sig = jnp.clip(singular_values, a_min=0.0)
        thresh = jnp.maximum(atol, rtol * sig[0])
        keep = sig > thresh
        k = jnp.sum(keep).astype(jnp.int32)
        log["omega_r"] = (k / omega.shape[0])  # scalar int32
        # save_times_tracked[idx] = -1
        return True

    return symplectic_rank_callback


def get_qgt_spectrum_callback(save_times):
    save_times_tracked = save_times.copy()

    def qgt_spectrum_callback(step, log, driver):
        hit = np.isclose(step, save_times_tracked, atol=driver.dt)
        if np.any(hit):
            if driver._last_qgt is not None:
                qgt = driver._last_qgt
                evals = jnp.linalg.eigvalsh(qgt.to_dense())
            else:
                qgt_constructor = partial(QGTJacobianDense, mode="complex")
                qgt = qgt_constructor(driver.state)
                evals = jnp.linalg.eigvalsh(qgt.to_dense())
            log["qgt_spectrum"] = evals
            # plt.plot(np.abs(np.sort(evals)))
            # plt.yscale('log')
            # plt.show()
            idx = np.where(np.isclose(step, save_times_tracked, atol=driver.dt))[0]
            save_times_tracked[idx] = -1
        return True

    return qgt_spectrum_callback


def get_parameter_save_callback(save_times, logger, starting_idx=0):
    save_times_tracked = save_times.copy()
    def parameter_save_callback(s, log, driver):
        step= driver.t
        log["t"] = step
        hit = np.isclose(step, save_times_tracked, atol=driver.dt)
        if np.any(hit):
            idx = np.where(np.isclose(step, save_times_tracked, atol=driver.dt))[0]
            logger.save_parameters(
                driver.state, var_name=f"_{idx[0] + starting_idx}"
            )
            print("Saved parameters...")
            save_times_tracked[idx] = -1
        return True

    return parameter_save_callback



def get_acceptance_rate_callback():

    def acceptance_rate_callback(step, log, driver):
        log["acceptance_rate"] = driver.state.sampler_state.acceptance
        return True

    return acceptance_rate_callback


def get_parameter_save_callback_fidelity(save_every, logger, starting_idx=0):

    def parameter_save_callback(step, log, driver):
        if step % save_every == 0:
            logger.save_parameters(driver.state, var_name=f"_{step+starting_idx}.npy")
            print(
                f"Saved parameters...t = {driver.t:1.4f} , step = {step+starting_idx}"
            )
        return True

    return parameter_save_callback


def get_flush_callback(save_times, logger, starting_idx=0):
    save_times_tracked = save_times.copy()

    def flush_callback(step, log, driver):
        hit = np.isclose(step, save_times_tracked, atol=driver.dt)
        if np.any(hit):
            idx = np.where(np.isclose(step, save_times_tracked, atol=driver.dt))[0]
            logger.flush(driver.state, var_name=f"_{idx[0] + starting_idx}")
            print("Saved parameters...")
            save_times_tracked[idx] = -1
        return True

    return flush_callback


def get_rhat_callback():
    def rhat_callback(step, log, driver):
        try:
            log["R_hat"] = driver._loss_stats.R_hat
        except AttributeError:
            log["R_hat"] = np.nan
        return True

    return rhat_callback

def get_schmitt_callback():
    def schmitt_callback(step, log, driver):
        try:
            dt = driver.integrator._state.dt
            log["dt"] = dt
            log["r_squared"] = driver._rmd
            log["snr"] = driver._snr
            if np.isnan(dt):
                print("NaN detected in dt! Breaking...")
                return False
            # try:
            #     log['R_squared'] += driver._rmd
            # except KeyError:
            #     log['R_squared'] = driver._rmd
            # if log['R_squared'] > threshold:
            #     print(f"Simulation error exceeded {threshold:1.3f}")
            #     return False
        except AttributeError:
            log["dt"] = np.nan
            log["r_squared"] = np.nan
            log["snr"] = np.nan
        return True

    return schmitt_callback


def get_umbrella_monitor_callback(save_times, save_path):
    save_times_tracked = save_times.copy()

    def umbrella_monitor_callback(step, log, driver):
        # Populate monitoring metrics from driver's self._monitor (make_monitor_dict)
        try:
            dt = driver.integrator._state.dt
            log["dt"] = dt
        except AttributeError:
            log["dt"] = np.nan
        try:
            monitor = driver._monitor
        except AttributeError:
            raise ValueError("No monitor found in driver, callback can't be used.")

        # Scalars: convert possible JAX arrays to Python floats
        def _to_float(x, default=np.nan):
            try:
                return float(np.array(x))
            except Exception:
                return default

        log["r_squared"] = _to_float(monitor.get("rmd", np.nan))
        # ESS as fraction in [0,1] for plotting, plus absolute ESS
        log["ess_bridge"] = _to_float(monitor.get("ess_bridge", np.nan))

        log["snr_min"] = _to_float(monitor.get("snr_min", np.nan))
        log["snr_10p"] = _to_float(monitor.get("snr_10p", np.nan))
        log["snr_med"] = _to_float(monitor.get("snr_med", np.nan))
        log["snrF_min"] = _to_float(monitor.get("snrF_min", np.nan))
        log["snrF_med"] = _to_float(monitor.get("snrF_med", np.nan))
        # Current bridge parameter q (kept in [0,1])
        log["q_bridge"] = _to_float(driver.q, np.nan)
        hit = np.isclose(step, save_times_tracked, atol=driver.dt)
        if np.any(hit):
            idx = np.where(np.isclose(step, save_times_tracked, atol=driver.dt))[0]
            save_times_tracked[idx] = -1
            log["snr"] = monitor.get("snr", np.nan)
            log["snr_F"] = monitor.get("snr_F", np.nan)
            ev = monitor.get("ev", np.array([np.nan]))
            ev_reg = monitor.get("ev_reg", np.array([np.nan]))
            np.save(f"{save_path}/ev_{idx[0]}.npy", ev)
            np.save(f"{save_path}/ev_reg_{idx[0]}.npy", ev_reg)

        return True

    return umbrella_monitor_callback


def get_tdvp_monitor_callback(save_times, save_path):
    save_times_tracked = save_times.copy()

    def tdvp_monitor_callback(step, log, driver):
        # Populate monitoring metrics from driver's self._monitor (make_monitor_dict)
        try:
            dt = driver.integrator._state.dt
            log["dt"] = dt
        except AttributeError:
            log["dt"] = np.nan
        try:
            monitor = driver._monitor
        except AttributeError:
            raise ValueError("No monitor found in driver, callback can't be used.")

        # Scalars: convert possible JAX arrays to Python floats
        def _to_float(x, default=np.nan):
            try:
                return float(np.array(x))
            except Exception:
                return default

        log["r_squared"] = _to_float(monitor.get("rmd", np.nan))
        # ESS as fraction in [0,1] for plotting, plus absolute ESS

        log["snr_min"] = _to_float(monitor.get("snr_min", np.nan))
        log["snr_10p"] = _to_float(monitor.get("snr_10p", np.nan))
        log["snr_med"] = _to_float(monitor.get("snr_med", np.nan))
        log["snrF_min"] = _to_float(monitor.get("snrF_min", np.nan))
        log["snrF_med"] = _to_float(monitor.get("snrF_med", np.nan))
        hit = np.isclose(step, save_times_tracked, atol=driver.dt)
        # ev = monitor.get("ev", np.array([np.nan]))
        # ev_reg = monitor.get("ev_reg", np.array([np.nan]))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(ev,ev_reg, label='reg', color='red')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend(loc='lower right')
        # plt.ylim([1e-14, 1e4])
        # plt.savefig("/tmp/fig.png")
        if np.any(hit):
            idx = np.where(np.isclose(step, save_times_tracked, atol=driver.dt))[0]
            save_times_tracked[idx] = -1
            log["snr"] = monitor.get("snr", np.nan)
            log["snr_F"] = monitor.get("snr_F", np.nan)
            ev = monitor.get("ev", np.array([np.nan]))
            ev_reg = monitor.get("ev_reg", np.array([np.nan]))
            np.save(f"{save_path}/ev_{idx[0]}.npy", ev)
            np.save(f"{save_path}/ev_reg_{idx[0]}.npy", ev_reg)
            # print(f"Saving monitor {step:1.5f}")

        return True

    return tdvp_monitor_callback


def get_adjust_dt_force_callback(r_squared_tol, dt_limits):
    def adjust_dt_force_callback(step, log, driver):
        try:
            dt = driver.integrator._state.dt
            # print(f"r = {driver._rmd} dt = {dt}")
            log["dt"] = dt
            # if dt >= dt_limits[0] and driver._rmd > r_squared_tol * 3:
            #     new_dt = dt * 0.8
            # elif dt <= dt_limits[1] and driver._rmd < r_squared_tol / 3:
            #     new_dt = dt * 1.2
            # else:
            #     new_dt = dt
            # driver.ode_solver = driver.ode_solver.replace(dt=new_dt,
            #                                               integrator_params=IntegratorParameters(
            #                                                   dt=new_dt, ))
            return True
        except AttributeError:
            log["dt"] = np.nan
            return True

    return adjust_dt_force_callback


def get_complex_structure(state, rtol: float = 1e-10):
    # omega_fn = partial(QGTJacobianDense, mode='imag')
    # g_fn = partial(QGTJacobianDense, mode='complex')
    # omega = omega_fn(state).to_dense()
    # g = g_fn(state).to_dense()
    # J = -pinv(g, omega)
    # s, U = jnp.linalg.eigh(g)
    # S_inv = jnp.where(jnp.abs(s / s[-1]) > rtol, jnp.reciprocal(s), 0.0)
    # J = U @ jnp.diag(S_inv) @ (U.conj().T @ omega)
    g = nk.optimizer.qgt.QGTJacobianDense(state, mode="complex").to_dense()
    omega = nk.optimizer.qgt.QGTJacobianDense(state, mode="imag").to_dense()
    J = -pinv(g, omega, rtol=rtol)
    return J
