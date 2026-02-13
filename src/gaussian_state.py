from typing import Any, Callable, Dict, Optional, Sequence, Union, Tuple
from functools import partial, reduce
from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


# For the pfaffian
@jax.jit
def _householder_n(x: jax.Array, n: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
    arange = jnp.arange(x.size)
    xn = x[n]
    x = jnp.where(arange <= n, jnp.zeros_like(x), x)
    sigma = jnp.vdot(x, x)
    norm_x = jnp.sqrt(xn.conj() * xn + sigma)

    phase = jnp.where(xn == 0.0, 1.0, xn / jnp.abs(xn))
    vn = xn + phase * norm_x
    alpha = -phase * norm_x

    v = jnp.where(arange == n, vn, x)
    v /= jnp.linalg.norm(v)

    cond = sigma == 0.0
    v = jnp.where(cond, jnp.zeros_like(x), v)
    tau = jnp.where(cond, 0, 2)
    alpha = jnp.where(cond, xn, alpha)

    return v, tau, alpha


@partial(jax.custom_jvp)
def slog_pfaffian(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    n = A.shape[0]
    if n % 2 == 1:
        return jnp.array(0, dtype=A.dtype)

    def body_fun(i, val):
        A, sign_val, logpf_val = val
        v, tau, alpha = _householder_n(A[:, i], i + 1)
        w = tau * A @ v.conj()
        A += jnp.outer(v, w) - jnp.outer(w, v)
        logpf_val += jnp.log(jnp.abs(1 - tau))
        logpf_val += jnp.where(i % 2 == 0, jnp.log(jnp.abs(-alpha)), 0.0)
        sign_val *= (
            (1 - tau)
            / jnp.abs(1 - tau)
            * jnp.where(i % 2 == 0, (-alpha) / jnp.abs(-alpha), 1.0)
        )
        return A, sign_val, logpf_val

    init_val = (A, jnp.array(1.0, dtype=A.dtype), jnp.array(0.0, dtype=jnp.float64))
    A, sign_val, logpf_val = jax.lax.fori_loop(0, A.shape[0] - 2, body_fun, init_val)
    logpf_val += jnp.log(jnp.abs(A[n - 2, n - 1]))
    sign_val *= (A[n - 2, n - 1]) / jnp.abs(A[n - 2, n - 1])

    return sign_val, logpf_val


@slog_pfaffian.defjvp
def slog_pfaffian_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents
    sign_pfaffian, log_pfaffian = slog_pfaffian(A)
    det_dot = jnp.einsum("...ij,...ji->...", jnp.linalg.inv(A), A_dot)
    pfaffian_dot = det_dot / 2
    return (sign_pfaffian, log_pfaffian), (
        1.0j * jnp.imag(pfaffian_dot) * sign_pfaffian,
        jnp.real(pfaffian_dot),
    )


slog_pfaffian = jax.jit(slog_pfaffian)


def gen_plus_jax(n):
    n = int(n)
    N = 2 * n
    H = jnp.zeros((N, N), dtype=jnp.complex128)
    idx = 2 * jnp.arange(n)
    H = H.at[idx, idx + 1].set(1.0j)
    Hsym = H - H.T
    e, v = jnp.linalg.eigh(Hsym)
    return v[:, :n]


def gen_v_from_zz(zz):
    zz = jnp.asarray(zz)
    n = zz.shape[-1]
    N = 2 * n
    sgn = jnp.sign(zz)
    sgn = sgn.at[..., -1].set(-sgn[..., -1])  # the parity change a sign here

    norm = 1.0 / jnp.sqrt(2.0)
    top = (-1j * sgn) * norm
    bot = (1.0 + 0j) * norm

    a_idx = 2 * jnp.arange(n) + 1  # (n,)
    b_idx = (2 * jnp.arange(n) + 2) % N
    col_idx = jnp.arange(n)

    # make v with batch dims
    batch_shape = zz.shape[:-1]
    v = jnp.zeros(batch_shape + (N, n), dtype=jnp.complex128)

    # scatter into the last two axes
    v = v.at[..., a_idx, col_idx].set(top)
    v = v.at[..., b_idx, col_idx].set(bot)  # bot broadcasts to (..., n)
    return v


def green_function_from_two(L, R):
    Lh = jnp.swapaxes(L.conj(), -1, -2)  # (..., n, N)
    M = Lh @ R  # (..., n, n)
    X = jnp.linalg.solve(M, Lh)  # batch-solve supported
    gbar = R @ X  # (..., N, N)
    N = gbar.shape[-1]
    I = jnp.eye(N, dtype=gbar.dtype)
    return I - 2 * gbar


def green_function_from_s_plus(s, plus_state):
    # s_i * s_{i+1}
    zz_val = s * jnp.roll(s, -1, axis=-1)
    v = gen_v_from_zz(zz_val)
    return green_function_from_two(v, plus_state)


def logeta_and_g_from_H(H):
    H_hermitian = 1.0j * (H - H.T) / 2
    e, v = jnp.linalg.eigh(H_hermitian)
    green_function = v @ jnp.diag(1.0j * jnp.tan(e / 2.0)) @ v.conj().T
    e_pos = e[: e.shape[-1] // 2]
    val = jnp.cos(e_pos / 2.0)
    logeta = jnp.sum(jnp.log(val.astype(jnp.complex128)))
    return logeta, green_function


def log_eta_propagation(G1, G2, logeta1, logeta2):
    logeta1 = jnp.asarray(logeta1)
    logeta2 = jnp.asarray(logeta2)

    A = (G1 - G1.T) * 0.5
    D = (G2 - G2.T) * 0.5

    ndim = G1.shape[0]
    I = jnp.eye(ndim, dtype=jnp.complex128)

    # pfmat = jnp.zeros((2 * ndim, 2 * ndim), dtype=jnp.complex128)
    # pfmat = jax.lax.dynamic_update_slice(pfmat, A, (0, 0))
    # pfmat = jax.lax.dynamic_update_slice(pfmat, D, (ndim, ndim))
    # pfmat = jax.lax.dynamic_update_slice(pfmat, -I, (0, ndim))
    # pfmat = jax.lax.dynamic_update_slice(pfmat, I, (ndim, 0))
    pfmat = jnp.block([[A, -I], [I, D]])

    # eta1 * eta2 * pf(pfmat) * (-1) ** (ndim // 2)
    sign_pref = (-1) ** (ndim // 2)
    sign_pf, logabs_pf = slog_pfaffian(pfmat)
    combined_sign = jnp.asarray(sign_pref, dtype=jnp.complex128) * sign_pf
    log_combined_sign = jnp.log(combined_sign)
    return logeta1 + logeta2 + log_combined_sign + logabs_pf


def amplitude(H1, H2, s, plus_state):

    amp = jnp.exp(logamp)
    return amp


import flax.linen as nn


class GaussianState(nn.Module):

    @nn.compact
    def __call__(self, x) -> Any:
        n = x.shape[-1]
        H1 = self.param(
            "H1",
            nn.initializers.normal(0.0001),
            (2 * n, 2 * n),
            float,
        )
        H2 = self.param(
            "H2",
            nn.initializers.normal(0.0001),
            (2 * n, 2 * n),
            float,
        )
        plus_state = gen_plus_jax(n)

        def batch_fun(_x):
            Gs = green_function_from_s_plus(_x, plus_state)
            logeta1, Gh1 = logeta_and_g_from_H(H1)
            logeta2, Gh2 = logeta_and_g_from_H(H2)
            logoverlap1 = log_eta_propagation(Gh1, Gs, logeta1, 0.0)
            logoverlap2 = log_eta_propagation(Gh2, Gs, logeta2, 0.0)
            s_prod = jnp.prod(_x, axis=-1).astype(jnp.complex128)
            logamp = logsumexp(
                jnp.stack([logoverlap1, jnp.log(s_prod) + logoverlap2], axis=0), axis=0
            )
            return logamp

        return jax.vmap(batch_fun)(x)


if __name__ == "__main__":
    import netket as nk
    from metropolis import LocalDoubleFlipRule

    n_samples = 2**10
    seed = 100
    N = 16
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)

    sampler = nk.sampler.MetropolisSampler(
        hilbert, LocalDoubleFlipRule(), n_chains=n_samples, 
    )

    model = GaussianState()

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=n_samples,
        seed=seed,
        sampler_seed=seed,
        n_discard_per_chain=0
    )
    for i in range(100):
        print(i)
        vstate.sample()

    # print(vstate.to_array())
