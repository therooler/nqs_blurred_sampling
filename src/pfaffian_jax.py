from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


__all__ = ["slog_pfaffian"]

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