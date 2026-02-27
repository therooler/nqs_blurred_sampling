from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from pfaffian_jax import slog_pfaffian

__all__ = [
    "FGOState",
    "green_function_from_two",
    "logeta_g_expH_from_H",
    "log_eta_propagation",
    "expH_times_fgo_state",
    "fgo_state_times_fgo_state",
]


Array = jax.Array


@jax.jit
def green_function_from_two(L: Array, R: Array) -> Array:
    Lh = jnp.swapaxes(L.conj(), -1, -2)    
    M = Lh @ R                              
    X = jnp.linalg.solve(M, Lh)             
    gbar = R @ X                            
    N = gbar.shape[-1]
    return gbar.T - gbar

@jax.jit
def uvT_from_two(L: Array, R: Array) -> Array:
    Lh = jnp.swapaxes(L.conj(), -1, -2)    
    M = Lh @ R                              
    X = jnp.linalg.solve(M, Lh)       
    return -R, X

# @jax.jit
# def green_function_from_two(L: Array, R: Array) -> Array:
#     Lh = jnp.swapaxes(L.conj(), -1, -2)    
#     M = Lh @ R                              
#     X = jnp.linalg.solve(M, Lh)             
#     gbar = R @ X                            
#     N = gbar.shape[-1]
#     I = jnp.eye(N, dtype=gbar.dtype)
#     return I - 2 * gbar

@jax.jit
def logeta_g_expH_from_H(H):
    H_hermitian = 1.0j * (H - H.T) / 2
    e, v = jnp.linalg.eigh(H_hermitian)
    green_function = v @ jnp.diag(1.0j * jnp.tan(e / 2.0)) @ v.conj().T
    e_pos = e[: e.shape[-1] // 2]
    val = jnp.cos(e_pos / 2.0)
    logeta = jnp.sum(jnp.log(val.astype(jnp.complex128)))
    return logeta, green_function, v @ jnp.diag(jnp.exp(-1.0j * e)) @ v.conj().T


@jax.jit
def log_eta_propagation(G1, G2, logeta1, logeta2):
    logeta1 = jnp.asarray(logeta1)
    logeta2 = jnp.asarray(logeta2)

    A = (G1 - G1.T) * 0.5
    D = (G2 - G2.T) * 0.5

    ndim = G1.shape[0]
    I = jnp.eye(ndim, dtype=jnp.complex128)

    pfmat = jnp.block(
        [
            [A, -I],
            [I, D],
        ]
    ).astype(jnp.complex128)

    # eta1 * eta2 * pf(pfmat) * (-1) ** (ndim // 2)
    sign_pref = (-1) ** (ndim // 2)
    sign_pf, logabs_pf = slog_pfaffian(pfmat)
    combined_sign = jnp.asarray(sign_pref, dtype=jnp.complex128) * sign_pf
    log_combined_sign = jnp.log(combined_sign)
    return logeta1 + logeta2 + log_combined_sign + logabs_pf

# res = exp(1/4 H\gamma \gamma) \ket{right} \bra{left}
# @jax.jit
# def expH_times_fgo_state(H, fgo_tuple):
#     logeta_state, ket, bra = fgo_tuple
#     G_state = green_function_from_two(bra, ket)
#     logeta_expH, G_expH, expH = logeta_g_expH_from_H(H)
#     logeta_new = log_eta_propagation(G_expH, G_state, logeta_expH, logeta_state)
#     ket_new = expH @ ket
#     ket_new, _ = jnp.linalg.qr(ket_new)
#     return logeta_new, ket_new, bra


# @jax.jit
def expH_times_fgo_state(H, fgo_tuple):
    logeta_state, ket, bra = fgo_tuple
    u, vT = uvT_from_two(bra, ket)
    logeta_expH, G_expH, expH = logeta_g_expH_from_H(H)
    B = -jnp.eye(vT.shape[0]) + u.T @ G_expH @ vT.T
    signpf, logpf = slog_pfaffian(jnp.block([[u.T @ G_expH @ u, B], 
                             [-B.T, vT @ G_expH @ vT.T]]))
    logeta_new = logeta_expH + logeta_state + logpf + jnp.log(signpf.astype(jnp.complex128))
    ket_new = expH @ ket
    ket_new, _ = jnp.linalg.qr(ket_new)
    return logeta_new, ket_new, bra


# res = exp(1/4 H\gamma \gamma) \ket{right} \bra{left}
# \ket_2\bra_2\ket_1\bra 1
@jax.jit
def fgo_state_times_fgo_state(fgo1_tuple, fgo2_tuple):
    logeta_1, ket_1, bra_1 = fgo1_tuple
    logeta_2, ket_2, bra_2 = fgo2_tuple
    G_1 = green_function_from_two(bra_1, ket_1)
    G_2 = green_function_from_two(bra_2, ket_2)
    logeta_new = log_eta_propagation(G_1, G_2, logeta_1, logeta_2)
    return logeta_new, ket_2, bra_1


@dataclass(frozen=True)
class FGOState:
    """State for fermionic Gaussian operator: (logeta, ket, bra)."""
    logeta: Array
    ket: Array  # (..., N, n) or (N, n)
    bra: Array  # (..., N, n) or (N, n)

    def _tuple(self) -> Tuple[Array, Array, Array]:
        return self.logeta, self.ket, self.bra
    
    def green(self) -> Array:
        """Green's function for this state."""
        return green_function_from_two(self.bra, self.ket)
    
    def apply_quadratic(self, H: Array) -> "FGOState":
        """Return exp(1/4 H γγ) * self (acting on ket)."""
        return FGOState(*expH_times_fgo_state(H, self._tuple()))
    
    def __matmul__(self, other: "FGOState") -> "FGOState":
        """Operator product: self @ other."""
        return FGOState(*fgo_state_times_fgo_state(self._tuple(), other._tuple()))