from typing import Any, Callable, Dict, Optional, Sequence, Union, Tuple
from functools import partial, reduce
from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Any, Tuple
import flax.linen as nn


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
        A = A + jnp.outer(v, w) - jnp.outer(w, v)

        # Ensure scalar operations preserve type/shape consistency
        log_term = jnp.log(jnp.abs(1 - tau))
        log_alpha_term = jnp.where(
            i % 2 == 0, jnp.log(jnp.abs(-alpha)), jnp.array(0.0, dtype=logpf_val.dtype)
        )
        logpf_val = logpf_val + log_term + log_alpha_term

        sign_term = (1 - tau) / jnp.abs(1 - tau)
        alpha_sign_term = jnp.where(
            i % 2 == 0, (-alpha) / jnp.abs(-alpha), jnp.array(1.0, dtype=sign_val.dtype)
        )
        sign_val = sign_val * sign_term * alpha_sign_term
        return (A, sign_val, logpf_val)

    # Initialize with explicit types
    init_sign = jnp.array(1.0, dtype=A.dtype)
    init_logpf = jnp.array(0.0, dtype=jnp.float64)

    # Apply pvary to scalars to mark them as varying along the sharding axis
    init_sign = jax.lax.pvary(init_sign, ("S",))
    init_logpf = jax.lax.pvary(init_logpf, ("S",))

    init_val = (A, init_sign, init_logpf)
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


def majorana_plus_state(n, dtype=jnp.complex128):
    """
    construct the |+> state
    X_i = 1, i = 0, ..., n-1
    """
    N = 2 * n
    norm = jnp.array(1 / jnp.sqrt(2.0), dtype=dtype)
    k = jnp.arange(n)
    e_even = jax.nn.one_hot(2 * k, N, dtype=dtype).T
    e_odd = jax.nn.one_hot(2 * k + 1, N, dtype=dtype).T
    return (-1j * norm) * e_even + (1.0 * norm) * e_odd


def majorana_minus_state(n, dtype=jnp.complex128):
    """
    construct the |-> state
    X_i = 1, i = 0, ..., n-2;  X_{n-1} = -1
    """
    N = 2 * n
    norm = jnp.array(1 / jnp.sqrt(2.0), dtype=dtype)
    k = jnp.arange(n)
    e_even = jax.nn.one_hot(2 * k, N, dtype=dtype).T
    e_odd = jax.nn.one_hot(2 * k + 1, N, dtype=dtype).T
    # deal with the last X operator
    phase_even = jnp.where(k == (n - 1), 1j, -1j).astype(dtype)
    return (phase_even * norm) * e_even + (1.0 * norm) * e_odd


@partial(jax.jit, static_argnums=(1,))
def majorana_zz_state(zz, PX=1):
    """
    generate {ZZ, PX} stablizer state in Majorana representation.
    zz (±1) is the eigenvalue of Z_i Z_{i+1}
    PX is the eigenvalue of parity \prod X_i
    """
    zz = jnp.asarray(zz)
    n = zz.shape[-1]
    N = 2 * n

    sgn = jnp.sign(zz)
    sgn = sgn.at[..., -1].set(-PX * sgn[..., -1])  # parity convention

    norm = 1.0 / jnp.sqrt(2.0)
    top = (-1j * sgn) * norm
    bot = (1.0 + 0j) * norm

    a_idx = 2 * jnp.arange(n) + 1  # (n,)
    b_idx = (2 * jnp.arange(n) + 2) % N
    col_idx = jnp.arange(n)

    batch_shape = zz.shape[:-1]
    v = jnp.zeros(batch_shape + (N, n), dtype=jnp.complex128)
    v = v.at[..., a_idx, col_idx].set(top)
    v = v.at[..., b_idx, col_idx].set(bot)
    return v


@partial(jax.jit, static_argnums=(1,))
def majorana_state_from_spins(s, PX=1):
    """
    generate the Z basis product state with parity PX
    |s>+PX|\tilde s>, \tilde s is all fliped state of s
    """
    zz_val = s * jnp.roll(s, -1, axis=-1)
    return majorana_zz_state(zz_val, PX)


@jax.jit
def calculate_gaussian_log_trace_and_green(H):
    """
    H: skew-symmetric real matrix
    for the gaussian operator O = exp(1/4 H_ij \gamma_i\gamma_j)
    output: log trace \Tr[O] and the green function G_ij = \Tr[O \gamma_i \gamma_j] / \Tr[O]
    """
    N = H.shape[0]
    H_hermitian = 0.5j * (H - H.T)
    e, v = jnp.linalg.eigh(H_hermitian)
    D = 1.0j * jnp.tan(e / 2.0)
    green_function = (v * D) @ v.conj().T
    green_function = 0.5 * (green_function - green_function.T)
    log_trace = jnp.sum(jnp.log(jnp.cos(0.5 * e[: N // 2])))
    return log_trace.astype(jnp.complex128), green_function


@jax.jit
def uvt_from_braket(L, R):
    """
    given <L| and |R>, return the u, vT
    that G = u @ vT - uT @ v is the green function from |R><L|
    G = I - 2 R @ inv(L^H @ R) @ L^H
      = - R @ inv(L^H @ R) @ L^H - (R @ inv(L^H @ R) @ L^H)^T
    """
    Lh = jnp.swapaxes(L.conj(), -1, -2)
    M = Lh @ R
    X = jnp.linalg.solve(M, Lh)
    return -R, X


@jax.jit
def expectation_gaussian_braket(bra, Gh, ket):
    """
    calculate the <bra|O|ket>/<bra|ket>/trace(O)=
                  pf[[Gh -I],[I, u @ vT - uT @ v]] * (-1)^(N/2)
    using the equality
        pf(A + uCuT) = pf(A) * pf(C^-1 + uT A^-1 u)
        pf[[Gh -I],[I, u @ vT - uT @ v]] =  (pf[[Gh -I],[I, 0]] * (-1)^(N/2))->1
        * pf[[u.T G u, -I + u.T G v],[I + v.T G u, v.T G v]]
    """
    u, vT = uvt_from_braket(bra, ket)
    I = jnp.eye(u.shape[-1], dtype=u.dtype)
    A = u.T @ Gh @ u
    Y = Gh @ vT.T
    D = vT @ Y
    B = -I + u.T @ Y

    # pf[[A   , B],   = pf(A) pf(D + B.T @ A^-1 @ B)
    #    [-B.T, D]]
    X = jnp.linalg.solve(A, B)
    S = D + B.T @ X
    signA, logA = slog_pfaffian(A)
    signS, logS = slog_pfaffian(S)
    signpf = signA * signS
    logpf = logA + logS

    return logpf + jnp.log(signpf.astype(jnp.complex128))


@partial(jax.jit, static_argnums=(1,))
def skew_from_vec(v, N):
    # v: (k,)  where k = N*(N-1)//2
    U = jnp.zeros((N, N), dtype=v.dtype)
    iu = jnp.triu_indices(N, k=1)
    U = U.at[iu].set(v)
    return U - U.T


def skew_banded_pbc_from_vec(v, N: int, k: int):
    """
    N x N skew-symmetric with PBC, nonzero couplings only for ring-distance < k.
    We parameterize directed edges i -> (i+d)%N for d=1..k-1, then antisymmetrize.

    v length must be N*(k-1).

    Assumption: k <= N  (typically k <= N//2+1 to avoid "double-counting" distances).
    """
    if k < 2 or k > N:
        raise ValueError(f"k must satisfy 2 <= k <= N. Got k={k}, N={N}")

    expected = N * (k - 1)
    if v.shape[0] != expected:
        raise ValueError(
            f"v has length {v.shape[0]}, expected {expected} for N={N}, k={k}"
        )

    M = jnp.zeros((N, N), dtype=v.dtype)

    # Build indices for all directed edges i -> (i+d)%N
    ii = jnp.repeat(jnp.arange(N), k - 1)  # shape (N*(k-1),)
    ds = jnp.tile(jnp.arange(1, k), N)  # 1..k-1 repeated for each i
    jj = (ii + ds) % N

    M = M.at[ii, jj].set(v)
    M = M - M.T  # enforce skew-symmetry
    return M


class GaussianState(nn.Module):
    param_dtype: Any = jnp.float64
    parametrization: str = "dense"  # "dense" or "banded"
    band_k: int = 10  # only used if banded

    @nn.compact
    def __call__(self, x) -> Any:
        n = x.shape[-1]
        N = 2 * n
        # parameterization
        if self.parametrization == "dense":
            # compress the skew-symmetric matrix into a vector for parameterization
            num = (2 * n - 1) * n
            H1_raw = self.param(
                "H1_raw",
                nn.initializers.normal(0.001),
                (num,),
                self.param_dtype,
            )
            H2_raw = self.param(
                "H2_raw",
                nn.initializers.normal(0.001),
                (num,),
                self.param_dtype,
            )
            H1 = skew_from_vec(H1_raw, N)
            H2 = skew_from_vec(H2_raw, N)
        else:
            k = self.band_k
            num = N * (k - 1)
            H1_raw = self.param(
                "H1_raw",
                nn.initializers.normal(0.001),
                (num,),
                self.param_dtype,
            )
            H2_raw = self.param(
                "H2_raw",
                nn.initializers.normal(0.001),
                (num,),
                self.param_dtype,
            )
            H1 = skew_banded_pbc_from_vec(H1_raw, N, k)
            H2 = skew_banded_pbc_from_vec(H2_raw, N, k)

        # generate the plus state and FGO (independent for x)
        plus_state = majorana_plus_state(n)
        logtrace1, Gh1 = calculate_gaussian_log_trace_and_green(H1)
        logtrace2, Gh2 = calculate_gaussian_log_trace_and_green(H2)

        # compute the log amplitude for each configuration in x
        def batch_fun(_x):
            s_plus_state = majorana_state_from_spins(_x, PX=1)
            logoverlap1 = (
                expectation_gaussian_braket(s_plus_state, Gh1, plus_state) + logtrace1
            )
            logoverlap2 = (
                expectation_gaussian_braket(s_plus_state, Gh2, plus_state) + logtrace2
            )

            s_prod = jnp.prod(_x, axis=-1).astype(jnp.complex128)
            logamp = logsumexp(
                jnp.stack([logoverlap1, jnp.log(s_prod) + logoverlap2], axis=0), axis=0
            )
            return logamp

        return jax.vmap(batch_fun)(x)


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
    v = majorana_zz_state(zz_val)
    return green_function_from_two(v, plus_state)


def log_eta_propagation(G1, G2, logeta1, logeta2):
    logeta1 = jnp.asarray(logeta1)
    logeta2 = jnp.asarray(logeta2)

    A = (G1 - G1.T) * 0.5
    D = (G2 - G2.T) * 0.5

    ndim = G1.shape[0]
    I = jnp.eye(ndim, dtype=jnp.complex128)

    pfmat = jnp.block([[A, -I], [I, D]])

    # eta1 * eta2 * pf(pfmat) * (-1) ** (ndim // 2)
    sign_pref = (-1) ** (ndim // 2)
    sign_pf, logabs_pf = slog_pfaffian(pfmat)
    combined_sign = jnp.asarray(sign_pref, dtype=jnp.complex128) * sign_pf
    log_combined_sign = jnp.log(combined_sign)
    return logeta1 + logeta2 + log_combined_sign + logabs_pf


def gen_v_from_zz_px(zz, PX=1):
    zz = jnp.asarray(zz)
    n = zz.shape[-1]
    N = 2 * n
    sgn = jnp.sign(zz)
    sgn = sgn.at[..., -1].set(-PX * sgn[..., -1])  # the parity change a sign here

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


def green_function_from_s_state(s, state, PX=1):
    # s_i * s_{i+1}
    zz_val = s * jnp.roll(s, -1, axis=-1)
    v = gen_v_from_zz_px(zz_val, PX)
    return green_function_from_two(v, state)


def gen_green_init_s_proj(s, PX=1):
    zz_val = s * jnp.roll(s, -1, axis=-1)
    v = gen_v_from_zz_px(zz_val, PX)
    return v, green_function_from_two(v, v)


def logeta_g_expH_from_H(H):
    H_hermitian = 1.0j * (H - H.T) / 2
    e, v = jnp.linalg.eigh(H_hermitian)
    green_function = v @ jnp.diag(1.0j * jnp.tan(e / 2.0)) @ v.conj().T
    green_function = jnp.real(green_function)
    e_pos = e[: e.shape[-1] // 2]
    val = jnp.cos(e_pos / 2.0)
    logeta = jnp.sum(jnp.log(val.astype(jnp.complex128)))
    return logeta, green_function, v @ jnp.diag(jnp.exp(-1.0j * e)) @ v.conj().T


class EpsilonState(nn.Module):
    param_dtype: Any = jnp.float64
    s0: Tuple = ()

    @nn.compact
    def __call__(self, x) -> Any:
        n = x.shape[-1]
        H_plus = self.param(
            "H1",
            nn.initializers.normal(0.0001),
            (2 * n, 2 * n),
            self.param_dtype,
        )
        H_minus = self.param(
            "H2",
            nn.initializers.normal(0.0001),
            (2 * n, 2 * n),
            self.param_dtype,
        )
        plus_state = majorana_plus_state(n)
        minus_state = majorana_minus_state(n)
        s0 = jnp.array(self.s0)
        v_plus, Gz_plus = gen_green_init_s_proj(s0, PX=1)
        v_minus, Gz_minus = gen_green_init_s_proj(s0, PX=-1)
        logeta_expH_plus, G_expH_plus, expH_plus = logeta_g_expH_from_H(H_plus)
        logeta_expH_minus, G_expH_minus, expH_minus = logeta_g_expH_from_H(H_minus)
        logeta_Ghz_plus = log_eta_propagation(
                G_expH_plus, Gz_plus, logeta_expH_plus, 0.0
            )
        expH_v_plus = expH_plus @ v_plus
        Ghz_plus = green_function_from_two(v_plus, expH_v_plus)

        logeta_Ghz_minus = log_eta_propagation(
                G_expH_minus, Gz_minus, logeta_expH_minus, 0.0
            )
        expH_v_minus = expH_minus @ v_minus
        Ghz_minus = green_function_from_two(v_minus, expH_v_minus)

        def batch_fun(_x):
            Gs_plus = green_function_from_s_state(_x, plus_state, PX=1)
            Gs_minus = green_function_from_s_state(_x, minus_state, PX=-1)

            logoverlap_plus = log_eta_propagation(
                Ghz_plus, Gs_plus, logeta_Ghz_plus, 0.0
            )

            logoverlap_minus = log_eta_propagation(
                Ghz_minus, Gs_minus, logeta_Ghz_minus, 0.0
            )

            logamp = logsumexp(
                jnp.stack(
                    [
                        logoverlap_minus,
                        jnp.log(
                            _x[-1].astype(jnp.complex128)
                            * s0[-1].astype(jnp.complex128)
                        )
                        + logoverlap_plus,
                    ],
                    axis=0,
                ),
                axis=0,
            )
            return logamp

        return jax.vmap(batch_fun)(x)


if __name__ == "__main__":
    import netket as nk
    from metropolis import LocalDoubleFlipRule

    n_samples = 2**10
    seed = 100
    N = 4
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)

    sampler = nk.sampler.MetropolisSampler(
        hilbert,
        LocalDoubleFlipRule(),
        n_chains=n_samples,
    )
    s0 = (1, -1, -1, -1)

    for i in range(N):
        string = ["I"] * N
        string[i] = "Z"
        if i == 0:
            sigma_zs = nk.operator.PauliStringsJax(hilbert, "".join(string), 1.0 / N)
        else:
            sigma_zs += nk.operator.PauliStringsJax(hilbert, "".join(string), 1.0 / N)

    model = EpsilonState(s0=s0)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=n_samples,
        seed=seed,
        sampler_seed=seed,
        n_discard_per_chain=0,
    )
    for i in range(10):
        # print(i)
        vstate.sample()
    print(vstate.expect(sigma_zs))

    print(vstate.to_array())
