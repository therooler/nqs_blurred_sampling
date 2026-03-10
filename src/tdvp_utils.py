import jax
import jax.numpy as jnp
import netket.jax as nkjax
import jax.scipy as jsp
from functools import partial
from netket.utils.types import Sequence, PyTree, Array
from netket.operator import AbstractOperator


@jax.jit
def make_monitor_dict(rmd, ess, snr, snr_F, ev, ev_reg):
    """
    Build a small diagnostics dict for logging / monitoring.

    Args
    ----
    rmd                : (nsamples,)
    importance_weights : (n_samples,)
    snr                : (n_params,)    # SNR in eigenbasis of S (from _impl)
    snr_F              : (n_params,)    # SNR in parameter basis (O_var["snr_F"])
    ev                 : (n_params,)    # Eigenvalues of QGT
    eta_p              : (n_params,)    # Regularized eigenvalues

    Returns
    -------
    metrics : dict of scalars (JAX arrays)
    """

    # Clean SNRs: replace inf/NaN with 0 for summary stats
    def _clean(x):
        x = jnp.where(jnp.isfinite(x), x, 0.0)
        return x

    snr_clean = _clean(snr)
    snrF_clean = _clean(snr_F)
    ev_clean = _clean(ev)
    ev_reg_clean = _clean(ev_reg)
    # Eigenbasis SNR summaries
    snr_min = jnp.min(snr_clean)
    snr_med = jnp.median(snr_clean)

    # 10th percentile of snr (lower tail)
    snr_sorted = jnp.sort(snr_clean)
    n_modes = snr_sorted.shape[0]
    idx_10p = jnp.maximum(0, (n_modes * 10) // 100)
    snr_10p = snr_sorted[idx_10p]

    # Parameter-basis SNR summaries
    snrF_min = jnp.min(snrF_clean)
    snrF_med = jnp.median(snrF_clean)

    # jax.debug.print("ev:\n{}", ev)
    # jax.debug.print("eta_p:\n{}",ev_reg)
    metrics = {
        "rmd": rmd,
        "ess_bridge": ess,
        "snr_min": snr_min,
        "snr_10p": snr_10p,
        "snr_med": snr_med,
        "snrF_min": snrF_min,
        "snrF_med": snrF_med,
        "snr": snr,
        "snr_F": snr_F,
        "ev": ev_clean,
        "ev_reg": ev_reg_clean,
    }
    return metrics


@jax.jit
def ess_from_weights(w):
    s1_sq = jnp.mean(w, axis=0) ** 2
    s2 = jnp.mean(w**2, axis=0)
    # Return normalized ESS in [0, 1]
    return (s1_sq / (s2 + jnp.finfo(w.dtype).eps)).squeeze()


@jax.jit
def ess_from_weights_var(w):
    # sum over the sample axis

    s1_sq = jnp.mean(w, axis=0) ** 2
    s2 = jnp.mean(w**2, axis=0)
    # jax.debug.print("w {} s1_sq {} s2 {}",w, s1_sq, s2 )
    return ((s1_sq / (s2 - s1_sq + jnp.finfo(w.dtype).eps))).squeeze()


@partial(jax.jit, static_argnames=("apply_fn", "chunk_size", "diagonal_mels"))
def blurred_sample(
    x: Array, key, params, q: float, apply_fn, op: AbstractOperator, chunk_size, diagonal_mels:bool=True,
):
    """One-step "bridge" proposal with importance weights.

    For each input configuration ``x[i]``, this kernel constructs a simple mixture proposal:

    - with probability ``q`` it keeps the configuration unchanged;
    - with probability ``1-q`` it proposes a *single* random connected configuration sampled
      uniformly from ``op.get_conn_padded(x[i])``.

    The returned scalar weight ``w_bridge`` corrects expectations from this mixture proposal to
    the target density :math:`p(\sigma) \propto |\psi(\sigma)|^2` (computed from
    ``apply_fn({'params': params}, ·).real``).

    Parameters
    ----------
    x:
        Array of shape ``(batch, n_dof)`` (or generally ``(batch, ...)``) containing the input
        configurations.
    key:
        JAX PRNGKey.
    params:
        Parameters passed to ``apply_fn``.
    q:
        Mixture parameter in ``[0, 1]`` controlling the probability of *staying* at the current
        configuration.
    apply_fn:
        Callable such that ``apply_fn({'params': params}, x)`` returns ``log(psi(x))`` (possibly
        complex). Only the real part is used to form :math:`|\psi|^2`.
    op:
        Operator providing ``get_conn_padded`` returning connected configurations and matrix
        elements.
    chunk_size:
        If not ``None``, evaluates the per-sample function with ``nkjax.apply_chunked``.

    Returns
    -------
    x_p:
        Array with the same shape as ``x`` containing the proposed (or unchanged) configurations.
    w_bridge:
        Array of shape ``(batch,)`` with importance weights
        :math:`w = p_{\mathrm{target}}(x_p) / p_{\mathrm{mix}}(x_p)`, where
        :math:`p_{\mathrm{target}}(\sigma) \propto |\psi(\sigma)|^2` and
        :math:`p_{\mathrm{mix}}(\sigma) = q\,p_{\mathrm{target}}(\sigma) + (1-q)\,\frac{1}{n}\sum_j p_{\mathrm{target}}(\sigma_j)`.
    E_loc:
        Local energy estimate for each proposed configuration ``x_p[i]``.
    """
    batch_size = x.shape[0]
    # rng for u1, u2 per configuration
    c = jax.random.uniform(key, shape=(batch_size, 2))
    if diagonal_mels:
        def get_blurred_sample_and_Eloc(_in):
            _x, rng = _in
            u1, u2 = rng
            _x_shape = _x.shape
            _x = _x.reshape(-1)
            # Connected elements of Hamiltonian
            x_conn, _ = op.get_conn_padded(_x)
            # NOTE: get_conn_padded(_x) can contain diagonal elements, which correspond to "stay" configuration
            # For Ising, the first element will be diagonal, we therefore only have nconn-1 off-diagonal elements
            n_conn = x_conn.shape[-2] - 1
            idx = jnp.floor(u2 * n_conn).astype(jnp.int32)
            # Only choose from off-diagonal elements
            proposed = x_conn[idx + 1]
            # choose a whether to flip or stay
            x_p = jnp.where(u1 > q, _x, proposed)  # equivalent to u1 < 1-q
            x_p_conn, mels = op.get_conn_padded(x_p)
            # log |psi| for flipped and all neighbors
            logpsi_stay = apply_fn({"params": params}, x_p)
            logpsi_all = apply_fn({"params": params}, x_p_conn)
            # target density ∝ |psi|^2
            logp_stay = 2.0 * logpsi_stay.real
            logp_all = 2.0 * logpsi_all.real  # (n,)
            # stable mixture weight: (1-q)*p(stay) + (q/n)*sum_j p(all_flipped_j)
            log_term_main = jnp.log1p(-q) + logp_stay
            log_term_flips = (
                jnp.log(q) - jnp.log(n_conn) + jsp.special.logsumexp(logp_all[1:])
            )
            log_w_bridge = jsp.special.logsumexp(jnp.stack([log_term_main, log_term_flips]))
            w_bridge = jnp.exp(logp_stay - log_w_bridge)  # scalar
            # Calculate local energies
            E_loc = jnp.sum(
                mels * jnp.exp(logpsi_all - jnp.expand_dims(logpsi_stay, -1)), axis=-1
            )
            return x_p.reshape(_x_shape), w_bridge, jnp.atleast_1d(E_loc)
    else:
        def get_blurred_sample_and_Eloc(_in):
            _x, rng = _in
            u1, u2 = rng
            _x_shape = _x.shape
            _x = _x.reshape(-1)
            # Connected elements of Hamiltonian
            x_conn, _ = op.get_conn_padded(_x)
            # no diagonal elements
            n_conn = x_conn.shape[-2]
            idx = jnp.floor(u2 * n_conn).astype(jnp.int32)
            # Only choose from off-diagonal elements
            proposed = x_conn[idx]
            # choose whether to flip or stay
            x_p = jnp.where(u1 > q, _x, proposed)  # equivalent to u1 < 1-q
            x_p_conn, mels = op.get_conn_padded(x_p)
            # log |psi| for flipped and all neighbors
            logpsi_stay = apply_fn({"params": params}, x_p)
            logpsi_all = apply_fn({"params": params}, x_p_conn)
            # target density ∝ |psi|^2
            logp_stay = 2.0 * logpsi_stay.real
            logp_all = 2.0 * logpsi_all.real  # (n,)
            # stable mixture weight: (1-q)*p(stay) + (q/n)*sum_j p(all_flipped_j)
            log_term_main = jnp.log1p(-q) + logp_stay
            log_term_flips = (
                jnp.log(q) - jnp.log(n_conn) + jsp.special.logsumexp(logp_all)
            )
            log_w_bridge = jsp.special.logsumexp(jnp.stack([log_term_main, log_term_flips]))
            w_bridge = jnp.exp(logp_stay - log_w_bridge)  # scalar
            # Calculate local energies
            E_loc = jnp.sum(
                mels * jnp.exp(logpsi_all - jnp.expand_dims(logpsi_stay, -1)), axis=-1
            )
            return x_p.reshape(_x_shape), w_bridge, jnp.atleast_1d(E_loc)
    vmapped_get_blurred_sample_and_weight = jax.vmap(
        get_blurred_sample_and_Eloc, in_axes=0
    )
    if chunk_size is None:
        return vmapped_get_blurred_sample_and_weight((x, c))
    else:
        return nkjax.apply_chunked(
            vmapped_get_blurred_sample_and_weight, in_axes=0, chunk_size=chunk_size, axis_0_is_sharded=False
        )((x, c))

def random_flip_uniform_k(key, x):
    ns = x.shape[-1]
    key_k, key_perm = jax.random.split(key)
    k = jax.random.randint(key_k, shape=(), minval=0, maxval=ns+1)
    perm = jax.random.permutation(key_perm, ns)
    m_order = jnp.arange(ns) < k 
    mask = jnp.zeros(ns, dtype=bool).at[perm].set(m_order)
    return jnp.where(mask, -1, 1)

@partial(jax.jit, static_argnames=("apply_fn", "chunk_size"))
def randomized_blurred_sample(
    x: Array, key, params, q1: float, q2: float, flip_prob: float, apply_fn, op: AbstractOperator, chunk_size
):
    """One-step "bridge" proposal with importance weights.

    For each input configuration ``x[i]``, this kernel constructs a simple mixture proposal:

    - randomly generate a flip mask with bournolli distribution flip_prob;
    - with probability ``q`` it keeps the configuration unchanged;
    - with probability ``1-q`` it flip the spin using the flip mask. 

    The returned scalar weight ``w_bridge`` corrects expectations from this mixture proposal to
    the target density :math:`p(\sigma) \propto |\psi(\sigma)|^2` (computed from
    ``apply_fn({'params': params}, ·).real``).

    Parameters
    ----------
    x:
        Array of shape ``(batch, n_dof)`` (or generally ``(batch, ...)``) containing the input
        configurations.
    key:
        JAX PRNGKey.
    params:
        Parameters passed to ``apply_fn``.
    q:
        Mixture parameter in ``[0, 1]`` controlling the probability of *staying* at the current
        configuration.
    apply_fn:
        Callable such that ``apply_fn({'params': params}, x)`` returns ``log(psi(x))`` (possibly
        complex). Only the real part is used to form :math:`|\psi|^2`.
    op:
        Operator providing ``get_conn_padded`` returning connected configurations and matrix
        elements.
    chunk_size:
        If not ``None``, evaluates the per-sample function with ``nkjax.apply_chunked``.

    Returns
    -------
    x_p:
        Array with the same shape as ``x`` containing the proposed (or unchanged) configurations.
    w_bridge:
        Array of shape ``(batch,)`` with importance weights
        :math:`w = p_{\mathrm{target}}(x_p) / p_{\mathrm{mix}}(x_p)`, where
        :math:`p_{\mathrm{target}}(\sigma) \propto |\psi(\sigma)|^2` and
        :math:`p_{\mathrm{mix}}(\sigma) = q\,p_{\mathrm{target}}(\sigma) + (1-q)\,\frac{1}{n}\sum_j p_{\mathrm{target}}(\sigma_j)`.
    E_loc:
        Local energy estimate for each proposed configuration ``x_p[i]``.
    """
    batch_size = x.shape[0]
    key, subkey1, subkey2 = jax.random.split(key, 3)
    c = jax.random.uniform(subkey1, shape=(batch_size, 3))
    keys = jax.random.split(subkey2, batch_size)

    def get_blurred_sample_and_Eloc(_in):
        _x, rng, key = _in
        u1, u2, u3 = rng
        _x_shape = _x.shape
        _x = _x.reshape(-1)
        flip = random_flip_uniform_k(key, _x)
        # flip = 1 - 2 * jax.random.bernoulli(key, flip_prob, _x.shape)
        flip_proposed = flip * _x

        # Connected elements of Hamiltonian
        x_conn, _ = op.get_conn_padded(_x)
        # NOTE: get_conn_padded(_x) can contain diagonal elements, which correspond to "stay" configuration
        # For Ising, the first element will be diagonal, we therefore only have nconn-1 off-diagonal elements
        n_conn = x_conn.shape[-2] - 1
        idx = jnp.floor(u1 * n_conn).astype(jnp.int32)
        conn_proposed = x_conn[idx + 1]

        proposed = jnp.where(u2 > q1/(q1 + q2), flip_proposed, conn_proposed)
        q = q1 + q2
        x_p = jnp.where(u3 > q, _x, proposed)
        
        # log |psi| for the blurred sample and its flipped configuration
        logpsi_stay = apply_fn({"params": params}, x_p)
        logp_stay = 2.0 * logpsi_stay.real
        log_term_main = jnp.log1p(-q) + logp_stay

        logpsi_flip = apply_fn({"params": params}, flip * x_p)
        logp_flip = 2.0 * logpsi_flip.real 
        log_term_flip = jnp.log(q2) + logp_flip 

        x_p_conn, mels = op.get_conn_padded(x_p)
        logpsi_all = apply_fn({"params": params}, x_p_conn)
        logp_all = 2.0 * logpsi_all.real  # (n,)
        # stable mixture weight: (1-q)*p(stay) + (q/n)*sum_j p(all_flipped_j)
        log_term_conn = (
            jnp.log(q1) - jnp.log(n_conn) + jsp.special.logsumexp(logp_all[1:])
        )

        # stable mixture weight: (1-q)*p(stay) + (q/n)*sum_j p(all_flipped_j)
        log_w_bridge = jsp.special.logsumexp(jnp.stack([log_term_main, log_term_flip, log_term_conn]))
        w_bridge = jnp.exp(logp_stay - log_w_bridge)  # scalar

        # Calculate local energies
        E_loc = jnp.sum(
            mels * jnp.exp(logpsi_all - jnp.expand_dims(logpsi_stay, -1)), axis=-1
        )
        return x_p.reshape(_x_shape), w_bridge, jnp.atleast_1d(E_loc)

    vmapped_get_blurred_sample_and_weight = jax.vmap(
        get_blurred_sample_and_Eloc, in_axes=0
    )
    if chunk_size is None:
        return vmapped_get_blurred_sample_and_weight((x, c, keys))
    else:
        return nkjax.apply_chunked(
            vmapped_get_blurred_sample_and_weight, in_axes=0, chunk_size=64
        )((x, c, keys))