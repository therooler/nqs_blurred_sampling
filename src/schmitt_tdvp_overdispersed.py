# Copyright 2020, 2021  The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import jax.scipy as jsp
from netket import stats
from netket.operator import AbstractOperator, DiscreteJaxOperator
from netket.optimizer.qgt.qgt_jacobian import QGTJacobian_DefaultConstructor
from netket.optimizer.qgt.qgt_jacobian_dense import convert_tree_to_dense_format
from netket.utils.api_utils import partial_from_kwargs
from netket.vqs import VariationalState, VariationalMixedState, MCState
from netket.jax import tree_cast
from netket.utils import timing, mpi, HashablePartial
from netket.utils.types import Sequence, PyTree, Array
import netket.jax as nkjax

from netket.experimental.driver.tdvp_common import TDVPBaseDriver, odefun
from netket.experimental.dynamics._solver import AbstractSolver

from tdvp_utils import make_monitor_dict, overdispersed_sample, ess_from_weights
from jax.sharding import PartitionSpec as P

import platform
import os

system = platform.system()
if system == "Linux" and os.environ.get("ENABLE_JAXMG") == "1":
    try:
        from jaxmg import syevd

        JAXMG_ENABLED = True
    except ModuleNotFoundError:
        JAXMG_ENABLED = False
else:
    JAXMG_ENABLED = False


class TDVPSchmittOverdispersed(TDVPBaseDriver):
    r"""
    Variational time evolution using the time-dependent variational principle with
    overdispersed importance sampling.

    Instead of the bridge/blur kernel used in :class:`TDVPSchmittBlur`, this driver
    samples from the overdispersed distribution

    .. math::

        q(\sigma) \propto |\psi(\sigma)|^\alpha

    and corrects expectations back to the target density
    :math:`p(\sigma) \propto |\psi(\sigma)|^2` via importance weights

    .. math::

        w(\sigma) = |\psi(\sigma)|^{2-\alpha} = \exp\!\big((2-\alpha)\,\mathrm{Re}\,\log\psi(\sigma)\big).

    For :math:`\alpha < 2` the proposal is overdispersed relative to the VMC target,
    which can reduce variance in the gradient estimator at the cost of a lower effective
    sample size.  Setting :math:`\alpha = 2` recovers standard VMC with unit weights.

    The sampling from :math:`|\psi|^\alpha` is handled externally: pass a
    ``sampling_state`` whose model returns :math:`(\alpha/2)\,\log\psi(x)` so that
    NetKet's Metropolis sampler accepts configurations according to :math:`|\psi|^\alpha`.
    If ``sampling_state`` is ``None``, the driver falls back to the variational state's
    own samples (which are from :math:`|\psi|^2`), effectively making :math:`\alpha = 2`.

    The Schmitt regularization of the QGT (eigendecomposition + SVD/SNR cutoffs) is
    identical to :class:`TDVPSchmittBlur`.
    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        integrator: AbstractSolver = None,
        *,
        alpha: float = 1.0,
        t0: float = 0.0,
        propagation_type: str = "real",
        holomorphic: bool | None = None,
        diag_shift: float = 0.0,
        diag_scale: float | None = None,
        error_norm: str | Callable = "qgt",
        rcond: float = 1e-14,
        rcond_smooth: float = 1e-8,
        snr_atol: float | None = None,
        sampling_state: VariationalState = None,
        distributed_eigh: bool = False,
    ):
        r"""
        Initializes the overdispersed TDVP driver.

        Args:
            operator: The generator of the dynamics.
            variational_state: The variational state.
            integrator: ODE integrator configuration.
            alpha: Exponent of the proposal distribution q(σ) ∝ |ψ(σ)|^alpha.
                Must satisfy 0 < alpha ≤ 2.  alpha=2 recovers standard VMC.
            t0: Initial time.
            propagation_type: "real" for real-time SE, "imag" for imaginary-time SE.
            error_norm: Norm for adaptive integrator error estimation.
            holomorphic: Flag indicating a holomorphic wave function.
            diag_shift: Diagonal shift of the QGT.
            diag_scale: Optional rescaling of the diagonal shift.
            rcond: Hard cut-off ratio for small QGT eigenvalues.
            rcond_smooth: Smooth cut-off ratio for QGT eigenvalues (ε_SVD).
            snr_atol: SNR regularization threshold (ε_SNR).  None disables it.
            sampling_state: VariationalState whose model returns (alpha/2)*logψ so
                that its Metropolis sampler draws from |ψ|^alpha.  If None, uses
                the variational state's own samples (equivalent to alpha=2).
            distributed_eigh: Use jaxmg distributed eigendecomposition (Linux only,
                requires ENABLE_JAXMG=1 and jaxmg installed).
        """
        self.propagation_type = propagation_type
        if isinstance(variational_state, VariationalMixedState):
            if propagation_type == "real":
                self._loss_grad_factor = 1.0
            else:
                raise ValueError(
                    "only real-time Lindblad evolution is supported for mixed states"
                )
        else:
            if propagation_type == "real":
                self._loss_grad_factor = -1.0j
            elif propagation_type == "imag":
                self._loss_grad_factor = -1.0
            else:
                raise ValueError("propagation_type must be one of 'real', 'imag'")

        self.rcond = rcond
        self.rcond_smooth = rcond_smooth
        self.snr_atol = snr_atol

        self.diag_shift = diag_shift
        self.holomorphic = holomorphic
        self.diag_scale = diag_scale

        self._monitor = {}

        if not (0 < alpha <= 2):
            raise ValueError(f"`alpha` must satisfy 0 < alpha <= 2, received {alpha}")
        self.alpha = alpha

        if distributed_eigh and not JAXMG_ENABLED:
            raise ImportError(
                "distributed_eigh=True requires jaxmg to be installed and enabled. "
                "Please install jaxmg (pip install jaxmg) and set the environment variable "
                "ENABLE_JAXMG=1 before running. This feature is only available on Linux systems."
            )
        self.distributed_eigh = distributed_eigh

        if sampling_state is not None:
            if not isinstance(sampling_state, VariationalState):
                raise ValueError(
                    f"Expected `sampling_state` to be a VariationalState, received {type(sampling_state)}"
                )
            sampling_structure = jax.tree_util.tree_structure(sampling_state.parameters)
            variational_structure = jax.tree_util.tree_structure(
                variational_state.parameters
            )
            if sampling_structure != variational_structure:
                raise ValueError(
                    f"Parameter structures of sampling_state and variational_state do not match. "
                    f"sampling_state structure: {sampling_structure}, "
                    f"variational_state structure: {variational_structure}"
                )
            sampling_leaves = jax.tree_util.tree_leaves(sampling_state.parameters)
            self.sampling_dtype = sampling_leaves[0].dtype
            self.sampling_state = sampling_state
        else:
            self.sampling_state = None
            self.sampling_dtype = None

        super().__init__(
            operator, variational_state, integrator, t0=t0, error_norm=error_norm
        )

    def _iter(
        self,
        T: float,
        tstops: Sequence[float] | None = None,
        callback: Callable | None = None,
    ):
        """
        Implementation of :code:`iter`. This method accepts an additional `callback` object, which
        is called after every accepted step.
        """
        t_end = self.t + T
        if tstops is not None and (
            np.any(np.less(tstops, self.t)) or np.any(np.greater(tstops, t_end))
        ):
            raise ValueError(
                f"All tstops must be in range [t, t + T]=[{self.t}, {t_end}]"
            )

        if tstops is not None and len(tstops) > 0:
            tstops = np.sort(tstops)
            always_stop = False
        else:
            tstops = []
            always_stop = True

        while self.t < t_end:
            if always_stop or (
                len(tstops) > 0
                and (np.isclose(self.t, tstops[0]) or self.t > tstops[0])
            ):
                self._stop_count += 1
                yield self.t
                tstops = tstops[1:]

            step_accepted = False
            while not step_accepted:
                if not always_stop and len(tstops) > 0:
                    max_dt = tstops[0] - self.t
                else:
                    max_dt = None
                step_accepted = self._integrator.step(max_dt=max_dt)
                if self._integrator.errors:
                    raise RuntimeError(
                        f"ODE integrator: {self._integrator.errors.message()}"
                    )
            self._step_count += 1
            if callback:
                callback()

        if (always_stop and np.isclose(self.t, t_end)) or (
            len(tstops) > 0 and np.isclose(tstops[0], t_end)
        ):
            yield self.t


# Copyright notice:
# The function `_impl` below includes lines copied from the jVMC repository
# found at github.com/markusschmitt/vmc_jax and licensed according to
# MIT License, Copyright (c) 2021 Markus Schmitt


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "n_samples",
        "rcond",
        "rcond_smooth",
        "snr_atol",
        "distributed_eigh",
    ),
)
def _impl(
    parameters,
    n_samples,
    E_loc,
    S,
    importance_weights,
    rhs_coeff,
    rcond,
    rcond_smooth,
    snr_atol,
    distributed_eigh,
):
    E = stats.statistics(importance_weights * E_loc)
    ΔE_loc = E_loc.reshape(-1, 1) - E.mean

    stack_jacobian = S.mode == "complex"

    O = S.O
    if stack_jacobian:
        O = O.reshape(-1, 2, S.O.shape[-1])
        O = O[:, 0, :] + 1j * O[:, 1, :]
    O = O * jnp.sqrt(
        importance_weights / importance_weights.shape[0]
    )
    Sd = S.to_dense()
    if distributed_eigh:
        Sd = jax.lax.with_sharding_constraint(Sd, P("S", None))
        mesh = jax.sharding.get_abstract_mesh()
        ev, V = syevd(Sd, T_A=1024, mesh=mesh, in_specs=(P("S", None),))
    else:
        ev, V = jnp.linalg.eigh(Sd)

    OEdata = O.conj() * ΔE_loc
    OE_mean = stats.mean(OEdata, axis=0)
    OE_var = stats.var(OEdata, axis=0)
    eps = jnp.finfo(O.dtype).eps
    snr_F = jnp.where(
        OE_var <= eps,
        jnp.inf,
        jnp.abs(OE_mean) * jnp.sqrt(n_samples) / jnp.sqrt(OE_var + eps),
    )
    F = stats.sum(OEdata, axis=0)
    Q = jnp.tensordot(V.conj().T, O.T, axes=1).T
    QEdata = Q.conj() * ΔE_loc
    rho = V.conj().T @ F

    sigma_k = jnp.maximum(jnp.sqrt(stats.var(QEdata, axis=0)), rcond)
    snr = jnp.where(
        sigma_k <= eps,
        jnp.inf,
        jnp.abs(rho) * jnp.sqrt(n_samples) / sigma_k,
    )

    ev_inv = jnp.where(jnp.abs(ev / ev[-1]) > rcond, 1.0 / ev, 0.0)
    regularizer = 1.0 / (1.0 + (rcond_smooth / jnp.abs(ev / ev[-1])) ** 6)
    if snr_atol is not None:
        regularizer = regularizer * (1.0 / (1.0 + (snr_atol / snr) ** 6))

    eta_p = ev_inv * regularizer * rhs_coeff * rho
    update = V @ eta_p

    rmd = jnp.linalg.norm(Sd.dot(update) - rhs_coeff * F) / jnp.linalg.norm(F)

    y, reassemble = convert_tree_to_dense_format(parameters, S.mode)
    update_tree = reassemble(update if jnp.iscomplexobj(y) else update.real)

    dw = tree_cast(update_tree, parameters)
    ev_reg = jnp.where(
        ev_inv * regularizer < 1.0 / rcond, 1.0 / (ev_inv * regularizer), jnp.nan
    )

    return E, dw, rmd, snr, snr_F, ev, ev_reg


@odefun.dispatch
def odefun_custom(
    state: MCState, self: TDVPSchmittOverdispersed, t, w, *, stage=0
):  # noqa: F811
    # pylint: disable=protected-access

    state.parameters = w
    state.reset()
    chunk_size = getattr(state, "chunk_size", None)

    op_t = self.generator(t)

    if self.sampling_state is not None:
        self.sampling_state.parameters = tree_cast(w, self.sampling_state.parameters)
        self.sampling_state.reset()
        samples = self.sampling_state.samples
    else:
        samples = state.samples

    samples_q, importance_weights, E_loc = HashablePartial(
        overdispersed_sample,
        apply_fn=state._apply_fun,
        op=op_t,
        alpha=self.alpha,
        chunk_size=chunk_size,
    )(samples, w)

    ess = ess_from_weights(importance_weights)
    importance_weights = importance_weights / jnp.mean(importance_weights)
    importance_weights = importance_weights.reshape(samples_q.shape[:-1])

    self._S = partial_from_kwargs(
        QGTJacobian_DefaultConstructor,
        exclusive_arg_names=(("mode", "holomorphic")),
    )(
        state._apply_fun,
        state.parameters,
        state.model_state,
        samples_q,
        pdf=importance_weights / importance_weights.size,
        dense=True,
        diag_shift=self.diag_shift,
        diag_scale=self.diag_scale,
        holomorphic=self.holomorphic,
        chunk_size=chunk_size,
    )

    (
        self._loss_stats,
        self._dw,
        self._rmd,
        self._snr,
        self._snr_F,
        self._ev,
        self._ev_reg,
    ) = _impl(
        state.parameters,
        state.n_samples,
        E_loc,
        self._S,
        importance_weights,
        self._loss_grad_factor,
        self.rcond,
        self.rcond_smooth,
        self.snr_atol,
        self.distributed_eigh,
    )
    self._monitor = make_monitor_dict(
        self._rmd, ess, self._snr, self._snr_F, self._ev, self._ev_reg
    )
    if stage == 0:
        self._last_qgt = self._S

    return self._dw