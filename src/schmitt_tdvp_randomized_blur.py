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

from src.tdvp_utils import make_monitor_dict, randomized_blurred_sample, ess_from_weights
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


class TDVPSchmittRandomizedBlur(TDVPBaseDriver):
    r"""
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :class:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.

    This driver, which only works with standard MCState variational states, uses the regularization
    procedure described in `M. Schmitt's PRL 125 <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.125.100503>`_ .

    With the force vector

    .. math::

        F_k=\langle \mathcal O_{\theta_k}^* E_{loc}^{\theta}\rangle_c

    and the quantum Fisher matrix

    .. math::

        S_{k,k'} = \langle \mathcal O_{\theta_k} (\mathcal O_{\theta_{k'}})^*\rangle_c

    and for real parameters :math:`\theta\in\mathbb R`, the TDVP equation reads

    .. math::

        q\big[S_{k,k'}\big]\theta_{k'} = -q\big[xF_k\big]

    Here, either :math:`q=\text{Re}` or :math:`q=\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.

    For ground state search a regularization controlled by a parameter :math:`\rho` can be included
    by increasing the diagonal entries and solving

    .. math::

        q\big[(1+\rho\delta_{k,k'})S_{k,k'}\big]\theta_{k'} = -q\big[F_k\big]

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

    .. math::

        S = V\Sigma V^\dagger

    with a diagonal matrix :math:`\Sigma_{kk}=\sigma_k`
    Assuming that :math:`\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed
    from the regularized inverted eigenvalues

    .. math::

        \tilde\sigma_k^{-1}=\frac{1}{\Big(1+\big(\frac{\epsilon_{SVD}}{\sigma_j/\sigma_1}\big)^6\Big)\Big(1+\big(\frac{\epsilon_{SNR}}{\text{SNR}(\rho_k)}\big)^6\Big)}

    with :math:`\text{SNR}(\rho_k)` the signal-to-noise ratio of
    :math:`\rho_k=V_{k,k'}^{\dagger}F_{k'}` (see
    `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).


    .. note::

        This TDVP Driver uses the time-integrators from the `nkx.dynamics` module, which are
        automatically executed under a `jax.jit` context.

        When running computations on GPU, this can lead to infinite hangs or extremely long
        compilation times. In those cases, you might try setting the configuration variable
        `nk.config.netket_experimental_disable_ode_jit = True` to mitigate those issues.

    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        integrator: AbstractSolver = None,
        *,
        q1: float = 0.1,
        q2: float = 0.1,
        flip_prob: float = 0.1,
        t0: float = 0.0,
        propagation_type: str = "real",
        holomorphic: bool | None = None,
        diag_shift: float = 0.0,
        diag_scale: float | None = None,
        error_norm: str | Callable = "qgt",
        rcond: float = 1e-14,
        rcond_smooth: float = 1e-8,
        snr_atol: float | None,
        sampling_state: VariationalState = None,
        distributed_eigh: bool = False,
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (Hamiltonian for pure states,
                Lindbladian for density operators).
            variational_state: The variational state.
            integrator: Configuration of the algorithm used for solving the ODE.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schrödinger equation (SE), "imag" for the imaginary-time SE.
            error_norm: Norm function used to calculate the error with adaptive integrators.
                Can be either "euclidean" for the standard L2 vector norm :math:`w^\dagger w`,
                "maximum" for the maximum norm :math:`\max_i |w_i|`
                or "qgt", in which case the scalar product induced by the QGT :math:`S` is used
                to compute the norm :math:`\Vert w \Vert^2_S = w^\dagger S w` as suggested
                in PRL 125, 100503 (2020).
                Additionally, it possible to pass a custom function with signature
                :code:`norm(x: PyTree) -> float`
                which maps a PyTree of parameters :code:`x` to the corresponding norm.
                Note that norm is used in jax.jit-compiled code.
            holomorphic: a flag to indicate that the wavefunction is holomorphic.
            diag_shift: diagonal shift of the quantum geometric tensor (QGT)
            diag_scale: If not None rescales the diagonal shift of the QGT
            rcond : Cut-off ratio for small singular :math:`\sigma_k` values of the
                Quantum Geometric Tensor. For the purposes of rank determination,
                singular values are treated as zero if they are smaller than rcond times
                the largest singular value :code:`\sigma_{max}`.
            rcond_smooth : Smooth cut-off ratio for singular values of the Quantum Geometric
                Tensor. This regularization parameter used with a similar effect to `rcond`
                but with a softer curve. See :math:`\epsilon_{SVD}` in the formula
                above.
            snr_atol: Noise regularisation absolute tolerance, meaning that eigenvalues of
                the S matrix that have a signal to noise ratio above this quantity will be
                (soft) truncated. This is :math:`\epsilon_{SNR}` in the formulas above.

        """
        self.propagation_type = propagation_type
        if isinstance(variational_state, VariationalMixedState):
            # assuming Lindblad Dynamics
            # TODO: support density-matrix imaginary time evolution
            if propagation_type == "real":
                self._loss_grad_factor = 1.0
            else:
                raise ValueError(
                    "only real-time Lindblad evolution is supported for " "mixed states"
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

        if not (0 < q1 < 1):
            raise ValueError(f"`q1` must satisfy 0 < q1 < 1, received {q1}")
        self.q1 = q1
        if not (0 < q2 < 1):
            raise ValueError(f"`q2` must satisfy 0 < q2 < 1, received {q2}")
        self.q2 = q2
        self.q = [q1, q2]
        self.flip_prob = flip_prob

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
            # Check that sampling_state and variational_state have the same parameter structure
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
            # Store the dtype of sampling_state parameters
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
        Implementation of :code:`iter`. This method accepts and additional `callback` object, which
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
            # print("dt:", self.dt)
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
            # optionally call callback
            if callback:
                callback()

        # Yield one last time if the remaining tstop is at t_end
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
    )  # Undo PDF since it's already in ΔE_loc
    Sd = S.to_dense()
    if distributed_eigh:
        Sd = jax.lax.with_sharding_constraint(Sd, P("S", None))
        mesh = jax.sharding.get_abstract_mesh()
        print("Using syevd")
        ev, V = syevd(Sd, T_A=1024, mesh=mesh, in_specs=(P("S", None),))
    else:

        ev, V = jnp.linalg.eigh(Sd)

    OEdata = O.conj() * ΔE_loc
    # SNR of the force estimator F = sum_i O_i^* ΔE_i
    OE_mean = stats.mean(OEdata, axis=0)
    OE_var = stats.var(OEdata, axis=0)
    eps = jnp.finfo(O.dtype).eps
    snr_F = jnp.where(
        OE_var <= eps,
        jnp.inf,
        jnp.abs(OE_mean) * jnp.sqrt(n_samples) / jnp.sqrt(OE_var + eps),
    )
    F = stats.sum(OEdata, axis=0)
    # eigenbasis
    Q = jnp.tensordot(V.conj().T, O.T, axes=1).T
    QEdata = Q.conj() * ΔE_loc
    rho = V.conj().T @ F

    # Compute the SNR according to Eq. 21 but taking care of where sigma_k is zero
    sigma_k = jnp.maximum(jnp.sqrt(stats.var(QEdata, axis=0)), rcond)
    # Here we are hardcoding the case where rho==0 and sigma_k==0 to have infinite snr.
    # This is an arbitrary choice, but avoids generating NaNs in the snr calculation.
    # See netket#1959 and #1960 for more details.
    snr = jnp.where(
        sigma_k <= eps,
        jnp.inf,
        jnp.abs(rho) * jnp.sqrt(n_samples) / sigma_k,
    )

    # Discard eigenvalues below numerical precision
    ev_inv = jnp.where(jnp.abs(ev / ev[-1]) > rcond, 1.0 / ev, 0.0)
    # Set regularizer for singular value cutoff
    regularizer = 1.0 / (1.0 + (rcond_smooth / jnp.abs(ev / ev[-1])) ** 6)
    if snr_atol is not None:
        # Construct a soft cutoff based on the SNR
        regularizer = regularizer * (1.0 / (1.0 + (snr_atol / snr) ** 6))

    # solve the linear system by hand
    eta_p = ev_inv * regularizer * rhs_coeff * rho
    # convert back to the parameter space
    update = V @ eta_p

    # remainder of the solution
    rmd = jnp.linalg.norm(Sd.dot(update) - rhs_coeff * F) / jnp.linalg.norm(F)

    y, reassemble = convert_tree_to_dense_format(parameters, S.mode)
    update_tree = reassemble(update if jnp.iscomplexobj(y) else update.real)

    # If parameters are real, then take only real part of the gradient (if it's complex)
    dw = tree_cast(update_tree, parameters)
    ev_reg = jnp.where(
        ev_inv * regularizer < 1.0 / rcond, 1.0 / (ev_inv * regularizer), jnp.nan
    )

    return E, dw, rmd, snr, snr_F, ev, ev_reg


@odefun.dispatch
def odefun_custom(
    state: MCState, self: TDVPSchmittRandomizedBlur, t, w, *, stage=0
):  # noqa: F811
    # pylint: disable=protected-access

    state.parameters = w
    state.reset()
    chunk_size = getattr(state, "chunk_size", None)

    # Generator and schedules
    op_t = self.generator(t)
    # Allow sampling in lower precision
    if self.sampling_state is not None:
        # Cast parameters to sampling_state dtype and assign
        self.sampling_state.parameters = tree_cast(w, self.sampling_state.parameters)
        self.sampling_state.reset()
        samples = self.sampling_state.samples
    else:
        samples = self.state.samples
    # Bridge kernel
    state._sampler_seed, key = jax.random.split(state._sampler_seed, 2)

    samples_q, importance_weights, E_loc = HashablePartial(
        randomized_blurred_sample,
        apply_fn=state._apply_fun,
        op=op_t,
        q1=self.q1,
        q2=self.q2,
        flip_prob=self.flip_prob,
        chunk_size=chunk_size,
    )(samples, key, w)
    # Monitor ESS of the combined weights
    ess = ess_from_weights(importance_weights)
    # print("ESS", ess, "q", self.q)
    # Normalize weights for use as a pdf
    importance_weights = importance_weights / jnp.mean(importance_weights)

    # Get S-matrix
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
    return ((s1_sq / (s2 - s1_sq + jnp.finfo(w.dtype).eps))).squeeze()


def main():
    import netket as nk
    from netket.experimental.dynamics import RK45
    import flax.linen as nn

    N = 4
    alpha = 1
    n_samples = 2**12
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N)
    graph = nk.graph.Chain(N, pbc=True)
    hamiltonian = nk.operator.IsingJax(hilbert=hilbert, graph=graph, h=1, J=-1.0)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_samples)
    model = nk.models.RBM(
        alpha=alpha, param_dtype=complex, kernel_init=nn.initializers.normal(1e-1)
    )
    vstate = nk.vqs.MCState(
        sampler=sampler, model=model, n_samples=n_samples, seed=100, sampler_seed=100
    )

    T = 0.5

    integrator = RK45(1e-3, adaptive=True, rtol=1e-4, dt_limits=(1e-4, 1e-2))
    tvmc_kwargs = {}
    driver = TDVPSchmittRandomizedBlur(
        hamiltonian,
        vstate,
        integrator,
        t0=0,
        q=0.2,
        snr_atol=2,
        rcond=1e-14,
        rcond_smooth=1e-10,
        distributed_eigh=True,
        **tvmc_kwargs,
    )

    driver.run(
        T,
        show_progress=True,
        timeit=True,
    )


if __name__ == "__main__":
    main()
