import jax
from dataclasses import dataclass

from netket.hilbert.random import flip_state

from netket.sampler.rules import MetropolisRule


@dataclass
class LocalDoubleFlipRule(MetropolisRule):
    r""""""

    def transition(rule, sampler, machine, parameters, state, key, σ):
        double_flip, select_1, accept_1, select_2, accept_2 = jax.random.split(key, 5)

        n_chains = σ.shape[0]
        hilb = sampler.hilbert
        u = jax.random.uniform(double_flip, shape=())
        indxs = jax.random.randint(
            select_1, shape=(n_chains,), minval=0, maxval=hilb.size
        )
        σp, _ = flip_state(hilb, accept_1, σ, indxs)

        def true_branch(x):
            indxs = jax.random.randint(
                select_2, shape=(n_chains,), minval=0, maxval=hilb.size
            )
            x, _ = flip_state(hilb, accept_2, x, indxs)
            return x

        def false_branch(x):
            return x

        σpp = jax.lax.cond(u < 0.5, true_branch, false_branch, operand=σp)

        return σpp, None
