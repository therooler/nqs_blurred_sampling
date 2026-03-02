import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn

class DressedRBM(nn.Module):
    rbm: nn.Module
    amp_init: float = 1e-4
   
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        logpsi_rbm = self.rbm(x) 

        ns = x.shape[-1]
        sum_x = jnp.sum(x, axis=-1)

        amp_spin = self.param(
            "amp_spin",
            lambda key, shape: jnp.array([0.0]*(ns-1), dtype = jnp.complex128),
            (ns-1,)
        )

        # global shared log-amplitude for all sectors n_up>=1
        eps = self.param(
            "eps",
            lambda key: jnp.asarray(self.amp_init, dtype=jnp.float64).astype(jnp.complex128)
        )
        a_s =  jnp.concatenate((jnp.array([1.0, eps], dtype = jnp.complex128), 
                                self.amp_init + amp_spin))
        index = (ns + sum_x)//2
        logF =  jnp.log(a_s[index]) 

        return logpsi_rbm + logF