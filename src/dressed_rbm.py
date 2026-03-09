import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn

class DressedRBM(nn.Module):
    rbm: nn.Module
    amp_init: float = 1e-4
    correlation: float = 0.0
   

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        logpsi_rbm = self.rbm(x) 

        ns = x.shape[-1]
        sum_x = jnp.sum(x, axis=-1)
        index = (ns + sum_x)//2
        baseline = jnp.array([1.0]+ [self.amp_init]*(ns) , dtype = jnp.complex128)

        thetas = self.param(
            "thetas",
            lambda key, shape: jnp.array([0.0]*(ns), dtype = jnp.complex128),
            (ns,)
        )

        eps = jnp.concatenate([jnp.array([0.0], dtype = jnp.complex128), thetas])
        
        logF =  jnp.log(baseline[index] + eps[index] + self.correlation * jnp.where(index==3, eps[1], 0.0))
        
        # logF =  jnp.log(baseline[index] + eps[index] + eps[(index+1)%(ns+1)])  
        return logpsi_rbm + logF