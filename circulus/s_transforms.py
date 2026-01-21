import jax
import jax.numpy as jnp

@jax.jit
def s_to_y(S: jax.Array, z0: float = 1.0) -> jax.Array:
    """
    Utility: Converts an S-parameter matrix to an Admittance (Y) matrix.
    Formula: Y = (1/z0) * (I - S) * (I + S)^-1
    """
    n = S.shape[-1]
    I = jnp.eye(n, dtype=S.dtype)
    return (1.0 / z0) * (I - S) @ jnp.linalg.inv(I + S)