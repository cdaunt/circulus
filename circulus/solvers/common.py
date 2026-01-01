import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['num_vars'])
def _extract_diagonal(rows, cols, values, num_vars):
    """Extracts the diagonal elements of a sparse matrix (COO format)."""
    # Initialize diagonal with zeros
    diag = jnp.zeros(num_vars)
    # Mask for diagonal elements
    mask = (rows == cols)
    # Use index_add to sum values where row==col
    diag = diag.at[rows].add(jnp.where(mask, values, 0.0))
    return diag

@partial(jax.jit, static_argnames=['num_vars'])
def _sparse_matvec(rows, cols, values, vec, num_vars):
    """Computes A @ x for a sparse matrix A."""
    # Gather values from x based on column indices
    input_vals = vec[cols]
    # Multiply by matrix values
    products = values * input_vals
    # Scatter sum into result based on row indices
    res = jnp.zeros(num_vars)
    return res.at[rows].add(products)