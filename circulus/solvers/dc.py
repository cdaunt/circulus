import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
from jax.experimental import sparse # Required for Dense Solver assembly
from jax import lax
from functools import partial
import optimistix as optx

# Import types for type hinting
from circulus.compiler import ComponentGroup
from circulus.solvers.common import _extract_diagonal, _sparse_matvec

def solve_dc_op_sparse(
    component_groups: list[ComponentGroup], 
    num_vars: int, 
    t0: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6
):
    """
    Solves the algebraic system f(x, t0) = 0.
    Ignores dynamic terms (dq/dt).
    """
    
    # Pre-calculate static structure (same as in transient, but useful here too)
    all_rows = []
    all_cols = []
    for g in component_groups:
        all_rows.append(g.jac_rows)
        all_cols.append(g.jac_cols)
    
    # Add a dummy zero at the end to ensure concatenation works safely
    static_rows = jnp.concatenate(all_rows + [jnp.array([0])])
    static_cols = jnp.concatenate(all_cols + [jnp.array([0])])

    def op_step(y_guess, _):
        total_f = jnp.zeros(num_vars)
        jac_vals_list = []

        # --- Vectorized Assembly (DC Version) ---
        for group in component_groups:
            # 1. Get Local Variables
            v_locs = y_guess[group.var_indices]
            
            # 2. Physics Evaluation (t is fixed)
            physics_fn = partial(group.physics_func, t=t0)
            
            # 3. Compute Value and Jacobian
            # We explicitly define a wrapper that returns ONLY the resistive part 'f'
            # because jax.jacfwd needs to know exactly what to differentiate.
            def get_f_only(v, p):
                return physics_fn(v, p)[0]
            
            # Calculate Value (N, width)
            f_loc = jax.vmap(get_f_only)(v_locs, group.params)
            
            # Calculate Jacobian (N, width, width) using Forward Mode AD
            # jacfwd is required here (not value_and_grad) because output is a vector.
            df_loc = jax.vmap(jax.jacfwd(get_f_only))(v_locs, group.params)

            # 4. Scatter Add
            total_f = total_f.at[group.eq_indices].add(f_loc)
            jac_vals_list.append(df_loc.reshape(-1))

        # --- Linear Solve ---
        # G_leak to prevent singular matrix if nodes are floating (e.g. between capacitors)
        G_leak = 1e-9 
        
        # Flatten Jacobian values
        all_vals = jnp.concatenate(jac_vals_list + [jnp.array([G_leak])])
        
        # Add leakage to diagonal explicitly in residual and matrix
        # (This is a simplified Gmin stepping approach)
        total_f = total_f + y_guess * G_leak
        
        # Solve J * delta = -f
        
        # 1. Extract Diagonal for Preconditioner
        diag_vals = _extract_diagonal(static_rows, static_cols, all_vals, num_vars)
        # Avoid division by zero
        safe_diag = jnp.where(jnp.abs(diag_vals) < 1e-12, 1.0, diag_vals)
        precond_inv = 1.0 / safe_diag

        def matvec(x):
            mv = _sparse_matvec(static_rows, static_cols, all_vals, x, num_vars)
            return mv + x * G_leak # Add the implicit leakage diagonal

        delta, info = jax.scipy.sparse.linalg.gmres(
            matvec,
            -total_f,
            M=lambda x: precond_inv * x,
            tol=1e-6,
            maxiter=100
        )
        
        return y_guess + delta

    # Run the Fixed Point Solver
    solver = optx.FixedPointIteration(rtol=tol, atol=tol)
    sol = optx.fixed_point(op_step, solver, jnp.zeros(num_vars), max_steps=max_iter)
    
    return sol.value


def solve_dc_op_dense(
    component_groups: list, 
    num_vars: int, 
    t0: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    g_leak = 1e-9
):
    """
    Optimized DENSE OP Solver:
    - Direct Dense Scatter (Faster)
    - Soft Damping (Prevents divergence on cold start)
    - Diagonal G_leak (Prevents singular matrix on floating nodes)
    """
    
    # --- 1. Pre-calculate indices (Flattened) ---
    all_rows_list = []
    all_cols_list = []
    for g in component_groups:
        all_rows_list.append(g.jac_rows.reshape(-1))
        all_cols_list.append(g.jac_cols.reshape(-1))
    
    # Optimization: Apply G_leak to ALL nodes (diagonal), not just ground.
    # This prevents singular matrices if a node is "floating" (e.g. between capacitors).
    diag_indices = jnp.arange(num_vars)
    
    static_rows = jnp.concatenate(all_rows_list + [diag_indices])
    static_cols = jnp.concatenate(all_cols_list + [diag_indices])

    def op_step(y_guess, _):
        total_f = jnp.zeros(num_vars)
        J_dense = jnp.zeros((num_vars, num_vars))
        jac_vals_list = []

        # --- Vectorized Assembly ---
        for group in component_groups:
            v_locs = y_guess[group.var_indices]
            
            # Define physics function at t0
            def get_f_only(v, p): 
                # [0] = currents (f), ignore charges (q) for DC
                return group.physics_func(v, p, t=t0)[0]
            
            # Primal Evaluation
            f_loc = jax.vmap(get_f_only)(v_locs, group.params)
            
            # Jacobian Evaluation
            # jacfwd is robust for vector-valued outputs
            df_loc = jax.vmap(jax.jacfwd(get_f_only))(v_locs, group.params)

            total_f = total_f.at[group.eq_indices].add(f_loc)
            jac_vals_list.append(df_loc.reshape(-1))

        # --- G_leak Handling (Stabilization) ---
        # Add small conductance to every node's diagonal
        total_f = total_f + y_guess * g_leak
        leak_vals = jnp.full(num_vars, g_leak) 

        # --- Optimization 1: Direct Dense Scatter ---
        all_vals = jnp.concatenate(jac_vals_list + [leak_vals])
        
        # Directly scatter into the dense matrix (Faster than BCOO->Dense)
        J_dense = J_dense.at[static_rows, static_cols].add(all_vals)
        
        # --- Direct Solve ---
        delta = jnp.linalg.solve(J_dense, -total_f)
        
        # --- Optimization 2: Soft Damping ---
        # Prevents "overshoot" where the solver jumps to 1000V in one step.
        max_change = jnp.max(jnp.abs(delta))
        
        # If change > 2.0V, scale the whole vector down
        damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))
        
        return y_guess + (delta * damping_factor)

    # Solve
    solver = optx.FixedPointIteration(rtol=tol, atol=tol)
    sol = optx.fixed_point(op_step, solver, jnp.zeros(num_vars), max_steps=max_iter, throw=False)
    
    return sol.value