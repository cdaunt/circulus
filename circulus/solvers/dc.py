import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import optimistix as optx

from circulus.compiler import ComponentGroup
from circulus.solvers.common import _extract_diagonal, _sparse_matvec


def solve_dc_op_sparse(
    component_groups: list, 
    num_vars: int, 
    t0: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6
):
    """
    Optimized Sparse DC OP Solver:
    - Uses jax.ops.segment_sum for ultra-fast sparse matrix ops.
    - Implements Soft Damping to handle exponential components from 0V start.
    - Handles floating nodes via diagonal leakage.
    """
    
    # --- 1. Pre-calculate Indices (Static) ---
    all_rows_list = []
    all_cols_list = []
    for g in component_groups:
        all_rows_list.append(g.jac_rows.reshape(-1))
        all_cols_list.append(g.jac_cols.reshape(-1))
        
    # Add Ground Stiffener Index (0,0) explicitly to structure
    static_rows = jnp.concatenate(all_rows_list + [jnp.array([0])])
    static_cols = jnp.concatenate(all_cols_list + [jnp.array([0])])
    
    # Pre-calculate Mask for fast diagonal extraction
    # This allows us to find diagonal elements in one fast vector operation
    diag_mask = (static_rows == static_cols)

    def op_step(y_guess, _):
        total_f = jnp.zeros(num_vars)
        jac_vals_list = []

        # --- Vectorized Assembly ---
        for group in component_groups:
            v_locs = y_guess[group.var_indices]
            
            # Physics at t0 (DC only, ignores time derivatives)
            def get_f_only(v, p): 
                return group.physics_func(v, p, t=t0)[0]
            
            # Primal Eval
            f_loc = jax.vmap(get_f_only)(v_locs, group.params)
            
            # Jacobian Eval (Forward Mode)
            # Necessary because outputs are vectors
            df_loc = jax.vmap(jax.jacfwd(get_f_only))(v_locs, group.params)

            total_f = total_f.at[group.eq_indices].add(f_loc)
            jac_vals_list.append(df_loc.reshape(-1))

        # --- Linear System Setup ---
        # 1. Ground Constraint (Node 0) - "Stiff" connection to 0V
        G_stiff = 1e9
        total_f = total_f.at[0].add(G_stiff * y_guess[0])
        
        # 2. Global Leakage (G_leak)
        # Adds 1nS conductance to ground for EVERY node. 
        # Prevents singular matrix errors on floating nodes (e.g. between capacitors).
        G_leak = 1e-9
        total_f = total_f + y_guess * G_leak
        
        # Flatten Jacobian Values
        all_vals = jnp.concatenate(jac_vals_list + [jnp.array([G_stiff])])

        # --- OPTIMIZATION 1: Fast Diagonal Extraction ---
        # Extract physical diagonal sums using segment_sum
        diag_vals_physical = jax.ops.segment_sum(
            all_vals * diag_mask, static_rows, num_segments=num_vars
        )
        # Add leakage to diagonal (Effective Jacobian Diagonal)
        diag_vals_total = diag_vals_physical + G_leak
        
        # Compute Preconditioner (Inverse Diagonal)
        inv_diag = jnp.where(jnp.abs(diag_vals_total) < 1e-12, 1.0, 1.0 / diag_vals_total)

        # --- OPTIMIZATION 2: Fast Matvec ---
        def matvec(x):
            # 1. Sparse Matrix Multiply (J_physical * x)
            x_gathered = x[static_cols]
            products = all_vals * x_gathered
            Ax = jax.ops.segment_sum(products, static_rows, num_segments=num_vars)
            
            # 2. Add Leakage Term implicitly ((J + G_leak*I) * x)
            return Ax + (x * G_leak)

        # --- GMRES Solve ---
        # Initial guess: 1-step Jacobi update
        delta_guess = -total_f * inv_diag
        
        delta, _ = jax.scipy.sparse.linalg.gmres(
            matvec,
            -total_f,
            x0=delta_guess,
            M=lambda x: inv_diag * x, # Apply Preconditioner
            tol=1e-6,
            maxiter=100,
            restart=20
        )
        
        # --- OPTIMIZATION 3: Soft Damping ---
        # Critical for DC convergence. If the solver wants to change a node 
        # by 500V (common in first step), scale it down to 2V.
        max_change = jnp.max(jnp.abs(delta))
        damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))
        
        return y_guess + (delta * damping_factor)

    # Solve
    solver = optx.FixedPointIteration(rtol=tol, atol=tol)
    sol = optx.fixed_point(op_step, solver, jnp.zeros(num_vars), max_steps=max_iter, throw=False)
    
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