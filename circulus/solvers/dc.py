import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import optimistix as optx
import equinox as eqx

from circulus.compiler import ComponentGroup
from circulus.solvers.common import _extract_diagonal, _sparse_matvec


def _infer_dtype(component_groups, default_dtype):
    for group in component_groups.values():
        for leaf in jax.tree.leaves(group.params):
            if jnp.iscomplexobj(leaf):
                return jnp.complex128
    return default_dtype


def solve_dc_op_sparse(
    component_groups: dict[str, ComponentGroup],
    num_vars: int, 
    t0: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    dtype = None
):
    """
    Optimized Sparse DC OP Solver:
    - Uses jax.ops.segment_sum for ultra-fast sparse matrix ops.
    - Implements Soft Damping to handle exponential components from 0V start.
    - Handles floating nodes via diagonal leakage.
    """
    
    if dtype is None:
        dtype = _infer_dtype(component_groups, jnp.float64)
    
    # --- 1. Pre-calculate Indices (Static) ---
    all_rows_list = []
    all_cols_list = []
    for g_name, g in component_groups.items():
        all_rows_list.append(g.jac_rows.reshape(-1))
        all_cols_list.append(g.jac_cols.reshape(-1))
        
    # Add Ground Stiffener Index (0,0) explicitly to structure
    static_rows = jnp.concatenate(all_rows_list + [jnp.array([0])])
    static_cols = jnp.concatenate(all_cols_list + [jnp.array([0])])
    
    # Pre-calculate Mask for fast diagonal extraction
    # This allows us to find diagonal elements in one fast vector operation
    diag_mask = (static_rows == static_cols)

    def op_step(y_guess, _):
        total_f = jnp.zeros(num_vars, dtype=dtype)
        jac_vals_list = []

        # --- Vectorized Assembly ---
        for group_name, group in component_groups.items():
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
    sol = optx.fixed_point(op_step, solver, jnp.zeros(num_vars, dtype=dtype), max_steps=max_iter, throw=False)
    
    return sol.value

def solve_dc_op_dense(
    component_groups: dict[str, ComponentGroup],
    num_vars: int, 
    t0: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    g_leak = 1e-9,
    dtype = None
):
    """
    Optimized DENSE OP Solver:
    - Direct Dense Scatter (Faster)
    - Soft Damping (Prevents divergence on cold start)
    - Diagonal G_leak (Prevents singular matrix on floating nodes)
    """
    
    if dtype is None:
        dtype = _infer_dtype(component_groups, jnp.float64)
    
    is_complex = jnp.issubdtype(dtype, jnp.complexfloating)
    
    # --- 1. Pre-calculate indices (Flattened) ---
    all_rows_list = []
    all_cols_list = []
    for g_name, g in component_groups.items():
        all_rows_list.append(g.jac_rows.reshape(-1))
        all_cols_list.append(g.jac_cols.reshape(-1))
        
    base_rows = jnp.concatenate(all_rows_list)
    base_cols = jnp.concatenate(all_cols_list)
    
    if is_complex:
        # Expand to 2N x 2N Real system
        # Blocks: [[RR, RI], [IR, II]]
        r, c = base_rows, base_cols
        N = num_vars
        static_rows = jnp.concatenate([r, r, r+N, r+N])
        static_cols = jnp.concatenate([c, c+N, c, c+N])
        
        # Diagonal Leakage Indices (Real and Imag parts)
        diag_idx = jnp.arange(num_vars)
        diag_rows = jnp.concatenate([diag_idx, diag_idx + N])
        diag_cols = diag_rows
        
        # Ground Stiffener Indices (Real and Imag parts of Node 0)
        gnd_rows = jnp.array([0, N])
        gnd_cols = jnp.array([0, N])
    else:
        static_rows = base_rows
        static_cols = base_cols
        diag_rows = jnp.arange(num_vars)
        diag_cols = diag_rows
        
        # Ground Stiffener Index (Node 0)
        gnd_rows = jnp.array([0])
        gnd_cols = jnp.array([0])
    
    # Combine Physics + Leakage + Ground indices
    static_rows = jnp.concatenate([static_rows, diag_rows, gnd_rows])
    static_cols = jnp.concatenate([static_cols, diag_cols, gnd_cols])

    @jax.jit
    def op_step(y_guess, _):
        sys_size = 2 * num_vars if is_complex else num_vars
        # Underlying linear system is always Real
        real_dtype = jnp.array(0, dtype=dtype).real.dtype
        
        total_f = jnp.zeros(sys_size, dtype=real_dtype)
        J_dense = jnp.zeros((sys_size, sys_size), dtype=real_dtype)
        
        vals_rr, vals_ri, vals_ir, vals_ii = [], [], [], []
        vals_real = []

        # --- Vectorized Assembly ---
        for group_name, group in component_groups.items():
            
            if is_complex:
                # Reconstruct Complex State for Physics
                y_real = y_guess[:num_vars]
                y_imag = y_guess[num_vars:]
                y_c = y_real + 1j * y_imag
                v_locs = y_c[group.var_indices]

                # Split Physics for Non-Holomorphic Support (|E|^2)
                def get_f_split(v_r, v_i, p):
                    v = v_r + 1j * v_i
                    f = group.physics_func(v, p, t=t0)[0]
                    return f.real, f.imag

                # Jacobian: Returns ((dFr/dVr, dFr/dVi), (dFi/dVr, dFi/dVi))
                jac_tuple = jax.vmap(jax.jacfwd(get_f_split, argnums=(0,1)))(
                    v_locs.real, v_locs.imag, group.params
                )
                ((J_rr, J_ri), (J_ir, J_ii)) = jac_tuple
                
                vals_rr.append(J_rr.reshape(-1))
                vals_ri.append(J_ri.reshape(-1))
                vals_ir.append(J_ir.reshape(-1))
                vals_ii.append(J_ii.reshape(-1))
                
                # Residuals
                fr, fi = jax.vmap(get_f_split)(v_locs.real, v_locs.imag, group.params)
                total_f = total_f.at[group.eq_indices].add(fr)
                total_f = total_f.at[group.eq_indices + num_vars].add(fi)
            else:
                # Standard Real Physics
                v_locs = y_guess[group.var_indices]
                def get_f_only(v, p): return group.physics_func(v, p, t=t0)[0]
                f_loc = jax.vmap(get_f_only)(v_locs, group.params)
                df_loc = jax.vmap(jax.jacfwd(get_f_only))(v_locs, group.params)
                total_f = total_f.at[group.eq_indices].add(f_loc)
                vals_real.append(df_loc.reshape(-1))

        # --- G_leak Handling (Stabilization) ---
        leak_vals = jnp.full(len(diag_rows), g_leak, dtype=real_dtype)
        
        # --- Ground Constraint ---
        G_stiff = 1e9
        
        if is_complex:
             # Add to residual (Force V_gnd = 0)
             total_f = total_f.at[0].add(G_stiff * y_guess[0]) # Real part of Node 0
             total_f = total_f.at[num_vars].add(G_stiff * y_guess[num_vars]) # Imag part of Node 0
             
             total_f = total_f + y_guess * g_leak
             
             stiff_vals = jnp.array([G_stiff, G_stiff], dtype=real_dtype)
             all_vals = jnp.concatenate(vals_rr + vals_ri + vals_ir + vals_ii + [leak_vals, stiff_vals])
        else:
             # Add to residual (Force V_gnd = 0)
             total_f = total_f.at[0].add(G_stiff * y_guess[0])
             total_f = total_f + y_guess * g_leak
             stiff_vals = jnp.array([G_stiff], dtype=real_dtype)
             all_vals = jnp.concatenate(vals_real + [leak_vals, stiff_vals])

        # --- Optimization 1: Direct Dense Scatter ---
        # Directly scatter into the dense matrix (Faster than BCOO->Dense)
        J_dense = J_dense.at[static_rows, static_cols].add(all_vals)
        
        # --- Direct Solve ---
        delta = jnp.linalg.solve(J_dense, -total_f)
        
        # --- Optimization 2: Soft Damping ---
        # Prevents "overshoot" where the solver jumps to 1000V in one step.
        max_change = jnp.max(jnp.abs(delta))
        
        damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))
        
        return y_guess + (delta * damping_factor)

    # Solve
    solver = optx.FixedPointIteration(rtol=tol, atol=tol)
    
    if is_complex:
        # Initialize with Flat Real Zeros (2N)
        y0_flat = jnp.zeros(2 * num_vars, dtype=jnp.array(0, dtype=dtype).real.dtype)
        sol = optx.fixed_point(op_step, solver, y0_flat, max_steps=max_iter, throw=False)
        # Reconstruct Complex Result
        return sol.value[:num_vars] + 1j * sol.value[num_vars:]
    else:
        sol = optx.fixed_point(op_step, solver, jnp.zeros(num_vars, dtype=dtype), max_steps=max_iter, throw=False)
        return sol.value

@jax.jit
def s_to_y(S: jax.Array, z0: float = 1.0) -> jax.Array:
    """
    Utility: Converts an S-parameter matrix to an Admittance (Y) matrix.
    Formula: Y = (1/z0) * (I - S) * (I + S)^-1
    
    Args:
        S: (N, N) complex S-matrix.
        z0: Characteristic impedance (default 1.0).
    
    Returns:
        Y: (N, N) complex Y-matrix.
    """
    n = S.shape[-1]
    I = jnp.eye(n, dtype=S.dtype)
    return (1.0 / z0) * (I - S) @ jnp.linalg.inv(I + S)