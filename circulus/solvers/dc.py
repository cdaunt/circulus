import functools
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import optimistix as optx
import equinox as eqx
from typing import Dict, List, Tuple, Any, Optional, Union

from circulus.compiler import ComponentGroup

# --- Helpers ---

def _infer_dtype(component_groups: Union[Dict[str, ComponentGroup], List[ComponentGroup]], default_dtype):
    """Detects if any component parameters are complex, forcing complex mode."""
    # Handle both dict (legacy) and list (sorted)
    iterator = component_groups.values() if isinstance(component_groups, dict) else component_groups
    
    for group in iterator:
        for leaf in jax.tree.leaves(group.params):
            if jnp.iscomplexobj(leaf):
                return jnp.complex128
    return default_dtype

def _apply_soft_damping(y_guess, delta):
    """
    Applies soft damping to updates to prevent divergence on cold starts.
    Scales down large voltage jumps (e.g. > 2V) significantly.
    """
    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))
    return y_guess + (delta * damping_factor)

# --- Assembly Functions (Framework Style) ---

def _assemble_dc_real(y_guess, groups_list: List[ComponentGroup], t0):
    """
    Assembles the DC residual vector and Jacobian values for a Real system.
    Iterates over a sorted list of groups to ensure alignment with static indices.
    Returns: (total_f, all_vals)
    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    vals_list = []

    for group in groups_list:
        v_locs = y_guess[group.var_indices]
        
        # Physics at t0 (DC only)
        def physics_dc(v, p): 
            # Expecting physics_func to return (f, q). We only need f for DC.
            return group.physics_func(v, p, t=t0)[0] 
        
        f_l = jax.vmap(physics_dc)(v_locs, group.params)
        df_l = jax.vmap(jax.jacfwd(physics_dc))(v_locs, group.params)
        
        total_f = total_f.at[group.eq_indices].add(f_l)
        vals_list.append(df_l.reshape(-1))

    # Ground Constraint (Stiffness)
    G_stiff = 1e9
    vals_list.append(jnp.array([G_stiff], dtype=y_guess.dtype))
    
    all_vals = jnp.concatenate(vals_list)
    return total_f, all_vals

def _assemble_dc_complex(y_guess, groups_list: List[ComponentGroup], t0):
    """
    Assembles the DC residual vector and Jacobian values for a Complex system.
    Iterates over a sorted list of groups to ensure alignment with static indices.
    Returns: (total_f, all_vals)
    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real = y_guess[:half_size]
    y_imag = y_guess[half_size:]
    
    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    
    # Separate lists for each block type
    vals_rr, vals_ri, vals_ir, vals_ii = [], [], [], []

    for group in groups_list:
        v_locs_real = y_real[group.var_indices]
        v_locs_imag = y_imag[group.var_indices]

        # Split Physics for Non-Holomorphic Support
        def physics_split(v_r, v_i, p):
            v = v_r + 1j * v_i
            f, _ = group.physics_func(v, p, t=t0) # Ignore q
            return f.real, f.imag

        # Jacobian: Returns ((dFr/dVr, dFr/dVi), (dFi/dVr, dFi/dVi))
        jac_res = jax.vmap(jax.jacfwd(physics_split, argnums=(0,1)))(
            v_locs_real, v_locs_imag, group.params
        )
        ((dfr_vr, dfr_vi), (dfi_vr, dfi_vi)) = jac_res
        
        # Collect blocks
        vals_rr.append(dfr_vr.reshape(-1))
        vals_ri.append(dfr_vi.reshape(-1))
        vals_ir.append(dfi_vr.reshape(-1))
        vals_ii.append(dfi_vi.reshape(-1))
        
        # Residuals
        fr, fi = jax.vmap(physics_split)(v_locs_real, v_locs_imag, group.params)
        total_f = total_f.at[group.eq_indices].add(fr)
        total_f = total_f.at[group.eq_indices + half_size].add(fi)

    # Ground Constraint (Stiffness) - Real and Imag parts
    G_stiff = 1e9
    stiff_vals = jnp.array([G_stiff], dtype=jnp.float64)

    # Concatenate in Block Order: RR, RI, IR, II, Ground_Real, Ground_Imag
    all_vals = jnp.concatenate(
        vals_rr + vals_ri + vals_ir + vals_ii + 
        [stiff_vals, stiff_vals]
    )
    return total_f, all_vals

import functools
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import optimistix as optx
import equinox as eqx
from typing import Dict, List, Tuple, Any, Optional, Union

from circulus.compiler import ComponentGroup

# --- Helpers ---

def _infer_dtype(component_groups: Union[Dict[str, ComponentGroup], List[ComponentGroup]], default_dtype):
    """Detects if any component parameters are complex, forcing complex mode."""
    # Handle both dict (legacy) and list (sorted)
    iterator = component_groups.values() if isinstance(component_groups, dict) else component_groups
    
    for group in iterator:
        for leaf in jax.tree.leaves(group.params):
            if jnp.iscomplexobj(leaf):
                return jnp.complex128
    return default_dtype

def _apply_soft_damping(y_guess, delta):
    """
    Applies soft damping to updates to prevent divergence on cold starts.
    Scales down large voltage jumps (e.g. > 2V) significantly.
    """
    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))
    return y_guess + (delta * damping_factor)

# --- Assembly Functions (Framework Style) ---

def _assemble_dc_real(y_guess, groups_list: List[ComponentGroup], t0):
    """
    Assembles the DC residual vector and Jacobian values for a Real system.
    Iterates over a sorted list of groups to ensure alignment with static indices.
    Returns: (total_f, all_vals)
    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    vals_list = []

    for group in groups_list:
        v_locs = y_guess[group.var_indices]
        
        # Physics at t0 (DC only)
        def physics_dc(v, p): 
            # Expecting physics_func to return (f, q). We only need f for DC.
            return group.physics_func(v, p, t=t0)[0] 
        
        f_l = jax.vmap(physics_dc)(v_locs, group.params)
        df_l = jax.vmap(jax.jacfwd(physics_dc))(v_locs, group.params)
        
        total_f = total_f.at[group.eq_indices].add(f_l)
        vals_list.append(df_l.reshape(-1))

    # Ground Constraint (Stiffness)
    G_stiff = 1e9
    vals_list.append(jnp.array([G_stiff], dtype=y_guess.dtype))
    
    all_vals = jnp.concatenate(vals_list)
    return total_f, all_vals

def _assemble_dc_complex(y_guess, groups_list: List[ComponentGroup], t0):
    """
    Assembles the DC residual vector and Jacobian values for a Complex system.
    Iterates over a sorted list of groups to ensure alignment with static indices.
    Returns: (total_f, all_vals)
    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real = y_guess[:half_size]
    y_imag = y_guess[half_size:]
    
    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    
    # Separate lists for each block type
    vals_rr, vals_ri, vals_ir, vals_ii = [], [], [], []

    for group in groups_list:
        v_locs_real = y_real[group.var_indices]
        v_locs_imag = y_imag[group.var_indices]

        # Split Physics for Non-Holomorphic Support
        def physics_split(v_r, v_i, p):
            v = v_r + 1j * v_i
            f, _ = group.physics_func(v, p, t=t0) # Ignore q
            return f.real, f.imag

        # Jacobian: Returns ((dFr/dVr, dFr/dVi), (dFi/dVr, dFi/dVi))
        jac_res = jax.vmap(jax.jacfwd(physics_split, argnums=(0,1)))(
            v_locs_real, v_locs_imag, group.params
        )
        ((dfr_vr, dfr_vi), (dfi_vr, dfi_vi)) = jac_res
        
        # Collect blocks
        vals_rr.append(dfr_vr.reshape(-1))
        vals_ri.append(dfr_vi.reshape(-1))
        vals_ir.append(dfi_vr.reshape(-1))
        vals_ii.append(dfi_vi.reshape(-1))
        
        # Residuals
        fr, fi = jax.vmap(physics_split)(v_locs_real, v_locs_imag, group.params)
        total_f = total_f.at[group.eq_indices].add(fr)
        total_f = total_f.at[group.eq_indices + half_size].add(fi)

    # Ground Constraint (Stiffness) - Real and Imag parts
    G_stiff = 1e9
    stiff_vals = jnp.array([G_stiff], dtype=jnp.float64)

    # Concatenate in Block Order: RR, RI, IR, II, Ground_Real, Ground_Imag
    all_vals = jnp.concatenate(
        vals_rr + vals_ri + vals_ir + vals_ii + 
        [stiff_vals, stiff_vals]
    )
    return total_f, all_vals

# --- Step Functions (JIT-Compiled) ---

@jax.jit
def _dc_step_real_dense(y_guess, args):
    (groups_list, t0, static_rows, static_cols, g_leak) = args
    
    total_f, all_vals = _assemble_dc_real(y_guess, groups_list, t0)
    
    # 1. Add Ground & Leakage to Residual
    residual = total_f
    residual = residual.at[0].add(1e9 * y_guess[0])
    residual = residual + (y_guess * g_leak)
    
    # 2. Dense Solve
    sys_size = y_guess.shape[0]
    J_dense = jnp.zeros((sys_size, sys_size), dtype=residual.dtype)
    J_dense = J_dense.at[static_rows, static_cols].add(all_vals)
    
    # Add leakage diagonal
    diag_idx = jnp.arange(sys_size)
    J_dense = J_dense.at[diag_idx, diag_idx].add(g_leak)
    
    delta = jnp.linalg.solve(J_dense, -residual)
    return _apply_soft_damping(y_guess, delta)

@jax.jit
def _dc_step_real_sparse(y_guess, args):
    (groups_list, t0, static_rows, static_cols, diag_mask, g_leak) = args
    
    total_f, all_vals = _assemble_dc_real(y_guess, groups_list, t0)
    
    # 1. Add Ground & Leakage to Residual
    residual = total_f
    residual = residual.at[0].add(1e9 * y_guess[0])
    residual = residual + (y_guess * g_leak)
    
    # 2. Sparse Solve
    sys_size = y_guess.shape[0]
    
    # Extract Diagonals for Preconditioner
    diag_vals = jax.ops.segment_sum(
        all_vals * diag_mask, static_rows, num_segments=sys_size
    )
    diag_vals = diag_vals + g_leak
    inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-12, 1.0, 1.0 / diag_vals)

    def matvec(x):
        x_gathered = x[static_cols]
        products = all_vals * x_gathered
        Ax = jax.ops.segment_sum(products, static_rows, num_segments=sys_size)
        return Ax + (x * g_leak)

    delta_guess = -residual * inv_diag
    delta, _ = jax.scipy.sparse.linalg.gmres(
        matvec, -residual, x0=delta_guess,
        M=lambda x: inv_diag * x, tol=1e-6, maxiter=100, restart=20
    )
    return _apply_soft_damping(y_guess, delta)

@jax.jit
def _dc_step_complex_dense(y_guess, args):
    (groups_list, t0, static_rows, static_cols, g_leak) = args
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    
    total_f, all_vals = _assemble_dc_complex(y_guess, groups_list, t0)
    
    # 1. Add Ground & Leakage to Residual
    residual = total_f
    residual = residual.at[0].add(1e9 * y_guess[0]) # Real Ground
    residual = residual.at[half_size].add(1e9 * y_guess[half_size]) # Imag Ground
    residual = residual + (y_guess * g_leak)
    
    J_dense = jnp.zeros((sys_size, sys_size), dtype=residual.dtype)
    J_dense = J_dense.at[static_rows, static_cols].add(all_vals)
    
    # Add leakage diagonal
    diag_idx = jnp.arange(sys_size)
    J_dense = J_dense.at[diag_idx, diag_idx].add(g_leak)
    
    delta = jnp.linalg.solve(J_dense, -residual)
    return _apply_soft_damping(y_guess, delta)

@jax.jit
def _dc_step_complex_sparse(y_guess, args):
    (groups_list, t0, static_rows, static_cols, diag_mask, g_leak) = args
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    
    total_f, all_vals = _assemble_dc_complex(y_guess, groups_list, t0)
    
    # 1. Add Ground & Leakage to Residual
    residual = total_f
    residual = residual.at[0].add(1e9 * y_guess[0]) # Real Ground
    residual = residual.at[half_size].add(1e9 * y_guess[half_size]) # Imag Ground
    residual = residual + (y_guess * g_leak)
    
    diag_vals = jax.ops.segment_sum(
        all_vals * diag_mask, static_rows, num_segments=sys_size
    )
    diag_vals = diag_vals + g_leak
    inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-12, 1.0, 1.0 / diag_vals)

    def matvec(x):
        x_gathered = x[static_cols]
        products = all_vals * x_gathered
        Ax = jax.ops.segment_sum(products, static_rows, num_segments=sys_size)
        return Ax + (x * g_leak)

    delta_guess = -residual * inv_diag
    delta, _ = jax.scipy.sparse.linalg.gmres(
        matvec, -residual, x0=delta_guess,
        M=lambda x: inv_diag * x, tol=1e-6, maxiter=100, restart=20
    )
    return _apply_soft_damping(y_guess, delta)

# --- Main Solver Entry Point ---

def solve_operating_point(
    component_groups: Dict[str, ComponentGroup],
    num_vars: int,
    t0: float = 0.0,
    method: str = 'dense',
    max_iter: int = 50,
    tol: float = 1e-6,
    g_leak: float = 1e-9,
    dtype = None
):
    """
    Unified DC Operating Point solver.
    Solves the circuit at time t0 assuming d/dt = 0.
    
    Args:
        component_groups: Dictionary of ComponentGroups.
        num_vars: Number of nodes in the circuit.
        t0: Time at which to solve (default 0.0).
        method: 'dense' or 'sparse'.
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance.
        g_leak: Leakage conductance for floating node stabilization.
        dtype: Optional dtype (inferred if None).
    """
    if dtype is None:
        dtype = _infer_dtype(component_groups, jnp.float64)
        
    is_complex = jnp.issubdtype(dtype, jnp.complexfloating)
    
    # --- CRITICAL: Enforce Deterministic Order ---
    # Sort groups by name (key) to ensure the index construction (outside JIT)
    # matches the value assembly (inside JIT), regardless of dict flattening.
    sorted_keys = sorted(component_groups.keys())
    groups_list = [component_groups[k] for k in sorted_keys]
    
    # --- Index Pre-calculation ---
    all_rows_list = []
    all_cols_list = []
    
    # Use the sorted list!
    for g in groups_list:
        all_rows_list.append(g.jac_rows.reshape(-1))
        all_cols_list.append(g.jac_cols.reshape(-1))
        
    base_rows = jnp.concatenate(all_rows_list)
    base_cols = jnp.concatenate(all_cols_list)
    
    if is_complex:
        # Expand 2N indices
        r, c = base_rows, base_cols
        N = num_vars
        # Blocks: RR, RI, IR, II, G_real, G_imag
        static_rows = jnp.concatenate([r, r, r+N, r+N, jnp.array([0]), jnp.array([N])])
        static_cols = jnp.concatenate([c, c+N, c, c+N, jnp.array([0]), jnp.array([N])])
        
        # Initial guess (Real + Imag flattened)
        y0_flat = jnp.zeros(2 * num_vars, dtype=jnp.float64)
    else:
        # Standard indices + Ground
        static_rows = jnp.concatenate([base_rows, jnp.array([0])])
        static_cols = jnp.concatenate([base_cols, jnp.array([0])])
        
        y0_flat = jnp.zeros(num_vars, dtype=dtype)
        
    # Diag mask for sparse preconditioner
    diag_mask = (static_rows == static_cols)
    
    # --- Solve ---
    solver = optx.FixedPointIteration(rtol=tol, atol=tol)
    
    if is_complex:
        if method == 'dense':
            step_args = (groups_list, t0, static_rows, static_cols, g_leak)
            solver_fn = _dc_step_complex_dense
        else:
            step_args = (groups_list, t0, static_rows, static_cols, diag_mask, g_leak)
            solver_fn = _dc_step_complex_sparse
    else:
        if method == 'dense':
            step_args = (groups_list, t0, static_rows, static_cols, g_leak)
            solver_fn = _dc_step_real_dense
        else:
            step_args = (groups_list, t0, static_rows, static_cols, diag_mask, g_leak)
            solver_fn = _dc_step_real_sparse
            
    sol = optx.fixed_point(solver_fn, solver, y0_flat, args=step_args, max_steps=max_iter, throw=False)
    
    # --- Result Formatting ---
    if is_complex:
        # Reconstruct complex vector from flat [Real, Imag]
        return sol.value[:num_vars] + 1j * sol.value[num_vars:]
    else:
        return sol.value

# --- Legacy Wrappers ---

def solve_dc_op_dense(component_groups, num_vars, t0=0.0, max_iter=50, tol=1e-6, g_leak=1e-9, dtype=None):
    return solve_operating_point(component_groups, num_vars, t0, 'dense', max_iter, tol, g_leak, dtype)

def solve_dc_op_sparse(component_groups, num_vars, t0=0.0, max_iter=50, tol=1e-6, dtype=None):
    return solve_operating_point(component_groups, num_vars, t0, 'sparse', max_iter, tol, 1e-9, dtype)

@jax.jit
def s_to_y(S: jax.Array, z0: float = 1.0) -> jax.Array:
    """
    Utility: Converts an S-parameter matrix to an Admittance (Y) matrix.
    Formula: Y = (1/z0) * (I - S) * (I + S)^-1
    """
    n = S.shape[-1]
    I = jnp.eye(n, dtype=S.dtype)
    return (1.0 / z0) * (I - S) @ jnp.linalg.inv(I + S)