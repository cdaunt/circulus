import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx
import equinox as eqx
from functools import partial
from typing import Tuple, Any

# --- 1. Common Logic (JIT-able Functions) ---

# --- HELPER ---
def _compute_history(component_groups, y_c, t, num_vars):
    """
    Computes total charge Q at time t.
    Returns:
        Flat Real Array (2N) if complex
        Real Array (N) if real
    """
    if jnp.iscomplexobj(y_c):
        total_q = jnp.zeros(2 * num_vars, dtype=jnp.float64) # Real, Imag unrolled
    else:
        total_q = jnp.zeros(num_vars, dtype=y_c.dtype)
        
    for group in component_groups.values():
        v_locs = y_c[group.var_indices]
        # Physics func returns (f, q). We need q.
        _, q_l = jax.vmap(lambda v, p: group.physics_func(v, p, t=t))(v_locs, group.params)
        
        if jnp.iscomplexobj(y_c):
             # Map Real charge to first half, Imag charge to second half
             total_q = total_q.at[group.eq_indices].add(q_l.real)
             total_q = total_q.at[group.eq_indices + num_vars].add(q_l.imag)
        else:
             total_q = total_q.at[group.eq_indices].add(q_l)
             
    return total_q

def _assemble_system_real(y_guess, component_groups, t1, dt, num_vars):
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    
    vals_list = []

    # Sort to ensure order matches 'init' construction
    # (Assuming component_groups is a standard dict, iteration order might vary 
    # unless sorted or inserted deterministically. Sorting keys is safest.)
    sorted_keys = sorted(component_groups.keys())
    
    for k in sorted_keys:
        group = component_groups[k]
        v_locs = y_guess[group.var_indices]
        def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)
        
        (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
        (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
        
        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        
        j_eff = df_l + (dq_l / dt)
        vals_list.append(j_eff.reshape(-1))

    # REMOVED: Ground Constraint Appending
    # Reason: The solver 'step' function handles this manually via .at[0,0].add(1e9)
    # Adding it here would cause a shape mismatch with static_rows.

    all_vals = jnp.concatenate(vals_list)
    return total_f, total_q, all_vals


def _prepare_step_state(y0, is_complex, num_vars):
    """Prepares the state vectors for the solver step."""
    # 1. Promote Real(N) to Complex(N) if needed
    if is_complex and not jnp.iscomplexobj(y0) and y0.shape[0] == num_vars:
        y0 = y0.astype(jnp.complex128)
    
    # 2. Extract Physics State (y_c)
    if is_complex and not jnp.iscomplexobj(y0) and y0.shape[0] == 2 * num_vars:
        y_c = y0[:num_vars] + 1j * y0[num_vars:]
    else:
        y_c = y0

    # 3. Prepare Solver Guess (Flattened 2N or Real N)
    if is_complex and jnp.iscomplexobj(y0):
        y_solver_init = jnp.concatenate([y0.real, y0.imag])
    else:
        y_solver_init = y0

    return y0, y_c, y_solver_init

def _assemble_system_real(y_guess, component_groups, t1, dt, num_vars):
    """
    Assembles the Jacobian and Residual for Real-valued systems.
    Does NOT append ground constraints (handled in solver step).
    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    
    vals_list = []

    # Sort keys to ensure deterministic ordering matching 'init'
    sorted_keys = sorted(component_groups.keys())
    
    for k in sorted_keys:
        group = component_groups[k]
        v_locs = y_guess[group.var_indices]
        def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)
        
        (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
        (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
        
        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        
        j_eff = df_l + (dq_l / dt)
        vals_list.append(j_eff.reshape(-1))

    all_vals = jnp.concatenate(vals_list)
    return total_f, total_q, all_vals

def _assemble_system_real(y_guess, component_groups, t1, dt, num_vars):
    """
    Assembles the Jacobian and Residual for Real-valued systems.
    Does NOT append ground constraints (handled in solver step).
    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    
    vals_list = []

    # Sort keys to ensure deterministic ordering matching 'init'
    sorted_keys = sorted(component_groups.keys())
    
    for k in sorted_keys:
        group = component_groups[k]
        v_locs = y_guess[group.var_indices]
        def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)
        
        (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
        (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
        
        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        
        j_eff = df_l + (dq_l / dt)
        vals_list.append(j_eff.reshape(-1))

    all_vals = jnp.concatenate(vals_list)
    return total_f, total_q, all_vals

def _assemble_system_complex(y_guess, component_groups, t1, dt, num_vars):
    """
    Assembles the Jacobian and Residual for Complex systems (Unrolled to Real).
    Sorts Jacobian values into [RR, RI, IR, II] blocks to match 'init' indices.
    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real = y_guess[:half_size]
    y_imag = y_guess[half_size:]
    
    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)
    
    # Buckets to match the 'init' index structure: [RR, RI, IR, II]
    vals_rr = []
    vals_ri = []
    vals_ir = []
    vals_ii = []

    sorted_keys = sorted(component_groups.keys())

    for k in sorted_keys:
        group = component_groups[k]
        v_locs_real = y_real[group.var_indices]
        v_locs_imag = y_imag[group.var_indices]

        # Split Physics for Non-Holomorphic Support
        def physics_split(v_r, v_i, p):
            v = v_r + 1j * v_i
            f, q = group.physics_func(v, p, t=t1)
            return f.real, f.imag, q.real, q.imag

        # Jacobian: Returns 4 tuples of derivatives
        jac_res = jax.vmap(jax.jacfwd(physics_split, argnums=(0,1)))(
            v_locs_real, v_locs_imag, group.params
        )
        ((dfr_vr, dfr_vi), (dfi_vr, dfi_vi), (dqr_vr, dqr_vi), (dqi_vr, dqi_vi)) = jac_res
        
        # Effective Jacobian Blocks: J = df/dv + (1/dt)*dq/dv
        vals_rr.append((dfr_vr + dqr_vr/dt).reshape(-1))
        vals_ri.append((dfr_vi + dqr_vi/dt).reshape(-1))
        vals_ir.append((dfi_vr + dqi_vr/dt).reshape(-1))
        vals_ii.append((dfi_vi + dqi_vi/dt).reshape(-1))
        
        # Residuals
        fr, fi, qr, qi = jax.vmap(physics_split)(v_locs_real, v_locs_imag, group.params)
        total_f = total_f.at[group.eq_indices].add(fr).at[group.eq_indices + half_size].add(fi)
        total_q = total_q.at[group.eq_indices].add(qr).at[group.eq_indices + half_size].add(qi)

    # Concatenate in the exact order defined in 'init': RR, RI, IR, II
    all_vals = jnp.concatenate(
        vals_rr + vals_ri + vals_ir + vals_ii
    )
    
    return total_f, total_q, all_vals


# ==============================================================================
# 2. SOLVER STEPS (CORRECTED)
# ==============================================================================

@jax.jit
def _dense_newton_step_real(y_guess, args):
    (component_groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    
    total_f, total_q, all_vals = _assemble_system_real(y_guess, component_groups, t1, dt, num_vars)
    
    # 1. Residual Construction
    residual = total_f + (total_q - q_prev) / dt
    residual = residual.at[0].add(1e9 * y_guess[0]) # Real Ground

    # 2. Jacobian Construction
    sys_size = y_guess.shape[0]
    J_dense = jnp.zeros((sys_size, sys_size), dtype=residual.dtype)
    J_dense = J_dense.at[static_rows, static_cols].add(all_vals)

    # --- FIX: Add Ground Stiffness to Jacobian ---
    J_dense = J_dense.at[0, 0].add(1e9)

    delta = jnp.linalg.solve(J_dense, -residual)

    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 1.0 / (max_change + 1e-9))
    
    return y_guess + (delta * damping_factor)

@jax.jit
def _dense_newton_step_complex(y_guess, args):
    (component_groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    
    # y_guess is flat real (2N). _assemble must handle this input correctly.
    total_f, total_q, all_vals = _assemble_system_complex(y_guess, component_groups, t1, dt, num_vars)
    
    # FIX: q_prev is ALREADY flat real (2N) from _compute_history. Do not re-expand.
    residual = total_f + (total_q - q_prev) / dt
    
    # 1. Residual Constraints
    residual = residual.at[0].add(1e9 * y_guess[0])
    residual = residual.at[half_size].add(1e9 * y_guess[half_size])

    # 2. Jacobian
    J_dense = jnp.zeros((sys_size, sys_size), dtype=residual.dtype)
    J_dense = J_dense.at[static_rows, static_cols].add(all_vals)

    # 3. Manual Ground Stiffness
    J_dense = J_dense.at[0, 0].add(1e9)
    J_dense = J_dense.at[half_size, half_size].add(1e9)

    delta = jnp.linalg.solve(J_dense, -residual)

    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 1.0 / (max_change + 1e-9))
    
    return y_guess + (delta * damping_factor)

@jax.jit
def _sparse_newton_step_real(y_guess, args):
    (component_groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    sys_size = y_guess.shape[0]

    total_f, total_q, all_vals = _assemble_system_real(y_guess, component_groups, t1, dt, num_vars)
    
    # 1. Residual
    residual = total_f + (total_q - q_prev) / dt
    residual = residual.at[0].add(1e9 * y_guess[0])

    # 2. Preconditioner Fix
    diag_vals = jax.ops.segment_sum(
        all_vals * diag_mask, static_rows, num_segments=sys_size
    )
    # --- FIX: Add stiffness to diagonal estimate ---
    diag_vals = diag_vals.at[0].add(1e9)
    inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-9, 1.0, 1.0 / diag_vals)

    # 3. MatVec Fix
    def matvec(x):
        x_gathered = x[static_cols]
        products = all_vals * x_gathered
        Ax = jax.ops.segment_sum(products, static_rows, num_segments=sys_size)
        
        # --- FIX: Add stiffness to linear operator ---
        Ax = Ax.at[0].add(1e9 * x[0])
        return Ax

    delta_guess = -residual * inv_diag 
    
    # Switched to BiCGSTAB
    delta, _ = jax.scipy.sparse.linalg.bicgstab(
        matvec, -residual, x0=delta_guess,
        M=lambda x: inv_diag * x, tol=1e-5, maxiter=50
    )

    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))

    return y_guess + (delta * damping_factor)

@jax.jit
def _sparse_newton_step_complex(y_guess, args):
    (component_groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2

    total_f, total_q, all_vals = _assemble_system_complex(y_guess, component_groups, t1, dt, num_vars)
    
    # FIX: q_prev is ALREADY flat real (2N) from _compute_history. Do not re-expand.
    residual = total_f + (total_q - q_prev) / dt
    
    residual = residual.at[0].add(1e9 * y_guess[0])
    residual = residual.at[half_size].add(1e9 * y_guess[half_size])

    # Preconditioner
    diag_vals = jax.ops.segment_sum(
        all_vals * diag_mask, static_rows, num_segments=sys_size
    )
    # Manual Ground Stiffness
    diag_vals = diag_vals.at[0].add(1e9)
    diag_vals = diag_vals.at[half_size].add(1e9)
    
    inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-9, 1.0, 1.0 / diag_vals)

    def matvec(x):
        x_gathered = x[static_cols]
        products = all_vals * x_gathered
        Ax = jax.ops.segment_sum(products, static_rows, num_segments=sys_size)
        
        # Manual Ground Stiffness
        Ax = Ax.at[0].add(1e9 * x[0])
        Ax = Ax.at[half_size].add(1e9 * x[half_size])
        return Ax

    delta_guess = -residual * inv_diag 
    
    delta, _ = jax.scipy.sparse.linalg.bicgstab(
        matvec, -residual, x0=delta_guess,
        M=lambda x: inv_diag * x, tol=1e-5, maxiter=50
    )

    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))

    return y_guess + (delta * damping_factor)

# ==============================================================================
# 3. UNIFIED SOLVER CLASS
# ==============================================================================

class TransientSolverState(eqx.Module):
    static_rows: jax.Array
    static_cols: jax.Array
    diag_mask: jax.Array | None
    history: tuple
    is_complex_mode: bool = eqx.field(static=True)

class VectorizedTransientSolver(diffrax.AbstractSolver):
    mode: str = eqx.field(static=True)
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars = args

        # --- 1. Index Pre-calculation ---
        all_rows_list = []
        all_cols_list = []
        # Sort keys to ensure deterministic ordering
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            all_rows_list.append(g.jac_rows.reshape(-1))
            all_cols_list.append(g.jac_cols.reshape(-1))
            
        base_rows = jnp.concatenate(all_rows_list)
        base_cols = jnp.concatenate(all_cols_list)

        # --- 2. Complex Unrolling Check ---
        # If y0 is (2N,) real, we treat it as unrolled complex.
        is_complex = jnp.iscomplexobj(y0) or (y0.shape[0] == 2 * num_vars)
        
        # Also check parameters just in case y0 started real (e.g. all zeros)
        if not is_complex:
            for group in component_groups.values():
                for leaf in jax.tree.leaves(group.params):
                    if jnp.iscomplexobj(leaf):
                        is_complex = True
                        break
        
        # --- 3. Build Static Indices ---
        if is_complex:
            r, c = base_rows, base_cols
            N = num_vars
            # Unrolled Blocks: RR, RI, IR, II
            # CRITICAL: Do NOT append Ground indices (0, N) here.
            # The solver handles ground manually via .at[].add()
            static_rows = jnp.concatenate([r, r, r+N, r+N])
            static_cols = jnp.concatenate([c, c+N, c, c+N])
        else:
            static_rows = base_rows
            static_cols = base_cols

        diag_mask = (static_rows == static_cols)

        # --- 4. Prepare History (Always Flat Real) ---
        # Ensure history is stored as a Flat Real array to satisfy Diffrax
        if jnp.iscomplexobj(y0):
            y0_flat = jnp.concatenate([y0.real, y0.imag])
        elif is_complex and y0.shape[0] == num_vars:
            # Promote size N real -> size 2N real (Imag part 0)
            y0_flat = jnp.concatenate([y0, jnp.zeros_like(y0)])
        else:
            # Already flat real (N or 2N)
            y0_flat = y0
            
        history = (y0_flat, 1.0) # (y_prev, dt_prev)
        
        return TransientSolverState(static_rows, static_cols, diag_mask, history, is_complex)
    
    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state.history
        is_complex = solver_state.is_complex_mode

        # --- 1. Standardization: Ensure y0 is Flat Real ---
        if jnp.iscomplexobj(y0):
            # If Diffrax passed complex (rare if init is correct), flatten it
            y0_flat = jnp.concatenate([y0.real, y0.imag])
        else:
            y0_flat = y0

        # --- 2. Physics Prep: Need Complex for History Calc ---
        if is_complex:
            # Reconstruct complex view for physics function
            # y0_flat is size 2N -> y_c is size N complex
            y_c = y0_flat[:num_vars] + 1j * y0_flat[num_vars:]
        else:
            y_c = y0_flat

        # --- 3. Predictor (1st Order) ---
        rate_prev = (y0_flat - y_prev_step) / dt_prev
        y_pred = y0_flat + rate_prev * dt
        y_solver_init = y_pred

        # --- 4. Calculate History (Charge at t0) ---
        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        # --- 5. Newton Solve ---
        # We pass y_solver_init (Flat Real) and get back Flat Real
        step_args = (component_groups, t1, dt, q_prev, solver_state.static_rows, 
                     solver_state.static_cols, solver_state.diag_mask, num_vars)

        if self.mode == 'dense':
            solver_fn = _dense_newton_step_complex if is_complex else _dense_newton_step_real
        else:
            solver_fn = _sparse_newton_step_complex if is_complex else _sparse_newton_step_real

        # Wrapper to fixed point
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(solver_fn, solver, y_solver_init, args=step_args, max_steps=30, throw=False)

        # --- 6. Output Formatting ---
        y_next_flat = sol.value
        
        # CRITICAL FIX: Return Flat Real arrays to Diffrax. 
        # Do NOT re-complexify y_next here.
        y_next = y_next_flat
        y_error = y_next_flat - y_pred

        # Update history with the flat real state
        new_state = eqx.tree_at(lambda s: s.history, solver_state, (y_next_flat, dt))
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )

        return y_next, y_error, {"y0": y0_flat, "y1": y_next}, new_state, result
    
    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
