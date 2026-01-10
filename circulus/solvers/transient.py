import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optimistix as optx
from functools import partial

# ==============================================================================
# 1. ASSEMBLY KERNELS (PHYSICS)
# ==============================================================================

def _assemble_system_real(y_guess, component_groups, t1, dt):
    """Assembles Jacobian/Residual for Real systems."""
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    vals_list = []

    # Deterministic order via sorted keys
    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v_locs = y_guess[group.var_indices]
        
        # Physics & Derivatives
        def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)
        (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
        (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
        
        # Accumulate
        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        j_eff = df_l + (dq_l / dt)
        vals_list.append(j_eff.reshape(-1))

    return total_f, total_q, jnp.concatenate(vals_list)

def _assemble_system_complex(y_guess, component_groups, t1, dt):
    """Assembles Jacobian/Residual for Unrolled Complex systems (Block Format)."""
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]
    
    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)
    
    # Block Accumulators: [RR, RI, IR, II]
    vals_blocks = [[], [], [], []] 

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        # 1. Split Physics (Real -> Complex -> Real)
        def physics_split(vr, vi, p):
            v = vr + 1j * vi
            f, q = group.physics_func(v, p, t=t1)
            return f.real, f.imag, q.real, q.imag

        # 2. Primal & Residuals
        fr, fi, qr, qi = jax.vmap(physics_split)(v_r, v_i, group.params)
        
        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

        # 3. Jacobian (4 blocks)
        jac_res = jax.vmap(jax.jacfwd(physics_split, argnums=(0,1)))(v_r, v_i, group.params)
        ((dfr_r, dfr_i), (dfi_r, dfi_i), (dqr_r, dqr_i), (dqi_r, dqi_i)) = jac_res
        
        # J_eff = df/dv + (1/dt)*dq/dv
        vals_blocks[0].append((dfr_r + dqr_r/dt).reshape(-1)) # RR
        vals_blocks[1].append((dfr_i + dqr_i/dt).reshape(-1)) # RI
        vals_blocks[2].append((dfi_r + dqi_r/dt).reshape(-1)) # IR
        vals_blocks[3].append((dfi_i + dqi_i/dt).reshape(-1)) # II

    # Concatenate blocks in RR, RI, IR, II order to match 'init' indices
    all_vals = jnp.concatenate([jnp.concatenate(b) for b in vals_blocks])
    return total_f, total_q, all_vals

def _compute_history(component_groups, y_c, t, num_vars):
    """Computes total charge Q at time t (Initial Condition)."""
    is_complex = jnp.iscomplexobj(y_c)
    total_q = jnp.zeros(2 * num_vars if is_complex else num_vars, dtype=jnp.float64 if is_complex else y_c.dtype)
        
    for group in component_groups.values():
        v_locs = y_c[group.var_indices]
        _, q_l = jax.vmap(lambda v, p: group.physics_func(v, p, t=t))(v_locs, group.params)
        
        if is_complex:
             total_q = total_q.at[group.eq_indices].add(q_l.real)
             total_q = total_q.at[group.eq_indices + num_vars].add(q_l.imag)
        else:
             total_q = total_q.at[group.eq_indices].add(q_l)
    return total_q

# ==============================================================================
# 2. UNIFIED SOLVER KERNEL
# ==============================================================================

@partial(jax.jit, static_argnames=['is_complex', 'mode'])
def _unified_newton_step(y_guess, args, is_complex=False, mode='dense'):
    (groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    sys_size = y_guess.shape[0]

    # --- 1. Assemble System ---
    if is_complex:
        total_f, total_q, all_vals = _assemble_system_complex(y_guess, groups, t1, dt)
        # FIX: q_prev is already flat real (2N) from _compute_history. Do not re-expand.
        q_prev_flat = q_prev 
        ground_indices = [0, sys_size // 2] # Real Ground, Imag Ground
    else:
        total_f, total_q, all_vals = _assemble_system_real(y_guess, groups, t1, dt)
        q_prev_flat = q_prev
        ground_indices = [0] # Real Ground

    # --- 2. Construct Residual ---
    residual = total_f + (total_q - q_prev_flat) / dt
    
    # Apply Ground Constraints to Residual
    for idx in ground_indices:
        residual = residual.at[idx].add(1e9 * y_guess[idx])

    # --- 3. Solve (Dense vs Sparse) ---
    if mode == 'dense':
        J = jnp.zeros((sys_size, sys_size), dtype=residual.dtype)
        J = J.at[static_rows, static_cols].add(all_vals)
        
        # Ground Stiffness (Derivative)
        for idx in ground_indices:
            J = J.at[idx, idx].add(1e9)
            
        delta = jnp.linalg.solve(J, -residual)
        
    else: # Sparse
        # Preconditioner
        diag_vals = jax.ops.segment_sum(all_vals * diag_mask, static_rows, num_segments=sys_size)
        for idx in ground_indices:
            diag_vals = diag_vals.at[idx].add(1e9)
        
        # Guard against zero diagonal
        inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-12, 1.0, 1.0 / diag_vals)

        # Linear Operator
        def matvec(x):
            x_gathered = x[static_cols]
            Ax = jax.ops.segment_sum(all_vals * x_gathered, static_rows, num_segments=sys_size)
            for idx in ground_indices:
                Ax = Ax.at[idx].add(1e9 * x[idx])
            return Ax

        delta_guess = -residual * inv_diag
        
        # TUNING: Increased limits for steady-state robustness
        delta, _ = jax.scipy.sparse.linalg.bicgstab(
            matvec, -residual, x0=delta_guess, 
            M=lambda x: inv_diag * x, tol=1e-6, maxiter=1000
        )

    # --- 4. Damping ---
    max_change = jnp.max(jnp.abs(delta))
    # Heuristic: Stricter damping for sparse/complex to prevent oscillation
    safe_factor = 2.0 if mode == 'sparse' else 1.0 
    damping = jnp.minimum(1.0, safe_factor / (max_change + 1e-9))
    
    return y_guess + (delta * damping)


# ==============================================================================
# 3. SOLVER CLASS
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

    def order(self, terms): return 1

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars = args

        # --- 1. Index Pre-calculation ---
        all_rows, all_cols = [], []
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            all_rows.append(g.jac_rows.reshape(-1))
            all_cols.append(g.jac_cols.reshape(-1))
            
        base_rows = jnp.concatenate(all_rows)
        base_cols = jnp.concatenate(all_cols)

        # --- 2. Detect Complexity ---
        # If y0 is size 2N, we assume it's an unrolled complex state
        is_complex = jnp.iscomplexobj(y0) or (y0.shape[0] == 2 * num_vars)
        if not is_complex: # Double check params
            for g in component_groups.values():
                if any(jnp.iscomplexobj(p) for p in jax.tree.leaves(g.params)):
                    is_complex = True; break

        # --- 3. Build Block Indices ---
        if is_complex:
            r, c = base_rows, base_cols
            N = num_vars
            # RR, RI, IR, II Blocks. NO Ground indices here (handled in step)
            static_rows = jnp.concatenate([r, r, r+N, r+N])
            static_cols = jnp.concatenate([c, c+N, c, c+N])
        else:
            static_rows, static_cols = base_rows, base_cols

        # --- 4. History Setup ---
        # Normalize y0 to flat real
        if jnp.iscomplexobj(y0):
            y0_flat = jnp.concatenate([y0.real, y0.imag])
        elif is_complex and y0.shape[0] == num_vars:
            y0_flat = jnp.concatenate([y0, jnp.zeros_like(y0)])
        else:
            y0_flat = y0
            
        return TransientSolverState(
            static_rows, static_cols, (static_rows == static_cols), 
            (y0_flat, 1.0), is_complex
        )
    
    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state.history
        is_complex = solver_state.is_complex_mode

        # 1. Normalize State (Complex -> Flat Real)
        y0_flat = y0
        if jnp.iscomplexobj(y0): 
            y0_flat = jnp.concatenate([y0.real, y0.imag])

        # 2. Physics Prep (Flat Real -> Complex View)
        y_c = y0_flat
        if is_complex:
            y_c = y0_flat[:num_vars] + 1j * y0_flat[num_vars:]

        # 3. Predictor & History
        # Linear prediction for better initial guess: y_pred = y0 + y' * dt
        # Guard dt_prev to prevent division instability
        rate = (y0_flat - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0_flat + rate * dt
        
        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        # 4. Newton Solve (Unified)
        step_args = (component_groups, t1, dt, q_prev, solver_state.static_rows, 
                     solver_state.static_cols, solver_state.diag_mask, num_vars)
        
        # Bind static args for JIT compilation
        solver_fn = partial(_unified_newton_step, is_complex=is_complex, mode=self.mode)

        # TUNING: Increased limits for large steps
        solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
        # Start Newton from Predictor
        sol = optx.fixed_point(solver_fn, solver, y_pred, args=step_args, max_steps=100, throw=False)

        # 5. Result
        y_next = sol.value # Already Flat Real
        
        # Error Estimate: Difference between Implicit Solution and Explicit Predictor
        # Approximates local truncation error O(dt^2) for Order 1 methods.
        y_error = y_next - y_pred

        # Update history with the state at the START of the step (y0_flat)
        # This enables the next step to calculate the slope (y_next - y0_flat)
        new_state = eqx.tree_at(lambda s: s.history, solver_state, (y0_flat, dt))
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )

        return y_next, y_error, {"y0": y0_flat, "y1": y_next}, new_state, result
    
    def func(self, terms, t0, y0, args): return terms.vf(t0, y0, args)