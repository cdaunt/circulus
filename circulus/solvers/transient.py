import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx
import equinox as eqx
from functools import partial
from typing import Tuple, Any

# --- 1. Common Logic (JIT-able Functions) ---
def _compute_history(component_groups, y0, t0, num_vars):
    """
    Computes the history vector (q_prev) at time t0.
    Shared by both Dense and Sparse solvers.
    """
    solver_dtype = y0.dtype
    q_prev = jnp.zeros(y0.shape[0], dtype=solver_dtype)

    for group in component_groups.values():
        v_locs = y0[group.var_indices]
        
        # Bind t=t0 to the physics function
        physics_at_t0 = partial(group.physics_func, t=t0)
        
        # Vectorized physics evaluation
        _, q_locs = jax.vmap(physics_at_t0)(v_locs, group.params)
        
        # Scatter add results
        q_prev = q_prev.at[group.eq_indices].add(q_locs)
        
    return q_prev

def _assemble_system_real(y_guess, component_groups, t1, dt, num_vars):
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    
    vals_list = []

    for group in component_groups.values():
        v_locs = y_guess[group.var_indices]
        def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)
        
        (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
        (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
        
        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        
        j_eff = df_l + (dq_l / dt)
        vals_list.append(j_eff.reshape(-1))

    # Ground Constraint (Stiffness)
    G_stiff = 1e9
    vals_list.append(jnp.array([G_stiff]))

    all_vals = jnp.concatenate(vals_list)
    return total_f, total_q, all_vals

def _assemble_system_complex(y_guess, component_groups, t1, dt, num_vars):
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real = y_guess[:half_size]
    y_imag = y_guess[half_size:]
    
    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)
    vals_list = []

    for group in component_groups.values():
        v_locs_real = y_real[group.var_indices]
        v_locs_imag = y_imag[group.var_indices]

        # Split Physics for Non-Holomorphic Support
        def physics_split(v_r, v_i, p):
            v = v_r + 1j * v_i
            f, q = group.physics_func(v, p, t=t1)
            return f.real, f.imag, q.real, q.imag

        # Jacobian: Returns 4 tuples of (f_r, f_i, q_r, q_i) derivs
        jac_res = jax.vmap(jax.jacfwd(physics_split, argnums=(0,1)))(
            v_locs_real, v_locs_imag, group.params
        )
        ((dfr_vr, dfr_vi), (dfi_vr, dfi_vi), (dqr_vr, dqr_vi), (dqi_vr, dqi_vi)) = jac_res
        
        # Effective Jacobian: J = df/dv + (1/dt)*dq/dv
        vals_list.append((dfr_vr + dqr_vr/dt).reshape(-1))
        vals_list.append((dfr_vi + dqr_vi/dt).reshape(-1))
        vals_list.append((dfi_vr + dqi_vr/dt).reshape(-1))
        vals_list.append((dfi_vi + dqi_vi/dt).reshape(-1))
        
        # Residuals
        fr, fi, qr, qi = jax.vmap(physics_split)(v_locs_real, v_locs_imag, group.params)
        total_f = total_f.at[group.eq_indices].add(fr).at[group.eq_indices + half_size].add(fi)
        total_q = total_q.at[group.eq_indices].add(qr).at[group.eq_indices + half_size].add(qi)

    # Ground Constraint (Stiffness)
    G_stiff = 1e9
    vals_list.append(jnp.array([G_stiff])) # Real
    vals_list.append(jnp.array([G_stiff])) # Imag

    all_vals = jnp.concatenate(vals_list)
    return total_f, total_q, all_vals


# --- 2. Newton Step Strategies ---
def _dense_newton_step_real(y_guess, args):
    (component_groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    
    total_f, total_q, all_vals = _assemble_system_real(y_guess, component_groups, t1, dt, num_vars)
    residual = total_f + (total_q - q_prev) / dt
    residual = residual.at[0].add(1e9 * y_guess[0]) # Real Ground

    sys_size = y_guess.shape[0]
    J_dense = jnp.zeros((sys_size, sys_size), dtype=residual.dtype)
    J_dense = J_dense.at[static_rows, static_cols].add(all_vals)

    delta = jnp.linalg.solve(J_dense, -residual)

    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 1.0 / (max_change + 1e-9))
    
    return y_guess + (delta * damping_factor)

def _dense_newton_step_complex(y_guess, args):
    (component_groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    
    total_f, total_q, all_vals = _assemble_system_complex(y_guess, component_groups, t1, dt, num_vars)
    
    q_prev_expanded = jnp.concatenate([q_prev.real, q_prev.imag])
    residual = total_f + (total_q - q_prev_expanded) / dt
    residual = residual.at[0].add(1e9 * y_guess[0]) # Real Ground
    
    half_size = y_guess.shape[0] // 2
    residual = residual.at[half_size].add(1e9 * y_guess[half_size]) # Imag Ground

    sys_size = y_guess.shape[0]
    J_dense = jnp.zeros((sys_size, sys_size), dtype=residual.dtype)
    J_dense = J_dense.at[static_rows, static_cols].add(all_vals)

    delta = jnp.linalg.solve(J_dense, -residual)

    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 1.0 / (max_change + 1e-9))
    
    return y_guess + (delta * damping_factor)


def _sparse_newton_step_real(y_guess, args):
    (component_groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    sys_size = y_guess.shape[0]

    total_f, total_q, all_vals = _assemble_system_real(y_guess, component_groups, t1, dt, num_vars)
    residual = total_f + (total_q - q_prev) / dt
    residual = residual.at[0].add(1e9 * y_guess[0])

    # Preconditioner
    diag_vals = jax.ops.segment_sum(
        all_vals * diag_mask, static_rows, num_segments=sys_size
    )
    inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-9, 1.0, 1.0 / diag_vals)

    def matvec(x):
        x_gathered = x[static_cols]
        products = all_vals * x_gathered
        return jax.ops.segment_sum(products, static_rows, num_segments=sys_size)

    delta_guess = -residual * inv_diag 
    delta, _ = jax.scipy.sparse.linalg.gmres(
        matvec, -residual, x0=delta_guess,
        M=lambda x: inv_diag * x, tol=1e-5, maxiter=50, restart=10
    )

    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))

    return y_guess + (delta * damping_factor)


def _sparse_newton_step_complex(y_guess, args):
    (component_groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars) = args
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2

    total_f, total_q, all_vals = _assemble_system_complex(y_guess, component_groups, t1, dt, num_vars)
    
    q_prev_expanded = jnp.concatenate([q_prev.real, q_prev.imag])
    residual = total_f + (total_q - q_prev_expanded) / dt
    residual = residual.at[0].add(1e9 * y_guess[0])
    residual = residual.at[half_size].add(1e9 * y_guess[half_size])

    # Preconditioner
    diag_vals = jax.ops.segment_sum(
        all_vals * diag_mask, static_rows, num_segments=sys_size
    )
    inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-9, 1.0, 1.0 / diag_vals)

    def matvec(x):
        x_gathered = x[static_cols]
        products = all_vals * x_gathered
        return jax.ops.segment_sum(products, static_rows, num_segments=sys_size)

    delta_guess = -residual * inv_diag 
    delta, _ = jax.scipy.sparse.linalg.gmres(
        matvec, -residual, x0=delta_guess,
        M=lambda x: inv_diag * x, tol=1e-5, maxiter=50, restart=10
    )

    max_change = jnp.max(jnp.abs(delta))
    damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))

    return y_guess + (delta * damping_factor)


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


# --- 3. Unified Solver Class ---

class TransientSolverState(eqx.Module):
    static_rows: jax.Array
    static_cols: jax.Array
    diag_mask: jax.Array | None
    history: tuple
    is_complex_mode: bool = eqx.field(static=True)

class VectorizedTransientSolver(diffrax.AbstractSolver):
    """
    Unified Transient Solver supporting both Dense and Sparse strategies.
    mode: 'dense' (Direct LU) or 'sparse' (GMRES + Jacobi).
    """
    mode: str = eqx.field(static=True)
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars = args

        # --- Pre-computed Structure ---
        all_rows_list = []
        all_cols_list = []
        for g in component_groups.values():
            all_rows_list.append(g.jac_rows.reshape(-1))
            all_cols_list.append(g.jac_cols.reshape(-1))
            
        base_rows = jnp.concatenate(all_rows_list)
        base_cols = jnp.concatenate(all_cols_list)

        # Check for Complex Mode
        is_complex = jnp.iscomplexobj(y0) or (y0.shape[0] == 2 * num_vars)
        if not is_complex:
            for group in component_groups.values():
                for leaf in jax.tree.leaves(group.params):
                    if jnp.iscomplexobj(leaf):
                        is_complex = True
                        break
        
        # Expand indices for Complex Mode (2N x 2N)
        if is_complex:
            r, c = base_rows, base_cols
            N = num_vars
            # RR, RI, IR, II, G_real, G_imag
            static_rows = jnp.concatenate([r, r, r+N, r+N, jnp.array([0]), jnp.array([N])])
            static_cols = jnp.concatenate([c, c+N, c, c+N, jnp.array([0]), jnp.array([N])])
        else:
            static_rows = jnp.concatenate([base_rows, jnp.array([0])])
            static_cols = jnp.concatenate([base_cols, jnp.array([0])])

        # Diag Mask (Only needed for Sparse, but cheap to compute)
        diag_mask = (static_rows == static_cols)

        # Ensure history is consistent with mode
        if is_complex and not jnp.iscomplexobj(y0) and y0.shape[0] == num_vars:
            y0_hist = y0.astype(jnp.complex128)
        else:
            y0_hist = y0
            
        history = (y0_hist, 1.0)
        return TransientSolverState(static_rows, static_cols, diag_mask, history, is_complex)

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state.history
        is_complex = solver_state.is_complex_mode

        y0, y_c, y_solver_init = _prepare_step_state(y0, is_complex, num_vars)

        # 1. History (t0)
        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        # 2. Newton Step
        step_args = (component_groups, t1, dt, q_prev, solver_state.static_rows, 
                     solver_state.static_cols, solver_state.diag_mask, num_vars)

        if self.mode == 'dense':
            solver_fn = _dense_newton_step_complex if is_complex else _dense_newton_step_real
        else:
            solver_fn = _sparse_newton_step_complex if is_complex else _sparse_newton_step_real

        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        
        sol = optx.fixed_point(solver_fn, solver, y_solver_init, args=step_args, max_steps=30, throw=False)

        # 3. Result & PID
        y_next_flat = sol.value
        if is_complex and jnp.iscomplexobj(y0):
            y_next = y_next_flat[:num_vars] + 1j * y_next_flat[num_vars:]
        else:
            y_next = y_next_flat

        rate_prev = (y0 - y_prev_step) / dt_prev
        y_pred = y0 + rate_prev * dt
        y_error = y_next - y_pred

        new_state = eqx.tree_at(lambda s: s.history, solver_state, (y0, dt))
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )

        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
