import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx
import equinox as eqx
from functools import partial


class DenseSolverState(eqx.Module):
    static_rows: jax.Array
    static_cols: jax.Array
    history: tuple
    is_complex_mode: bool = eqx.field(static=True)

class VectorizedDenseSolver(diffrax.AbstractSolver):
    """
    High-Performance Dense Solver:
    - Uses Vectorized Group Assembly (like VectorizedSparseSolver).
    - Uses Dense LU Decomposition (jnp.linalg.solve) for maximum stability.
    - Best for small-to-medium circuits (< 2000 nodes) where stability > memory.
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):

        component_groups, num_vars = args

        # --- 2. Pre-computed Structure (Static) ---
        # We collect all indices once to avoid concatenation inside the Newton loop
        all_rows_list = []
        all_cols_list = []
        for g_name, g in component_groups.items():
            all_rows_list.append(g.jac_rows.reshape(-1))
            all_cols_list.append(g.jac_cols.reshape(-1))
            
        static_rows = jnp.concatenate(all_rows_list + [jnp.array([0])])
        static_cols = jnp.concatenate(all_cols_list + [jnp.array([0])])
        
        # Check for Complex Mode (via y0 OR parameters)
        is_complex = jnp.iscomplexobj(y0)
        if not is_complex:
            for group in component_groups.values():
                for leaf in jax.tree.leaves(group.params):
                    if jnp.iscomplexobj(leaf):
                        is_complex = True
                        break
        
        base_rows = jnp.concatenate(all_rows_list)
        base_cols = jnp.concatenate(all_cols_list)
        
        if is_complex:
            # Expand to 2N x 2N
            r, c = base_rows, base_cols
            N = num_vars
            static_rows = jnp.concatenate([r, r, r+N, r+N, jnp.array([0]), jnp.array([N])])
            static_cols = jnp.concatenate([c, c+N, c, c+N, jnp.array([0]), jnp.array([N])])
        else:
            static_rows = jnp.concatenate([base_rows, jnp.array([0])])
            static_cols = jnp.concatenate([base_cols, jnp.array([0])])

        if is_complex:
            y0 = y0.astype(jnp.complex128)

        # Store is_complex decision in solver_state to ensure step() agrees with init()
        history = (y0, 1.0)
        return DenseSolverState(
            static_rows=static_rows, 
            static_cols=static_cols, 
            history=history, 
            is_complex_mode=is_complex
        )

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        # Unpack: args is a list of ComponentGroups, not individual blocks
        component_groups, num_vars = args
        static_rows = solver_state.static_rows
        static_cols = solver_state.static_cols
        history = solver_state.history
        is_complex_mode = solver_state.is_complex_mode
        
        y_prev_step, dt_prev = history
        dt = t1 - t0
        
        # Ensure y0 matches the mode decided in init()
        # If init() saw complex params, we MUST upgrade y0 to complex here,
        # otherwise the solver will try to fit a 2N system into N variables.
        if is_complex_mode:
            y0 = y0.astype(jnp.complex128)
        
        solver_dtype = y0.dtype
        
        # --- 1. Vectorized History Calculation (at t0) ---
        q_prev = jnp.zeros(num_vars, dtype=solver_dtype)
        
        for group_name, group in component_groups.items():
            v_locs = y0[group.var_indices]
            
            # Bind t=t0 to the physics function
            physics_at_t0 = partial(group.physics_func, t=t0)
            
            # Single vmap over all components of this type
            _, q_locs = jax.vmap(physics_at_t0)(v_locs, group.params)
            
            # Scatter add results
            q_prev = q_prev.at[group.eq_indices].add(q_locs)


# --- 2. The Vectorized Dense Newton Step ---
        def dense_newton_step(y_guess, args):
            # y_guess is always REAL here (flattened if complex mode)
            sys_size = 2 * num_vars if is_complex_mode else num_vars
            
            # We build a Real system (2N if complex)
            total_f = jnp.zeros(sys_size, dtype=jnp.float64 if is_complex_mode else solver_dtype)
            total_q = jnp.zeros(sys_size, dtype=jnp.float64 if is_complex_mode else solver_dtype)
            J_dense = jnp.zeros((sys_size, sys_size), dtype=jnp.float64 if is_complex_mode else solver_dtype)
            
            vals_rr, vals_ri, vals_ir, vals_ii = [], [], [], []

            # --- Vectorized Assembly ---
            for group_name, group in component_groups.items():
                
                if is_complex_mode:
                    # Reconstruct Complex State for Physics
                    y_real = y_guess[:num_vars]
                    y_imag = y_guess[num_vars:]
                    y_c = y_real + 1j * y_imag
                    v_locs = y_c[group.var_indices]

                    # Split Physics for Non-Holomorphic Support
                    def physics_split(v_r, v_i, p):
                        v = v_r + 1j * v_i
                        f, q = group.physics_func(v, p, t=t1)
                        return f.real, f.imag, q.real, q.imag

                    # Jacobian: Returns 4 tuples of (f_r, f_i, q_r, q_i) derivs
                    jac_res = jax.vmap(jax.jacfwd(physics_split, argnums=(0,1)))(
                        v_locs.real, v_locs.imag, group.params
                    )
                    # Unpack the massive tuple structure
                    ((dfr_vr, dfr_vi), (dfi_vr, dfi_vi), (dqr_vr, dqr_vi), (dqi_vr, dqi_vi)) = jac_res
                    
                    # Effective Jacobian: J = df/dv + (1/dt)*dq/dv
                    vals_rr.append((dfr_vr + dqr_vr/dt).reshape(-1))
                    vals_ri.append((dfr_vi + dqr_vi/dt).reshape(-1))
                    vals_ir.append((dfi_vr + dqi_vr/dt).reshape(-1))
                    vals_ii.append((dfi_vi + dqi_vi/dt).reshape(-1))
                    
                    # Residuals
                    fr, fi, qr, qi = jax.vmap(physics_split)(v_locs.real, v_locs.imag, group.params)
                    total_f = total_f.at[group.eq_indices].add(fr).at[group.eq_indices + num_vars].add(fi)
                    total_q = total_q.at[group.eq_indices].add(qr).at[group.eq_indices + num_vars].add(qi)
                    
                else:
                    # Standard Real
                    v_locs = y_guess[group.var_indices]
                    def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)
                    (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
                    (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
                    
                    total_f = total_f.at[group.eq_indices].add(f_l)
                    total_q = total_q.at[group.eq_indices].add(q_l)
                    vals_rr.append((df_l + dq_l/dt).reshape(-1))

            # --- Solve Preparation ---
            # q_prev must be expanded if complex
            if is_complex_mode:
                q_prev_expanded = jnp.concatenate([q_prev.real, q_prev.imag])
                residual = total_f + (total_q - q_prev_expanded) / dt
            else:
                residual = total_f + (total_q - q_prev) / dt
            
            # Ground Constraint
            G_stiff = 1e9
            residual = residual.at[0].add(G_stiff * y_guess[0]) # Real part
            if is_complex_mode:
                residual = residual.at[num_vars].add(G_stiff * y_guess[num_vars]) # Imag Ground
            
            # --- optimization 2: Direct Dense Scatter (Preserved) ---
            if is_complex_mode:
                # Order: RR, RI, IR, II, G_stiff_Real, G_stiff_Imag
                all_vals = jnp.concatenate(vals_rr + vals_ri + vals_ir + vals_ii + [jnp.array([G_stiff]), jnp.array([G_stiff])])
            else:
                all_vals = jnp.concatenate(vals_rr + [jnp.array([G_stiff])])
            
            # Scatter values directly into the dense matrix
            J_dense = J_dense.at[static_rows, static_cols].add(all_vals)
            
            delta = jnp.linalg.solve(J_dense, -residual)
            
            # --- Soft Damping (Better than Hard Clipping) ---
            # Calculate how "aggressive" this step is
            max_change = jnp.max(jnp.abs(delta))

            damping_limit = 1.0 
            damping_factor = jnp.minimum(1.0, damping_limit / (max_change + 1e-9))

            # Apply scaling. This preserves the 'direction' of the update, 
            # which is better for the PID controller than hard clipping.
            delta_damped = delta * damping_factor

            return y_guess + delta_damped

        # --- 3. Solver Loop ---
        # Prepare Initial Guess (Flattened if complex)
        if is_complex_mode:
            y0_solver = jnp.concatenate([y0.real, y0.imag])
        else:
            y0_solver = y0
            
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(dense_newton_step, solver, y0_solver, max_steps=30, throw=False)

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )

        # Reconstruct Result
        if is_complex_mode:
            y_next = sol.value[:num_vars] + 1j * sol.value[num_vars:]
        else:
            y_next = sol.value

        # Error calculation for PID Controller
        rate_prev = (y0 - y_prev_step) / dt_prev
        y_pred = y0 + rate_prev * dt
        
        # The error is the difference between the Newton solution and the Linear Prediction
        # This scales with curvature (d2y/dt2), not slope (dy/dt)
        y_error = y_next - y_pred

        # Update History for next step
        new_history = (y0, dt)
        new_state = DenseSolverState(
            static_rows=static_rows, 
            static_cols=static_cols, 
            history=new_history, 
            is_complex_mode=is_complex_mode
        )
        
        # We must return the *next* step's value, not the function evaluation
        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result
    
    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
    