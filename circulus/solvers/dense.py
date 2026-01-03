import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx
from functools import partial


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

        history = (y0, 1.0)

        return (static_rows, static_cols, history) 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        # Unpack: args is a list of ComponentGroups, not individual blocks
        component_groups, num_vars = args
        static_rows, static_cols, history = solver_state
        y_prev_step, dt_prev = history
        dt = t1 - t0
        
        # --- 1. Vectorized History Calculation (at t0) ---
        q_prev = jnp.zeros(num_vars)
        
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
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            
            # Initialize Dense Jacobian
            J_dense = jnp.zeros((num_vars, num_vars))
            jac_vals_list = []

            # --- Vectorized Assembly ---
            for group_name, group in component_groups.items():
                v_locs = y_guess[group.var_indices]
                
                def physics_at_t1(v, p):
                    return group.physics_func(v, p, t=t1)
                
                # --- CORRECTION HERE ---
                # 1. Evaluate the Physics (Primal)
                (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
                
                # 2. Evaluate the Jacobian (Tangent)
                # We strictly need the full matrix for Direct Solves
                (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
                
                # Accumulate Residuals
                total_f = total_f.at[group.eq_indices].add(f_l)
                total_q = total_q.at[group.eq_indices].add(q_l)
                
                # Calculate Effective Jacobian Values
                j_eff = df_l + (dq_l / dt)
                jac_vals_list.append(j_eff.reshape(-1))

            # --- Solve Preparation ---
            residual = total_f + (total_q - q_prev) / dt
            
            # Ground Constraint
            G_stiff = 1e9
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            # --- optimization 2: Direct Dense Scatter (Preserved) ---
            all_vals = jnp.concatenate(jac_vals_list + [jnp.array([G_stiff])])
            
            # Scatter values directly into the dense matrix
            J_dense = J_dense.at[static_rows, static_cols].add(all_vals)
            
            # Inside dense_newton_step ...
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
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(dense_newton_step, solver, y0, max_steps=30, throw=False)

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )

        # Error calculation for PID Controller
        rate_prev = (y0 - y_prev_step) / dt_prev
        y_pred = y0 + rate_prev * dt
        
        # The error is the difference between the Newton solution and the Linear Prediction
        # This scales with curvature (d2y/dt2), not slope (dy/dt)
        y_next = sol.value
        y_error = y_next - y_pred

        # Update History for next step
        new_history = (y0, dt)
        new_state = (static_rows, static_cols,  new_history)
        
        # We must return the *next* step's value, not the function evaluation
        return sol.value, y_error, {"y0": y0, "y1": sol.value}, new_state, result
    
    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
    