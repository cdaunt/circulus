import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx
from functools import partial
from circulus.solvers.common import _extract_diagonal, _sparse_matvec

@jax.jit
def _compute_preconditioner(diag_vals):
    return jnp.where(jnp.abs(diag_vals) < 1e-8, 1.0, 1.0 / diag_vals)


class VectorizedSparseSolver(diffrax.AbstractSolver):
    """
    High-Performance Sparse Solver (GMRES + Jacobi Preconditioner):
    - Uses jax.ops.segment_sum for ultra-fast Matrix-Vector products.
    - Implements Soft Damping for stability.
    - Supports Adaptive Time Stepping (PID).
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
            return 1

    def init(self, terms, t0, t1, y0, args):
        component_groups, num_vars = args
        
        # --- 1. Pre-compute Structure ---
        all_rows_list = []
        all_cols_list = []
        for g in component_groups:
            all_rows_list.append(g.jac_rows.reshape(-1))
            all_cols_list.append(g.jac_cols.reshape(-1))
            
        # Add Ground Stiffener Index (0,0)
        static_rows = jnp.concatenate(all_rows_list + [jnp.array([0])])
        static_cols = jnp.concatenate(all_cols_list + [jnp.array([0])])
        
        # --- Pre-calculate Preconditioner Mask ---
        # Used to extract the diagonal quickly using segment_sum
        diag_mask = (static_rows == static_cols)
        
        return (static_rows, static_cols, diag_mask)

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        static_rows, static_cols, diag_mask = solver_state
        dt = t1 - t0
        
        # --- 1. History (t0) ---
        q_prev = jnp.zeros(num_vars)
        for group in component_groups:
            v_locs = y0[group.var_indices]
            physics_at_t0 = partial(group.physics_func, t=t0)
            _, q_locs = jax.vmap(physics_at_t0)(v_locs, group.params)
            q_prev = q_prev.at[group.eq_indices].add(q_locs)

        # --- 2. Sparse Newton Step (t1) ---
        def sparse_newton_step(y_guess, args):
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            jac_vals_list = []
            
            # --- Vectorized Assembly ---
            for group in component_groups:
                v_locs = y_guess[group.var_indices]
                
                def physics_at_t1(v, p):
                    return group.physics_func(v, p, t=t1)
                
                # Evaluate Physics & Jacobian
                (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
                (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
                
                total_f = total_f.at[group.eq_indices].add(f_l)
                total_q = total_q.at[group.eq_indices].add(q_l)
                
                j_eff = df_l + (dq_l / dt)
                jac_vals_list.append(j_eff.reshape(-1))
            
            # --- Residual Calculation ---
            residual = total_f + (total_q - q_prev) / dt
            
            # Ground Constraint
            G_stiff = 1e6
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            # Flatten Matrix Values
            all_vals = jnp.concatenate(jac_vals_list + [jnp.array([G_stiff])])
            
            # --- OPTIMIZATION 1: Fast Diagonal Extraction ---
            # Instead of a custom function, we use segment_sum with the mask we made in init.
            # This handles summing duplicate diagonal entries (parallel resistors) automatically.
            diag_vals = jax.ops.segment_sum(
                all_vals * diag_mask, static_rows, num_segments=num_vars
            )
            
            # Compute Preconditioner (Inverse Diagonal)
            # Add epsilon to prevent div/0 on floating nodes
            inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-9, 1.0, 1.0 / diag_vals)
            
            # --- OPTIMIZATION 2: Fast Matvec ---
            # Replacing _sparse_matvec with inline segment_sum avoids overhead
            def matvec(x):
                x_gathered = x[static_cols]
                products = all_vals * x_gathered
                return jax.ops.segment_sum(products, static_rows, num_segments=num_vars)

            # --- GMRES Solve ---
            # Use Preconditioned Residual as initial guess (1-step Jacobi)
            delta_guess = -residual * inv_diag 
            
            delta, _ = jax.scipy.sparse.linalg.gmres(
                matvec,
                -residual,
                x0=delta_guess,
                M=lambda x: inv_diag * x, # Apply preconditioner
                tol=1e-5, # Slightly loose tolerance is fine for Newton steps
                maxiter=50,
                restart=10
            )
            
            # --- OPTIMIZATION 3: Soft Damping ---
            # Replaced your "divergence check" (which zeros the update) with scaling.
            # This is much more robust for nonlinear circuits.
            max_change = jnp.max(jnp.abs(delta))
            damping_factor = jnp.minimum(1.0, 2.0 / (max_change + 1e-9))
            
            return y_guess + (delta * damping_factor)

        # --- 3. Solver Loop ---
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(sparse_newton_step, solver, y0, max_steps=20, throw=False)
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )
        
        # --- PID Error Estimate ---
        y_error = sol.value - y0
        
        return sol.value, y_error, {"y0": y0, "y1": sol.value}, solver_state, result
    
    def func(self, terms, t0, y0, args):
            return terms.vf(t0, y0, args)