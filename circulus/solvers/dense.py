import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx

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
        return None 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        # Unpack: args is a list of ComponentGroups, not individual blocks
        component_groups, num_vars = args
        dt = t1 - t0
        
        # --- 1. Vectorized History Calculation (at t0) ---
        q_prev = jnp.zeros(num_vars)
        
        for group in component_groups:
            v_locs = y0[group.var_indices]
            
            # Bind t=t0 to the physics function
            def physics_at_t0(v, p):
                return group.physics_func(v, p, t=t0)
            
            # Single vmap over all components of this type
            _, q_locs = jax.vmap(physics_at_t0)(v_locs, group.params)
            
            # Scatter add results
            q_prev = q_prev.at[group.eq_indices].add(q_locs)

        # --- 2. Pre-computed Structure (Static) ---
        # We collect all indices once to avoid concatenation inside the Newton loop
        all_rows_list = []
        all_cols_list = []
        for g in component_groups:
            all_rows_list.append(g.jac_rows.reshape(-1))
            all_cols_list.append(g.jac_cols.reshape(-1))
            
        static_rows = jnp.concatenate(all_rows_list + [jnp.array([0])])
        static_cols = jnp.concatenate(all_cols_list + [jnp.array([0])])

        # --- 3. The Vectorized Dense Newton Step (at t1) ---
        def dense_newton_step(y_guess, args):
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            jac_vals_list = []
            
            # --- Vectorized Assembly ---
            for group in component_groups:
                v_locs = y_guess[group.var_indices]
                
                # Bind t=t1 to the physics function
                def physics_at_t1(v, p):
                    return group.physics_func(v, p, t=t1)
                
                # Massively parallel physics eval
                (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
                
                # Massively parallel Jacobian eval
                (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
                
                # Accumulate
                total_f = total_f.at[group.eq_indices].add(f_l)
                total_q = total_q.at[group.eq_indices].add(q_l)
                
                # Jacobian Entries
                j_eff = df_l + (dq_l / dt)
                jac_vals_list.append(j_eff.reshape(-1))
            
            # --- Solve Preparation ---
            residual = total_f + (total_q - q_prev) / dt
            
            # Boundary Condition (Ground Node 0)
            # 1e9 is safe for Dense solvers (64-bit float limit is much higher)
            G_stiff = 1e9
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            all_vals = jnp.concatenate(jac_vals_list + [jnp.array([G_stiff])])
            
            # --- Sparse Assembly -> Dense Solve ---
            # We use BCOO to handle the "Scatter-Add" logic efficiently
            J_sparse = sparse.BCOO(
                (all_vals, jnp.stack([static_rows, static_cols], axis=1)),
                shape=(num_vars, num_vars)
            )
            
            # Convert to Dense Matrix (The key difference)
            J_dense = J_sparse.todense()
            
            # Dense LU Solve (Robust)
            delta = jnp.linalg.solve(J_dense, -residual)
            
            return y_guess + delta

        # --- 4. Solver Loop (Optimistix) ---
        # We use FixedPointIteration to drive our custom Newton step
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        
        # Max steps can be lower for dense solvers as they are exact linear solves
        sol = optx.fixed_point(dense_newton_step, solver, y0, max_steps=20, throw=False)
        
        # --- 5. Result Handling ---
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )
        
        return sol.value, None, {"y0": y0, "y1": sol.value}, None, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
    