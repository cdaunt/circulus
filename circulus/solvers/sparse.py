import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx
from functools import partial

# --- JIT Compiled Kernels (Same as before) ---
@partial(jax.jit, static_argnames=['num_vars'])
def _sparse_matvec_jit(row_idx, col_idx, data, x, num_vars):
    prods = data * x[col_idx]
    return jnp.zeros(num_vars).at[row_idx].add(prods)

@partial(jax.jit, static_argnames=['num_vars'])
def _extract_diagonal_jit(row_idx, col_idx, data, num_vars):
    diag_mask = (row_idx == col_idx)
    return jnp.zeros(num_vars).at[row_idx].add(jnp.where(diag_mask, data, 0.0))

@jax.jit
def _compute_preconditioner_jit(diag_vals):
    return jnp.where(jnp.abs(diag_vals) < 1e-8, 1.0, 1.0 / diag_vals)


@jax.jit
def _check_convergence_jit(delta, residual_check_norm):
    """JIT-compiled convergence check."""
    is_diverged = jnp.isnan(residual_check_norm) | (residual_check_norm > 1.0)
    return jax.lax.select(is_diverged, jnp.zeros_like(delta), delta)


class UltraFastSparseSolverVectorized(diffrax.AbstractSolver):
    """
    Maximum performance: Requires pre-computed sparse structure.
    Use this if you're solving the same topology repeatedly.
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        # Could cache sparse structure here for repeated solves
        return None 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        dt = t1 - t0
        
        # History computation
        q_prev = jnp.zeros(num_vars)
        for block in blocks:
            v_loc = y0[block.var_indices]
            _, q_loc = jax.vmap(block.physics_func)(v_loc, block.params)
            q_prev = q_prev.at[block.eq_indices].add(q_loc)

        # Pre-compute static structure once
        static_rows = jnp.concatenate([b.jac_rows for b in blocks] + [jnp.array([0])])
        static_cols = jnp.concatenate([b.jac_cols for b in blocks] + [jnp.array([0])])

        def sparse_newton_step(y_guess, args):
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            jac_vals_list = []
            
            # --- 1. Vectorized Assembly Loop ---
            # Instead of looping 10,000 times for each resistor instance, 
            # we loop ONCE per component TYPE (e.g., 1 iteration for all Resistors).
            for group in component_groups:
                # 1a. Gather Voltages
                # group.var_indices has shape (Num_Components, Nodes_Per_Comp)
                v_locs = y_guess[group.var_indices]
                
                # 1b. Massively Parallel Physics Evaluation
                # jax.vmap automatically vectorizes over the '0' axis of v_locs and group.params
                (f_l, q_l) = jax.vmap(group.physics_func)(v_locs, group.params)
                
                # 1c. Massively Parallel Jacobian Evaluation
                # jacfwd computes the Jacobian for the entire batch efficiently
                (df_l, dq_l) = jax.vmap(jax.jacfwd(group.physics_func))(v_locs, group.params)
                
                # 1d. Scatter Add Results (Vectorized)
                # We scatter the results back into the global residual vectors.
                # .at[].add handles duplicate indices correctly (e.g. KCL sums at shared nodes).
                total_f = total_f.at[group.eq_indices].add(f_l)
                total_q = total_q.at[group.eq_indices].add(q_l)
                
                # 1e. Store Jacobian Values
                # Effective Jacobian: J = df/dv + (1/dt) * dq/dv
                # We flatten the result to 1D to match the pre-computed static coordinate arrays
                j_eff = df_l + (dq_l / dt)
                jac_vals_list.append(j_eff.reshape(-1))
            
            # --- 2. Residual Calculation ---
            # Residual Equation: F(v) + (Q(v) - Q_prev)/dt = 0
            residual = total_f + (total_q - q_prev) / dt
            
            # Boundary Condition (Ground at Node 0)
            G_stiff = 1e6
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            # --- 3. Flatten Jacobian for Solver ---
            # Concatenate all the 1D arrays we collected into one massive values array
            # This aligns perfectly with static_rows/static_cols calculated in `init` or `step` preamble
            all_vals = jnp.concatenate(jac_vals_list + [jnp.array([G_stiff])])
            
            # --- 4. Preconditioner Setup (JIT Compiled) ---
            # _extract_diagonal_jit uses the pre-calculated static_rows/cols (O(1) python overhead)
            diag_vals = _extract_diagonal_jit(static_rows, static_cols, all_vals, num_vars)
            safe_diag_inv = _compute_preconditioner_jit(diag_vals)
            
            # Define matrix-vector product using JIT kernel (Bypasses BCOO overhead)
            def matvec(x):
                return _sparse_matvec_jit(static_rows, static_cols, all_vals, x, num_vars)
            
            def precond(x):
                return safe_diag_inv * x

            # --- 5. Initial Guess for Newton Step ---
            # A simple diagonal-scaled step helps GMRES start closer to the solution
            delta_guess = -safe_diag_inv * residual

            # --- 6. Linear Solve (GMRES) ---
            delta, _ = jax.scipy.sparse.linalg.gmres(
                matvec,
                -residual,
                x0=delta_guess,
                M=precond,
                tol=1e-8,
                maxiter=500,  # Lower count per step is often fine for Newton
                restart=30
            )
            
            # --- 7. Convergence Check & Safety ---
            # Re-calculate residual norm manually to verify convergence
            check_norm = jnp.linalg.norm(matvec(delta) + residual)
            
            # Safe update: if solver diverged, return 0.0 (stagnation) rather than NaNs
            # This allows the non-linear solver (Optimistix) to detect stagnation cleanly
            delta_safe = _check_convergence_jit(delta, check_norm)
            
            return y_guess + delta_safe
            # ...
            for group in component_groups:
                v_locs = y_guess[group.var_indices] # Shape: (N, num_nodes)
                
                # Since group.params contains arrays (N,), vmap handles them naturally
                (f_l, q_l) = jax.vmap(group.physics_func)(v_locs, group.params)
                
                # ... rest is the same ...
                (df_l, dq_l) = jax.vmap(jax.jacfwd(block.physics_func))(v_loc, block.params)
                
                total_f = total_f.at[block.eq_indices].add(f_l)
                total_q = total_q.at[block.eq_indices].add(q_l)
                jac_vals_all.append((df_l + dq_l / dt).reshape(-1))
            
            residual = total_f + (total_q - q_prev) / dt
            residual = residual.at[0].add(1e6 * y_guess[0])
            
            all_vals = jnp.concatenate([jnp.concatenate(jac_vals_all), jnp.array([1e6])])
            
            # Everything after assembly can be JIT-compiled
            diag_vals = _extract_diagonal_jit(static_rows, static_cols, all_vals, num_vars)
            safe_diag_inv = _compute_preconditioner_jit(diag_vals)
            
            def matvec(x):
                return _sparse_matvec_jit(static_rows, static_cols, all_vals, x, num_vars)
            
            delta, _ = jax.scipy.sparse.linalg.gmres(
                matvec,
                -residual,
                x0=-0.01 * safe_diag_inv * residual,
                M=lambda x: safe_diag_inv * x,
                tol=1e-8,
                maxiter=1000,
                restart=50
            )
            
            check_norm = jnp.linalg.norm(matvec(delta) + residual)
            delta_safe = _check_convergence_jit(delta, check_norm)
            
            return y_guess + delta_safe

        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(sparse_newton_step, solver, y0, max_steps=20, throw=False)
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )
        
        return sol.value, None, {"y0": y0, "y1": sol.value}, None, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
    

class VectorizedSparseSolver(diffrax.AbstractSolver):


    """
    High-Performance Solver: 
    - Replaces Python loops with JAX vectorization.
    - Requires 'args' to be grouped by component type.
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        # UNPACKING ARGS:
        # Instead of a list of blocks, we expect a list of 'ComponentGroups'.
        # Each group contains ALL resistors, or ALL diodes, stacked together.
        component_groups, num_vars = args
        dt = t1 - t0
        
        # --- 1. Vectorized History Calculation ---
        # We calculate q_prev for ALL components in parallel groups
        q_prev = jnp.zeros(num_vars)
        
        for group in component_groups:
            # Gather all voltages for this component type at once
            # group.var_indices is shape (num_components, num_nodes_per_comp)
            v_locs = y0[group.var_indices]

            physics_at_t0 = partial(group.physics_func, t=t0)
            
            # Single vmap for thousands of components
            _, q_locs = jax.vmap(physics_at_t0)(v_locs, group.params)
            
            # Scatter add results (vectorized)
            q_prev = q_prev.at[group.eq_indices].add(q_locs)

        # --- 2. Pre-computed Structure (Static) ---
        # This should ideally be done in 'init', but is cheap enough here if cached
        all_rows_list = []
        all_cols_list = []
        for g in component_groups:
            all_rows_list.append(g.jac_rows.reshape(-1))
            all_cols_list.append(g.jac_cols.reshape(-1))
            
        static_rows = jnp.concatenate(all_rows_list + [jnp.array([0])])
        static_cols = jnp.concatenate(all_cols_list + [jnp.array([0])])

        # --- 3. The Vectorized Newton Step ---
        def sparse_newton_step(y_guess, args):
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            jac_vals_list = []
            
            # --- Vectorized Assembly ---
            # Loop over TYPES (e.g. 3 loops) instead of INSTANCES (e.g. 10,000 loops)
            for group in component_groups:
                v_locs = y_guess[group.var_indices]

                physics_at_t1 = partial(group.physics_func, t=t1)
                
                # Massively parallel physics eval
                (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
                
                # Massively parallel Jacobian eval
                # Note: group.physics_func must handle batched inputs
                (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
                
                # Accumulate Residuals
                total_f = total_f.at[group.eq_indices].add(f_l)
                total_q = total_q.at[group.eq_indices].add(q_l)
                
                # Flatten Jacobians
                j_eff = df_l + (dq_l / dt)
                jac_vals_list.append(j_eff.reshape(-1))
            
            # --- Solve Preparation ---
            residual = total_f + (total_q - q_prev) / dt
            
            # Boundary Condition
            G_stiff = 1e6
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            # Concatenate all Jacobian values
            all_vals = jnp.concatenate(jac_vals_list + [jnp.array([G_stiff])])
            
            # --- JIT-Compiled Linear Solve (Same as before) ---
            diag_vals = _extract_diagonal_jit(static_rows, static_cols, all_vals, num_vars)
            safe_diag_inv = _compute_preconditioner_jit(diag_vals)
            
            def matvec(x):
                return _sparse_matvec_jit(static_rows, static_cols, all_vals, x, num_vars)
            
            # Better initial guess for Newton: Scale residual by diagonal
            delta_guess = -safe_diag_inv * residual
            
            delta, _ = jax.scipy.sparse.linalg.gmres(
                matvec,
                -residual,
                x0=delta_guess, 
                M=lambda x: safe_diag_inv * x,
                tol=1e-8,
                maxiter=500, # Lower maxiter per step often suffices
                restart=30
            )
            
            # Safety Check
            check_norm = jnp.linalg.norm(matvec(delta) + residual)
            # Divergence check with looser tolerance for intermediate steps
            delta_safe = jax.lax.select(
                (check_norm > 1.0) | jnp.isnan(check_norm), 
                jnp.zeros_like(delta), 
                delta
            )
            
            return y_guess + delta_safe

        # Solver Loop
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(sparse_newton_step, solver, y0, max_steps=20, throw=False)
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )
        
        return sol.value, None, {"y0": y0, "y1": sol.value}, None, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
    
