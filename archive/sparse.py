import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx
from functools import partial

class SparseNLSolver(diffrax.AbstractSolver):
    """
    Implements Backward Euler using GMRES with Safe Jacobi Preconditioning.
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        blocks, num_vars = args
        dt = t1 - t0
        
        # 1. Prepare History
        q_prev = jnp.zeros(num_vars)
        for block in blocks:
            v_loc = y0[block.var_indices]
            _, q_loc = jax.vmap(block.physics_func)(v_loc, block.params)
            q_prev = q_prev.at[block.eq_indices].add(q_loc)

        # --- HELPER: Manual Sparse Matrix-Vector Product ---
        # This bypasses jax.experimental.sparse.bcoo bugs
        def sparse_matvec(bcoo_indices, bcoo_data, x, shape):
            # Gather x values involved in multiplication
            row_idx = bcoo_indices[:, 0]
            col_idx = bcoo_indices[:, 1]
            
            # Element-wise multiplication
            # val * x[col]
            prods = bcoo_data * x[col_idx]
            
            # Scatter sum into result (Accumulate rows)
            # This handles duplicate indices automatically!
            return jnp.zeros(shape).at[row_idx].add(prods)

        def sparse_newton_step(y_guess, args):
            # ... [Assembly code same as before] ...
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            jac_vals_all = [] 
            
            for block in blocks:
                v_loc = y_guess[block.var_indices]
                (f_l, q_l) = jax.vmap(block.physics_func)(v_loc, block.params)
                (df_l, dq_l) = jax.vmap(jax.jacfwd(block.physics_func))(v_loc, block.params)
                total_f = total_f.at[block.eq_indices].add(f_l)
                total_q = total_q.at[block.eq_indices].add(q_l)
                j_eff = df_l + (dq_l / dt)
                jac_vals_all.append(j_eff.reshape(-1))
            
            residual = total_f + (total_q - q_prev) / dt
            
            # Boundary Conditions
            G_stiff = 1e6
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            # Flatten Arrays for Sparse Matrix
            all_rows = jnp.concatenate([b.jac_rows for b in blocks] + [jnp.array([0])])
            all_cols = jnp.concatenate([b.jac_cols for b in blocks] + [jnp.array([0])])
            all_vals = jnp.concatenate([jnp.concatenate(jac_vals_all), jnp.array([G_stiff])])
            
            # Stack indices for BCOO format
            indices = jnp.stack([all_rows, all_cols], axis=1)

            # --- Preconditioner (Safe Jacobi) ---
            diag_mask = (all_rows == all_cols)
            raw_diag = jnp.zeros(num_vars).at[all_rows].add(
                jnp.where(diag_mask, all_vals, 0.0)
            )
            safe_diag_inv = jnp.where(jnp.abs(raw_diag) < 1e-6, 1.0, 1.0 / raw_diag)
            
            def precond(x):
                return safe_diag_inv * x
            
            # --- Solve with GMRES (Manual MatVec) ---
            # We pass our custom function instead of J_sparse @ x
            delta, info = jax.scipy.sparse.linalg.gmres(
                lambda x: sparse_matvec(indices, all_vals, x, num_vars),  # <--- FIX HERE
                -residual,
                x0=jnp.zeros_like(residual),
                M=precond,
                tol=1e-8,
                maxiter=1000,
                restart=50
            )
            
            # Convergence Check
            # Re-calculate residual using the same manual matvec
            check_res = sparse_matvec(indices, all_vals, delta, num_vars) + residual
            check_norm = jnp.linalg.norm(check_res)
            
            is_diverged = jnp.isnan(check_norm) | (check_norm > 1.0)
            delta_safe = jax.lax.select(is_diverged, jnp.zeros_like(delta), delta)
            
            return y_guess + delta_safe

        # 3. Non-Linear Loop
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

class SparseNLSolver2(diffrax.AbstractSolver):
    """
    Improved Backward Euler using GMRES with enhanced stability.
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        blocks, num_vars = args
        dt = t1 - t0
        
        # 1. Prepare History
        q_prev = jnp.zeros(num_vars)
        for block in blocks:
            v_loc = y0[block.var_indices]
            _, q_loc = jax.vmap(block.physics_func)(v_loc, block.params)
            q_prev = q_prev.at[block.eq_indices].add(q_loc)

        # --- Optimized Sparse Matrix-Vector Product ---
        def sparse_matvec(indices, data, x):
            """Efficient sparse matvec with automatic duplicate handling."""
            col_idx = indices[:, 1]
            row_idx = indices[:, 0]
            prods = data * x[col_idx]
            return jnp.zeros(num_vars).at[row_idx].add(prods)

        def sparse_newton_step(y_guess, args):
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            jac_vals_all = [] 
            
            # --- Assembly Loop ---
            for block in blocks:
                v_loc = y_guess[block.var_indices]
                (f_l, q_l) = jax.vmap(block.physics_func)(v_loc, block.params)
                (df_l, dq_l) = jax.vmap(jax.jacfwd(block.physics_func))(v_loc, block.params)
                
                total_f = total_f.at[block.eq_indices].add(f_l)
                total_q = total_q.at[block.eq_indices].add(q_l)
                
                j_eff = df_l + (dq_l / dt)
                jac_vals_all.append(j_eff.reshape(-1))
            
            residual = total_f + (total_q - q_prev) / dt
            
            # Boundary Conditions
            G_stiff = 1e6
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            # Flatten Arrays for Sparse Matrix
            all_rows = jnp.concatenate([b.jac_rows for b in blocks] + [jnp.array([0])])
            all_cols = jnp.concatenate([b.jac_cols for b in blocks] + [jnp.array([0])])
            all_vals = jnp.concatenate([jnp.concatenate(jac_vals_all), jnp.array([G_stiff])])
            
            indices = jnp.stack([all_rows, all_cols], axis=1)

            # --- IMPROVEMENT 1: Better Preconditioner ---
            # Instead of just diagonal, consider ILU-like approximation
            diag_mask = (all_rows == all_cols)
            raw_diag = jnp.zeros(num_vars).at[all_rows].add(
                jnp.where(diag_mask, all_vals, 0.0)
            )
            
            # More robust diagonal regularization
            diag_abs = jnp.abs(raw_diag)
            diag_scale = jnp.maximum(diag_abs, 1e-10)  # Better than fixed threshold
            safe_diag_inv = 1.0 / (raw_diag + 1e-12 * jnp.sign(raw_diag))
            
            # Detect badly conditioned entries
            is_small = diag_abs < 1e-8
            safe_diag_inv = jnp.where(is_small, 1.0, safe_diag_inv)
            
            def precond(x):
                return safe_diag_inv * x
            
            # --- IMPROVEMENT 2: Adaptive Tolerance ---
            # Scale tolerance based on problem size and residual
            residual_scale = jnp.linalg.norm(residual)
            adaptive_tol = jnp.maximum(1e-10, 1e-6 * residual_scale / jnp.sqrt(num_vars))
            
            # --- IMPROVEMENT 3: Better Initial Guess ---
            # Use scaled residual as initial guess (better than zeros)
            x0_guess = -0.01 * precond(residual)
            
            # --- Solve with GMRES ---
            delta, info = jax.scipy.sparse.linalg.gmres(
                lambda x: sparse_matvec(indices, all_vals, x),
                -residual,
                x0=x0_guess,  # Better initial guess
                M=precond,
                tol=adaptive_tol,  # Adaptive tolerance
                maxiter=2000,  # Increased for harder problems
                restart=100  # Larger restart for better convergence
            )
            
            # --- IMPROVEMENT 4: Better Convergence Check ---
            actual_residual = sparse_matvec(indices, all_vals, delta) + residual
            residual_norm = jnp.linalg.norm(actual_residual)
            relative_residual = residual_norm / (jnp.linalg.norm(residual) + 1e-12)
            
            # Multiple failure criteria
            has_nan = jnp.any(jnp.isnan(delta))
            has_inf = jnp.any(jnp.isinf(delta))
            poor_convergence = relative_residual > 0.1  # More reasonable than absolute 1.0
            
            is_diverged = has_nan | has_inf | poor_convergence
            
            # --- IMPROVEMENT 5: Damped Update on Failure ---
            # Instead of zero, return damped step to help outer iteration
            damping_factor = jax.lax.select(
                relative_residual > 1.0,
                0.1,  # Heavy damping if really bad
                jax.lax.select(
                    relative_residual > 0.5,
                    0.5,  # Moderate damping
                    1.0   # No damping if reasonable
                )
            )
            
            delta_safe = jax.lax.select(
                is_diverged, 
                damping_factor * delta,  # Adaptive damping
                delta
            )
            
            return y_guess + delta_safe

        # --- IMPROVEMENT 6: Better Nonlinear Solver Settings ---
        solver = optx.FixedPointIteration(
            rtol=1e-6,  # Tighter tolerance
            atol=1e-8,  # Absolute tolerance
        )
        sol = optx.fixed_point(
            sparse_newton_step, 
            solver, 
            y0, 
            max_steps=50,  # More iterations
            throw=False
        )
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )
        
        return sol.value, None, {"y0": y0, "y1": sol.value}, None, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class AdaptiveSparseSolver(diffrax.AbstractSolver):
    """
    Version with row-scaling preconditioner for better conditioning.
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        blocks, num_vars = args
        dt = t1 - t0
        
        q_prev = jnp.zeros(num_vars)
        for block in blocks:
            v_loc = y0[block.var_indices]
            _, q_loc = jax.vmap(block.physics_func)(v_loc, block.params)
            q_prev = q_prev.at[block.eq_indices].add(q_loc)

        def sparse_matvec(indices, data, x):
            col_idx = indices[:, 1]
            row_idx = indices[:, 0]
            prods = data * x[col_idx]
            return jnp.zeros(num_vars).at[row_idx].add(prods)

        def sparse_newton_step(y_guess, args):
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            jac_vals_all = [] 
            
            for block in blocks:
                v_loc = y_guess[block.var_indices]
                (f_l, q_l) = jax.vmap(block.physics_func)(v_loc, block.params)
                (df_l, dq_l) = jax.vmap(jax.jacfwd(block.physics_func))(v_loc, block.params)
                
                total_f = total_f.at[block.eq_indices].add(f_l)
                total_q = total_q.at[block.eq_indices].add(q_l)
                
                j_eff = df_l + (dq_l / dt)
                jac_vals_all.append(j_eff.reshape(-1))
            
            residual = total_f + (total_q - q_prev) / dt
            
            G_stiff = 1e6
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            all_rows = jnp.concatenate([b.jac_rows for b in blocks] + [jnp.array([0])])
            all_cols = jnp.concatenate([b.jac_cols for b in blocks] + [jnp.array([0])])
            all_vals = jnp.concatenate([jnp.concatenate(jac_vals_all), jnp.array([G_stiff])])
            
            indices = jnp.stack([all_rows, all_cols], axis=1)

            # --- ROW-SCALING PRECONDITIONER ---
            # Compute row norms for better scaling
            row_norms = jnp.zeros(num_vars).at[all_rows].add(jnp.abs(all_vals))
            row_scale = 1.0 / (row_norms + 1e-12)
            
            # Also extract diagonal
            diag_mask = (all_rows == all_cols)
            raw_diag = jnp.zeros(num_vars).at[all_rows].add(
                jnp.where(diag_mask, all_vals, 0.0)
            )
            diag_scale = 1.0 / (jnp.abs(raw_diag) + 1e-12)
            
            # Combined preconditioner: geometric mean of row and diagonal scaling
            combined_scale = jnp.sqrt(row_scale * diag_scale)
            
            def precond(x):
                return combined_scale * x
            
            # Apply left preconditioning to system
            # Solve: P^-1 A x = P^-1 b
            def precond_matvec(x):
                return combined_scale * sparse_matvec(indices, all_vals, x)
            
            precond_rhs = combined_scale * (-residual)
            
            # Adaptive tolerance
            residual_scale = jnp.linalg.norm(residual)
            adaptive_tol = jnp.maximum(1e-10, 1e-7 * residual_scale)
            
            delta, info = jax.scipy.sparse.linalg.gmres(
                precond_matvec,
                precond_rhs,
                x0=jnp.zeros_like(residual),
                tol=adaptive_tol,
                maxiter=2000,
                restart=100
            )
            
            # Convergence check
            actual_residual = sparse_matvec(indices, all_vals, delta) + residual
            relative_residual = jnp.linalg.norm(actual_residual) / (jnp.linalg.norm(residual) + 1e-12)
            
            is_diverged = (
                jnp.any(jnp.isnan(delta)) | 
                jnp.any(jnp.isinf(delta)) |
                (relative_residual > 0.1)
            )
            
            # Adaptive damping
            damping = jnp.where(relative_residual > 1.0, 0.1, 
                      jnp.where(relative_residual > 0.5, 0.5, 1.0))
            
            delta_safe = jax.lax.select(is_diverged, damping * delta, delta)
            
            return y_guess + delta_safe

        solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-8)
        sol = optx.fixed_point(sparse_newton_step, solver, y0, max_steps=50, throw=False)
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )
        
        return sol.value, None, {"y0": y0, "y1": sol.value}, None, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


from functools import partial

@partial(jax.jit, static_argnames=['num_vars'])
def _sparse_matvec_jit(row_idx, col_idx, data, x, num_vars):
    """JIT-compiled sparse matrix-vector product."""
    prods = data * x[col_idx]
    return jnp.zeros(num_vars).at[row_idx].add(prods)


@partial(jax.jit, static_argnames=['num_vars'])
def _extract_diagonal_jit(row_idx, col_idx, data, num_vars):
    """JIT-compiled diagonal extraction."""
    diag_mask = (row_idx == col_idx)
    return jnp.zeros(num_vars).at[row_idx].add(
        jnp.where(diag_mask, data, 0.0)
    )


@jax.jit
def _compute_preconditioner_jit(diag_vals):
    """JIT-compiled preconditioner computation."""
    diag_abs = jnp.abs(diag_vals)
    return jnp.where(
        diag_abs < 1e-8,
        1.0,  # Identity for bad rows
        1.0 / diag_vals
    )


@jax.jit
def _check_convergence_jit(delta, residual_check_norm):
    """JIT-compiled convergence check."""
    is_diverged = jnp.isnan(residual_check_norm) | (residual_check_norm > 1.0)
    return jax.lax.select(is_diverged, jnp.zeros_like(delta), delta)


class UltraFastSparseSolver(diffrax.AbstractSolver):
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
        blocks, num_vars = args
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
            jac_vals_all = [] 
            
            # Assembly loop (cannot be JIT'd due to variable block structure)
            for block in blocks:
                v_loc = y_guess[block.var_indices]
                (f_l, q_l) = jax.vmap(block.physics_func)(v_loc, block.params)
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