import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx
import inspect
from typing import NamedTuple, Callable, List, Dict, Any
import matplotlib.pyplot as plt



import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx

class SparseTaxSolver(diffrax.AbstractSolver):
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

class SparseTaxSolverClaude(diffrax.AbstractSolver):
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


# ============================================================================
# ADVANCED VERSION: Adaptive Preconditioner
# ============================================================================

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


# ============================================================================
# PERFORMANCE-OPTIMIZED VERSION
# ============================================================================


# PERFORMANCE-OPTIMIZED VERSION
# ============================================================================

# ============================================================================
# JIT-OPTIMIZED HELPER FUNCTIONS (MODULE LEVEL)
# ============================================================================


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


# ============================================================================
# PERFORMANCE-OPTIMIZED VERSION (WITH EXTERNAL JIT)
# ============================================================================

class FastSparseSolver(diffrax.AbstractSolver):
    """
    Optimized for speed: JIT-compiled helper functions at module level.
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        blocks, num_vars = args
        dt = t1 - t0
        
        # Efficient history computation
        q_prev = jnp.zeros(num_vars)
        for block in blocks:
            v_loc = y0[block.var_indices]
            _, q_loc = jax.vmap(block.physics_func)(v_loc, block.params)
            q_prev = q_prev.at[block.eq_indices].add(q_loc)

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
                jac_vals_all.append((df_l + dq_l / dt).reshape(-1))
            
            residual = total_f + (total_q - q_prev) / dt
            
            # Boundary condition
            G_stiff = 1e6
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            # Assemble sparse matrix
            all_rows = jnp.concatenate([b.jac_rows for b in blocks] + [jnp.array([0])])
            all_cols = jnp.concatenate([b.jac_cols for b in blocks] + [jnp.array([0])])
            all_vals = jnp.concatenate([jnp.concatenate(jac_vals_all), jnp.array([G_stiff])])
            
            # Use JIT-compiled diagonal extraction
            diag_vals = _extract_diagonal_jit(all_rows, all_cols, all_vals, num_vars)
            safe_diag_inv = _compute_preconditioner_jit(diag_vals)
            
            def precond(x):
                return safe_diag_inv * x
            
            # Use JIT-compiled matvec in GMRES
            def matvec(x):
                return _sparse_matvec_jit(all_rows, all_cols, all_vals, x, num_vars)
            
            # GMRES with better initial guess
            x0_guess = -0.01 * precond(residual)
            
            delta, _ = jax.scipy.sparse.linalg.gmres(
                matvec,
                -residual,
                x0=x0_guess,
                M=precond,
                tol=1e-8,
                maxiter=1000,
                restart=50
            )
            
            # Use JIT-compiled convergence check
            check_res_norm = jnp.linalg.norm(matvec(delta) + residual)
            delta_safe = _check_convergence_jit(delta, check_res_norm)
            
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


# ============================================================================
# ALTERNATIVE: Pre-computed Structure Solver
# ============================================================================

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
    

class DenseTaxSolver(diffrax.AbstractSolver):
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

        def dense_newton_step(y_guess, args):
            total_f = jnp.zeros(num_vars)
            total_q = jnp.zeros(num_vars)
            jac_vals_all = [] 
            
            for block in blocks:
                v_loc = y_guess[block.var_indices]
                
                # Compute Values and Jacobians
                (f_l, q_l) = jax.vmap(block.physics_func)(v_loc, block.params)
                (df_l, dq_l) = jax.vmap(jax.jacfwd(block.physics_func))(v_loc, block.params)

                total_f = total_f.at[block.eq_indices].add(f_l)
                total_q = total_q.at[block.eq_indices].add(q_l)
                
                j_eff = df_l + (dq_l / dt)
                jac_vals_all.append(j_eff.reshape(-1))
            
            residual = total_f + (total_q - q_prev) / dt
            
            # Boundary Condition (Ground Node 0)
            G_stiff = 1e8
            residual = residual.at[0].add(G_stiff * y_guess[0])
            
            all_rows = jnp.concatenate([b.jac_rows for b in blocks] + [jnp.array([0])])
            all_cols = jnp.concatenate([b.jac_cols for b in blocks] + [jnp.array([0])])
            all_vals = jnp.concatenate([jnp.concatenate(jac_vals_all), jnp.array([G_stiff])])
            
            # Sparse Assembly -> Dense Solve
            J_sparse = sparse.BCOO(
                (all_vals, jnp.stack([all_rows, all_cols], axis=1)),
                shape=(num_vars, num_vars)
            )
            J_dense = J_sparse.todense()
            
            delta = jnp.linalg.solve(J_dense, -residual)
            return y_guess + delta

        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(dense_newton_step, solver, y0, max_steps=50, throw=False)
        
        # Result Code Handling
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )
        
        return sol.value, None, {"y0": y0, "y1": sol.value}, None, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
    

################################

def get_jacobian_diagonal(y, blocks, dt, num_vars):
    """
    Computes the EXACT diagonal of the Jacobian matrix for Jacobi Preconditioning.
    J_diag = diag( df/dy + (1/dt)*dq/dy )
    
    This is O(N) memory and fast to compute.
    """
    total_diag = jnp.zeros(num_vars)

    for block in blocks:
        v_loc = y[block.var_indices]
        
        # Compute local Jacobians (Dense small blocks)
        # jax.jacfwd returns (Batch, Vars, Vars)
        df_loc, dq_loc = jax.vmap(jax.jacfwd(block.physics_func))(v_loc, block.params)
        
        # Effective Local Jacobian
        J_loc = df_loc + (dq_loc / dt)
        
        # Extract diagonals: Shape (Batch, Vars)
        J_diag_vals = jax.vmap(jnp.diagonal)(J_loc)
        
        # Scatter add to global diagonal vector
        # Assumes MNA diagonal aligns with eq_indices
        total_diag = total_diag.at[block.eq_indices].add(J_diag_vals)

    # Boundary Condition Penalty Diagonal
    # Lowered from 1e9 to 1e6 to improve condition number for BiCGSTAB
    G_stiff = 1e6 
    total_diag = total_diag.at[0].add(G_stiff)
    
    # FIX: Handle Zero Diagonals (e.g. Voltage Sources, Inductors)
    # If diagonal is ~0, using 1e-10 results in 1e10 scaling (Explosion).
    # Instead, we use 1.0 (Identity scaling) for these constraint rows.
    total_diag = jnp.where(jnp.abs(total_diag) < 1e-6, 1.0, total_diag)
    
    return total_diag

def residual_fn(y, blocks, q_prev, dt, num_vars):
    """
    Calculates the DAE Residual: R(y) = f(y) + (q(y) - q_prev)/dt
    """
    total_f = jnp.zeros(num_vars)
    total_q = jnp.zeros(num_vars)

    for block in blocks:
        v_loc = y[block.var_indices]
        # Calculate Physics
        f_l, q_l = jax.vmap(block.physics_func)(v_loc, block.params)

        # Scatter Add
        total_f = total_f.at[block.eq_indices].add(f_l)
        total_q = total_q.at[block.eq_indices].add(q_l)

    R = total_f + (total_q - q_prev) / dt

    # Boundary condition (Ground Node 0) - Penalty Method
    G_stiff = 1e6 # Matched to preconditioner
    R = R.at[0].add(G_stiff * y[0])

    return R

def matvec_product(v, y, blocks, q_prev, dt, num_vars):
    """
    Computes Jacobian-Vector Product (J*v) WITHOUT forming J.
    """
    # JVP is Directional Derivative
    _, Jv = jax.jvp(
        lambda y_: residual_fn(y_, blocks, q_prev, dt, num_vars),
        (y,),
        (v,)
    )
    return Jv

def newton_krylov_step(y, args):
    """
    Performs one Newton update: y_new = y - J_inv * R
    Includes a simple Backtracking Line Search to ensure global convergence.
    """
    blocks, q_prev, dt, num_vars = args
    
    # 1. Calculate Residual
    R = residual_fn(y, blocks, q_prev, dt, num_vars)
    initial_norm = jnp.linalg.norm(R)
    
    # 2. Calculate Preconditioner (Inverse Diagonal)
    J_diag = get_jacobian_diagonal(y, blocks, dt, num_vars)
    inv_diag = 1.0 / J_diag
    
    def preconditioner(x):
        return x * inv_diag

    # 3. Define Linear Operator A(x) = J * x
    def linear_op(v):
        return matvec_product(v, y, blocks, q_prev, dt, num_vars)
    
    # 4. Solve Linear System (BiCGSTAB)
    delta, info = jax.scipy.sparse.linalg.bicgstab(
        linear_op,
        -R,
        M=preconditioner, 
        tol=1e-6,
        maxiter=1000
    )
    
    # 5. Backtracking Line Search (Robustness against bad y0)
    # Check 3 candidates: Full step, Half step, Small step
    # This prevents divergence when initial guess is far from solution.
    
    alphas = jnp.array([1.0, 0.5, 0.125])
    
    def evaluate_step(alpha):
        y_test = y + alpha * delta
        R_test = residual_fn(y_test, blocks, q_prev, dt, num_vars)
        return jnp.linalg.norm(R_test)
        
    norms = jax.vmap(evaluate_step)(alphas)
    
    # Find best step size
    best_idx = jnp.argmin(norms)
    best_alpha = alphas[best_idx]
    
    # Only take the step if it improves things (or if we are stuck)
    # If all norms are worse, take a tiny step to nudge it.
    final_alpha = jnp.where(norms[best_idx] < initial_norm, best_alpha, 0.01)
    
    return y + final_alpha * delta

# =========================================================================
# SOLVER CLASS
# =========================================================================

class MatrixFreeTaxSolver(diffrax.AbstractSolver):
    """
    Jacobian-Free Newton-Krylov (JFNK) Solver with Jacobi Preconditioning.
    """
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        blocks, num_vars = args
        dt = t1 - t0

        # 1. Precompute History (Q_prev)
        q_prev = jnp.zeros(num_vars)
        for block in blocks:
            v_loc = y0[block.var_indices]
            _, q_l = jax.vmap(block.physics_func)(v_loc, block.params)
            q_prev = q_prev.at[block.eq_indices].add(q_l)

        # 2. Setup Fixed Point Solver
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)

        # 3. Solve
        sol = optx.fixed_point(
            newton_krylov_step,
            solver,
            y0,
            args=(blocks, q_prev, dt, num_vars),
            max_steps=50,
            throw=False,
        )

        # 4. Result Handling
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )

        return sol.value, None, {"y0": y0, "y1": sol.value}, None, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)