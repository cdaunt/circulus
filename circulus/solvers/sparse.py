import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx
from functools import partial
import equinox as eqx
from circulus.solvers.common import _extract_diagonal, _sparse_matvec

@jax.jit
def _compute_preconditioner(diag_vals):
    return jnp.where(jnp.abs(diag_vals) < 1e-8, 1.0, 1.0 / diag_vals)


class SparseSolverState(eqx.Module):
    static_rows: jax.Array
    static_cols: jax.Array
    diag_mask: jax.Array
    history: tuple
    is_complex_mode: bool = eqx.field(static=True)

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

        # --- Pre-calculate Preconditioner Mask ---
        # Used to extract the diagonal quickly using segment_sum
        diag_mask = (static_rows == static_cols)
        
        return SparseSolverState(static_rows, static_cols, diag_mask, history, is_complex)


    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        static_rows = solver_state.static_rows
        static_cols = solver_state.static_cols
        history = solver_state.history
        is_complex_mode = solver_state.is_complex_mode
        diag_mask = solver_state.diag_mask
        
        y_prev_step, dt_prev = history
        dt = t1 - t0

        if is_complex_mode:
            y0 = y0.astype(jnp.complex128)
        
        solver_dtype = y0.dtype
        
        # --- 1. History (t0) ---
        q_prev = jnp.zeros(num_vars, dtype=solver_dtype)

        for group_name, group in component_groups.items():
            v_locs = y0[group.var_indices]

            physics_at_t0 = partial(group.physics_func, t=t0)
            
            _, q_locs = jax.vmap(physics_at_t0)(v_locs, group.params)
           
            q_prev = q_prev.at[group.eq_indices].add(q_locs)

        # --- 2. Sparse Newton Step (t1) ---
        def sparse_newton_step(y_guess, args):

            sys_size = 2 * num_vars if is_complex_mode else num_vars

            total_f = jnp.zeros(sys_size, dtype=jnp.float64 if is_complex_mode else solver_dtype)
            total_q = jnp.zeros(sys_size, dtype=jnp.float64 if is_complex_mode else solver_dtype)
            jac_vals_list = []
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
                    v_locs = y_guess[group.var_indices]
                    def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)
                    
                    # Evaluate Physics & Jacobian
                    (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
                    (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
                    
                    total_f = total_f.at[group.eq_indices].add(f_l)
                    total_q = total_q.at[group.eq_indices].add(q_l)
                    
                    j_eff = df_l + (dq_l / dt)
                    jac_vals_list.append(j_eff.reshape(-1))
            
            # --- Residual Calculation ---
            if is_complex_mode:
                q_prev_expanded = jnp.concatenate([q_prev.real, q_prev.imag])
                residual = total_f + (total_q - q_prev_expanded) / dt
            else:
                residual = total_f + (total_q - q_prev) / dt
            
            # Ground Constraint
            G_stiff = 1e6
            residual = residual.at[0].add(G_stiff * y_guess[0])
            if is_complex_mode:
                residual = residual.at[num_vars].add(G_stiff * y_guess[num_vars])
            
            # Flatten Matrix Values
            if is_complex_mode:
                all_vals = jnp.concatenate(vals_rr + vals_ri + vals_ir + vals_ii + [jnp.array([G_stiff]), jnp.array([G_stiff])])
            else:
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
        # Error calculation for PID Controller
        rate_prev = (y0 - y_prev_step) / dt_prev
        y_pred = y0 + rate_prev * dt
        
        # The error is the difference between the Newton solution and the Linear Prediction
        # This scales with curvature (d2y/dt2), not slope (dy/dt)
        y_next = sol.value
        y_error = y_next - y_pred

        # Update History for next step
        new_history = (y0, dt)
        new_state = SparseSolverState(static_rows, static_cols,  diag_mask, new_history, is_complex_mode=is_complex_mode)
        
        return sol.value, y_error, {"y0": y0, "y1": sol.value}, new_state, result
    
    def func(self, terms, t0, y0, args):
            return terms.vf(t0, y0, args)