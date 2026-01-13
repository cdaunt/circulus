import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from circulus.solvers.strategies import CircuitLinearSolver,_assemble_system_complex, _assemble_system_real


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
# VECTORIZED TRANSIENT SOLVER
# ==============================================================================

class VectorizedTransientSolver(diffrax.AbstractSolver):
    """
    Transient Circuit Solver.
    Uses a custom Newton loop via Optimistix FixedPointIteration.
    """
    linear_solver: CircuitLinearSolver
    
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms): return 1

    def init(self, terms, t0, t1, y0, args):
        if jnp.iscomplexobj(y0):
            y0_flat = jnp.concatenate([y0.real, y0.imag])
        else:
            y0_flat = y0
        return (y0_flat, 1.0) 

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state
        
        # 1. State Normalization
        y0_flat = y0
        if jnp.iscomplexobj(y0): 
            y0_flat = jnp.concatenate([y0.real, y0.imag])
            
        # 2. Predictor
        rate = (y0_flat - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0_flat + rate * dt
        
        # 3. Compute History
        is_complex = (y0_flat.shape[0] == 2 * num_vars)
        y_c = y0_flat[:num_vars] + 1j * y0_flat[num_vars:] if is_complex else y0_flat
        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        # 4. Define One Newton Step
        def newton_update_step(y, _):
            # A. Assemble Physics
            if is_complex:
                total_f, total_q, all_vals = _assemble_system_complex(y, component_groups, t1, dt)
                ground_indices = [0, num_vars]
            else:
                total_f, total_q, all_vals = _assemble_system_real(y, component_groups, t1, dt)
                ground_indices = [0]
            
            # B. Transient Residual
            residual = total_f + (total_q - q_prev) / dt
            
            # C. Apply Ground Constraints (RHS)
            for idx in ground_indices:
                residual = residual.at[idx].add(1e9 * y[idx])
            
            # D. Solve Linear System (FIXED)
            #    We call _solve_impl directly. 
            #    The solver expects the Jacobian values and the NEGATIVE residual (RHS).
            sol = self.linear_solver._solve_impl(all_vals, -residual)
            
            delta = sol.value
            
            # E. Damping
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, 0.2 / (max_change + 1e-9))
            
            return y + delta * damping

        # 5. Run Newton Loop
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(
            newton_update_step, 
            solver, 
            y_pred, 
            max_steps=20, 
            throw=False
        )
        
        y_next = sol.value
        y_error = y_next - y_pred 
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )

        return y_next, y_error, {"y0": y0_flat, "y1": y_next}, (y0_flat, dt), result
    
    def func(self, terms, t0, y0, args): return terms.vf(t0, y0, args)
