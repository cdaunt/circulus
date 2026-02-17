import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from circulus.solvers.linear import CircuitLinearSolver
from circulus.solvers.assembly import (
    _assemble_system_complex,
    _assemble_system_real,
    _assemble_residual_only_real,
    _assemble_residual_only_complex,
)
from klujax import free_numeric
from diffrax import ConstantStepSize
from typing import Callable
from jax.typing import ArrayLike


def _compute_history(component_groups, y_c, t, num_vars):
    """Computes total charge Q at time t (Initial Condition)."""
    is_complex = jnp.iscomplexobj(y_c)
    total_q = jnp.zeros(
        2 * num_vars if is_complex else num_vars,
        dtype=jnp.float64 if is_complex else y_c.dtype,
    )

    for group in component_groups.values():
        v_locs = y_c[group.var_indices]
        _, q_l = jax.vmap(lambda v, p: group.physics_func(y=v, args=p, t=t))(
            v_locs, group.params
        )

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
    Transient solver that works strictly on FLAT (Real) vectors.
    Delegates complexity handling to the 'linear_solver' strategy.
    """

    linear_solver: CircuitLinearSolver

    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return (y0, 1.0)

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state

        # 0. Check Mode from Linear Solver Strategy
        #    (We trust the strategy knows if it's 2N or N size)
        is_complex = getattr(self.linear_solver, "is_complex", False)

        # 1. Predictor (1st order extrapolation on Flat Vector)
        rate = (y0 - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0 + rate * dt

        # 2. Compute History
        #    _compute_history expects a Complex vector if physics is complex.
        #    We reconstruct it temporarily here.
        if is_complex:
            y_c = y0[:num_vars] + 1j * y0[num_vars:]
        else:
            y_c = y0

        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        # Assemble Jacobian at y_pred for solver setup (e.g. factorization)
        # if is_complex:
        #     _, _, init_vals = _assemble_system_complex(y_pred, component_groups, t1, dt)
        # else:
        #     _, _, init_vals = _assemble_system_real(y_pred, component_groups, t1, dt)

        # solver_callable = self.linear_solver.get_transient_callable(init_vals)

        # 3. Define One Newton Step

        def newton_update_step(y, _):
            # A. Assemble Physics
            #    Both assemblers now expect FLAT vectors 'y'.
            if is_complex:
                total_f, total_q, all_vals = _assemble_system_complex(
                    y, component_groups, t1, dt
                )
                ground_indices = [0, num_vars]  # Real and Imag ground nodes
            else:
                total_f, total_q, all_vals = _assemble_system_real(
                    y, component_groups, t1, dt
                )
                ground_indices = [0]

            # B. Transient Residual: I + dQ/dt
            residual = total_f + (total_q - q_prev) / dt

            # C. Apply Ground Constraints (RHS)
            for idx in ground_indices:
                residual = residual.at[idx].add(1e9 * y[idx])

            # D. Solve Linear System
            #    Strategy handles the flat 2N system automatically
            sol = self.linear_solver._solve_impl(all_vals, -residual)
            delta = sol.value

            # E. Damping
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, 0.5 / (max_change + 1e-9))

            return y + delta * damping

        # 4. Run Newton Loop (On Flat Vectors)
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(
            newton_update_step, solver, y_pred, max_steps=20, throw=False
        )

        y_next = sol.value
        y_error = y_next - y_pred

        # Map result to Diffrax
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )

        return y_next, y_error, {"y0": y0, "y1": y_next}, (y0, dt), result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class FactorizedTransientSolver(VectorizedTransientSolver):
    # Frozen Jacobian converges linearly, so we need more steps.
    newton_max_steps: int = 100

    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars = args
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state

        is_complex = getattr(self.linear_solver, "is_complex", False)

        # 1. Predictor
        rate = (y0 - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0 + rate * dt

        # 2. Compute History
        if is_complex:
            y_c = y0[:num_vars] + 1j * y0[num_vars:]
        else:
            y_c = y0

        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        # 3. Assemble and Factor ONCE
        if is_complex:
            _, _, frozen_jac_vals = _assemble_system_complex(
                y_pred, component_groups, t1, dt
            )
            ground_indices = [0, num_vars]
        else:
            _, _, frozen_jac_vals = _assemble_system_real(
                y_pred, component_groups, t1, dt
            )
            ground_indices = [0]

        # Factor ONCE
        numeric_handle = self.linear_solver.factor_jacobian(frozen_jac_vals)

        # 4. Newton iterations with frozen Jacobian
        def newton_update_step(y, _):
            if is_complex:
                total_f, total_q = _assemble_residual_only_complex(
                    y, component_groups, t1, dt
                )
            else:
                total_f, total_q = _assemble_residual_only_real(
                    y, component_groups, t1, dt
                )

            residual = total_f + (total_q - q_prev) / dt

            for idx in ground_indices:
                residual = residual.at[idx].add(1e9 * y[idx])

            sol = self.linear_solver.solve_with_frozen_jacobian(
                -residual, numeric_handle
            )
            delta = sol.value

            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, 0.5 / (max_change + 1e-9))

            return y + delta * damping

        # 5. Run Newton Loop
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(
            newton_update_step,
            solver,
            y_pred,
            max_steps=self.newton_max_steps,
            throw=False,
        )

        # 6. Free the numeric handle (now returns int32, can be traced!)
        _ = free_numeric(numeric_handle)

        y_next = sol.value
        y_error = y_next - y_pred

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )

        return y_next, y_error, {"y0": y0, "y1": y_next}, (y0, dt), result


def setup_transient(
    groups: list, linear_strategy: CircuitLinearSolver, transient_solver=None
) -> Callable[..., diffrax.Solution]:  # Replace Any with diffrax.Solution if available
    """Configures and returns a function for executing transient analysis.

    This function acts as a factory, preparing a transient solver that is
    pre-configured with the circuit's linear strategy. It returns a callable
    that executes the time-domain simulation using `diffrax.diffeqsolve`.

    Args:
        groups (list): A list of component groups that define the circuit.
        linear_strategy (CircuitLinearSolver): The configured linear solver
            strategy, typically obtained from `analyze_circuit`.
        transient_solver (optional): The transient solver class to use.
            If None, `VectorizedTransientSolver` will be used.

    Returns:
        Callable[..., Any]: A function that executes the transient analysis.
        This returned function accepts the following arguments:

            t0 (float): The start time of the simulation.
            t1 (float): The end time of the simulation.
            dt0 (float): The initial time step for the solver.
            y0 (ArrayLike): The initial state vector of the system.
            saveat (diffrax.SaveAt, optional): Specifies time points at which
                to save the solution. Defaults to None.
            max_steps (int, optional): The maximum number of steps the solver
                can take. Defaults to 100000.
            throw (bool, optional): If True, the solver will raise an error on
                failure. Defaults to False.
            term (diffrax.AbstractTerm, optional): The term defining the ODE.
                Defaults to a zero-value ODETerm.
            stepsize_controller (diffrax.AbstractStepSizeController, optional):
                The step size controller. Defaults to `ConstantStepSize()`.
            **kwargs: Additional keyword arguments to pass directly to
                `diffrax.diffeqsolve`.
    """

    if transient_solver is None:
        transient_solver = VectorizedTransientSolver

    tsolver = transient_solver(linear_solver=linear_strategy)

    sys_size = (
        linear_strategy.sys_size // 2
        if linear_strategy.is_complex
        else linear_strategy.sys_size
    )

    def _execute_transient(
        *,
        t0: float,
        t1: float,
        dt0: float,
        y0: ArrayLike,
        saveat: diffrax.SaveAt = None,
        max_steps: int = 100000,
        throw: bool = False,
        **kwargs,
    ) -> diffrax.Solution:
        """Executes the transient simulation for the pre-configured circuit."""

        term = kwargs.pop("term", diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y)))
        solver = kwargs.pop("solver", tsolver)
        args = kwargs.pop("args", (groups, sys_size))
        stepsize_controller = kwargs.pop("stepsize_controller", ConstantStepSize())

        sol = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=saveat,
            max_steps=max_steps,
            throw=throw,
            stepsize_controller=stepsize_controller,
            **kwargs,
        )

        return sol

    return _execute_transient
