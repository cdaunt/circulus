import jax
import jax.numpy as jnp
import diffrax
import pytest

from circulus.compiler import compile_netlist
from circulus.solvers import strategies as st
from circulus.solvers.transient import VectorizedTransientSolver

solvers = [st.KLUSolver, st.SparseSolver, st.DenseSolver]

@pytest.mark.parametrize("solver", solvers, ids=lambda x: x.__name__)
def test_short_transient_runs_float(simple_lrc_netlist, solver):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Solve DC operating point and run a very short transient with zero forcing
    linear_strat = solver.from_circuit(groups, sys_size, is_complex=False)
    y_guess = jnp.zeros(sys_size)
    y_op = linear_strat.solve_dc(groups, y_guess)

    assert y_op.shape[0] == sys_size

    solver = VectorizedTransientSolver(linear_solver=linear_strat)

    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    t_max = 1e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 5))

    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_max, dt0=1e-3 * t_max,
        y0=y_op, args=(groups, sys_size), saveat=saveat, max_steps=1000
    )

    assert sol.ys.shape == (5, sys_size)
    assert jnp.isfinite(sol.ys).all()

@pytest.mark.parametrize("solver", solvers, ids=lambda x: x.__name__)
def test_short_transient_runs_complex(simple_optical_netlist, solver):
    net_dict, models_map = simple_optical_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Solve DC operating point and run a very short transient with zero forcing
    linear_strat = solver.from_circuit(groups, sys_size, is_complex=True)

    print("2. Solving DC Operating Point...")

    y_guess_flat = jnp.zeros(sys_size * 2, dtype=jnp.float64)
    y_op_flat = linear_strat.solve_dc(groups, y_guess_flat)

    assert y_op_flat.shape[0] == 2*sys_size

    solver = VectorizedTransientSolver(linear_solver=linear_strat)
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    t_max = 1.0e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 500))

    sol = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=0.0,
        t1=t_max,
        dt0=1e-13,
        y0=y_op_flat,
        args=(groups, sys_size),
        saveat=saveat,
        max_steps=100000,
        throw=False,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6)
    )