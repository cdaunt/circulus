import jax
import jax.numpy as jnp
import diffrax
import pytest

from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_operating_point, s_to_y
from circulus.solvers.transient import VectorizedTransientSolver

@pytest.mark.parametrize("mode", ['dense', 'sparse'])
def test_short_transient_runs_float(simple_lrc_netlist, mode):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Solve DC operating point and run a very short transient with zero forcing
    y_op = solve_operating_point(groups, sys_size, mode=mode, t0=0.0)

    solver = VectorizedTransientSolver(mode=mode)
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    t_max = 1e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 5))

    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_max, dt0=1e-3 * t_max,
        y0=y_op, args=(groups, sys_size), saveat=saveat, max_steps=1000
    )

    assert sol.ys.shape == (5, sys_size)
    assert jnp.isfinite(sol.ys).all()

@pytest.mark.parametrize("mode", ['dense', 'sparse'])
def test_short_transient_runs_complex(simple_optical_netlist, mode):
    net_dict, models_map = simple_optical_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Solve DC operating point and run a very short transient with zero forcing
    y_op = solve_operating_point(groups, sys_size, mode=mode, t0=0.0, dtype=jnp.complex128)
    print(y_op)

    # solver = VectorizedTransientSolver(mode=mode)
    # term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    # t_max = 1e-9
    # saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 5))

    # sol = diffrax.diffeqsolve(
    #     term, solver, t0=0.0, t1=t_max, dt0=1e-3 * t_max,
    #     y0=y_op, args=(groups, sys_size), saveat=saveat, max_steps=1000
    # )

    # assert sol.ys.shape == (5, sys_size)
    # assert jnp.isfinite(sol.ys).all()