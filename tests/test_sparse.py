import jax
import jax.numpy as jnp
import diffrax
import pytest

from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_dc_op_dense
from circulus.solvers.sparse import VectorizedSparseSolver as SparseSolver


jax.config.update("jax_enable_x64", True)


def test_sparse_short_transient_runs(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Solve DC operating point and run a very short transient with zero forcing
    y_op = solve_dc_op_dense(groups, sys_size, t0=0.0)

    solver = SparseSolver()
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    t_max = 1e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 5))

    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_max, dt0=1e-3 * t_max,
        y0=y_op, args=(groups, sys_size), saveat=saveat, max_steps=1000
    )

    assert sol.ys.shape == (5, sys_size)
    assert jnp.isfinite(sol.ys).all()
