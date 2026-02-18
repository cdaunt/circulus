from functools import partial

import jax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.netlist import Netlist
from circulax.solvers.linear import backends

solvers = set(backends.values())


# assemble_total_f is local to this module and uses jax
def assemble_total_f(component_groups: list, y, t:float=0.0)-> None:
    total_f = jnp.zeros(y.shape[0])
    for group in component_groups.values():
        v_locs = y[group.var_indices]
        physics_fn = partial(group.physics_func, t=t)

        def get_f_only(v, p):  # noqa: ANN202
            return physics_fn(y=v, args=p)[0]  # noqa: B023

        f_loc = jax.vmap(get_f_only)(v_locs, group.params)
        total_f = total_f.at[group.eq_indices].add(f_loc)
    return total_f


@pytest.mark.parametrize("solver", solvers, ids=lambda x: x.__name__)
def test_solve_operation_point_residual_small(simple_lrc_netlist: Netlist, solver)-> None:  # noqa: ANN001
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    linear_strat = solver.from_circuit(groups, sys_size, is_complex=False)
    y_guess = jnp.zeros(sys_size)
    y_op = linear_strat.solve_dc(groups, y_guess)

    assert y_op.shape[0] == sys_size

    total_f = assemble_total_f(groups, y_op, t=0.0)
    G_leak = 1e-9
    residual = total_f + y_op * G_leak

    # Residual should be very small for a converged DC solution
    assert jnp.linalg.norm(residual) < 1e-6


@pytest.mark.parametrize("solver", solvers, ids=lambda x: x.__name__)
def test_solve_operation_point_residual_small_complex(simple_optical_netlist: Netlist, solver)-> None:  # noqa: ANN001
    net_dict, models_map = simple_optical_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    linear_strat = solver.from_circuit(groups, sys_size, is_complex=True)
    y_guess = jnp.ones(2 * sys_size)
    y_op = linear_strat.solve_dc(groups, y_guess)

    assert y_op.shape[0] == 2 * sys_size
