import jax
import jax.numpy as jnp
from functools import partial
import pytest

from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_dc_op_dense, solve_dc_op_sparse, solve_operating_point

# assemble_total_f is local to this module and uses jax
def assemble_total_f(component_groups, y, t=0.0):
    total_f = jnp.zeros(y.shape[0])
    for group_name, group in component_groups.items():
        v_locs = y[group.var_indices]
        physics_fn = partial(group.physics_func, t=t)

        def get_f_only(v, p):
            return physics_fn(v, p)[0]

        f_loc = jax.vmap(get_f_only)(v_locs, group.params)
        total_f = total_f.at[group.eq_indices].add(f_loc)
    return total_f

@pytest.mark.parametrize("mode", ['dense', 'sparse'])
def test_solve_operation_point_residual_small(simple_lrc_netlist, mode):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    y_op = solve_operating_point(groups, sys_size, t0=0.0, mode=mode)
    assert y_op.shape[0] == sys_size

    total_f = assemble_total_f(groups, y_op, t=0.0)
    G_leak = 1e-9
    residual = total_f + y_op * G_leak

    # Residual should be very small for a converged DC solution
    assert jnp.linalg.norm(residual) < 1e-6

@pytest.mark.parametrize("mode", ['dense', 'sparse'])
def test_solve_operation_point_residual_small_complex(simple_optical_netlist, mode):
    net_dict, models_map = simple_optical_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    y_op = solve_operating_point(groups, sys_size, t0=0.0, mode=mode, dtype=jnp.complex128)
    assert y_op.shape[0] == sys_size

