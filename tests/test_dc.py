import jax
import jax.numpy as jnp
from functools import partial
import pytest

from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_dc_op_dense

# assemble_total_f is local to this module and uses jax
def assemble_total_f(component_groups, y, t=0.0):
    total_f = jnp.zeros(y.shape[0])
    for group in component_groups:
        v_locs = y[group.var_indices]
        physics_fn = partial(group.physics_func, t=t)

        def get_f_only(v, p):
            return physics_fn(v, p)[0]

        f_loc = jax.vmap(get_f_only)(v_locs, group.params)
        total_f = total_f.at[group.eq_indices].add(f_loc)
    return total_f


def assemble_total_f(component_groups, y, t=0.0):
    total_f = jnp.zeros(y.shape[0])
    for group in component_groups:
        v_locs = y[group.var_indices]
        physics_fn = partial(group.physics_func, t=t)

        def get_f_only(v, p):
            return physics_fn(v, p)[0]

        f_loc = jax.vmap(get_f_only)(v_locs, group.params)
        total_f = total_f.at[group.eq_indices].add(f_loc)
    return total_f


def test_solve_dc_op_dense_residual_small(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    y_op = solve_dc_op_dense(groups, sys_size, t0=0.0)
    assert y_op.shape[0] == sys_size

    total_f = assemble_total_f(groups, y_op, t=0.0)
    G_leak = 1e-9
    residual = total_f + y_op * G_leak

    # Residual should be very small for a converged DC solution
    assert jnp.linalg.norm(residual) < 1e-6
