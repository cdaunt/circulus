import jax
import jax.numpy as jnp
import pytest

from circulus.models import (
    diode, current_source, vcvs, vccs,
    resistor, capacitor, voltage_source, inductor
    , ccvs, cccs, ideal_opamp)

jax.config.update("jax_enable_x64", True)


def test_resistor_current_symmetry_and_ohms_law():
    # v_a - v_b = I*R
    R = 10.0
    v_a = 5.0
    v_b = 0.0
    vars = jnp.array([v_a, v_b])

    f, q = resistor(vars, params={'R': R})
    # f[0] is current leaving node a, equals (v_a - v_b)/R
    expected_i = (v_a - v_b) / R
    assert jnp.isclose(f[0], expected_i)
    # Opposite at node b
    assert jnp.isclose(f[1], -expected_i)


def test_capacitor_charge_and_zero_current_in_dc():
    C = 1e-11
    v_a = 2.0
    v_b = 1.0
    vars = jnp.array([v_a, v_b])

    f, q = capacitor(vars, params={'C': C})
    # In this model f are currents (dynamic terms are placed in q). For steady-state, f should be zero
    assert jnp.allclose(f, jnp.array([0.0, 0.0]))

    expected_q = C * (v_a - v_b)
    assert jnp.isclose(q[0], expected_q)
    assert jnp.isclose(q[1], -expected_q)


def test_voltage_source_constraint_and_delay_behavior():
    V = 5.0
    delay = 0.5
    # vars: [v_a, v_b, i_src]
    vars = jnp.array([0.0, 0.0, 0.0])

    # Before delay: output should be zero
    f0, q0 = voltage_source(vars, params={'V': V, 'delay': delay}, t=0.0)
    # constraint is stored as last entry of f
    constraint0 = f0[-1]
    assert jnp.isclose(constraint0, 0.0)

    # After delay: constraint equals v_a - v_b + V (with v_a=v_b=0 => V)
    f1, q1 = voltage_source(vars, params={'V': V, 'delay': delay}, t=1.0)
    constraint1 = f1[-1]
    assert jnp.isclose(constraint1, -V)


def test_inductor_branch_and_flux_relation():
    L = 1e-9
    v_a = 0.5
    v_b = 0.0
    i_branch = 0.1
    vars = jnp.array([v_a, v_b, i_branch])

    f, q = inductor(vars, params={'L': L})

    # f contains KCL contributions and branch equation
    # Branch equation (3rd) should be v_a - v_b
    assert jnp.isclose(f[2], v_a - v_b)

    # q for branch (3rd) equals -L * i_branch
    assert jnp.isclose(q[2], -L * i_branch)


def test_diode_iv_forward():
    # Forward-bias diode should produce a positive current
    vars = jnp.array([0.7, 0.0])
    f, q = diode(vars)
    assert f.shape == (2,)
    assert q.shape == (2,)
    assert f[0] > 0.0


def test_current_source_injection():
    vars = jnp.array([0.0, 0.0])
    f, q = current_source(vars, params={'I': 2.0})
    assert f[0] == 2.0
    assert f[1] == -2.0


def test_vcvs_constraint_enforced():
    # v_out+ - v_out- - A*(v_ctrl+ - v_ctrl-) == 0
    A = 10.0
    v_out_p = 1.0
    v_out_m = 0.0
    v_ctrl_p = 0.2
    v_ctrl_m = 0.0

    vars = jnp.array([v_out_p, v_out_m, 0.0, v_ctrl_p, v_ctrl_m])
    f, q = vcvs(vars, params={'A': A})

    constraint = f[2]
    expected = v_out_p - v_out_m - A * (v_ctrl_p - v_ctrl_m)
    assert jnp.isclose(constraint, expected)


def test_vccs_outputs_current():
    G = 0.5
    v_ctrl_p = 1.0
    v_ctrl_m = 0.0
    vars = jnp.array([0.0, 0.0, v_ctrl_p, v_ctrl_m])
    f, q = vccs(vars, params={'G': G})
    expected = G * (v_ctrl_p - v_ctrl_m)
    assert jnp.isclose(f[0], expected)
    assert jnp.isclose(f[1], -expected)


def test_ccvs_voltage_relation():
    R = 2.0
    v_out_p = 1.2
    v_out_m = 0.0
    i_src = 0.0
    i_ctrl = 0.1
    vars = jnp.array([v_out_p, v_out_m, i_src, i_ctrl])
    f, q = ccvs(vars, params={'R': R})
    constraint = f[2]
    expected = v_out_p - v_out_m - R * i_ctrl
    assert jnp.isclose(constraint, expected)


def test_cccs_current_gain():
    alpha = 3.0
    v_out_p = 0.0
    v_out_m = 0.0
    i_ctrl = 0.2
    vars = jnp.array([v_out_p, v_out_m, i_ctrl])
    f, q = cccs(vars, params={'alpha': alpha})
    expected = alpha * i_ctrl
    assert jnp.isclose(f[0], expected)
    assert jnp.isclose(f[1], -expected)





def test_ideal_opamp_behavior():
    A = 1e6
    v_out_p = 1.0
    v_out_m = 0.0
    i_src = 0.0
    v_in_p = 0.1
    v_in_m = 0.0
    vars = jnp.array([v_out_p, v_out_m, i_src, v_in_p, v_in_m])
    f, q = ideal_opamp(vars, params={'A': A})
    constraint = f[2]
    expected = v_out_p - v_out_m - A * (v_in_p - v_in_m)
    assert jnp.isclose(constraint, expected)


