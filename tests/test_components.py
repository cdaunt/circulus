import jax
import jax.numpy as jnp
import pytest
from collections import namedtuple

# Import components to be tested
from circulus.components import (
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    Diode,
    CurrentSource,
    VCVS,
    VCCS,
    CCVS,
    CCCS,
    IdealOpAmp,
)
from circulus.base_component import CircuitComponent

# Enable x64 for precision
jax.config.update("jax_enable_x64", True)

# Helper to call the physics method in a standardized way
def run_phys(component, v_dict, s_dict={}, t=0.0):
    # Create namedtuples on the fly to simulate the solver_bridge's unpacking
    if v_dict:
        Ports = namedtuple("Ports", v_dict.keys())
        v = Ports(**v_dict)
    else:
        v = ()
        
    if s_dict:
        States = namedtuple("States", s_dict.keys())
        s = States(**s_dict)
    else:
        s = ()
        
    return component.physics(v, s, t)

# --- Component Tests ---

def test_resistor():
    r = Resistor(R=10.0)
    v_dict = {'p1': 5.0, 'p2': 0.0}
    f, q = run_phys(r, v_dict)
    
    expected_i = (v_dict['p1'] - v_dict['p2']) / (r.R + 1e-12)
    assert jnp.isclose(f['p1'], expected_i)
    assert jnp.isclose(f['p2'], -expected_i)
    assert not q

def test_capacitor():
    c = Capacitor(C=1e-11)
    v_dict = {'p1': 2.0, 'p2': 1.0}
    f, q = run_phys(c, v_dict)

    assert not f # f should be an empty dict, meaning zero resistive current
    
    expected_q = c.C * (v_dict['p1'] - v_dict['p2'])
    assert jnp.isclose(q['p1'], expected_q)
    assert jnp.isclose(q['p2'], -expected_q)

def test_voltage_source_delay():
    vs = VoltageSource(V=5.0, delay=0.5)
    v_dict = {'p1': 0.0, 'p2': 0.0}
    s_dict = {'i_src': 0.0}

    # Before delay
    f0, q0 = run_phys(vs, v_dict, s_dict, t=0.0)
    assert jnp.isclose(f0['i_src'], 0.0) # Constraint should be (0-0) - 0 = 0
    assert not q0

    # After delay
    f1, q1 = run_phys(vs, v_dict, s_dict, t=1.0)
    expected_constraint = (v_dict['p1'] - v_dict['p2']) - vs.V
    assert jnp.isclose(f1['i_src'], expected_constraint)
    assert not q1

def test_inductor():
    ind = Inductor(L=1e-9)
    v_dict = {'p1': 0.5, 'p2': 0.0}
    s_dict = {'i_L': 0.1}
    
    f, q = run_phys(ind, v_dict, s_dict)
    
    # Check f (resistive part)
    assert jnp.isclose(f['p1'], s_dict['i_L'])
    assert jnp.isclose(f['p2'], -s_dict['i_L'])
    assert jnp.isclose(f['i_L'], v_dict['p1'] - v_dict['p2']) # Branch equation
    
    # Check q (reactive part)
    expected_flux_linkage = -ind.L * s_dict['i_L']
    assert jnp.isclose(q['i_L'], expected_flux_linkage)

def test_diode_forward_bias():
    d = Diode()
    v_dict = {'p1': 0.7, 'p2': 0.0}
    f, q = run_phys(d, v_dict)
    assert f['p1'] > 0.0
    assert jnp.isclose(f['p1'], -f['p2'])
    assert not q

def test_current_source():
    cs = CurrentSource(I=2.0)
    v_dict = {'p1': 0.0, 'p2': 0.0}
    f, q = run_phys(cs, v_dict)
    assert jnp.isclose(f['p1'], cs.I)
    assert jnp.isclose(f['p2'], -cs.I)
    assert not q

def test_vcvs():
    vcvs = VCVS(A=10.0)
    v_dict = {'out_p': 1.0, 'out_m': 0.0, 'ctrl_p': 0.2, 'ctrl_m': 0.0}
    s_dict = {'i_src': 0.0}
    f, q = run_phys(vcvs, v_dict, s_dict)
    
    expected_constraint = (v_dict['out_p'] - v_dict['out_m']) - vcvs.A * (v_dict['ctrl_p'] - v_dict['ctrl_m'])
    assert jnp.isclose(f['i_src'], expected_constraint)
    assert f['ctrl_p'] == 0.0
    assert f['ctrl_m'] == 0.0
    assert not q

def test_ideal_opamp():
    opamp = IdealOpAmp(A=1e6)
    v_dict = {'out_p': 1.0, 'out_m': 0.0, 'in_p': 0.1, 'in_m': 0.0}
    s_dict = {'i_src': 0.0}
    f, q = run_phys(opamp, v_dict, s_dict)
    
    expected_constraint = (v_dict['out_p'] - v_dict['out_m']) - opamp.A * (v_dict['in_p'] - v_dict['in_m'])
    assert jnp.isclose(f['i_src'], expected_constraint)
    assert f['in_p'] == 0.0
    assert f['in_m'] == 0.0
    assert not q

# --- Base Component Tests ---

def test_solver_bridge_resistor():
    r = Resistor(R=100.0)
    # vars_vec = [v_p1, v_p2]
    vars_vec = jnp.array([5.0, 1.0])
    
    f_vec, q_vec = Resistor.solver_bridge(vars_vec, r, t=0.0)
    
    # Expected current
    i = (5.0 - 1.0) / (100.0 + 1e-12)
    
    # f_vec should be [i, -i]
    assert f_vec.shape == (2,)
    assert jnp.allclose(f_vec, jnp.array([i, -i]))
    
    # q_vec should be [0, 0]
    assert q_vec.shape == (2,)
    assert jnp.allclose(q_vec, jnp.zeros(2))

def test_solver_bridge_capacitor():
    c = Capacitor(C=1e-9)
    # vars_vec = [v_p1, v_p2]
    vars_vec = jnp.array([3.0, 0.0])
    
    f_vec, q_vec = Capacitor.solver_bridge(vars_vec, c, t=0.0)
    
    # Expected charge
    q_val = 1e-9 * (3.0 - 0.0)
    
    # f_vec should be [0, 0]
    assert f_vec.shape == (2,)
    assert jnp.allclose(f_vec, jnp.zeros(2))
    
    # q_vec should be [q, -q]
    assert q_vec.shape == (2,)
    assert jnp.allclose(q_vec, jnp.array([q_val, -q_val]))

def test_subclass_init_creates_namedtuples():
    # Define a dummy component
    class MyComp(CircuitComponent):
        ports = ("a", "b")
        states = ("s1",)

        def physics(self, v, s, t):
            return {}, {}
    
    assert MyComp._VarsType_P is not None
    assert MyComp._VarsType_S is not None
    
    p = MyComp._VarsType_P(1, 2)
    assert p.a == 1 and p.b == 2
    
    s = MyComp._VarsType_S(10)
    assert s.s1 == 10