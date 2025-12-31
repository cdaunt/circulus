
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import diffrax
import optimistix as optx
import inspect
from typing import NamedTuple, Callable, List, Dict, Any
import matplotlib.pyplot as plt

# Enable 64-bit precision (Critical for Circuit Simulation)
jax.config.update("jax_enable_x64", True)


# =========================================================================
# SECTION 1: PHYSICS LIBRARIES
# Protocol: def model(params, vars=jnp.zeros(N))
# The default value of 'vars' tells the compiler the size of the model.
# =========================================================================


def resistor(vars=jnp.zeros(2), params={'R':0}):
    v_a, v_b = vars[0], vars[1]
    i = (v_a - v_b) / params['R']
    return jnp.array([i, -i]), jnp.array([0.0, 0.0])

def capacitor(vars=jnp.zeros(2), params={'C':1E-12}):
    v_a, v_b = vars[0], vars[1]
    q = params['C'] * (v_a - v_b)
    return jnp.array([0.0, 0.0]), jnp.array([q, -q])

def voltage_source(vars=jnp.zeros(3), params={'delay':0.0, 'V':0.0}, t=0.0):
    v_a, v_b = vars[0], vars[1]
    i_src    = vars[2]
    kcl = jnp.array([i_src, -i_src])
    v_output = jnp.array(jnp.where(t > params['delay'], params['V'], 0.0))
    constraint = v_a - v_b + v_output
    f_contrib = jnp.concatenate([kcl, jnp.array([constraint])])
    return f_contrib, jnp.zeros(3)

def inductor(vars=jnp.zeros(3), params={'L':1E-9}):
    v_a, v_b, i_branch = vars[0], vars[1], vars[2]
    f_kcl = jnp.array([i_branch, -i_branch])
    f_branch = v_a - v_b
    q_branch = -params['L'] * i_branch
    return jnp.concatenate([f_kcl, jnp.array([f_branch])]), \
           jnp.concatenate([jnp.zeros(2), jnp.array([q_branch])])