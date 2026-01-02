
import jax
import jax.numpy as jnp

# Enable 64-bit precision (Critical for Circuit Simulation)
jax.config.update("jax_enable_x64", True)


# =========================================================================
# SECTION 1: PHYSICS LIBRARIES
# Protocol: def model(params, vars=jnp.zeros(N))
# The default value of 'vars' tells the compiler the size of the model.
# =========================================================================

@jax.jit
def resistor(vars=jnp.zeros(2), params={'R':0}):
    v_a, v_b = vars[0], vars[1]
    i = (v_a - v_b) / params['R']
    return jnp.array([i, -i]), jnp.array([0.0, 0.0])

@jax.jit
def capacitor(vars=jnp.zeros(2), params={'C':1E-12}):
    v_a, v_b = vars[0], vars[1]
    q = params['C'] * (v_a - v_b)
    return jnp.array([0.0, 0.0]), jnp.array([q, -q])

@jax.jit
def voltage_source(vars=jnp.zeros(3), params={'delay':0.0, 'V':0.0}, t=0.0):
    v_a, v_b = vars[0], vars[1]
    i_src    = vars[2]
    kcl = jnp.array([i_src, -i_src])
    v_output = jnp.array(jnp.where(t >= params['delay'], params['V'], 0.0))
    constraint = v_a - v_b + v_output
    f_contrib = jnp.concatenate([kcl, jnp.array([constraint])])
    return f_contrib, jnp.zeros(3)

@jax.jit
def voltage_source_ac(vars=jnp.zeros(3), params={'delay':0.0, 'V':0.0, 'freq':1e6, 'phase':0.0}, t=0.0):
    v_a, v_b = vars[0], vars[1]
    i_src    = vars[2]
    kcl = jnp.array([i_src, -i_src])
    v_output = jnp.array(jnp.where(t >= params['delay'], params['V']*jnp.sin(2*jnp.pi*params['freq']*t + params['phase']), 0.0))
    constraint = v_a - v_b + v_output
    f_contrib = jnp.concatenate([kcl, jnp.array([constraint])])
    return f_contrib, jnp.zeros(3)

@jax.jit
def inductor(vars=jnp.zeros(3), params={'L':1E-9}):
    v_a, v_b, i_branch = vars[0], vars[1], vars[2]
    f_kcl = jnp.array([i_branch, -i_branch])
    f_branch = v_a - v_b
    q_branch = -params['L'] * i_branch
    return jnp.concatenate([f_kcl, jnp.array([f_branch])]), \
           jnp.concatenate([jnp.zeros(2), jnp.array([q_branch])])


@jax.jit
def current_source(vars=jnp.zeros(2), params={'I': 0.0}, t=0.0):
    """Independent current source from node A to node B.

    vars: [v_a, v_b]
    params: {'I': current_value}
    Returns KCL contributions and zero charges.
    """
    v_a, v_b = vars[0], vars[1]
    I_val = params.get('I', 0.0)
    return jnp.array([I_val, -I_val]), jnp.array([0.0, 0.0])


@jax.jit
def diode(vars=jnp.zeros(2), params={'Is':1e-12, 'n':1.0, 'Vt': 25.85e-3}):
    """Shockley diode model (I = Is * (exp(Vd/(n*Vt)) - 1)).

    vars: [v_a, v_b]
    params: {'Is', 'n', 'Vt'}
    """
    v_a, v_b = vars[0], vars[1]
    vd = v_a - v_b
    Is = params.get('Is', 1e-12)
    n = params.get('n', 1.0)
    Vt = params.get('Vt', 25.85e-3)

    i = Is * (jnp.exp(vd / (n * Vt)) - 1.0)
    return jnp.array([i, -i]), jnp.array([0.0, 0.0])


@jax.jit
def vcvs(vars=jnp.zeros(5), params={'A': 1.0}):
    """Voltage-Controlled Voltage Source (E element).

    vars: [v_out+, v_out-, i_src, v_ctrl+, v_ctrl-]
    params: {'A': gain}
    Equations: KCL at output nodes and voltage constraint
    v_out+ - v_out- - A*(v_ctrl+ - v_ctrl-) = 0
    """
    v_out_p, v_out_m, i_src, v_ctrl_p, v_ctrl_m = vars
    kcl = jnp.array([i_src, -i_src])
    constraint = v_out_p - v_out_m - params.get('A', 1.0) * (v_ctrl_p - v_ctrl_m)

    # Return residuals for each variable (constraint only affects third entry)
    return jnp.array([kcl[0], kcl[1], constraint, 0.0, 0.0]), jnp.zeros(5)


@jax.jit
def vccs(vars=jnp.zeros(4), params={'G': 0.0}):
    """Voltage-Controlled Current Source (G element).

    vars: [v_out+, v_out-, v_ctrl+, v_ctrl-]
    params: {'G': transconductance}
    Current injected from out+ to out- equals G*(v_ctrl+ - v_ctrl-).
    """
    v_out_p, v_out_m, v_ctrl_p, v_ctrl_m = vars
    i = params.get('G', 0.0) * (v_ctrl_p - v_ctrl_m)
    return jnp.array([i, -i, 0.0, 0.0]), jnp.zeros(4)


# -------------------------------------------------------------------------
# Additional SPICE-like components (default set)
# -------------------------------------------------------------------------

@jax.jit
def ccvs(vars=jnp.zeros(4), params={'R': 1.0}):
    """Current-Controlled Voltage Source (H element).

    vars: [v_out+, v_out-, i_src, i_ctrl]
    params: {'R': transresistance (V per Amp)}
    Constraint: v_out+ - v_out- - R * i_ctrl = 0
    """
    v_out_p, v_out_m, i_src, i_ctrl = vars
    constraint = v_out_p - v_out_m - params.get('R', 1.0) * i_ctrl
    # KCL: i_src and -i_src on the first two nodes, constraint on the third
    return jnp.array([i_src, -i_src, constraint, 0.0]), jnp.zeros(4)


@jax.jit
def cccs(vars=jnp.zeros(3), params={'alpha': 1.0}):
    """Current-Controlled Current Source (F element).

    vars: [v_out+, v_out-, i_ctrl]
    params: {'alpha': gain (A_out per A_ctrl)}
    """
    v_out_p, v_out_m, i_ctrl = vars
    i_out = params.get('alpha', 1.0) * i_ctrl
    return jnp.array([i_out, -i_out, 0.0]), jnp.zeros(3)


@jax.jit
def norton(vars=jnp.zeros(2), params={'I': 0.0, 'G': 0.0}):
    """Norton equivalent: current source in parallel with conductance.

    vars: [v_a, v_b]
    params: {'I': current from a->b, 'G': conductance between a and b}
    """
    v_a, v_b = vars
    I = params.get('I', 0.0)
    G = params.get('G', 0.0)
    i_ab = I + G * (v_a - v_b)
    return jnp.array([i_ab, -i_ab]), jnp.array([0.0, 0.0])


@jax.jit
def diode_series(vars=jnp.zeros(2), params={'Is':1e-12, 'n':1.0, 'Vt':25.85e-3, 'Rs': 0.0}):
    """Diode with series resistance approximation.

    i = Is*(exp(vd/(n*Vt)) - 1) + vd / Rs
    (Rs=0 allowed => pure diode)
    """
    v_a, v_b = vars
    vd = v_a - v_b
    Is = params.get('Is', 1e-12)
    n = params.get('n', 1.0)
    Vt = params.get('Vt', 25.85e-3)
    Rs = params.get('Rs', 0.0)

    j = Is * (jnp.exp(vd / (n * Vt)) - 1.0)
    r_term = jnp.where(Rs!=0, vd / Rs, 0.0)
    i = j + r_term
    return jnp.array([i, -i]), jnp.array([0.0, 0.0])


@jax.jit
def schottky(vars=jnp.zeros(2), params={'Is':1e-6, 'n':1.2, 'Vt':25.85e-3}):
    """Schottky diode (simple exponential model with larger Is).
    vars: [v_a, v_b]
    """
    v_a, v_b = vars
    vd = v_a - v_b
    Is = params.get('Is', 1e-6)
    n = params.get('n', 1.2)
    Vt = params.get('Vt', 25.85e-3)

    i = Is * (jnp.exp(vd / (n * Vt)) - 1.0)
    return jnp.array([i, -i]), jnp.array([0.0, 0.0])


@jax.jit
def ideal_opamp(vars=jnp.zeros(5), params={'A': 1e6}):
    """Idealized operational amplifier modeled as a very high gain VCVS.

    vars: [v_out+, v_out-, i_src, v_in+, v_in-]
    params: {'A': open-loop gain}
    Constraint: v_out+ - v_out- - A*(v_in+ - v_in-) = 0
    """
    v_out_p, v_out_m, i_src, v_in_p, v_in_m = vars
    constraint = v_out_p - v_out_m - params.get('A', 1e6) * (v_in_p - v_in_m)
    kcl = jnp.array([i_src, -i_src])
    return jnp.array([kcl[0], kcl[1], constraint, 0.0, 0.0]), jnp.zeros(5)


def behavioral_voltage_source(vars=jnp.zeros(3), params={'func': None}, t=0.0):
    """Behavioral voltage source (B element).

    vars: [v_out+, v_out-, i_src]
    params: {'func': callable(vars, params, t) -> voltage}

    Note: This function is intentionally NOT jitted because user-provided 'func'
    may be a Python callable that is not jax-transformable.
    """
    v_out_p, v_out_m, i_src = vars
    func = params.get('func', None)
    if func is None:
        raise ValueError("behavioral_voltage_source requires a 'func' parameter")
    v_val = func(vars, params, t)
    constraint = v_out_p - v_out_m - v_val
    return jnp.array([i_src, -i_src, constraint]), jnp.zeros(3)

