import jax
import jax.numpy as jnp
import inspect
import functools

@jax.jit
def smooth_pulse(vars=jnp.zeros(3), params={'delay': 1e-9, 'V': 1.0, 'tr': 1e-10}, t=0.0):
    """
    Pulse with finite rise time 'tr'. 
    Prevents PID controller from stalling on infinite gradients.
    """
    v_a, v_b, i_src = vars[0], vars[1], vars[2]
    
    # Sigmoid smoothing:
    # 0V -> V over approx 'tr' seconds
    # Center the transition at 'delay'
    k = 10.0 / params['tr'] 
    v_val = params['V'] * jax.nn.sigmoid(k * (t - params['delay']))
    
    # Polarity: Va - Vb - V_source = 0
    constraint = (v_a - v_b) - v_val
    return jnp.array([i_src, -i_src, constraint]), jnp.zeros(3)

@jax.jit
def resistor(vars=jnp.zeros(2), params={'R': 1e3}):
    v_a, v_b = vars[0], vars[1]
    # Protect against R=0 division by zero. 
    # Real 0-ohm links should use voltage_source(V=0)
    R_eff = params['R'] + 1e-12 
    i = (v_a - v_b) / R_eff
    return jnp.array([i, -i]), jnp.array([0.0, 0.0])

@jax.jit
def capacitor(vars=jnp.zeros(2), params={'C': 1e-12}):
    v_a, v_b = vars[0], vars[1]
    # Charge q = C * V
    q = params['C'] * (v_a - v_b)
    # Returns (Resistive=0, Reactive=q)
    return jnp.array([0.0, 0.0]), jnp.array([q, -q])

@jax.jit
def voltage_source(vars=jnp.zeros(3), params={'delay':0.0, 'V':0.0}, t=0.0):
    # vars: [NodeA, NodeB, Current_Through_Source]
    v_a, v_b = vars[0], vars[1]
    i_src    = vars[2]
    
    # KCL: Current leaves A, enters B
    kcl = jnp.array([i_src, -i_src])
    
    v_val = jnp.where(t >= params['delay'], params['V'], 0.0)

    constraint = (v_a - v_b) - v_val
    
    f_contrib = jnp.concatenate([kcl, jnp.array([constraint])])
    return f_contrib, jnp.zeros(3)

@jax.jit
def voltage_source_ac(vars=jnp.zeros(3), params={'delay':0.0, 'V':0.0, 'freq':1e6, 'phase':0.0}, t=0.0):
    v_a, v_b = vars[0], vars[1]
    i_src    = vars[2]
    kcl = jnp.array([i_src, -i_src])
    
    omega = 2.0 * jnp.pi * params['freq']
    v_ac = params['V'] * jnp.sin(omega * t + params['phase'])
    v_val = jnp.where(t >= params['delay'], v_ac, 0.0)
    
    constraint = (v_a - v_b) - v_val
    
    f_contrib = jnp.concatenate([kcl, jnp.array([constraint])])
    return f_contrib, jnp.zeros(3)

@jax.jit
def current_source(vars=jnp.zeros(2), params={'I': 0.0}, t=0.0):
    """
    Independent current source from node A to node B.
    
    vars: [v_a, v_b]
    params: {'I': current_value (Amps)}
    
    Current flows OUT of node A and INTO node B.
    """
    v_a, v_b = vars[0], vars[1]
    
    I_val = params['I']
    
    # KCL Contribution:
    # Node A loses current I (-I term in KCL sum? No.)
    # KCL is Sum(Currents LEAVING node) = 0.
    # If source pushes current A -> B:
    # At Node A: Current I is LEAVING. Contribution = +I
    # At Node B: Current I is ENTERING. Contribution = -I
    return jnp.array([I_val, -I_val]), jnp.array([0.0, 0.0])

def inductor(vars=jnp.zeros(3), params={'L': 1e-9}):
    # vars: [NodeA, NodeB, Current_Through_Inductor]
    v_a, v_b, i_branch = vars[0], vars[1], vars[2]
    
    f_kcl = jnp.array([i_branch, -i_branch])
    
    # Branch Equation: V_L - L*di/dt = 0
    # V_L = v_a - v_b
    # In our MNA format: f(y) + d/dt(q(y)) = 0
    # So: (v_a - v_b) + d/dt(-L * i_branch) = 0
    f_branch = v_a - v_b
    q_branch = -params['L'] * i_branch
    
    return jnp.concatenate([f_kcl, jnp.array([f_branch])]), \
           jnp.concatenate([jnp.zeros(2), jnp.array([q_branch])])




@jax.jit
def diode(vars=jnp.zeros(2), params={'Is':1e-12, 'n':1.0, 'Vt': 25.85e-3}):
    v_a, v_b = vars[0], vars[1]
    vd = v_a - v_b
    
    # Limit voltage to prevent overflow during early Newton iterations (Clipping)
    # This is distinct from solver damping, acting as a "physics safeguard"
    vd_safe = jnp.clip(vd, -5.0, 5.0) 
    
    i = params['Is'] * (jnp.exp(vd_safe / (params['n'] * params['Vt'])) - 1.0)
    return jnp.array([i, -i]), jnp.array([0.0, 0.0])

# Renamed to reflect physics reality
@jax.jit
def diode_parallel_leakage(vars=jnp.zeros(2), params={'Is':1e-12, 'n':1.0, 'Vt':25.85e-3, 'R_leak': 1e9}):
    v_a, v_b = vars[0], vars[1]
    vd = v_a - v_b
    
    i_diode = params['Is'] * (jnp.exp(vd / (params['n'] * params['Vt'])) - 1.0)
    i_leak = vd / params['R_leak']
    
    i_total = i_diode + i_leak
    return jnp.array([i_total, -i_total]), jnp.array([0.0, 0.0])


@jax.jit
def _junction_charge(v, Cj0, Vj, m):
    """
    Computes depletion charge Qj for a PN junction.
    Includes linearization near/above built-in potential Vj to prevent singularity.
    """
    # Threshold for linearization (usually 0.5 * Vj)
    fc = 0.5
    v_thresh = fc * Vj
    
    # Region 1: Reverse bias or small forward bias (Standard Model)
    # Q = -Cj0 * Vj / (1-m) * (1 - (1 - V/Vj)^(1-m))
    t1 = (1.0 - v / Vj)
    # Safe power for JIT (avoid NaN gradient if v > Vj inside this branch)
    q_normal = -Cj0 * Vj / (1.0 - m) * (1.0 - jnp.power(jnp.maximum(0.0, t1), 1.0 - m) - 1.0)
    
    # Region 2: Forward bias > v_thresh (Linear Extrapolation)
    # Q = Q(v_thresh) + C(v_thresh) * (v - v_thresh)
    # C(v_thresh) = Cj0 / (1 - fc)^m
    C_linear = Cj0 / jnp.power(1.0 - fc, m)
    # Q(v_thresh) calculation... simplified for performance:
    # We just ensure continuity of Charge and Capacitance.
    # For simulation, just defining C_eff is often enough, but here is the Q form:
    q_linear = q_normal # Approximate match at boundary
    # This is a simplified transition. For full SPICE exactness:
    # F2 = (1 - fc)^-(1+m)
    # Q = Q_thresh + Cj0 * (F1 + (1/F2) * (F3 * (V - fc*Vj) + 0.5 * m * (V^2 ...)))
    # We will use a simplified linear C for V > Vj/2 which is standard for high-speed stability.
    q_high = C_linear * (v - v_thresh) 
    
    # Soft Selection
    return jnp.where(v < v_thresh, q_normal, q_high)


@jax.jit
def nmos_level1(vars=jnp.zeros(3), 
                params={'Kp': 2e-5, 'W': 10e-6, 'L': 1e-6, 'Vth': 1.0, 'lambda': 0.0}, 
                t=0.0):
    """
    NMOS Level 1 (Shichman-Hodges) Model.
    vars: [Drain, Gate, Source]
    """
    vd, vg, vs = vars[0], vars[1], vars[2]
    
    vgs = vg - vs
    vds = vd - vs
    
    beta = params['Kp'] * (params['W'] / params['L'])
    vth = params['Vth']
    lam = params['lambda']
    
    # Region Logic
    cutoff_current = 0.0
    
    v_over = vgs - vth
    linear_current = beta * (v_over * vds - 0.5 * vds**2) * (1 + lam * vds)
    sat_current = (beta / 2.0) * (v_over**2) * (1 + lam * vds)
    
    # Smooth Selection
    i_ds = jnp.where(vgs <= vth, 
                     cutoff_current,
                     jnp.where(vds < v_over, linear_current, sat_current))

    return jnp.array([i_ds, 0.0, -i_ds]), jnp.array([0.0, 0.0, 0.0])


@jax.jit
def pmos_level1(vars=jnp.zeros(3), 
                params={'Kp': 1e-5, 'W': 20e-6, 'L': 1e-6, 'Vth': -1.0, 'lambda': 0.0}, 
                t=0.0):
    """
    PMOS Level 1 Model.
    vars: [Drain, Gate, Source]
    """
    vd, vg, vs = vars[0], vars[1], vars[2]
    
    vsg = vs - vg
    vsd = vs - vd
    
    beta = params['Kp'] * (params['W'] / params['L'])
    vth_abs = jnp.abs(params['Vth'])
    lam = params['lambda']
    
    cutoff_current = 0.0
    
    v_over = vsg - vth_abs
    linear_current = beta * (v_over * vsd - 0.5 * vsd**2) * (1 + lam * vsd)
    sat_current = (beta / 2.0) * (v_over**2) * (1 + lam * vsd)
    
    i_sd = jnp.where(vsg <= vth_abs,
                     cutoff_current,
                     jnp.where(vsd < v_over, linear_current, sat_current))
    
    return jnp.array([i_sd, 0.0, -i_sd]), jnp.array([0.0, 0.0, 0.0])


@jax.jit
def bjt_npn(vars=jnp.zeros(3), 
            params={'Is': 1e-12, 'BetaF': 100.0, 'BetaR': 1.0, 'Vt': 25.85e-3}, 
            t=0.0):
    """
    NPN BJT (Static Ebers-Moll).
    vars: [Collector, Base, Emitter]
    """
    vc, vb, ve = vars[0], vars[1], vars[2]
    
    vbe = vb - ve
    vbc = vb - vc
    
    Is = params['Is']
    Vt = params['Vt']
    alpha_f = params['BetaF'] / (1.0 + params['BetaF'])
    alpha_r = params['BetaR'] / (1.0 + params['BetaR'])
    
    # Safe Exponentials
    vbe_safe = jnp.clip(vbe, -5.0, 2.0)
    vbc_safe = jnp.clip(vbc, -5.0, 2.0)
    
    i_f = Is * (jnp.exp(vbe_safe / Vt) - 1.0)
    i_r = Is * (jnp.exp(vbc_safe / Vt) - 1.0)
    
    i_c = alpha_f * i_f - i_r
    i_e = -i_f + alpha_r * i_r
    i_b = -(i_c + i_e)
    
    return jnp.array([i_c, i_b, i_e]), jnp.array([0.0, 0.0, 0.0])


@jax.jit
def voltage_controlled_switch(vars=jnp.zeros(4), 
                              params={'Ron': 1.0, 'Roff': 1e6, 'Vt': 0.0}, 
                              t=0.0):
    """
    Smooth Voltage Controlled Switch.
    vars: [NodeA, NodeB, Vctrl+, Vctrl-]
    """
    va, vb, vcp, vcm = vars[0], vars[1], vars[2], vars[3]
    v_ctrl = vcp - vcm
    
    k = 10.0 # Switching sharpness
    s = jax.nn.sigmoid(k * (v_ctrl - params['Vt']))
    
    g_on = 1.0 / params['Ron']
    g_off = 1.0 / params['Roff']
    g_eff = g_off + (g_on - g_off) * s
    
    i = (va - vb) * g_eff
    
    return jnp.array([i, -i, 0.0, 0.0]), jnp.zeros(4)


@jax.jit
def zener_diode(vars=jnp.zeros(2), 
                params={'Vz': 5.0, 'Is': 1e-12, 'n': 1.0, 'Vt': 25.85e-3}):
    """
    Zener Diode (Reverse Breakdown).
    """
    v_a, v_b = vars[0], vars[1]
    vd = v_a - v_b
    
    Is = params['Is']
    n = params['n']
    Vt = params['Vt']
    Vz = params['Vz']
    
    i_fwd = Is * (jnp.exp(vd / (n * Vt)) - 1.0)
    
    # Soft breakdown logic
    i_rev = -Is * (jnp.exp(-(vd + Vz) / (n * Vt)) - 1.0)
    
    i_total = i_fwd + jnp.where(vd < -Vz, i_rev, 0.0)
    
    return jnp.array([i_total, -i_total]), jnp.array([0.0, 0.0])

@jax.jit
def nmos_level1_dynamic(vars=jnp.zeros(3), 
                        params={'Kp': 2e-5, 'W': 10e-6, 'L': 1e-6, 'Vth': 1.0, 'lambda': 0.0, 
                                'Cox': 1e-3, 'Cgd_ov': 1e-15, 'Cgs_ov': 1e-15}, 
                        t=0.0):
    """
    NMOS Level 1 with Meyer Capacitance.
    """
    vd, vg, vs = vars[0], vars[1], vars[2]
    vgs = vg - vs
    vds = vd - vs
    vgd = vg - vd
    
    # --- DC Part ---
    beta = params['Kp'] * (params['W'] / params['L'])
    vth = params['Vth']
    lam = params['lambda']
    
    v_over = vgs - vth
    cutoff = vgs <= vth
    saturation = vds >= v_over
    
    i_linear = beta * (v_over * vds - 0.5 * vds**2) * (1 + lam * vds)
    i_sat = (beta / 2.0) * (v_over**2) * (1 + lam * vds)
    i_ds = jnp.where(cutoff, 0.0, jnp.where(saturation, i_sat, i_linear))
    
    # --- Dynamic Part ---
    WL = params['W'] * params['L']
    Cox_total = params['Cox'] * WL 
    
    # Meyer Capacitance Logic
    Qg_cut = 0.0 
    Qg_sat = (2.0/3.0) * Cox_total * v_over
    Qg_lin = 0.5 * Cox_total * v_over 
    
    Qg = jnp.where(cutoff, Qg_cut, jnp.where(saturation, Qg_sat, Qg_lin))
    
    Qd = jnp.where(cutoff, 0.0, jnp.where(saturation, 0.0, -0.5 * Qg))
    Qs = -Qg - Qd 
    
    # Overlap Caps
    Cgd_ov = params['Cgd_ov']
    Cgs_ov = params['Cgs_ov']
    
    Q_gate_total   = Qg + Cgd_ov * vgd + Cgs_ov * vgs
    Q_drain_total  = Qd - Cgd_ov * vgd 
    Q_source_total = Qs - Cgs_ov * vgs
    
    return jnp.array([i_ds, 0.0, -i_ds]), \
           jnp.array([Q_drain_total, Q_gate_total, Q_source_total])


@jax.jit
def bjt_npn_dynamic(vars=jnp.zeros(3), 
                    params={'Is': 1e-12, 'BetaF': 100.0, 'BetaR': 1.0, 'Vt': 25.85e-3,
                            'Cje': 1e-12, 'Cjc': 1e-12, 'Vje': 0.75, 'Vjc': 0.75,
                            'Mje': 0.33, 'Mjc': 0.33, 'Tf': 0.0, 'Tr': 0.0}, 
                    t=0.0):
    """
    NPN BJT with Ebers-Moll + Charge Storage.
    """
    vc, vb, ve = vars[0], vars[1], vars[2]
    vbe = vb - ve
    vbc = vb - vc
    
    # --- DC Part ---
    Is = params['Is']
    Vt = params['Vt']
    BetaF = params['BetaF']
    BetaR = params['BetaR']
    
    vbe_safe = jnp.clip(vbe, -5.0, 2.0)
    vbc_safe = jnp.clip(vbc, -5.0, 2.0)
    
    i_f = Is * (jnp.exp(vbe_safe / Vt) - 1.0)
    i_r = Is * (jnp.exp(vbc_safe / Vt) - 1.0)
    
    alpha_f = BetaF / (1.0 + BetaF)
    alpha_r = BetaR / (1.0 + BetaR)
    
    i_c_dc = alpha_f * i_f - i_r
    i_e_dc = -i_f + alpha_r * i_r
    i_b_dc = -(i_c_dc + i_e_dc)
    
    # --- Dynamic Part ---
    # Helper for junction charge (assumed accessible in scope)
    Qje_depl = _junction_charge(vbe, params['Cje'], params['Vje'], params['Mje'])
    Qjc_depl = _junction_charge(vbc, params['Cjc'], params['Vjc'], params['Mjc'])
    
    # Diffusion Charge
    Qbe_diff = params['Tf'] * i_f
    Qbc_diff = params['Tr'] * i_r
    
    Q_be_total = Qje_depl + Qbe_diff
    Q_bc_total = Qjc_depl + Qbc_diff
    
    Q_base      = Q_be_total + Q_bc_total
    Q_collector = -Q_bc_total
    Q_emitter   = -Q_be_total
    
    return jnp.array([i_c_dc, i_b_dc, i_e_dc]), \
           jnp.array([Q_collector, Q_base, Q_emitter])


def vccs(vars=jnp.zeros(4), params={'G': 0.0}):
    """Voltage-Controlled Current Source (G element).

    vars: [v_out+, v_out-, v_ctrl+, v_ctrl-]
    params: {'G': transconductance}
    Current injected from out+ to out- equals G*(v_ctrl+ - v_ctrl-).
    """
    v_out_p, v_out_m, v_ctrl_p, v_ctrl_m = vars
    i = params.get('G', 0.0) * (v_ctrl_p - v_ctrl_m)
    return jnp.array([i, -i, 0.0, 0.0]), jnp.zeros(4)

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
