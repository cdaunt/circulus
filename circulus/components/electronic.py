import jax
import jax.numpy as jnp
import jax.nn as jnn
from typing import Callable, Any

# Adjust this import based on where you saved the decorator code
from circulus.components.base_component import component, source, Signals, States

# ===========================================================================
# Passive Components (Time-Invariant)
# ===========================================================================


@component(ports=("p1", "p2"))
def Resistor(signals: Signals, s: States, R: float = 1e3):
    """Ohm's Law: I = V/R"""
    i = (signals.p1 - signals.p2) / (R + 1e-12)
    return {"p1": i, "p2": -i}, {}


@component(ports=("p1", "p2"))
def Capacitor(signals: Signals, s: States, C: float = 1e-12):
    """
    Q = C * V.
    Returns Charge (q) so the solver computes I = dq/dt.
    """
    v_drop = signals.p1 - signals.p2
    q_val = C * v_drop
    return {}, {"p1": q_val, "p2": -q_val}


@component(ports=("p1", "p2"), states=("i_L",))
def Inductor(signals: Signals, s: States, L: float = 1e-9):
    """
    V = L * di/dt  ->  di_L/dt = V / L
    We treat Flux (phi) = L * i_L.
    """
    v_drop = signals.p1 - signals.p2
    # Flow: Current i_L flows p1 -> p2.
    # State Eq: i_L is the variable. We constrain its derivative via flux.
    # However, standard modified nodal analysis often writes:
    #   p1: i_L
    #   p2: -i_L
    #   branch: v1 - v2 - L*di/dt = 0
    #
    # In this (f, q) formulation:
    # f['i_L'] = v1 - v2 (Force / Potential)
    # q['i_L'] = -L * i_L (Momentum / Flux)
    # Result: (v1-v2) - d/dt(L*i_L) = 0  => v = L di/dt
    return ({"p1": s.i_L, "p2": -s.i_L, "i_L": v_drop}, {"i_L": -L * s.i_L})


# ===========================================================================
# Sources (Time-Dependent)
# ===========================================================================


@source(ports=("p1", "p2"), states=("i_src",))
def VoltageSource(
    signals: Signals, s: States, t: float, V: float = 0.0, delay: float = 0.0
):
    """Step voltage source."""
    v_val = jnp.where(t >= delay, V, 0.0)
    constraint = (signals.p1 - signals.p2) - v_val
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


@source(ports=("p1", "p2"), states=("i_src",))
def SmoothPulse(
    signals: Signals,
    s: States,
    t: float,
    V: float = 1.0,
    delay: float = 1e-9,
    tr: float = 1e-10,
):
    """Sigmoid-smoothed pulse."""
    k = 10.0 / tr
    v_val = V * jnn.sigmoid(k * (t - delay))
    constraint = (signals.p1 - signals.p2) - v_val
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


@source(ports=("p1", "p2"), states=("i_src",))
def VoltageSourceAC(
    signals: Signals,
    s: States,
    t: float,
    V: float = 0.0,
    freq: float = 1e6,
    phase: float = 0.0,
    delay: float = 0.0,
):
    """Sinusoidal voltage source."""
    omega = 2.0 * jnp.pi * freq
    v_ac = V * jnp.sin(omega * t + phase)
    v_val = jnp.where(t >= delay, v_ac, 0.0)
    constraint = (signals.p1 - signals.p2) - v_val
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


@component(ports=("p1", "p2"))
def CurrentSource(signals: Signals, s: States, I: float = 0.0):
    """Constant current source."""
    return {"p1": I, "p2": -I}, {}


# ===========================================================================
# Diodes
# ===========================================================================


@component(ports=("p1", "p2"))
def Diode(
    signals: Signals, s: States, Is: float = 1e-12, n: float = 1.0, Vt: float = 25.85e-3
):
    vd = signals.p1 - signals.p2
    # Clip for numerical stability
    vd_safe = jnp.clip(vd, -5.0, 5.0)
    i = Is * (jnp.exp(vd_safe / (n * Vt)) - 1.0)
    return {"p1": i, "p2": -i}, {}


@component(ports=("p1", "p2"))
def ZenerDiode(
    signals: Signals,
    s: States,
    Vz: float = 5.0,
    Is: float = 1e-12,
    n: float = 1.0,
    Vt: float = 25.85e-3,
):
    vd = signals.p1 - signals.p2
    i_fwd = Is * (jnp.exp(vd / (n * Vt)) - 1.0)
    # Zener breakdown modeled as reverse exponential
    i_rev = -Is * (jnp.exp(-(vd + Vz) / (n * Vt)) - 1.0)
    i_total = i_fwd + jnp.where(vd < -Vz, i_rev, 0.0)
    return {"p1": i_total, "p2": -i_total}, {}


# ===========================================================================
# Transistors (MOSFETs)
# ===========================================================================


def _nmos_current(v_d, v_g, v_s, Kp, W, L, Vth, lam):
    """Helper for NMOS DC physics."""
    vgs = v_g - v_s
    vds = v_d - v_s

    beta = Kp * (W / L)
    v_over = vgs - Vth

    linear_current = beta * (v_over * vds - 0.5 * vds**2) * (1 + lam * vds)
    sat_current = (beta / 2.0) * (v_over**2) * (1 + lam * vds)

    return jnp.where(
        vgs <= Vth, 0.0, jnp.where(vds < v_over, linear_current, sat_current)
    )


@component(ports=("d", "g", "s"))
def NMOS(
    signals: Signals,
    s: States,
    Kp: float = 2e-5,
    W: float = 10e-6,
    L: float = 1e-6,
    Vth: float = 1.0,
    lam: float = 0.0,
):
    i_ds = _nmos_current(signals.d, signals.g, signals.s, Kp, W, L, Vth, lam)
    return {"d": i_ds, "g": 0.0, "s": -i_ds}, {}


@component(ports=("d", "g", "s"))
def PMOS(
    signals: Signals,
    s: States,
    Kp: float = 1e-5,
    W: float = 20e-6,
    L: float = 1e-6,
    Vth: float = -1.0,
    lam: float = 0.0,
):
    vsg = signals.s - signals.g
    vsd = signals.s - signals.d

    beta = Kp * (W / L)
    vth_abs = jnp.abs(Vth)
    v_over = vsg - vth_abs

    linear_current = beta * (v_over * vsd - 0.5 * vsd**2) * (1 + lam * vsd)
    sat_current = (beta / 2.0) * (v_over**2) * (1 + lam * vsd)

    i_sd = jnp.where(
        vsg <= vth_abs, 0.0, jnp.where(vsd < v_over, linear_current, sat_current)
    )

    return {"d": -i_sd, "g": 0.0, "s": i_sd}, {}


@component(ports=("d", "g", "s"))
def NMOSDynamic(
    signals: Signals,
    s: States,
    Kp: float = 2e-5,
    W: float = 10e-6,
    L: float = 1e-6,
    Vth: float = 1.0,
    lam: float = 0.0,
    Cox: float = 1e-3,
    Cgd_ov: float = 1e-15,
    Cgs_ov: float = 1e-15,
):
    """NMOS with Meyer Capacitance Model."""
    # 1. DC Current
    i_ds = _nmos_current(signals.d, signals.g, signals.s, Kp, W, L, Vth, lam)
    f_dict = {"d": i_ds, "g": 0.0, "s": -i_ds}

    # 2. Charges
    vgs = signals.g - signals.s
    vds = signals.d - signals.s
    vgd = signals.g - signals.d

    WL = W * L
    Cox_total = Cox * WL
    v_over = vgs - Vth

    cutoff = vgs <= Vth
    saturation = vds >= v_over

    # Meyer Capacitance Logic
    Qg_cut = 0.0
    Qg_sat = (2.0 / 3.0) * Cox_total * v_over
    Qg_lin = 0.5 * Cox_total * v_over

    Qg = jnp.where(cutoff, Qg_cut, jnp.where(saturation, Qg_sat, Qg_lin))
    Qd = jnp.where(cutoff, 0.0, jnp.where(saturation, 0.0, -0.5 * Qg))
    Qs = -Qg - Qd

    Q_gate = Qg + Cgd_ov * vgd + Cgs_ov * vgs
    Q_drain = Qd - Cgd_ov * vgd
    Q_source = Qs - Cgs_ov * vgs

    return f_dict, {"d": Q_drain, "g": Q_gate, "s": Q_source}


# ===========================================================================
# BJTs
# ===========================================================================


def _junction_charge(v, Cj0, Vj, m):
    """JAX-compatible junction capacitance integration."""
    fc = 0.5
    v_thresh = fc * Vj
    # Standard SPICE depletion charge model
    q_normal = (
        -Cj0
        * Vj
        / (1.0 - m)
        * (1.0 - jnp.power(jnp.maximum(0.0, 1.0 - v / Vj), 1.0 - m) - 1.0)
    )

    # Linear extrapolation beyond threshold to prevent NaN
    C_linear = Cj0 / jnp.power(1.0 - fc, m)
    q_high = C_linear * (v - v_thresh) + q_normal

    return jnp.where(v < v_thresh, q_normal, q_high)


@component(ports=("c", "b", "e"))
def BJT_NPN(
    signals: Signals,
    s: States,
    Is: float = 1e-12,
    BetaF: float = 100.0,
    BetaR: float = 1.0,
    Vt: float = 25.85e-3,
):
    vbe = signals.b - signals.e
    vbc = signals.b - signals.c

    alpha_f = BetaF / (1.0 + BetaF)
    alpha_r = BetaR / (1.0 + BetaR)

    vbe_safe = jnp.clip(vbe, -5.0, 2.0)
    vbc_safe = jnp.clip(vbc, -5.0, 2.0)

    i_f = Is * (jnp.exp(vbe_safe / Vt) - 1.0)
    i_r = Is * (jnp.exp(vbc_safe / Vt) - 1.0)

    i_c = alpha_f * i_f - i_r
    i_e = -i_f + alpha_r * i_r
    i_b = -(i_c + i_e)

    return {"c": i_c, "b": i_b, "e": i_e}, {}


@component(ports=("c", "b", "e"))
def BJT_NPN_Dynamic(
    signals: Signals,
    s: States,
    Is: float = 1e-12,
    BetaF: float = 100.0,
    BetaR: float = 1.0,
    Vt: float = 25.85e-3,
    Cje: float = 1e-12,
    Cjc: float = 1e-12,
    Vje: float = 0.75,
    Vjc: float = 0.75,
    Mje: float = 0.33,
    Mjc: float = 0.33,
    Tf: float = 0.0,
    Tr: float = 0.0,
):
    # 1. DC Currents
    vbe = signals.b - signals.e
    vbc = signals.b - signals.c

    vbe_safe = jnp.clip(vbe, -5.0, 2.0)
    vbc_safe = jnp.clip(vbc, -5.0, 2.0)

    alpha_f = BetaF / (1.0 + BetaF)
    alpha_r = BetaR / (1.0 + BetaR)

    i_f = Is * (jnp.exp(vbe_safe / Vt) - 1.0)
    i_r = Is * (jnp.exp(vbc_safe / Vt) - 1.0)

    i_c = alpha_f * i_f - i_r
    i_e = -i_f + alpha_r * i_r
    i_b = -(i_c + i_e)

    f_dict = {"c": i_c, "b": i_b, "e": i_e}

    # 2. Dynamic Charges (Diffusion + Depletion)
    Qje_depl = _junction_charge(vbe, Cje, Vje, Mje)
    Qjc_depl = _junction_charge(vbc, Cjc, Vjc, Mjc)

    Qbe_diff = Tf * i_f
    Qbc_diff = Tr * i_r

    Q_be_total = Qje_depl + Qbe_diff
    Q_bc_total = Qjc_depl + Qbc_diff

    Q_base = Q_be_total + Q_bc_total
    Q_collector = -Q_bc_total
    Q_emitter = -Q_be_total

    return f_dict, {"c": Q_collector, "b": Q_base, "e": Q_emitter}


# ===========================================================================
# Controlled Sources & OpAmps
# ===========================================================================


@component(ports=("out_p", "out_m", "ctrl_p", "ctrl_m"), states=("i_src",))
def VCVS(signals: Signals, s: States, A: float = 1.0):
    """Voltage Controlled Voltage Source."""
    constraint = (signals.out_p - signals.out_m) - A * (signals.ctrl_p - signals.ctrl_m)
    return {
        "out_p": s.i_src,
        "out_m": -s.i_src,
        "ctrl_p": 0.0,
        "ctrl_m": 0.0,
        "i_src": constraint,
    }, {}


@component(ports=("out_p", "out_m", "ctrl_p", "ctrl_m"))
def VCCS(signals: Signals, s: States, G: float = 0.0):
    """Voltage Controlled Current Source."""
    i = G * (signals.ctrl_p - signals.ctrl_m)
    return {"out_p": i, "out_m": -i, "ctrl_p": 0.0, "ctrl_m": 0.0}, {}


@component(ports=("out_p", "out_m", "in_p", "in_m"), states=("i_src",))
def IdealOpAmp(signals: Signals, s: States, A: float = 1e6):
    constraint = (signals.out_p - signals.out_m) - A * (signals.in_p - signals.in_m)
    return {
        "out_p": s.i_src,
        "out_m": -s.i_src,
        "in_p": 0.0,
        "in_m": 0.0,
        "i_src": constraint,
    }, {}


@component(ports=("p1", "p2", "cp", "cm"))
def VoltageControlledSwitch(
    signals: Signals, s: States, Ron: float = 1.0, Roff: float = 1e6, Vt: float = 0.0
):
    v_ctrl = signals.cp - signals.cm
    k = 10.0
    sig = jnn.sigmoid(k * (v_ctrl - Vt))

    g_on = 1.0 / Ron
    g_off = 1.0 / Roff
    g_eff = g_off + (g_on - g_off) * sig

    i = (signals.p1 - signals.p2) * g_eff
    return {"p1": i, "p2": -i, "cp": 0.0, "cm": 0.0}, {}


@component(ports=("out_p", "out_m", "in_p", "in_m"), states=("i_src", "i_ctrl"))
def CCVS(signals: Signals, s: States, R: float = 1.0):
    """
    Current Controlled Voltage Source (Transresistance).

    Physics:
    1. Input side (in_p, in_m) acts as a short circuit (0V drop) to measure current 'i_ctrl'.
    2. Output side (out_p, out_m) acts as a voltage source V = R * i_ctrl.
    """
    # 1. Input Constraint: Short Circuit (v_in = 0)
    #    The variable 'i_ctrl' is the current flowing through this short.
    eq_in = signals.in_p - signals.in_m

    # 2. Output Constraint: Voltage Source (v_out - R*i_in = 0)
    #    The variable 'i_src' is the current delivered by this source.
    eq_out = (signals.out_p - signals.out_m) - (R * s.i_ctrl)

    return {
        # Output side flow (standard voltage source behavior)
        "out_p": s.i_src,
        "out_m": -s.i_src,
        # Input side flow (current 'i_ctrl' enters in_p, leaves in_m)
        "in_p": s.i_ctrl,
        "in_m": -s.i_ctrl,
        # Equations for the state variables
        "i_src": eq_out,  # Enforce V_out = R * I_in
        "i_ctrl": eq_in,  # Enforce V_in = 0
    }, {}


@component(ports=("out_p", "out_m", "in_p", "in_m"), states=("i_ctrl",))
def CCCS(signals: Signals, s: States, alpha: float = 1.0):
    """
    Current Controlled Current Source (Current Gain).

    Physics:
    1. Input side (in_p, in_m) acts as a short circuit to measure 'i_ctrl'.
    2. Output side pushes current I_out = alpha * i_ctrl.
    """
    # 1. Input Constraint: Short Circuit
    eq_in = signals.in_p - signals.in_m

    # 2. Output Current Calculation
    i_out = alpha * s.i_ctrl

    return {
        # Output side flow (Direct current injection)
        "out_p": i_out,
        "out_m": -i_out,
        # Input side flow (Short circuit current path)
        "in_p": s.i_ctrl,
        "in_m": -s.i_ctrl,
        # Equation for the state variable
        "i_ctrl": eq_in,  # Enforce V_in = 0
    }, {}
