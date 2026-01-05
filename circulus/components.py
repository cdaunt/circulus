import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from equinox import field
from typing import ClassVar, Tuple, Callable

from circulus.base_component import CircuitComponent


class Resistor(CircuitComponent):
    R: float | jax.Array = 1E3
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        # 1. Logic
        i = (v.p1 - v.p2) / (self.R + 1e-12)
        
        # 2. Return Currents (f)
        # No need to specify 'states' or 'charges' if they are zero.
        return {"p1": i, "p2": -i}

class Capacitor(CircuitComponent):
    C: float | jax.Array = 1E-12
    
    # Metadata
    ports = ("p1", "p2")
    
    def physics(self, v, s, t):
        """
        v: Ports (v.p1, v.p2)
        s: States (Empty for capacitor)
        t: Time
        """
        # 1. Physics Logic
        # Charge Q = C * (V_p1 - V_p2)
        q_val = self.C * (v.p1 - v.p2)
        
        # 2. Return (Resistive_f, Reactive_q)
        # First Dict (f): Resistive currents. Empty {} means 0.0.
        # Second Dict (q): Dynamic charges. Solver computes I = dq/dt automatically.
        return (
            {}, 
            {"p1": q_val, "p2": -q_val}
        )
    
class Inductor(CircuitComponent):
    L: float | jax.Array = 1E-9
    ports = ("p1", "p2")
    states = ("i_L",)

    def physics(self, v, s, t):
        # v = External Voltages, s = Internal States
        
        # Return Tuple: (Resistive_Dict, Reactive_Dict)
        return (
            # f (Currents / Residuals)
            {"p1": s.i_L, "p2": -s.i_L, "i_L": v.p1 - v.p2},
            
            # q (Charges / Fluxes) -> d/dt
            {"i_L": -self.L * s.i_L}
        )


class VoltageSource(CircuitComponent):
    V: float | jax.Array = 0.0
    delay: float | jax.Array = 0.0
    
    ports = ("p1", "p2")
    states = ("i_src",)  # Internal current variable

    def physics(self, v, s, t):
        # 1. Voltage Logic
        v_val = jnp.where(t >= self.delay, self.V, 0.0)
        
        # 2. Constraint: Potential Difference equals Source Voltage
        # (V_p1 - V_p2) - V_set = 0
        constraint = (v.p1 - v.p2) - v_val
        
        # 3. Return Logic
        # We return a single dictionary because there are no time derivatives (charges).
        # The solver interprets this as "Resistive/Static" equations (f).
        
        # KCL: Current 'i_src' leaves p1 (+), enters p2 (-)
        # Eq:  The constraint applies to the 'i_src' row
        return {
            "p1": s.i_src,    # Current leaving p1
            "p2": -s.i_src,   # Current entering p2
            "i_src": constraint # The voltage equation
        }
    
class VoltageSourceDynamic(CircuitComponent):
    """
    Voltage source defined by a time-dependent callable.
    The waveform function is static (not a learned parameter).
    """
    waveform: Callable = field(static=True)
    
    ports = ("p1", "p2")
    states = ("i_src",)

    def physics(self, v, s, t):
        v_val = self.waveform(t)
        constraint = (v.p1 - v.p2) - v_val
        return {
            "p1": s.i_src,
            "p2": -s.i_src,
            "i_src": constraint
        }

class SmoothPulse(CircuitComponent):
    delay: float | jax.Array = 1e-9
    V: float | jax.Array = 1.0
    tr: float | jax.Array = 1e-10
    
    ports = ("p1", "p2")
    states = ("i_src",)

    def physics(self, v, s, t):
        k = 10.0 / self.tr
        v_val = self.V * jnn.sigmoid(k * (t - self.delay))
        constraint = (v.p1 - v.p2) - v_val
        return {
            "p1": s.i_src,
            "p2": -s.i_src,
            "i_src": constraint
        }

class VoltageSourceAC(CircuitComponent):
    delay: float | jax.Array = 0.0
    V: float | jax.Array = 0.0
    freq: float | jax.Array = 1e6
    phase: float | jax.Array = 0.0
    
    ports = ("p1", "p2")
    states = ("i_src",)

    def physics(self, v, s, t):
        omega = 2.0 * jnp.pi * self.freq
        v_ac = self.V * jnp.sin(omega * t + self.phase)
        v_val = jnp.where(t >= self.delay, v_ac, 0.0)
        constraint = (v.p1 - v.p2) - v_val
        return {
            "p1": s.i_src,
            "p2": -s.i_src,
            "i_src": constraint
        }

class CurrentSource(CircuitComponent):
    I: float | jax.Array = 0.0
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        # Current I flows from p1 to p2 (leaving p1, entering p2)
        return {"p1": self.I, "p2": -self.I}

class Diode(CircuitComponent):
    Is: float | jax.Array = 1e-12
    n: float | jax.Array = 1.0
    Vt: float | jax.Array = 25.85e-3
    
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        vd = v.p1 - v.p2
        vd_safe = jnp.clip(vd, -5.0, 5.0)
        i = self.Is * (jnp.exp(vd_safe / (self.n * self.Vt)) - 1.0)
        return {"p1": i, "p2": -i}

class DiodeParallelLeakage(CircuitComponent):
    Is: float | jax.Array = 1e-12
    n: float | jax.Array = 1.0
    Vt: float | jax.Array = 25.85e-3
    R_leak: float | jax.Array = 1e9
    
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        vd = v.p1 - v.p2
        i_diode = self.Is * (jnp.exp(vd / (self.n * self.Vt)) - 1.0)
        i_leak = vd / self.R_leak
        i_total = i_diode + i_leak
        return {"p1": i_total, "p2": -i_total}

class NMOS(CircuitComponent):
    Kp: float | jax.Array = 2e-5
    W: float | jax.Array = 10e-6
    L: float | jax.Array = 1e-6
    Vth: float | jax.Array = 1.0
    lam: float | jax.Array = 0.0 # lambda
    
    ports = ("d", "g", "s")

    def physics(self, v, s, t):
        vgs = v.g - v.s
        vds = v.d - v.s
        
        beta = self.Kp * (self.W / self.L)
        
        v_over = vgs - self.Vth
        linear_current = beta * (v_over * vds - 0.5 * vds**2) * (1 + self.lam * vds)
        sat_current = (beta / 2.0) * (v_over**2) * (1 + self.lam * vds)
        
        i_ds = jnp.where(vgs <= self.Vth, 
                         0.0,
                         jnp.where(vds < v_over, linear_current, sat_current))
        
        return {"d": i_ds, "g": 0.0, "s": -i_ds}

class PMOS(CircuitComponent):
    Kp: float | jax.Array = 1e-5
    W: float | jax.Array = 20e-6
    L: float | jax.Array = 1e-6
    Vth: float | jax.Array = -1.0
    lam: float | jax.Array = 0.0
    
    ports = ("d", "g", "s")

    def physics(self, v, s, t):
        vsg = v.s - v.g
        vsd = v.s - v.d
        
        beta = self.Kp * (self.W / self.L)
        vth_abs = jnp.abs(self.Vth)
        
        v_over = vsg - vth_abs
        linear_current = beta * (v_over * vsd - 0.5 * vsd**2) * (1 + self.lam * vsd)
        sat_current = (beta / 2.0) * (v_over**2) * (1 + self.lam * vsd)
        
        i_sd = jnp.where(vsg <= vth_abs,
                         0.0,
                         jnp.where(vsd < v_over, linear_current, sat_current))
        
        return {"d": -i_sd, "g": 0.0, "s": i_sd}

class BJT_NPN(CircuitComponent):
    Is: float | jax.Array = 1e-12
    BetaF: float | jax.Array = 100.0
    BetaR: float | jax.Array = 1.0
    Vt: float | jax.Array = 25.85e-3
    
    ports = ("c", "b", "e")

    def physics(self, v, s, t):
        vbe = v.b - v.e
        vbc = v.b - v.c
        
        alpha_f = self.BetaF / (1.0 + self.BetaF)
        alpha_r = self.BetaR / (1.0 + self.BetaR)
        
        vbe_safe = jnp.clip(vbe, -5.0, 2.0)
        vbc_safe = jnp.clip(vbc, -5.0, 2.0)
        
        i_f = self.Is * (jnp.exp(vbe_safe / self.Vt) - 1.0)
        i_r = self.Is * (jnp.exp(vbc_safe / self.Vt) - 1.0)
        
        i_c = alpha_f * i_f - i_r
        i_e = -i_f + alpha_r * i_r
        i_b = -(i_c + i_e)
        
        return {"c": i_c, "b": i_b, "e": i_e}

class VoltageControlledSwitch(CircuitComponent):
    Ron: float | jax.Array = 1.0
    Roff: float | jax.Array = 1e6
    Vt: float | jax.Array = 0.0
    
    ports = ("p1", "p2", "cp", "cm")

    def physics(self, v, s, t):
        v_ctrl = v.cp - v.cm
        k = 10.0
        sig = jnn.sigmoid(k * (v_ctrl - self.Vt))
        
        g_on = 1.0 / self.Ron
        g_off = 1.0 / self.Roff
        g_eff = g_off + (g_on - g_off) * sig
        
        i = (v.p1 - v.p2) * g_eff
        return {"p1": i, "p2": -i, "cp": 0.0, "cm": 0.0}

class ZenerDiode(CircuitComponent):
    Vz: float | jax.Array = 5.0
    Is: float | jax.Array = 1e-12
    n: float | jax.Array = 1.0
    Vt: float | jax.Array = 25.85e-3
    
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        vd = v.p1 - v.p2
        i_fwd = self.Is * (jnp.exp(vd / (self.n * self.Vt)) - 1.0)
        i_rev = -self.Is * (jnp.exp(-(vd + self.Vz) / (self.n * self.Vt)) - 1.0)
        i_total = i_fwd + jnp.where(vd < -self.Vz, i_rev, 0.0)
        return {"p1": i_total, "p2": -i_total}

@jax.jit
def _junction_charge(v, Cj0, Vj, m):
    fc = 0.5
    v_thresh = fc * Vj
    t1 = (1.0 - v / Vj)
    q_normal = -Cj0 * Vj / (1.0 - m) * (1.0 - jnp.power(jnp.maximum(0.0, t1), 1.0 - m) - 1.0)
    C_linear = Cj0 / jnp.power(1.0 - fc, m)
    q_high = C_linear * (v - v_thresh) + q_normal
    return jnp.where(v < v_thresh, q_normal, q_high)

class NMOSDynamic(NMOS):
    Cox: float | jax.Array = 1e-3
    Cgd_ov: float | jax.Array = 1e-15
    Cgs_ov: float | jax.Array = 1e-15

    def physics(self, v, s, t):
        dc_res = super().physics(v, s, t)
        
        vgs = v.g - v.s
        vds = v.d - v.s
        vgd = v.g - v.d
        
        WL = self.W * self.L
        Cox_total = self.Cox * WL
        
        v_over = vgs - self.Vth
        cutoff = vgs <= self.Vth
        saturation = vds >= v_over
        
        Qg_cut = 0.0
        Qg_sat = (2.0/3.0) * Cox_total * v_over
        Qg_lin = 0.5 * Cox_total * v_over
        
        Qg = jnp.where(cutoff, Qg_cut, jnp.where(saturation, Qg_sat, Qg_lin))
        Qd = jnp.where(cutoff, 0.0, jnp.where(saturation, 0.0, -0.5 * Qg))
        Qs = -Qg - Qd
        
        Q_gate = Qg + self.Cgd_ov * vgd + self.Cgs_ov * vgs
        Q_drain = Qd - self.Cgd_ov * vgd
        Q_source = Qs - self.Cgs_ov * vgs
        
        return dc_res, {"d": Q_drain, "g": Q_gate, "s": Q_source}

class BJT_NPN_Dynamic(BJT_NPN):
    Cje: float | jax.Array = 1e-12
    Cjc: float | jax.Array = 1e-12
    Vje: float | jax.Array = 0.75
    Vjc: float | jax.Array = 0.75
    Mje: float | jax.Array = 0.33
    Mjc: float | jax.Array = 0.33
    Tf: float | jax.Array = 0.0
    Tr: float | jax.Array = 0.0

    def physics(self, v, s, t):
        dc_res = super().physics(v, s, t)
        
        vbe = v.b - v.e
        vbc = v.b - v.c
        
        vbe_safe = jnp.clip(vbe, -5.0, 2.0)
        vbc_safe = jnp.clip(vbc, -5.0, 2.0)
        i_f = self.Is * (jnp.exp(vbe_safe / self.Vt) - 1.0)
        i_r = self.Is * (jnp.exp(vbc_safe / self.Vt) - 1.0)
        
        Qje_depl = _junction_charge(vbe, self.Cje, self.Vje, self.Mje)
        Qjc_depl = _junction_charge(vbc, self.Cjc, self.Vjc, self.Mjc)
        
        Qbe_diff = self.Tf * i_f
        Qbc_diff = self.Tr * i_r
        
        Q_be_total = Qje_depl + Qbe_diff
        Q_bc_total = Qjc_depl + Qbc_diff
        
        Q_base = Q_be_total + Q_bc_total
        Q_collector = -Q_bc_total
        Q_emitter = -Q_be_total
        
        return dc_res, {"c": Q_collector, "b": Q_base, "e": Q_emitter}

class VCCS(CircuitComponent):
    G: float | jax.Array = 0.0
    ports = ("out_p", "out_m", "ctrl_p", "ctrl_m")

    def physics(self, v, s, t):
        i = self.G * (v.ctrl_p - v.ctrl_m)
        return {"out_p": i, "out_m": -i, "ctrl_p": 0.0, "ctrl_m": 0.0}

class VCVS(CircuitComponent):
    A: float | jax.Array = 1.0
    ports = ("out_p", "out_m", "ctrl_p", "ctrl_m")
    states = ("i_src",)

    def physics(self, v, s, t):
        constraint = (v.out_p - v.out_m) - self.A * (v.ctrl_p - v.ctrl_m)
        return {
            "out_p": s.i_src, "out_m": -s.i_src,
            "ctrl_p": 0.0, "ctrl_m": 0.0,
            "i_src": constraint
        }

class CCVS(CircuitComponent):
    R: float | jax.Array = 1.0
    ports = ("out_p", "out_m", "in_p", "in_m")
    states = ("i_src", "i_ctrl")

    def physics(self, v, s, t):
        # Input side is a short circuit measuring i_ctrl
        eq_in = v.in_p - v.in_m
        # Output side is voltage source dependent on i_ctrl
        eq_out = (v.out_p - v.out_m) - self.R * s.i_ctrl
        
        return {
            "out_p": s.i_src, "out_m": -s.i_src,
            "in_p": s.i_ctrl, "in_m": -s.i_ctrl,
            "i_src": eq_out,
            "i_ctrl": eq_in
        }

class CCCS(CircuitComponent):
    alpha: float | jax.Array = 1.0
    ports = ("out_p", "out_m", "in_p", "in_m")
    states = ("i_ctrl",)

    def physics(self, v, s, t):
        # Input side is short circuit
        eq_in = v.in_p - v.in_m
        # Output side is current source
        i_out = self.alpha * s.i_ctrl
        
        return {
            "out_p": i_out, "out_m": -i_out,
            "in_p": s.i_ctrl, "in_m": -s.i_ctrl,
            "i_ctrl": eq_in
        }

class Norton(CircuitComponent):
    I: float | jax.Array = 0.0
    G: float | jax.Array = 0.0
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        i_ab = self.I + self.G * (v.p1 - v.p2)
        return {"p1": i_ab, "p2": -i_ab}

class DiodeSeries(CircuitComponent):
    Is: float | jax.Array = 1e-12
    n: float | jax.Array = 1.0
    Vt: float | jax.Array = 25.85e-3
    Rs: float | jax.Array = 0.0
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        vd = v.p1 - v.p2
        j = self.Is * (jnp.exp(vd / (self.n * self.Vt)) - 1.0)
        r_term = jnp.where(self.Rs != 0, vd / self.Rs, 0.0)
        i = j + r_term
        return {"p1": i, "p2": -i}

class Schottky(CircuitComponent):
    Is: float | jax.Array = 1e-6
    n: float | jax.Array = 1.2
    Vt: float | jax.Array = 25.85e-3
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        vd = v.p1 - v.p2
        i = self.Is * (jnp.exp(vd / (self.n * self.Vt)) - 1.0)
        return {"p1": i, "p2": -i}

class IdealOpAmp(CircuitComponent):
    A: float | jax.Array = 1e6
    ports = ("out_p", "out_m", "in_p", "in_m")
    states = ("i_src",)

    def physics(self, v, s, t):
        constraint = (v.out_p - v.out_m) - self.A * (v.in_p - v.in_m)
        return {
            "out_p": s.i_src, "out_m": -s.i_src,
            "in_p": 0.0, "in_m": 0.0,
            "i_src": constraint
        }
if __name__ == "__main__":
    a = Resistor()
    print(a)