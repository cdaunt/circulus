import jax
import jax.numpy as jnp
import diffrax
import pytest

from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_dc_op_dense, s_to_y
from circulus.solvers.dense import VectorizedDenseSolver as DenseSolver
from circulus.base_component import CircuitComponent


def test_short_transient_runs(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Solve DC operating point and run a very short transient with zero forcing
    y_op = solve_dc_op_dense(groups, sys_size, t0=0.0)

    solver = DenseSolver()
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    t_max = 1e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 5))

    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_max, dt0=1e-3 * t_max,
        y0=y_op, args=(groups, sys_size), saveat=saveat, max_steps=1000
    )

    assert sol.ys.shape == (5, sys_size)
    assert jnp.isfinite(sol.ys).all()


class OpticalWaveguide(CircuitComponent):
    length_um: float = 100.0
    loss_dB_cm: float = 1.0
    neff: float = 2.4
    n_group: float = 4.0
    center_wavelength_nm: float = 1310.0
    wavelength_nm: float = 1310.0
    ports = ("p1", "p2")
    
    def physics(self, v, s, t):
        d_lam = self.wavelength_nm - self.center_wavelength_nm
        slope = (self.neff - self.n_group) / self.center_wavelength_nm
        n_eff_disp = self.neff + slope * d_lam
        phi = 2.0 * jnp.pi * n_eff_disp * (self.length_um / self.wavelength_nm) * 1000.0
        loss_val = self.loss_dB_cm * (self.length_um / 10000.0)
        T_mag = 10.0 ** (-loss_val / 20.0)
        T = T_mag * jnp.exp(-1j * phi)
        S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
        Y = s_to_y(S)
        v_vec = jnp.array([v.p1, v.p2], dtype=jnp.complex128)
        i_vec = Y @ v_vec
        return {"p1": i_vec[0], "p2": i_vec[1]}

class PulsedCurrentSource(CircuitComponent):
    I_mag: float = 1.0
    delay: float = 0.2e-9
    rise: float = 0.05e-9
    ports = ("p", "n")
    
    def physics(self, v, s, t):
        val = self.I_mag * jax.nn.sigmoid((t - self.delay)/self.rise)
        return {"p": -val, "n": val}

class Resistor(CircuitComponent):
    R: float = 50.0
    ports = ("p", "n")
    def physics(self, v, s, t):
        i = (v.p - v.n) / self.R
        return {"p": i, "n": -i}

def test_dense_complex_transient():
    models_map = {'waveguide': OpticalWaveguide, 'source': PulsedCurrentSource, 'resistor': Resistor, 'ground': lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "I1": {"component": "source", "settings": {"I_mag": 1.0, "delay": 0.1e-9}},
            "WG1": {"component": "waveguide", "settings": {"length_um": 100.0}},
            "R1": {"component": "resistor", "settings": {"R": 1.0}}
        },
        "connections": {"GND,p1": ("I1,n", "R1,n"), "I1,p": "WG1,p1", "WG1,p2": "R1,p"}
    }
    
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    y_op = solve_dc_op_dense(groups, sys_size, t0=0.0, dtype=jnp.complex128)
    
    solver = DenseSolver()
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))
    t_max = 0.5e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 20))
    
    sol = diffrax.diffeqsolve(term, solver, t0=0.0, t1=t_max, dt0=1e-11, y0=y_op, args=(groups, sys_size), saveat=saveat, max_steps=1000)
    
    assert sol.ys.shape == (20, sys_size)
    assert jnp.isfinite(sol.ys).all()
    assert jnp.abs(sol.ys[-1, port_map["R1,p"]]) > 0.1
