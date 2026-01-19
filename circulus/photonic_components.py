import jax
import jax.numpy as jnp
from circulus.base_component import CircuitComponent
from circulus.s_transforms import s_to_y

class OpticalWaveguide(CircuitComponent):
    length_um: float | jnp.ndarray = 100.0
    loss_dB_cm: float | jnp.ndarray = 1.0
    neff: float | jnp.ndarray = 2.4
    n_group: float | jnp.ndarray = 4.0
    center_wavelength_nm: float | jnp.ndarray = 1310.0
    wavelength_nm: float | jnp.ndarray = 1310.0
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
        return {"p1": i_vec[0], "p2": i_vec[1]}, {}

class Grating(CircuitComponent):
    center_wavelength_nm: float | jnp.ndarray = 1310.0
    peak_loss_dB: float | jnp.ndarray = 0.0
    bandwidth_1dB: float | jnp.ndarray = 20.0
    wavelength_nm: float | jnp.ndarray = 1310.0
    ports = ("grating", "waveguide")

    def physics(self, v, s, t):
        delta = self.wavelength_nm - self.center_wavelength_nm
        excess_loss = (delta / (0.5 * self.bandwidth_1dB))**2
        loss_dB = self.peak_loss_dB + excess_loss
        T = 10.0 ** (-loss_dB / 20.0)
        T = jnp.minimum(T, 0.9999)
        S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
        Y = s_to_y(S)
        v_vec = jnp.array([v.grating, v.waveguide], dtype=jnp.complex128)
        i_vec = Y @ v_vec
        return {"grating": i_vec[0], "waveguide": i_vec[1]}, {}

class Splitter(CircuitComponent):
    split_ratio: float | jnp.ndarray = 0.5
    ports = ("p1", "p2", "p3")
    
    def physics(self, v, s, t):
        r = jnp.sqrt(self.split_ratio)
        tc = jnp.sqrt(1.0 - self.split_ratio)
        S = jnp.array([
            [0.0, r, 1j*tc],
            [r, 0.0, 0.0],
            [1j*tc, 0.0, 0.0]
        ], dtype=jnp.complex128)
        Y = s_to_y(S)
        v_vec = jnp.array([v.p1, v.p2, v.p3], dtype=jnp.complex128)
        i_vec = Y @ v_vec
        return {"p1": i_vec[0], "p2": i_vec[1], "p3": i_vec[2]}, {}

class OpticalSource(CircuitComponent):
    """Phasor Source for DC/AC analysis"""
    power: float | jnp.ndarray = 1.0
    phase: float | jnp.ndarray = 0.0
    ports = ("p1", "p2")
    states = ("i_src",)

    def physics(self, v, s, t):
        v_val = jnp.sqrt(self.power) * jnp.exp(1j * self.phase)
        constraint = (v.p1 - v.p2) - v_val
        return {
            "p1": s.i_src, "p2": -s.i_src,
            "i_src": constraint
        }, {}

class OpticalSourcePulse(CircuitComponent):
    """Transient Source with sigmoid rise"""
    power: float | jnp.ndarray = 1.0
    phase: float | jnp.ndarray = 0.0
    delay: float | jnp.ndarray = 0.2e-9
    rise: float | jnp.ndarray = 0.05e-9
    ports = ("p1", "p2")
    states = ("i_src",)
    
    def physics(self, v, s, t):
        val = jnp.sqrt(self.power) * jax.nn.sigmoid((t - self.delay)/self.rise)
        v_val = val * jnp.exp(1j * self.phase)
        constraint = (v.p1 - v.p2) - v_val
        return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}