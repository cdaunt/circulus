import jax
import jax.numpy as jnp

from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_dc_op_dense, s_to_y
from circulus.utils import update_params_dict, update_group_params
import matplotlib.pyplot as plt
import time
from circulus.base_component import CircuitComponent
from circulus.components import VoltageSource, Resistor

class Grating(CircuitComponent):
    center_wavelength_nm: float | jnp.ndarray = 1310.0
    peak_loss_dB: float | jnp.ndarray = 0.0
    bandwidth_1dB: float | jnp.ndarray = 20.0
    
    # Explicit parameter for the simulation wavelength
    wavelength_nm: float | jnp.ndarray = 1310.0
    
    ports = ("grating", "waveguide")

    @jax.jit
    def physics(self, v, s, t):
        # Use the explicit parameter, ignoring 't' (time)
        delta = self.wavelength_nm - self.center_wavelength_nm
        # Loss increases by 1dB at delta = bandwidth/2
        # Loss = Peak + ((t - center) / (BW/2))^2
        excess_loss = (delta / (0.5 * self.bandwidth_1dB))**2
        loss_dB = self.peak_loss_dB + excess_loss
        
        # S-param magnitude
        T = 10.0 ** (-loss_dB / 20.0)
        # Clip to avoid singularity in Y-matrix (perfect transmission = infinite admittance)
        T = jnp.minimum(T, 0.9999)
        
        # S-matrix (Reciprocal, Symmetric)
        # S = [[0, T], [T, 0]]
        S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
        
        Y = s_to_y(S)
        
        # I = Y * V
        v_vec = jnp.array([v.grating, v.waveguide], dtype=jnp.complex128)
        i_vec = Y @ v_vec
        
        return {"grating": i_vec[0], "waveguide": i_vec[1]}

class OpticalWaveguide(CircuitComponent):
    length_um: float | jnp.ndarray = 100.0
    loss_dB_cm: float | jnp.ndarray = 1.0
    neff: float | jnp.ndarray = 2.4
    n_group: float | jnp.ndarray = 4.0
    center_wavelength_nm: float | jnp.ndarray = 1310.0
    
    wavelength_nm: float | jnp.ndarray = 1310.0
    
    ports = ("p1", "p2")
    
    @jax.jit
    def physics(self, v, s, t):
        # Dispersion: n_eff(lambda)
        d_lam = self.wavelength_nm - self.center_wavelength_nm
        slope = (self.neff - self.n_group) / self.center_wavelength_nm
        n_eff_disp = self.neff + slope * d_lam
        
        # Phase: phi = 2*pi * n * L / lambda
        # Units: L(um), lambda(nm). Factor 1000 converts nm to um in denominator
        phi = 2.0 * jnp.pi * n_eff_disp * (self.length_um / self.wavelength_nm) * 1000.0
        
        # Loss: dB/cm * L_cm
        loss_val = self.loss_dB_cm * (self.length_um / 10000.0)
        T_mag = 10.0 ** (-loss_val / 20.0)
        
        # Complex Transmission
        T = T_mag * jnp.exp(-1j * phi)
        
        S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
        Y = s_to_y(S)
        
        v_vec = jnp.array([v.p1, v.p2], dtype=jnp.complex128)
        i_vec = Y @ v_vec
        
        return {"p1": i_vec[0], "p2": i_vec[1]}

class Splitter(CircuitComponent):
    split_ratio: float | jnp.ndarray = 0.5 # Power ratio to port 2
    ports = ("p1", "p2", "p3") # In, Out1, Out2
    
    def physics(self, v, s, t):
        r = jnp.sqrt(self.split_ratio)
        tc = jnp.sqrt(1.0 - self.split_ratio)
        
        # 3-Port S-matrix (Behavioral Model)
        # p1 -> p2 (r), p1 -> p3 (j*t)
        S = jnp.array([
            [0.0, r, 1j*tc],
            [r, 0.0, 0.0],
            [1j*tc, 0.0, 0.0]
        ], dtype=jnp.complex128)
        
        Y = s_to_y(S)
        v_vec = jnp.array([v.p1, v.p2, v.p3], dtype=jnp.complex128)
        i_vec = Y @ v_vec
        
        return {"p1": i_vec[0], "p2": i_vec[1], "p3": i_vec[2]}

if __name__ == "__main__":
    print("\n--- DEMO: Photonic Splitter & Grating Link (Wavelength Sweep) ---")
    
    models_map = {
        'grating': Grating,
        'waveguide': OpticalWaveguide,
        'splitter': Splitter,
        'source': VoltageSource, # Acts as a Phasor Source if V is complex
        'resistor': Resistor,    # Acts as a Matched Load (Z0=1)
        'ground': lambda: 0
    }

    # Circuit: Laser -> Waveguide -> Splitter -> (Arm1, Arm2) -> Gratings -> Loads
    net_dict = {
        "instances": {
            "GND": {"component":"ground"},
            "Laser": {"component":"source", "settings":{"V": 1.0 + 0j}}, # 1.0 Amplitude, 0 Phase
            "WG_In": {"component":"waveguide", "settings":{"length_um": 100.0}},
            "Splitter": {"component":"splitter", "settings":{"split_ratio": 0.5}},
            
            "WG_Arm1": {"component":"waveguide", "settings":{"length_um": 500.0}},
            "GC1":   {"component":"grating", "settings":{"peak_loss_dB": 1.0, "bandwidth_1dB": 40.0}},
            "Load1":  {"component":"resistor", "settings":{"R": 1.0}},
            
            "WG_Arm2": {"component":"waveguide", "settings":{"length_um": 50.0}},
            "GC2":   {"component":"grating", "settings":{"peak_loss_dB": 1.0, "bandwidth_1dB": 40.0, "center_wavelength_nm":1300}},
            "Load2":  {"component":"resistor", "settings":{"R": 1.0}},
        },
        "connections": {
            "GND,p1": ("Laser,p2", "Load1,p2", "Load2,p2"),
            "Laser,p1": "WG_In,p1",
            "WG_In,p2": "Splitter,p1",
            
            # Arm 1
            "Splitter,p2": "WG_Arm1,p1",
            "WG_Arm1,p2": "GC1,grating",
            "GC1,waveguide": "Load1,p1",
            
            # Arm 2
            "Splitter,p3": "WG_Arm2,p1",
            "WG_Arm2,p2": "GC2,grating",
            "GC2,waveguide": "Load2,p1",
        },
    }

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    
    # --- Sweep Wavelength ---
    wavelengths = jnp.linspace(1260, 1360, 200)
    
    print("Sweeping Wavelength...")
    @jax.jit
    def solve_for_loss(val):
        g = groups
        # Update Gratings
        g = update_group_params(g, 'grating', 'wavelength_nm', val)
        # Update Waveguides
        g = update_group_params(g, 'waveguide', 'wavelength_nm', val)
        
        return solve_dc_op_dense(g, sys_size, dtype=jnp.complex128)
    
    # JAX VMAP
    #solve_for_loss(1310)
    start = time.time()
    solutions = jax.vmap(solve_for_loss)(wavelengths)
    total = time.time() - start
    print(total)

    # Extract Output Fields
    v_out1 = solutions[:, port_map["Load1,p1"]]
    v_out2 = solutions[:, port_map["Load2,p1"]]
    
    # Calculate Power (|E|^2)
    p_out1_db = 10.0 * jnp.log10(jnp.abs(v_out1)**2 + 1e-12)
    p_out2_db = 10.0 * jnp.log10(jnp.abs(v_out2)**2 + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, p_out1_db, 'b-', label='Port 1 (Split)')
    plt.plot(wavelengths, p_out2_db, 'r--', label='Port 2 (Split)')
    
    plt.title("Splitter + Grating Response")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Received Power (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()