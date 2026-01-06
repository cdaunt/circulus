import jax
import jax.numpy as jnp

from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_dc_op_dense
from circulus.utils import update_param_dict
import matplotlib.pyplot as plt

from circulus.base_component import CircuitComponent
from circulus.components import VoltageSource, Resistor

class Grating(CircuitComponent):
    loss_dB: float | jnp.ndarray = 0.0
    ports = ("grating", "waveguide")

    def physics(self, v, s, t):
        # Convert dB loss to linear transmission amplitude (T)
        # T = 10^(-loss_dB / 20)
        T = 10.0 ** (-self.loss_dB / 20.0)
        
        # S-Matrix to Y-Matrix conversion for a matched 2-port:
        # S = [[0, T], [T, 0]]
        # Y = (I - S) * (I + S)^-1
        # This handles the "flow" of the optical field through the component.
        
        # Singularity Check: T=1.0 implies a perfect short (Infinite Admittance).
        # We clamp slightly to keep numbers finite.
        T_safe = jnp.minimum(T, 0.999)
        
        denom = 1.0 - T_safe**2
        y_self = (1.0 + T_safe**2) / denom
        y_trans = -2.0 * T_safe / denom
        
        # I = Y * V
        i_g = y_self * v.grating + y_trans * v.waveguide
        i_w = y_trans * v.grating + y_self * v.waveguide
        
        return {"grating": i_g, "waveguide": i_w}

if __name__ == "__main__":
    print("\n--- DEMO: Photonic Grating Link (Complex Field Solve) ---")
    
    models_map = {
        #'grating': Grating,
        'grating': Grating,
        'source': VoltageSource, # Acts as a Phasor Source if V is complex
        'resistor': Resistor,    # Acts as a Matched Load (Z0=1)
        'ground': lambda: 0
    }

    # Simple Link: Source -> Grating -> Load
    net_dict = {
        "instances": {
            "GND": {"component":"ground"},
            "Laser": {"component":"source", "settings":{"V": 1.0 + 0j}}, # 1.0 Amplitude, 0 Phase
            "GC1":   {"component":"grating", "settings":{"loss_dB": 3.0}}, # 3dB Loss
            "Load":  {"component":"resistor", "settings":{"R": 1.0}},     # Matched Load (Z0=1)
        },
        "connections": {
            "GND,p1": ("Laser,p2", "Load,p2"),
            "Laser,p1": "GC1,grating",
            "GC1,waveguide": "Load,p1"
        },
    }

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    
    # --- Sweep Loss ---
    sweep_loss = jnp.linspace(0.0, 10.0, 50) # 0 to 10 dB
    
    def solve_for_loss(l_val):
        new_groups = update_param_dict(groups, 'grating', 'GC1', 'loss_dB', l_val)
        return solve_dc_op_dense(new_groups, sys_size, dtype=jnp.complex128)

    #print("Sweeping Grating Loss...")
    solutions = jax.vmap(solve_for_loss)(sweep_loss)
    
    # Extract Output Field at Load
    v_out = solutions[:, port_map["Load,p1"]]
    
    # Calculate Power (|E|^2)
    p_out = jnp.abs(v_out)**2
    p_out_db = 10.0 * jnp.log10(p_out + 1e-12)

    plt.figure(figsize=(8, 4))
    #plt.plot(sweep_voltage, p_out, 'b-', label='Output Power (dB)')
    #plt.plot(sweep_loss, -sweep_loss, 'r--', label='Ideal Loss')
    plt.plot(sweep_loss, p_out_db, 'b-', label='Output Power (dB)')
    plt.plot(sweep_loss, -sweep_loss, 'r--', label='Ideal Loss')
    
    plt.title("Grating Coupler Transmission Test")
    plt.xlabel("Grating Loss Setting (dB)")
    #plt.xlabel("Input signal (V)")
    plt.ylabel("Received Power (linear)")
    plt.legend()
    plt.grid(True)
    plt.show()