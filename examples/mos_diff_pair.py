import jax
import jax.numpy as jnp

from circulus.components import Resistor, NMOS, VoltageSource, CurrentSource
from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_dc_op_dense
from circulus.utils import update_param_dict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("\n--- DEMO 3: Differential Pair (DC Sweep) ---")
    
    # Models (Static models are fine for DC)
    models_map = {
        'nmos': NMOS,
        'resistor': Resistor,
        'source_dc': VoltageSource,
        'current_src': CurrentSource,
        'ground': lambda: 0
    }

    # Netlist: Diff Pair with Current Source Tail
    net_dict = {
        "instances": {
            "GND": {"component":"ground"},
            "VDD": {"component":"source_dc", "settings":{"V": 5.0}},
            "Iss": {"component":"current_src", "settings":{"I": 1e-3}}, # 1mA Tail
            "RD1": {"component":"resistor", "settings":{"R": 2000}},
            "RD2": {"component":"resistor", "settings":{"R": 2000}},
            "M1":  {"component":"nmos", "settings":{"W": 50e-6, "L": 1e-6}},
            "M2":  {"component":"nmos", "settings":{"W": 50e-6, "L": 1e-6}},
            "Vin1": {"component":"source_dc", "settings":{"V": 2.5}}, # We will override this
            "Vin2": {"component":"source_dc", "settings":{"V": 2.5}}, # Fixed Reference
        },
        "connections": {
            "GND,p1": ("VDD,p2", "Vin1,p2", "Vin2,p2", "Iss,p2"),
            # Tail
            "Iss,p1": ("M1,s", "M2,s"),
            # Drains to Resistors
            "M1,d": "RD1,p2",
            "M2,d": "RD2,p2",
            # Resistors to VDD
            "RD1,p1": "VDD,p1",
            "RD2,p1": "VDD,p1",
            # Inputs
            "Vin1,p1": "M1,g",
            "Vin2,p1": "M2,g",
        },
    }

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    
    # --- The DC Sweep Logic ---
    # We want to sweep Vin1 from 1.5V to 3.5V
    sweep_voltages = jnp.linspace(1.5, 3.5, 100)
    
    def solve_for_vin(v_in_val):
        
        # The index 1 corresponds to Vin1's voltage source instance in the instance dict
        # In future versions, there should be a map for this
        new_groups = update_param_dict(groups, 'source_dc', 'Vin1', 'V', v_in_val)
                
        return solve_dc_op_dense(new_groups, sys_size)

    # JAX VMAP allows us to solve 100 circuits in parallel on the GPU
    # Excellent performance on CPU as well due to JIT compilation
    print("Sweeping DC Operating Point...")
    solutions = jax.vmap(solve_for_vin)(sweep_voltages)
    
    v_out1 = solutions[:, port_map["RD1,p2"]]
    v_out2 = solutions[:, port_map["RD2,p2"]]
    v_diff = v_out1 - v_out2

    plt.figure(figsize=(8, 4))
    plt.plot(sweep_voltages, v_out1, 'r', label='V_out1 (Inv)')
    plt.plot(sweep_voltages, v_out2, 'b', label='V_out2 (Non-Inv)')
    plt.plot(sweep_voltages, v_diff, 'k--', label='V_diff')
    plt.axvline(2.5, color='gray', linestyle=':', label='Ref (2.5V)')
    plt.title("Differential Pair Transfer Characteristic")
    plt.xlabel("Input Voltage Vin1 (V)")
    plt.ylabel("Output Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.show()