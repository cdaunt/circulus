import jax
import jax.numpy as jnp
import diffrax

from circulus.components import Resistor, Diode, VoltageSourceAC
from circulus.compiler import compile_netlist
from circulus.solvers.transient import VectorizedTransientSolver
from circulus.solvers.dc import solve_dc_op_dense

import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("\n--- DEMO 1: Diode Clipper (Transient) ---")
    
    # 1. Define Models
    models_map = {
        'resistor': Resistor,
        'diode': Diode, # Uses your updated diode model
        'source_voltage': VoltageSourceAC,
        'ground': lambda: 0
    }

    # 2. Netlist: AC Source -> Resistor -> Parallel Diodes -> Ground
    net_dict = {
        "instances": {
            "GND": {"component":"ground"},
            "Vin": {"component":"source_voltage", "settings":{"V": 5.0, "freq": 1e3,}}, # 5V Amplitude
            "R1":  {"component":"resistor", "settings":{"R": 1000.0}},
            "D1":  {"component":"diode", "settings":{'Is':1e-14,}}, # Forward
            "D2":  {"component":"diode", "settings":{'Is':1e-14,}}, # Anti-parallel
        },
        "connections": {
            "GND,p1": ("Vin,p2", "D1,p2", "D2,p1"),
            "Vin,p1": "R1,p1",
            "R1,p2":  ("D1,p1", "D2,p2"), # Output Node
        },
    }

    # 3. Compile
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    y0 = solve_dc_op_dense(groups, sys_size)

    # 4. Simulate (2 Cycles of 1kHz = 2ms)
    t_max = 2e-3
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 1000))
    
    # Using PID controller for adaptive stepping around the sharp "clip" corners
    # controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y)), 
        VectorizedTransientSolver(mode='dense'),
        t0=0.0, t1=t_max, dt0=1e-6, y0=y0, args=(groups, sys_size),
        #stepsize_controller=controller, 
        saveat=saveat, max_steps=50000
    )

    # 5. Plot
    ts = sol.ts
    v_in = sol.ys[:, port_map["Vin,p1"]]
    v_out = sol.ys[:, port_map["R1,p2"]]

    plt.figure(figsize=(8, 4))
    plt.plot(ts*1000, v_in, 'k--', alpha=0.5, label='Input (5V Sine)')
    plt.plot(ts*1000, v_out, 'r-', linewidth=2, label='Output (Clipped)')
    plt.title("Diode Limiter: Hard Non-Linearity Test")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.show()
