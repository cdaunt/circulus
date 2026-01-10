import jax
import jax.numpy as jnp

from circulus.compiler import compile_netlist
from circulus.solvers.dc import solve_operating_point
import matplotlib.pyplot as plt
import time
import diffrax
from circulus.components import Resistor
from circulus.photonic_components import OpticalWaveguide, OpticalSourcePulse
from circulus.solvers.transient import VectorizedTransientSolver

if __name__ == "__main__":
    print("\n--- DEMO: Photonic Transient")
    
    models_map = {'waveguide': OpticalWaveguide, 'source': OpticalSourcePulse, 'resistor': Resistor, 'ground': lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "I1": {"component": "source", "settings": {"power": 1.0, "delay": 0.1e-9}},
            "WG1": {"component": "waveguide", "settings": {"length_um": 100.0}},
            "R1": {"component": "resistor", "settings": {"R": 1.0}} # circulus.components.Resistor defaults to 1k, we set 1.0
        },
        "connections": {"GND,p1": ("I1,p2", "R1,p2"), "I1,p1": "WG1,p1", "WG1,p2": "R1,p1"}
    }

    print("Compiling...")
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    print(port_map)
    
    print(f"Total System Size: {sys_size}")
    for g_name, g in groups.items():
        print(f"\nGroup: {g_name}")
        print(f"  Count: {g.var_indices.shape[0]}")
        print(f"  Var Indices Shape: {g.var_indices.shape}")
        print(f"  Sample Var Indices:\n{g.var_indices}")
        print(f"  Jacobian Rows Length: {len(g.jac_rows)}")

        # Simulation Config
    y_op = jnp.zeros(sys_size,dtype=jnp.complex128)

    print("2. Solving DC Operating Point...")
    # This solves the system at t=0 assuming capacitors are open
    #y_op = solve_operating_point(groups, sys_size, t0=0.0, dtype=jnp.complex128)
    #print(f"   OP Solution: {y_op}")
    
    # Flatten complex state to [Real, Imag] to avoid Diffrax complex warnings/instability
    y0_flat = jnp.concatenate([y_op.real, y_op.imag])

    solver = VectorizedTransientSolver(mode='dense')
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))
    
    t_max = 1e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 500))

    print("3. Running Simulation...")
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_max, dt0=1e-3*t_max, 
        y0=y0_flat, args=(groups, sys_size),

        saveat=saveat, max_steps=100000,
        progress_meter=diffrax.TqdmProgressMeter(refresh_steps=100)
    )

    node_out_idx = port_map["I1,p1"]
    node_in_idx  = port_map["R1,p1"]

    ts = sol.ts * 1e9 # Convert to ns
    v_in = sol.ys[:, node_in_idx]
    v_out = sol.ys[:, node_out_idx]

    plt.plot(ts, jnp.abs(v_out)**2)
    plt.xlabel("Time (ns)")
    plt.ylabel("Ouput Power (mW)")
    plt.grid()
    plt.show()

    