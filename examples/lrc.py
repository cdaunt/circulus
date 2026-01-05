import jax
import jax.numpy as jnp
import diffrax

from circulus.components import Resistor, Capacitor, Inductor, VoltageSource
from circulus.compiler import compile_netlist, build_net_map
#from circulus.solvers.sparse import VectorizedSparseSolver as SparseSolver
from circulus.solvers.dense import VectorizedDenseSolver as DenseSolver
from circulus.solvers.dc import solve_dc_op_dense
from circulus.netlist import draw_circuit_graph

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Enable 64-bit precision (Critical for Circuit Simulation)
    jax.config.update("jax_enable_x64", True)

    t_max = 3E-9

    models_map = {
            'resistor': Resistor,
            'capacitor': Capacitor,
            'inductor': Inductor,
            'source_voltage': VoltageSource,
            'ground': lambda: 0
        }

    # Complex Netlist
    net_dict = {
        "instances": {
            "GND": {"component":"ground"},
            "V1": {"component":"source_voltage", "settings":{"V": 1.0,"delay":1E-9}},
            "R1": {"component":"resistor", "settings":{"R": 10.0}},
            "C1": {"component":"capacitor", "settings":{"C": 1e-11}},
            "L1": {"component":"inductor", "settings":{"L": 5e-9}},
        },
        "connections": {
            "GND,p1": ("V1,p2", "C1,p2"),
            "V1,p1": "R1,p1",
            "R1,p2": "L1,p1",
            "L1,p2": "C1,p1",
        },
    }

    #draw_circuit_graph(net_dict)

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
    y0 = jnp.zeros(sys_size)

    print("2. Solving DC Operating Point...")
    # This solves the system at t=0 assuming capacitors are open
    y_op = solve_dc_op_dense(groups, sys_size, t0=0.0)
    #print(f"   OP Solution: {y_op}")
    
    solver = DenseSolver()
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))
    
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 500))
    
    print("3. Running Simulation...")
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t_max, dt0=1e-3*t_max, 
        y0=y_op, args=(groups, sys_size),

        saveat=saveat, max_steps=100000,
        progress_meter=diffrax.TqdmProgressMeter(refresh_steps=100)
    )

    # Visualization
    ts = sol.ts
    v_src = sol.ys[:, port_map["V1,p1"]]
    v_cap = sol.ys[:, port_map["C1,p1"]] # Node 3
    i_ind = sol.ys[:, 5] # Internal Variable for Inductor (Index 5)
    
    print("4. Plotting...")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ts, v_src, 'k--', label='Source V')
    ax1.plot(ts, v_cap, 'b-', label='Capacitor V')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(ts, i_ind, 'r:', label='Inductor I')
    ax2.set_ylabel('Current (A)')
    ax2.legend(loc='upper right')
    
    plt.title("Differentiable Simulation with Implicit Internals")
    plt.grid(True)
    plt.show()