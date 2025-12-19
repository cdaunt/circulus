import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

from circulus.models import resistor, capacitor, voltage_source, inductor
from circulus.compiler import Instance, compile_netlist_vectorized
from circulus.solvers.sparse import VectorizedSparseSolver as SparseSolver
from circulus.solvers.dense import VectorizedDenseSolver as DenseSolver



# Enable 64-bit precision (Critical for Circuit Simulation)
jax.config.update("jax_enable_x64", True)


def main():
    # Circuit: GND -- Vsrc -- N1 -- Res -- N2 -- Ind -- N3 -- Cap -- GND
    # Nodes: 0 (GND), 1, 2, 3
    
    # Define Netlist
    instances = [
        Instance("V1", voltage_source, [1, 0], {"V": 5.0, "delay":0.005}),   # Needs 1 internal (idx 4)
        Instance("R1", resistor,       [1, 2], {"R": 10.0}),  # Needs 0 internals
        Instance("L1", inductor,       [2, 3], {"L": 0.01}),  # Needs 1 internal (idx 5)
        Instance("C1", capacitor,      [3, 0], {"C": 1e-4}),  # Needs 0 internals
    ]
    
    print("1. Compiling Netlist...")
    # Node count is 4 (0,1,2,3).
    # Expected System Size: 4 Nodes + 1 (Vsrc current) + 1 (Ind current) = 6 Vars
    # blocks, sys_size = compile_netlist(instances, num_nodes=4)
    # print(f"   System Size: {sys_size} variables.")

    groups, sys_size = compile_netlist_vectorized(instances, num_nodes=4)
    print(f"   System Size: {sys_size} variables.")
    
    # Simulation Config
    y0 = jnp.zeros(sys_size)
    solver = DenseSolver()
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, 0.02, 500))
    
    print("2. Running Simulation...")
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=0.05, dt0=1e-5, 
        y0=y0, args=(groups, sys_size), 
        saveat=saveat, max_steps=5000
    )
    
    # Visualization
    ts = sol.ts
    v_src = sol.ys[:, 1] # Node 1
    v_cap = sol.ys[:, 3] # Node 3
    i_ind = sol.ys[:, 5] # Internal Variable for Inductor (Index 5)
    
    print("3. Plotting...")
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

if __name__ == "__main__":
    main()