import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# --- Import your Circuit Library ---
# Adjust these imports to match your file structure
from circulus.components import Resistor, VoltageSource
from circulus.compiler import compile_netlist
from circulus.solvers.strategies import DenseSolver, KLUSolver
import time

if __name__ == "__main__":
    # 1. Config
    jax.config.update("jax_enable_x64", True)
    
    models_map = {
        'resistor': Resistor,
        'source_voltage': VoltageSource,
        'ground': lambda: 0
    }

    # 2. Define Netlist: 3-Bit R-2R Ladder
    # V1 is the Reference (8.0V)
    # R = 1k, 2R = 2k
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V_REF": {"component": "source_voltage", "settings": {"V": 8.0}},
            
            # --- Stage 1 (MSB) ---
            "R_S1": {"component": "resistor", "settings": {"R": 1000.0}}, # Series R
            "R_P1": {"component": "resistor", "settings": {"R": 2000.0}}, # Parallel 2R
            
            # --- Stage 2 ---
            "R_S2": {"component": "resistor", "settings": {"R": 1000.0}},
            "R_P2": {"component": "resistor", "settings": {"R": 2000.0}},
            
            # --- Stage 3 (LSB) ---
            "R_S3": {"component": "resistor", "settings": {"R": 1000.0}},
            "R_P3": {"component": "resistor", "settings": {"R": 2000.0}},
            
            # --- Termination ---
            "R_TERM": {"component": "resistor", "settings": {"R": 2000.0}}, # 2R Termination
        },
        "connections": {
            # Reference -> Stage 1
            "GND,p1": ("V_REF,p2", "R_P1,p2", "R_P2,p2", "R_P3,p2", "R_TERM,p2"),
            
            # Stage 1 Node (V_ref -> R_S1 -> Node1)
            "V_REF,p1": "R_S1,p1",
            "R_S1,p2": ("R_P1,p1", "R_S2,p1"), # Node 1 splits to Parallel and Next Series
            
            # Stage 2 Node
            "R_S2,p2": ("R_P2,p1", "R_S3,p1"), # Node 2 splits
            
            # Stage 3 Node
            "R_S3,p2": ("R_P3,p1", "R_TERM,p1"), # Node 3 (End of chain)
        }
    }

    print("1. Compiling Circuit...")
    groups, num_vars, port_map = compile_netlist(net_dict, models_map)
    print(f"   System Size: {num_vars} variables")

    # 3. Setup Solver Strategy
    # We try to use KLU, fall back to Dense if not installed
    # try:
    #     linear_strategy = KLUSolver.from_circuit(groups, num_vars)
    #     print("   Using Strategy: KLU Sparse")
    # except ImportError:
    #     linear_strategy = DenseSolver.from_circuit(groups, num_vars)
    # print("   Using Strategy: Dense (KLU not found)")
    linear_strategy = KLUSolver.from_circuit(groups, num_vars)


    # 4. Run DC Analysis
    print("\n2. Solving DC Operating Point...")
    
    # We solve at t=0.0. The strategy automatically handles the solve.
    # We pass a zero guess.
    y_guess = jnp.zeros(num_vars)

    start = time.time()
    y_dc = linear_strategy.solve_dc(component_groups=groups, num_vars=num_vars, y_guess=y_guess)
    print(f"Time take = {time.time() - start:.4f}s")

    # 5. Verify Results (Theoretical R-2R Analysis)
    # In an R-2R ladder loaded with 2R at the end:
    # Voltage halves at every series node.
    # Input = 8.0V
    # Node 1 (after first R) should be 4.0V (but wait, check the loading...)
    
    # Let's trace nodes:
    # V_in = 8V. 
    # The structure here is V_in connected to R_S1.
    # Actually, R-2R usually drives the *parallels* with bits. 
    # This specific setup is a "Resistive Attenuator":
    # Total resistance seen by V_ref is 1k + (2k || (1k + (2k || (1k + 2k || 2k))))
    # Let's calculate backwards:
    # End: 2k || 2k = 1k.
    # Stage 3: 1k (series) + 1k (load) = 2k.  Parallel w/ 2k = 1k.
    # Stage 2: 1k (series) + 1k (load) = 2k.  Parallel w/ 2k = 1k.
    # Stage 1: 1k (series) + 1k (load) = 2k.
    # Total R = 2k.
    # V_Node1 = V_in * (1k_equivalent / (1k_series + 1k_equivalent)) = 8.0 * (1/2) = 4.0V
    # V_Node2 = V_Node1 * (1/2) = 2.0V
    # V_Node3 = V_Node2 * (1/2) = 1.0V

    print("\n3. Verification:")
    
    # Helper to get node voltage
    def get_v(name):
        idx = port_map[name]
        return float(y_dc[idx])

    v_n1 = get_v("R_S1,p2") # Node 1
    v_n2 = get_v("R_S2,p2") # Node 2
    v_n3 = get_v("R_S3,p2") # Node 3

    print(f"   V_REF:    8.0 V")
    print(f"   Node 1:   {v_n1:.4f} V  (Expected: 4.0000 V)")
    print(f"   Node 2:   {v_n2:.4f} V  (Expected: 2.0000 V)")
    print(f"   Node 3:   {v_n3:.4f} V  (Expected: 1.0000 V)")
    
    # Check Ground
    v_gnd = get_v("R_TERM,p2")
    print(f"   Ground:   {v_gnd:.1e} V  (Expected: 0.0)")

    # 6. Visualization
    nodes = ['Node 1', 'Node 2', 'Node 3']
    voltages = [v_n1, v_n2, v_n3]
    expected = [4.0, 2.0, 1.0]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(nodes))
    width = 0.35

    plt.bar(x - width/2, expected, width, label='Theoretical', color='gray', alpha=0.5)
    plt.bar(x + width/2, voltages, width, label='Solver Output', color='tab:blue', alpha=0.8)

    plt.ylabel('Voltage (V)')
    plt.title('DC Solver Accuracy Test (R-2R Ladder)')
    plt.xticks(x, nodes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Error calc
    max_err = np.max(np.abs(np.array(voltages) - np.array(expected)))
    plt.text(1, 3, f"Max Error: {max_err:.2e} V", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()

    if max_err < 1e-6:
        print("\n✅ DC Solver PASSED")
    else:
        print("\n❌ DC Solver FAILED")