# Tax: Differentiable Time-Domain Simulation in JAX

Tax is a fully differentiable, hardware-accelerated circuit simulator built on JAX and Diffrax. It is designed to be the time-domain sister project to sax. While sax solves for S-parameters in the frequency domain, tax solves for transient responses in the time domain using the exact same netlist definitions.

## Why another simulator? 
Standard tools (SPICE, Spectre) are "black boxes." You cannot backpropagate through them. tax allows you to differentiate through the simulation time-steps, enabling Gradient-Based Circuit Design and massive GPU acceleration for Monte Carlo analysis. 

## Key FeaturesGradient-Based Design: 
* Optimize component values (R, L, C, W/L) by backpropagating errors from the transient output directly to the parameters using JAX AD.
* Functional "Verilog-A" Models: Define component physics as pure Python functions. No manual Jacobian matrices required—JAX handles the derivatives.
* GPU Acceleration: Simulate thousands of circuit variations in parallel using jax.vmap.
* SAX Compatibility: Use the same recursive dictionary netlists for both Frequency (sax) and Time (tax) analysis. 
* State-of-the-Art Solvers: Built on Diffrax, utilizing implicit stiff solvers (Kvaerno, Rosenbrock) and adaptive step-size controllers.

# The Philosophy
Traditional simulators couple the Topology (Netlist), the Physics (Device Equations), and the Solver (MNA/Newton) into a single monolithic binary.tax decouples them completely:
* Topology: Defined via recursive dictionaries (compatible with sax).
* Physics: Defined as pure functions returning Current ($I$) and Charge/Flux ($Q$).Solver: 
* A generic DAE engine that consumes the topology and physics to minimize residuals.
 
# The "Functional Verilog-A" SignatureInstead of writing C++ or Verilog-A, you write a JAX function. 

The simulator automatically linearizes this to build the Jacobian.$$I_{total} = I(v, \theta) + \frac{d}{dt} Q(v, \theta)$$Pythondef diode_physics(v_local, params):
    # 1. Static Current (I)
    i = params['Is'] * (jnp.exp(v_local[0] / params['Vt']) - 1)
    
    # 2. Dynamic Charge (Q)
    q = params['Cj'] * v_local[0]
    
    # Return (Current Vector, Charge Vector)
    return jnp.array([i, -i]), jnp.array([q, -q])

## Quick Start

1. Define a NetlistWe use the sax format: a dictionary of instances and connections.
   
```python 
netlist = {
    "instances": {
        "V1": "source_pulse",
        "R1": "resistor",
        "C1": "capacitor",
        "D1": "diode"
    },
    "connections": {
        "V1,p": "R1,p1",
        "R1,p2": "D1,anode",
        "D1,anode": "C1,p1",
        "D1,cathode": "gnd",
        "C1,p2": "gnd",
        "V1,n": "gnd"
    }
}
```
2. Define the Model LibraryMap component names to physics functions.

```python 
import tax
import jax.numpy as jnp

models = {
    "resistor": tax.models.resistor,   # Pre-built models
    "capacitor": tax.models.capacitor,
    "diode": tax.models.diode,         # ...or define your own
    "source_pulse": tax.models.pulsed_source
}
1. Simulate (Transient)Python# Component Parameters
params = {
    "R1": {"R": 1000.0},
    "C1": {"C": 1e-9},
    "D1": {"Is": 1e-14},
    "V1": {"V_high": 5.0, "period": 1e-6}
}

# Run Simulation
solution = tax.solve(
    netlist, 
    models=models, 
    params=params, 
    t_end=10e-6, 
    dt=1e-9
)

# Plot results
import matplotlib.pyplot as plt
plt.plot(solution.ts, solution.ys["D1,anode"])
plt.show()
```
🔮 Advanced: Inverse Design (Optimization)Because tax is differentiable, you can ask JAX to find component values for you.

```python 
@jax.jit
@jax.value_and_grad
def loss_function(component_values):
    # 1. Update params
    params["R1"]["R"] = component_values
    
    # 2. Simulate
    sol = tax.solve(netlist, models, params, t_end=1e-6)
    
    # 3. Calculate Error (e.g., match a target rise time)
    v_out = sol.ys["C1,p1"]
    mse = jnp.mean((v_out - target_waveform)**2)
    return mse

# Optimization Step
grads = loss_function(1000.0)[1]
print(f"Gradient of Loss w.r.t Resistor: {grads}")

#Interop with SAXtax allows you to reuse your physics definitions for Frequency Domain analysis.Pythonimport sax

# Convert Time-Domain models to S-Parameter models automatically
sax_models = {
    k: tax.physics_to_sax(v) for k, v in models.items()
}

# Run S-Parameter simulation
s_params = sax.solve(netlist, models=sax_models)
```
## Roadmap Sparse Solver Backend: 
* Implementation of jax.experimental.sparse BCOO assembly for large netlists (>1000 nodes).
* BSIM/EKV Models: Implementation of standard transistor models in JAX.
* Noise Analysis: Transient noise simulation using Stochastic Differential Equations (SDEs) via Diffrax. 
* Verilog-A Parser: Experimental transpiler from Verilog-A to JAX.