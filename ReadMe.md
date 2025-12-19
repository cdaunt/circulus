# Circuitlus

A Differentiable, Functional Circuit Analyzer for the AI Era. 

Circuitlus is a next-generation circuit simulation engine built on JAX. It reimagines circuit analysis not as a monolithic black box (like SPICE), but as a modular, differentiable pipeline designed for optimization, inverse design, and AI integration. 

It serves as the time-domain sister to SAX (S-parameters for Photonics), using the same functional design philosophy to bring transient analysis to the differentiable world. 

## Why Circuitlus? 
Standard tools (SPICE, Spectre, Ngspice) rely on manual matrix stamping and opaque C++ solvers. Circuitlus takes a radically different approach:

| Feature | Legacy(SPICE) | Circuitlus |
| ----------- | ----------- | ----------- |
| Physics Definition    | Hardcoded C++ / Verilog-A | Pure Python Functions |
| Derivatives   | Manual Analytical Jacobian       |Automatic Differentiation (AD)|
| Solver State   | Fixed (Voltages/Currents)       |Dynamic (Inferred from function signatures)|
| Optimization  | Finite Difference (Slow)       |Backpropagation (Exact & Fast)|
| Matrix Backend  | CPU Sparse (KLU)       | GPU Sparse/Dense (JAX BCOO)|


## 🏛️ Architecture: The 3 Layers
Circuitlus strictly separates Physics, Topology, and Analysis, allowing you to swap out solvers or models without rewriting your netlist.

1. **Physics Layer ("Functional Components")**

Components are defined as Pure Functions. There are no classes, no stamp() methods, and no manual derivative calculations.

**The Protocol:** ```def model(params, vars=jnp.zeros(N))```: 
* ```vars```: The input state vector (Voltages + Internal states). The default value jnp.zeros(N) tells the compiler the size of the model.
* **Returns:** ```(Currents, Charges)``` tuple for DAE solving.
   
```python
import jax.numpy as jnp

# A Resistor needs 2 variables (Nodes A, B)
def resistor(params, vars=jnp.zeros(2)):
    v = vars[0] - vars[1]
    i = v / params['R']
    return jnp.array([i, -i]), jnp.array([0.0, 0.0])

# A Voltage Source needs 3 variables (Nodes A, B + Internal Current)
def voltage_source(params, vars=jnp.zeros(3)):
    i_src = vars[2]
    # KCL + Constraint Equation
    return jnp.array([i_src, -i_src, vars[0] - vars[1] - params['V']]), \
           jnp.zeros(3)
```

2. **Topology Layer ("The Compiler")**
The compiler inspects your netlist and your model signatures. It automatically:
1. Introspects models to determine how many internal variables (currents) they need.
2. Allocates indices in the global state vector.
3. Pre-calculates the Sparse Matrix indices (BCOO format) for massive parallel assembly.
```python
netlist = [
    Instance("V1", voltage_source, connections=[1, 0], params={"V": 5.0}),
    Instance("R1", resistor,       connections=[1, 2], params={"R": 100.0}),
]
# Compiler auto-detects that V1 needs an extra internal variable!
```

3. **Analysis Layer ("The Solver"):**
The solver is a generic DAE engine linking Diffrax (Time-stepping) and Optimistix (Root-finding).
* Transient: Solves $F(y) + \frac{d}{dt}Q(y) = 0$ using Implicit Backward Euler.
* DC Operating Point: Solves $F(y) = 0$ (automatically ignoring $Q$).
* Jacobian-Free: The solver builds the system Jacobian on-the-fly using ```jax.jacfwd``` and ```jax.experimental.sparse```.

🚀 Quick Start

Installation
```bash
uv install circuitlus
```

## Simulation Example

```python
import circuitlus as cl
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 1. Define Models (or import from circuitlus.models)
def capacitor(params, vars=jnp.zeros(2)):
    q = params['C'] * (vars[0] - vars[1])
    return jnp.array([0., 0.]), jnp.array([q, -q])

# 2. Define Netlist (RC Circuit)
# GND(0) -- Vsrc -- N1 -- Res -- N2 -- Cap -- GND
instances = [
    cl.Instance("V1", cl.models.voltage_source, [1, 0], {"V": 1.0}),
    cl.Instance("R1", cl.models.resistor,       [1, 2], {"R": 1e3}),
    cl.Instance("C1", capacitor,                [2, 0], {"C": 1e-6}),
]

# 3. Analyze
# Compiler handles variable allocation automatically
blocks, sys_size = cl.compile(instances, num_nodes=3)

# DC Operating Point (Warm Start)
y_dc = cl.solve_dc(blocks, sys_size)

# Transient Simulation
sol = cl.solve_transient(
    blocks, sys_size, 
    t_end=0.01, 
    y0=y_dc # Start from DC solution
)

# 4. Visualize
plt.plot(sol.ts, sol.ys[:, 2]) # Plot Node 2 (Capacitor)
plt.title("Step Response")
plt.show()

```

## Differentiable Design

Because Circuitlus is pure JAX, you can differentiate the entire simulation with respect to component parameters.

```python
@jax.jit
@jax.grad
def loss_fn(R_val):
    # Update params
    instances[1].params["R"] = R_val
    
    # Simulate
    blocks, size = cl.compile(instances, 3)
    sol = cl.solve_transient(blocks, size, t_end=0.01, y0=y_dc)
    
    # Calculate Error (e.g. Target Rise Time)
    v_out = sol.ys[:, 2]
    return jnp.mean((v_out - target_waveform)**2)

# Get gradient of Error w.r.t Resistance
grad_R = loss_fn(1000.0)
```

## Roadmap
[ ] Sparse Linear Solvers: Integration of advanced preconditioners for large-scale nets (>10k nodes).

[ ] Frequency Domain: Reuse time-domain models for AC analysis via linearization (cl.solve_ac).

[ ] Multi-physics: Support for Optical and Thermal ports (Generalized MNA).