# Circuitlus

A Differentiable, Functional Circuit Analyzer for the AI Era. 

Circuitlus is a next-generation circuit simulation engine built on JAX. It reimagines circuit analysis not as a monolithic black box (like SPICE), but as a modular, differentiable pipeline designed for optimization, inverse design, and AI integration. 

It serves as the time-domain sister to SAX (S-parameters for Photonics), using the same functional design philosophy to bring transient analysis to the differentiable world. 

## Why Circuitlus? 
Standard tools (SPICE, Spectre, Ngspice) rely on manual matrix stamping and opaque C++ solvers. Circuitlus takes a radically different approach:

| Feature | Legacy(SPICE) | Circuitlus |
| ----------- | ----------- | ----------- |
| Physics Definition    | Hardcoded C++ / Verilog-A | Equinox Modules (PyTrees) |
| Derivatives   | Manual Analytical Jacobian       |Automatic Differentiation (AD)|
| Solver State   | Fixed (Voltages/Currents)       |Dynamic (Inferred from function signatures)|
| Optimization  | Finite Difference (Slow)       |Backpropagation (Exact & Fast)|
| Matrix Backend  | CPU Sparse (KLU)       | GPU Sparse/Dense (JAX BCOO)|


## 🏛️ Architecture: The 3 Layers
Circuitlus strictly separates Physics, Topology, and Analysis, allowing you to swap out solvers or models without rewriting your netlist.

1. **Physics Layer ("Functional Components")**

Components are defined as **Equinox Modules**. They declare their parameters, ports, and internal states, and implement a `physics` method that returns the residuals (currents) and charges.

**The Protocol:**
   
```python
from circulus.base_component import CircuitComponent
import jax.numpy as jnp

class Resistor(CircuitComponent):
    R: float | jax.Array = 1000.0
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        # v.p1, v.p2 are voltages
        i = (v.p1 - v.p2) / self.R
        return {"p1": i, "p2": -i}

class Capacitor(CircuitComponent):
    C: float | jax.Array = 1e-6
    ports = ("p1", "p2")

    def physics(self, v, s, t):
        # Return (Resistive, Reactive) tuple
        q = self.C * (v.p1 - v.p2)
        return {}, {"p1": q, "p2": -q}
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
import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

from circulus.components import Resistor, Capacitor, VoltageSource
from circulus.compiler import compile_netlist
from circulus.solvers.dense import VectorizedDenseSolver
from circulus.solvers.dc import solve_dc_op_dense

# 1. Define Netlist (RC Circuit)
net_dict = {
    "instances": {
        "GND": {"component": "ground"},
        "V1":  {"component": "source",   "settings": {"V": 1.0, "delay": 1e-9}},
        "R1":  {"component": "resistor", "settings": {"R": 1000.0}},
        "C1":  {"component": "capacitor","settings": {"C": 1e-6}},
    },
    "connections": {
        "GND,p1": ("V1,p2", "C1,p2"),
        "V1,p1": "R1,p1",
        "R1,p2": "C1,p1"
    }
}

models_map = {
    "resistor": Resistor,
    "capacitor": Capacitor,
    "source": VoltageSource,
    "ground": lambda: 0
}

# 2. Compile
groups, sys_size, port_map = compile_netlist(net_dict, models_map)

# 3. Solve
# DC Operating Point (Warm Start)
y0 = solve_dc_op_dense(groups, sys_size)

# Transient Simulation
solver = VectorizedDenseSolver()
term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

sol = diffrax.diffeqsolve(
    term, solver, t0=0.0, t1=1e-5, dt0=1e-9, 
    y0=y0, args=(groups, sys_size),
    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
)

# 4. Visualize
node_idx = port_map["C1,p1"]
plt.plot(sol.ts, sol.ys[:, node_idx])
plt.title("Step Response")
plt.show()
```

<!-- ## Differentiable Design

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
``` -->

## Documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

To build and serve the documentation locally, you need to have the development dependencies installed. Then, run the following commands:

```bash
pixi run -e dev mkdocs serve
```

This will start a local server, and you can view the documentation by navigating to `http://127.0.0.1:8000` in your web browser.

## Roadmap
- [x] Sparse Linear Solvers: Integration of advanced preconditioners for large-scale nets (>10k nodes).

- [ ] Frequency Domain: Reuse time-domain models for AC analysis via linearization (cl.solve_ac).

- [ ] Multi-physics: Support for Optical and Thermal ports (Generalized MNA).