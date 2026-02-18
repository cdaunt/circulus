# **Circulus**

<img src="images/logo.svg" alt="logo" width="500">

## **A Differentiable, Functional Circuit Simulator based on JAX**
Circulus is a differentiable circuit simulation framework built on [JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html), [Optimistix](https://github.com/patrick-kidger/optimistix) and [Diffrax](https://docs.kidger.site/diffrax/). It treats circuit netlists as systems of Ordinary Differential Equations (ODEs), leveraging Diffrax's suite of numerical solvers for transient analysis.

By using JAX as its backend, Circulus provides:

**Native Differentiation**: Full support for forward and reverse-mode automatic differentiation through the solver, enabling gradient-based parameter optimization and inverse design.

**Hardware Acceleration**: Seamless execution on CPU, GPU, and TPU without code changes.

**Mixed-Domain Support**: Native complex-number handling for simultaneous simulation of electronic and photonic components.

**Modular Architecture**: A functional approach to simulation that integrates directly into machine learning and scientific computing workflows.

Standard tools (SPICE, Spectre, Ngspice) rely on established matrix stamping methods and CPU-bound sparse solvers. Circulus leverages the JAX ecosystem to offer specific advantages in optimization and hardware utilization:

| Feature | Legacy(SPICE) | Circulus |
| ----------- | ----------- | ----------- |
| Model Definition    | Hardcoded C++ / Verilog-A | Simple python functions |
| Derivatives   | Hardcoded (C) or Compiler-Generated (Verilog-A) | Automatic Differentiation (AD)|
| Solver Logic |	Fixed-step or heuristic-based |	Adaptive ODE stepping via Diffrax |
| Matrix Solver |	Monolithic CPU Sparse (KLU)	| Pluggable ([KLUJAX](https://github.com/flaport/klujax), Dense, or Custom) |
| Hardware Target |	CPU-bound	| Agnostic (CPU/GPU/TPU) |


## **Simulator setup**
Circulus strictly separates Physics, Topology, and Analysis, enabling the interchange of solvers or models without netlist modification.

### **Physics Layer**
Components are defined as simple Python functions wrapped with the ```@component``` decorator. This functional interface abstracts away the boilerplate, allowing users to define physics using simple voltage/current/field/flux relationships.

```python
from circulus.base_component import component, Signals, States
import jax.numpy as jnp

@component(ports=("p1", "p2"))
def Resistor(signals: Signals, s: States, R: float = 1e3):
    """Ohm's Law: I = V/R"""
    # signals.p1, signals.p2 are the nodal voltages
    i = (signals.p1 - signals.p2) / R
    # Return (Currents, Charges)
    return {"p1": i, "p2": -i}, {}

@component(ports=("p1", "p2"))
def Capacitor(signals: Signals, s: States, C: float = 1e-12):
    """
    Q = C * V.
    Returns Charge (q) so the solver computes I = dq/dt.
    """
    v_drop = signals.p1 - signals.p2
    q_val = C * v_drop
    return {}, {"p1": q_val, "p2": -q_val}
```

### **Topology**
The compiler inspects your netlist and your model signatures. It automatically:

1. Introspects models to determine how many internal variables (currents) they need.

2. Allocates indices in the global state vector.

3. Pre-calculates the Sparse Matrix indices (BCOO format) for batched/parallel assembly.

```python
netlist = [
    Instance("V1", voltage_source, connections=[1, 0], params={"V": 5.0}),
    Instance("R1", resistor,       connections=[1, 2], params={"R": 100.0}),
]
# Compiler auto-detects that V1 needs an extra internal variable!
```

### **Analysis**
The solver is a generic DAE engine linking Diffrax (Time-stepping) and Optimistix (Root-finding).

* Transient: Solves $F(y) + \frac{d}{dt}Q(y) = 0$ using Implicit Backward Euler.

* DC Operating Point: Solves $F(y) = 0$ (automatically ignoring $Q$).

* Jacobian-Free: The solver builds the system Jacobian on-the-fly using ```jax.jacfwd``` allowing for the simulation of arbitrary user-defined non-linearities without manual derivative derivation.The approach results in a more exact and stable simulation.

##  **Installation**

```sh
pip install circulus
```

## **Simulation Example**

```python
import jax
import diffrax

from circulus.components import Resistor, Capacitor, Inductor, VoltageSource
from circulus.compiler import compile_netlist
from circulus.solvers.transient import VectorizedTransientSolver
from circulus.solvers.strategies import DenseSolver
import jax.numpy as jnp

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

net_dict = {
    "instances": {
        "GND": {"component":"ground"},
        "V1": {"component":"source_voltage", "settings":{"V": 1.0,"delay":0.25E-9}},
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

jax.config.update("jax_enable_x64", True)


jax.config.update("jax_enable_x64", True)


models_map = {
    'resistor': Resistor,
    'capacitor': Capacitor,
    'inductor': Inductor,
    'source_voltage': VoltageSource,
    'ground': lambda: 0
}

print("Compiling...")
groups, sys_size, port_map = compile_netlist(net_dict, models_map)

print(port_map)

print(f"Total System Size: {sys_size}")
for g_name, g in groups.items():
    print(f"Group: {g_name}")
    print(f"  Count: {g.var_indices.shape[0]}")
    print(f"  Var Indices Shape: {g.var_indices.shape}")
    print(f"  Sample Var Indices:{g.var_indices}")
    print(f"  Jacobian Rows Length: {len(g.jac_rows)}")

print("2. Solving DC Operating Point...")
linear_strat = analyze_circuit(groups, sys_size, is_complex=False)

y_guess = jnp.zeros(sys_size)
y_op = linear_strat.solve_dc(groups,y_guess)

transient_sim = setup_transient(groups=groups, linear_strategy=linear_strat)
term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))


t_max = 3E-9
saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 500))
print("3. Running Simulation...")
sol = transient_sim(
    t0=0.0, t1=t_max, dt0=1e-3*t_max,
    y0=y_op,
    saveat=saveat, max_steps=100000,
    progress_meter=diffrax.TqdmProgressMeter(refresh_steps=100)
)

ts = sol.ts
v_src = sol.ys[:, port_map["V1,p1"]]
v_cap = sol.ys[:, port_map["C1,p1"]]
i_ind = sol.ys[:, 5]

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

ax2_ticks = ax2.get_yticks()
ax1_ticks = ax1.get_yticks()
ax2.set_yticks(jnp.linspace(ax2_ticks[0], ax2_ticks[-1], len(ax1_ticks)))
ax1.set_yticks(jnp.linspace(ax1_ticks[0], ax1_ticks[-1], len(ax1_ticks)))

plt.title("Impulse Response of LCR circuit")
plt.grid(True)
plt.show()
```

![output](images/lcr.png)

## **License**

Copyright © 2026, Chris Daunt, [Apache-2.0 License](https://github.com/cdaunt/circulus/blob/master/LICENSE)
