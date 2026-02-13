## Writing Components

Circulus uses a functional, JAX-first approach to component definition. Instead of inheriting from complex base classes, you define components as pure Python functions decorated with specific handlers.

This architecture ensures your components are automatically compatible with JIT compilation (```jax.jit```), vectorization (```jax.vmap```), and back-propagation (```jax.grad```).

### The Core Concept

Every component in Circulus is a function that calculates the instantaneous balance equations for a specific node or state. The function signature generally looks like this:

```python
def MyComponent(signals, s, [t], **params):
    # 1. Calculate physics
    # 2. Return (Flows, Storage)
```


### Arguments

 1) ```signals``` (Ports): A NamedTuple containing the potential (Voltage) at every port defined in the decorator. Accessed via dot notation (e.g., signals.p, signals.gate).

 2) ```s``` (States): A NamedTuple containing internal state variables (e.g., current through an inductor, internal node voltages).

 3) ```t``` (Time): Optional. Only present if you use the @source decorator.

 4) ```**params```: Keyword arguments defining the physical properties (Resistance, Length, Refractive Index).

### Return Values

The function must return a tuple of two dictionaries: ```(f_dict, q_dict)```.

* ```f_dict``` (The Flow/Balance Vector):

    * For Ports: Represents the "Flow" (Current) entering the node.

    * For States: Represents the algebraic constraint (should sum to 0).

* ```q_dict``` (The Storage Vector):

    * Represents the time-dependent quantity (Charge, Flux) stored in a variable.

    * The solver computes $\frac{d}{dt}(q\_dict)$.

1. Electronic Components (Time-Invariant)

Most passive components (Resistors, Transistors, Diodes) do not depend explicitly on time t. For these, use the @component decorator.

**Example: A Simple Resistor**

```python
import jax.numpy as jnp
from circulus.base_component import component, Signals, States

@component(ports=("p", "n"))
def Resistor(signals: Signals, s: States, R: float = 1e3):
    """
    Ohm's Law: I = V / R
    """
    # 1. Calculate the physics
    v_drop = signals.p - signals.n
    i = v_drop / (R + 1e-12)  # Add epsilon for stability
    
    # 2. Assign currents to ports
    # Current leaves 'p' and enters 'n' (Passive convention)
    f_dict = {
        "p": i, 
        "n": -i
    }
    
    # Resistors have no memory (no d/dt terms)
    q_dict = {}
    
    return f_dict, q_dict
```


**Example: A Capacitor (Time-Derivative)**

For reactive components, use the second return dictionary (```q_dict```) to define what is being differentiated.

```python
@component(ports=("p", "n"))
def Capacitor(signals: Signals, s: States, C: float = 1e-6):
    """
    I = C * dV/dt  =>  I = dQ/dt
    """
    v_drop = signals.p - signals.n
    
    # We define Charge (q)
    q_val = C * v_drop
    
    # The solver treats q_dict as the "Mass Matrix" side.
    # The entries in q_dict are differentiated with respect to time.
    # p: I_p = d(q_val)/dt
    return {}, {"p": q_val, "n": -q_val}
```


2. Time-Dependent Sources

If your component varies with time (e.g., AC source, Pulse generator), use the @source decorator. This injects ```t``` as the third argument.

**Example: AC Voltage Source**

Voltage sources require an Internal State variable (i_src) to represent the current flowing through the source. This is because the voltage is fixed, so the current is the unknown variable the solver must find.


```python
from circulus.base_component import source

@source(ports=("p", "n"), states=("i_src",))
def ACSource(signals: Signals, s: States, t: float, V: float = 1.0, freq: float = 60.0):
    # 1. Calculate Target Voltage based on time 't'
    target_v = V * jnp.sin(2 * jnp.pi * freq * t)
    
    # 2. Define the Constraint Equation
    # We want: (vp - vn) = target_v
    # Therefore: (vp - vn) - target_v = 0
    constraint = (signals.p - signals.n) - target_v
    
    return {
        # KCL: The unknown current 'i_src' leaves p and enters n
        "p": s.i_src,
        "n": -s.i_src,
        
        # Constraint: The solver adjusts 'i_src' until this equation equals 0
        "i_src": constraint
    }, {}
```


3. Photonic Components (Frequency Domain)

Circulus can simulate photonic circuits by treating them as complex-valued resistor networks. You typically start with an S-Matrix, convert it to an Admittance (Y) Matrix, and calculate currents via $I = Y \cdot V$.

**Example: Optical Waveguide**

```python
from circulus.s_transforms import s_to_y

@component(ports=("in", "out"))
def Waveguide(signals: Signals, s: States, length_um: float = 100.0, neff: float = 2.4, wl: float = 1.55):
    # 1. Physics: Calculate Phase Shift
    # Note: Use jnp (JAX numpy) for all math
    phi = 2.0 * jnp.pi * neff * length_um / wl
    
    # 2. Construct S-Matrix (Transmission)
    # T = exp(-j * phi)
    T = jnp.exp(-1j * phi)
    
    # S = [[0, T], 
    #      [T, 0]]
    S = jnp.array([
        [0.0, T], 
        [T, 0.0]
    ], dtype=jnp.complex128)
    
    # 3. Convert to Admittance (Y)
    Y = s_to_y(S)
    
    # 4. Calculate Currents: I = Y @ V
    # IMPORTANT: Cast inputs to complex128!
    v_vec = jnp.array([signals.in, signals.out], dtype=jnp.complex128)
    i_vec = Y @ v_vec
    
    return {"in": i_vec[0], "out": i_vec[1]}, {}
```


4. Integration with SAX

If you have existing models written for SAX, you can reuse them directly without rewriting physics logic using the @sax_component decorator.

```python
from circulus.sax_integration import sax_component

# 1. Define or Import a pure SAX model
def sax_coupler(coupling=0.5):
    kappa = coupling**0.5
    tau = (1 - coupling)**0.5
    return {
        ("in0", "out0"): tau,
        ("in0", "out1"): 1j*kappa,
        ("in1", "out0"): 1j*kappa,
        ("in1", "out1"): tau
    }

# 2. Convert to Circulus Component
# This automatically detects ports ('in0', 'in1', 'out0', 'out1')
Coupler = sax_component(sax_coupler)
```

5. Advanced: Under the Hood

For advanced users familiar with JAX and Equinox, it is helpful to understand what the @component decorator actually does.

It does not simply wrap your function. Instead, it dynamically generates a new class that inherits from equinox.Module.

The Transformation Process

When you write:

```python
@component(ports=("a", "b"))
def MyResistor(signals, s, R=100.0):
...


The decorator performs the following steps:

1. Introspection: It analyzes the function signature to identify parameters (R) and their default values (100.0).

2. Class Generation: It constructs a new eqx.Module class named MyResistor.

3. Field Registration: The parameters (R) become fields of this class. This allows JAX to differentiate with respect to R automatically.

4. Static Optimization: It creates a static _fast_physics method that unrolls dictionary lookups into raw array operations. This is what the solver calls inside jax.jit or jax.vmap.

