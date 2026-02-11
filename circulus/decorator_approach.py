import jax
import jax.numpy as jnp
import equinox as eqx
import inspect
from typing import Tuple, ClassVar, Any, Dict, Protocol, runtime_checkable
from collections import namedtuple

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class Signals(Protocol):
    def __getattr__(self, name: str) -> Any: ...

@runtime_checkable
class States(Protocol):
    def __getattr__(self, name: str) -> Any: ...


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class CircuitComponent(eqx.Module):
    ports:  ClassVar[Tuple[str, ...]] = ()
    states: ClassVar[Tuple[str, ...]] = ()
    
    # Configuration flags
    _uses_time: ClassVar[bool] = False 

    # Layout & Optimization
    _VarsType_P: ClassVar[Any] = None
    _VarsType_S: ClassVar[Any] = None
    _n_ports: ClassVar[int] = 0
    _fast_physics: ClassVar[Any] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.ports:
            cls._VarsType_P = namedtuple("Ports",  cls.ports)
        if cls.states:
            cls._VarsType_S = namedtuple("States", cls.states)
        cls._n_ports = len(cls.ports)

    def __call__(self, t: float, y: jax.Array = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Human Debug Entry Point.
        Accepts 't' regardless of component type, but only passes it to the
        physics logic if the component is a @source.
        """
        # 1. Unpack Inputs (Vector or Kwargs)
        if y is not None:
            n_p = self._n_ports
            signals = self._VarsType_P(*y[:n_p]) if self._VarsType_P else ()
            s       = self._VarsType_S(*y[n_p:]) if self._VarsType_S else ()
        else:
            def _get_args(names):
                return [kwargs.get(name, 0.0) for name in names]
            signals = self._VarsType_P(*_get_args(self.ports))  if self._VarsType_P else ()
            s       = self._VarsType_S(*_get_args(self.states)) if self._VarsType_S else ()

        # 2. Dispatch based on time-dependency
        if self._uses_time:
            return self.physics(signals, s, t)
        else:
            return self.physics(signals, s)

    def solver_call(self, t: float, y: jax.Array, args: Any = None) -> Tuple[jax.Array, jax.Array]:
        """
        Solver Entry Point.
        Always accepts (t, y, args) to satisfy Diffrax/ODEPACK interfaces.
        """
        return self._fast_physics(y, self, t)

    def physics(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The Builder (Shared logic for @component and @source)
# ---------------------------------------------------------------------------
def _build_component(fn, ports, states, uses_time: bool):
    """Internal builder to construct the class."""
    
    # 1. Validate Signature
    reserved = ("signals", "s", "t") if uses_time else ("signals", "s")
    
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    # Check length
    if len(params) < len(reserved):
        raise TypeError(f"Function '{fn.__name__}' must start with arguments {reserved}")
    
    # Check reserved prefix names
    for i, expected in enumerate(reserved):
        if params[i].name != expected:
             raise TypeError(
                 f"Arg #{i+1} of '{fn.__name__}' must be '{expected}', got '{params[i].name}'. "
                 f"Expected signature start: {reserved}"
             )

    param_specs = params[len(reserved):]

    # Check for specific misconfiguration: using 't' in a non-time component
    if not uses_time:
        for p in param_specs:
            if p.name == "t":
                raise TypeError(
                    f"Function '{fn.__name__}' has argument 't' but uses @component. "
                    f"Use @source for time-dependent components."
                )

    # Check that all remaining parameters have defaults
    for p in param_specs:
        if p.default is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{p.name}' in '{fn.__name__}' must have a default value.")

    # 2. Dry-Run
    _dummy_P  = namedtuple("Ports",  ports)(*([0.0] * len(ports)))   if ports  else ()
    _dummy_S  = namedtuple("States", states)(*([0.0] * len(states))) if states else ()
    _defaults = {p.name: p.default for p in param_specs}

    try:
        if uses_time:
            fn(_dummy_P, _dummy_S, 0.0, **_defaults)
        else:
            fn(_dummy_P, _dummy_S, **_defaults)
    except Exception as exc:
        raise TypeError(f"Dry-run of '{fn.__name__}' failed: {exc}") from exc

    # 3. Build Fast Closure
    n_p          = len(ports)
    full_keys    = ports + states
    _param_names = tuple(p.name for p in param_specs)
    _user_fn     = fn
    _PortsType   = namedtuple("Ports",  ports)  if ports  else None
    _StatesType  = namedtuple("States", states) if states else None
    
    if len(full_keys) == 0:
        _fast_physics = lambda v, i, t: (jnp.zeros(0), jnp.zeros(0))
    else:
        def _fast_physics(vars_vec, instance, t):
            signals = _PortsType(*vars_vec[:n_p])  if _PortsType  else ()
            s       = _StatesType(*vars_vec[n_p:]) if _StatesType else ()
            kwargs  = {name: getattr(instance, name) for name in _param_names}
            
            if uses_time:
                f_dict, q_dict = _user_fn(signals, s, t, **kwargs)
            else:
                f_dict, q_dict = _user_fn(signals, s, **kwargs)
            
            f_vals = [f_dict.get(k, 0.0) for k in full_keys]
            q_vals = [q_dict.get(k, 0.0) for k in full_keys]
            return jnp.array(f_vals), jnp.array(q_vals)

    # 4. User Physics Wrapper
    if uses_time:
        def physics(self, signals: Signals, s: States, t: float):
            kwargs = {name: getattr(self, name) for name in _param_names}
            return _user_fn(signals, s, t, **kwargs)
    else:
        def physics(self, signals: Signals, s: States):
            kwargs = {name: getattr(self, name) for name in _param_names}
            return _user_fn(signals, s, **kwargs)

    # 5. Assemble
    annotations = {p.name: (p.annotation if p.annotation is not inspect.Parameter.empty else Any) for p in param_specs}
    defaults    = {p.name: p.default for p in param_specs}

    namespace = {
        "__annotations__": annotations,
        "ports":           ports,
        "states":          states,
        "physics":         physics,
        "_fast_physics":   staticmethod(_fast_physics),
        "_uses_time":      uses_time,
        **defaults,
    }

    cls = type(fn.__name__, (CircuitComponent,), namespace)
    cls.__doc__ = fn.__doc__
    return cls

# ---------------------------------------------------------------------------
# Public Decorators
# ---------------------------------------------------------------------------

def component(ports: Tuple[str, ...] = (), states: Tuple[str, ...] = ()):
    """Standard time-invariant component. Signature: (signals, s, **kwargs)."""
    def decorator(fn):
        return _build_component(fn, ports, states, uses_time=False)
    return decorator


def source(ports: Tuple[str, ...] = (), states: Tuple[str, ...] = ()):
    """Time-dependent source. Signature: (signals, s, t, **kwargs)."""
    def decorator(fn):
        return _build_component(fn, ports, states, uses_time=True)
    return decorator


# ===========================================================================
# Examples
# ===========================================================================

# 1. Standard Component (No 't' in signature)
@component(ports=("p", "n"))
def Resistor(signals: Signals, s: States, R: float = 1e3):
    """Ohm's law. No time variable needed."""
    i = (signals.p - signals.n) / (R + 1e-12)
    return {"p": i, "n": -i}, {}


# 2. Time-Dependent Source (Has 't' in signature)
@source(ports=("p", "n"), states=("i_src",))
def SineWaveSource(signals: Signals, s: States, t: float, A: float = 1.0, f: float = 50.0):
    """V(t) = A * sin(2*pi*f*t)."""
    import jax.numpy as jnp
    
    target_v = A * jnp.sin(2 * jnp.pi * f * t)
    constraint = (signals.p - signals.n) - target_v
    
    return {
        "p": s.i_src,
        "n": -s.i_src,
        "i_src": constraint
    }, {}


# ===========================================================================
# Tests
# ===========================================================================

def test_component_no_time():
    """Ensure Resistor doesn't crash when called without 't' logic."""
    r = Resistor(R=100.0)
    
    # Debug Call (user passes 0.0 for t, but it's ignored internally)
    f, _ = r(0.0, p=10.0, n=0.0)
    assert jnp.allclose(f["p"], 0.1)
    
    # Solver Call (solver passes t, wrapper drops it)
    y = jnp.array([10.0, 0.0])
    f_vec, _ = r.solver_call(999.0, y) # t=999 shouldn't matter
    assert jnp.allclose(f_vec[0], 0.1)
    print("  [PASS] test_component_no_time")


def test_source_with_time():
    """Ensure SineWaveSource receives 't'."""
    src = SineWaveSource(A=10.0, f=1.0) # Period = 1.0s
    y_sat = jnp.array([0.0, 0.0, 0.0]) # [p, n, i_src]
    
    # t=0.25 -> sin(pi/2) = 1.0 -> target=10V
    # p=0, n=0 -> current constraint error = (0-0) - 10 = -10
    f_vec, _ = src.solver_call(0.25, y_sat)
    
    assert jnp.allclose(f_vec[2], -10.0)
    print("  [PASS] test_source_with_time")


def test_signature_validation():
    # Test 1: @component REJECTS a function with 't'
    try:
        @component(ports=("a",))
        def BadResistor(signals, s, t, R=1): return {}, {}
        assert False, "Should have raised TypeError"
    except TypeError as e:
        # Now we check for the specific advice about @source
        assert "Use @source" in str(e)
    
    # Test 2: @source REJECTS a function WITHOUT 't'
    try:
        @source(ports=("a",))
        def BadSource(signals, s, A=1): return {}, {}
        assert False, "Should have raised TypeError"
    except TypeError as e:
        # Validates the strict prefix check
        assert "Arg #3" in str(e) and "must be 't'" in str(e)
        
    print("  [PASS] test_signature_validation")

def test_vmap_over_inputs():
    """
    Scenario: Single Component, Batch of Inputs.
    Use case: Simulating one resistor across 1000 different timesteps or voltage conditions simultaneously.
    """
    r = Resistor(R=10.0)
    
    # Create a batch of 5 different voltage drops: [10V, 20V, 30V, 40V, 50V]
    # Input shape: (5, 2) -> 5 batches, 2 ports (p, n)
    # y vector structure: [p, n]
    p_values = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    n_values = jnp.zeros(5)
    
    # Stack into batch y: shape (5, 2)
    batch_y = jnp.stack([p_values, n_values], axis=1)
    
    # We vmap over the 'y' argument (axis 1 of signature (t, y))
    # in_axes: (None, 0) -> t is broadcasted, y is mapped over axis 0
    batch_run = jax.vmap(r.solver_call, in_axes=(None, 0))
    
    f_batch, q_batch = batch_run(0.0, batch_y)
    
    assert f_batch.shape == (5, 2) # (Batch, Ports)
    
    # Check logic: I = V/R = V/10
    expected_currents = p_values / 10.0
    assert jnp.allclose(f_batch[:, 0], expected_currents)
    print("  [PASS] test_vmap_over_inputs")


def test_vmap_over_params():
    """
    Scenario: Batch of Components, Single Input.
    Use case: Monte Carlo analysis (sweeping parameter tolerances).
    We create a batch of Resistors with different R values.
    """
    # 1. Create a "Stack" of Resistors using eqx.tree_at or manual mapping.
    #    Simpler way: Construct arrays of R and use vmap on the constructor if strictly functional,
    #    but since R is a field, we can construct the tree leaves manually.
    
    # R_values = [1, 10, 100]
    R_batch = jnp.array([1.0, 10.0, 100.0])
    
    # Helper to map constructor over R values
    # We use vmap to create a batch of PyTrees (Resistor instances)
    # Note: To vmap object creation, the class must be treated as a Pytree function.
    # An easier way with Equinox is to use eqx.filter_vmap on the function calling the object.
    
    def run_with_r(r_val, t, y):
        # Create resistor inside the vmapped function
        r = Resistor(R=r_val)
        return r.solver_call(t, y)

    # Vectorize over r_val (arg 0)
    # t (arg 1) is None (broadcast)
    # y (arg 2) is None (broadcast)
    run_sweep = jax.vmap(run_with_r, in_axes=(0, None, None))
    
    y = jnp.array([10.0, 0.0]) # 10V drop fixed
    
    f_batch, _ = run_sweep(R_batch, 0.0, y)
    
    # Results should be: 10/1=10A, 10/10=1A, 10/100=0.1A
    expected_i = jnp.array([10.0, 1.0, 0.1])
    
    assert jnp.allclose(f_batch[:, 0], expected_i)
    print("  [PASS] test_vmap_over_params")


def test_pmap_multi_device_simulation():
    """
    Scenario: Parallel Map.
    Checks if the code can run across multiple devices (or emulated devices).
    """
    # Get available devices (or 1 if CPU)
    n_devices = jax.local_device_count()
    r = Resistor(R=2.0)
    
    # Replicate inputs for each device
    # Shape: (n_devices, 2)
    y_per_device = jnp.tile(jnp.array([10.0, 0.0]), (n_devices, 1))
    t_per_device = jnp.zeros(n_devices)
    
    # pmap the solver call
    # We must replicate the 'r' object if it contains arrays, or rely on closure.
    # Safest way with Equinox modules + pmap:
    
    def step(t, y):
        return r.solver_call(t, y)
    
    # pmap over t and y (leading dimension)
    pmapped_step = jax.pmap(step)
    
    f_res, _ = pmapped_step(t_per_device, y_per_device)
    
    # Check shape: (n_devices, n_vars)
    assert f_res.shape == (n_devices, 2)
    
    # Result: 10V / 2Ohms = 5A
    assert jnp.allclose(f_res[0, 0], 5.0)
    print(f"  [PASS] test_pmap_multi_device_simulation ({n_devices} devices)")


def test_jit_control_flow_safety():
    """
    Ensure the component works inside JIT even with python control flow
    that was resolved at decoration time.
    """
    # This component has 'if' statements in the builder, but not in the runtime physics.
    # This test confirms that jax.jit doesn't complain.
    
    src = SineWaveSource(A=10.0, f=1.0)
    
    @jax.jit
    def fast_step(t, y):
        return src.solver_call(t, y)
        
    # First call triggers trace
    y = jnp.zeros(3)
    f1, _ = fast_step(0.25, y) # sin(pi/2)=1 -> target=10
    
    # Second call uses compiled kernel
    f2, _ = fast_step(0.25, y)
    
    assert jnp.allclose(f1, f2)
    assert jnp.allclose(f1[2], -10.0)
    print("  [PASS] test_jit_control_flow_safety")

if __name__ == "__main__":
    print("Running time-dependency tests...\n")
    test_vmap_over_inputs()
    test_vmap_over_params()
    test_pmap_multi_device_simulation()
    test_jit_control_flow_safety()
    test_component_no_time()
    test_source_with_time()
    test_signature_validation()
    print("\nAll tests passed.")