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
    ports: ClassVar[Tuple[str, ...]] = ()
    states: ClassVar[Tuple[str, ...]] = ()

    # Configuration flags
    _uses_time: ClassVar[bool] = False

    # Layout & Optimization
    _VarsType_P: ClassVar[Any] = None
    _VarsType_S: ClassVar[Any] = None
    _n_ports: ClassVar[int] = 0

    # The static closure
    _fast_physics: ClassVar[Any] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.ports:
            cls._VarsType_P = namedtuple("Ports", cls.ports)
        if cls.states:
            cls._VarsType_S = namedtuple("States", cls.states)
        cls._n_ports = len(cls.ports)

    def __call__(
        self, t: Any = 0.0, y: jax.Array = None, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Debug Entry Point (Instance Method).
        Passes 'self' as the parameters to the physics engine.
        """
        # --- Heuristic: Handle "Vector Shortcut" (component(y)) ---
        if y is None and not kwargs:
            is_scalar = False
            if isinstance(t, (int, float)):
                is_scalar = True
            elif hasattr(t, "shape") and t.shape == ():
                is_scalar = True

            if not is_scalar:
                y = t
                t = 0.0

        # --- 1. Unpack Inputs ---
        if y is not None:
            n_p = self._n_ports
            signals = self._VarsType_P(*y[:n_p]) if self._VarsType_P else ()
            s = self._VarsType_S(*y[n_p:]) if self._VarsType_S else ()
        else:

            def _get_args(names):
                return [kwargs.get(name, 0.0) for name in names]

            signals = (
                self._VarsType_P(*_get_args(self.ports)) if self._VarsType_P else ()
            )
            s = self._VarsType_S(*_get_args(self.states)) if self._VarsType_S else ()

        # --- 2. Dispatch ---
        # Note: We pass 'self' as the parameter container here!
        return self._invoke_physics(signals, s, t, self)

    @classmethod
    def solver_call(
        cls, t: float, y: jax.Array, args: Any
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Solver Entry Point (Class Method).

        Args:
            t: Time
            y: State Vector
            args: Parameter container. MUST be a Dictionary {'R': 100} or Object (like self).
                  CANNOT be a raw scalar.
        """
        # 'cls' is now the specific subclass (e.g. Resistor), so _fast_physics is valid.
        return cls._fast_physics(y, args, t)

    # -----------------------------------------------------------------------
    # Internal Dispatchers (wired up by decorator)
    # -----------------------------------------------------------------------
    def physics(self, *args, **kwargs):
        raise NotImplementedError

    def _invoke_physics(self, signals, s, t, params):
        """Trampoline for the debug __call__."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: Parameter Extraction
# ---------------------------------------------------------------------------
def _extract_param(container, name):
    """Get parameter from Object (attr) or Dict (item)."""
    if isinstance(container, dict):
        return container[name]
    else:
        return getattr(container, name)


# ---------------------------------------------------------------------------
# The Builder
# ---------------------------------------------------------------------------
def _build_component(fn, ports, states, uses_time: bool):
    reserved = ("signals", "s", "t") if uses_time else ("signals", "s")

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    # Validation (Checks signature...)
    if len(params) < len(reserved):
        raise TypeError(
            f"Function '{fn.__name__}' must start with arguments {reserved}"
        )
    for i, expected in enumerate(reserved):
        if params[i].name != expected:
            raise TypeError(f"Arg #{i + 1} must be '{expected}'")

    param_specs = params[len(reserved) :]

    if not uses_time:
        for p in param_specs:
            if p.name == "t":
                raise TypeError("Use @source for time-dependent components.")

    for p in param_specs:
        if p.default is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{p.name}' must have a default.")

    # Dry-Run
    _dummy_P = namedtuple("Ports", ports)(*([0.0] * len(ports))) if ports else ()
    _dummy_S = namedtuple("States", states)(*([0.0] * len(states))) if states else ()
    _defaults = {p.name: p.default for p in param_specs}

    try:
        if uses_time:
            fn(_dummy_P, _dummy_S, 0.0, **_defaults)
        else:
            fn(_dummy_P, _dummy_S, **_defaults)
    except Exception as exc:
        raise TypeError(f"Dry-run failed: {exc}") from exc

    # --- Build Fast Closure (Static) ---
    n_p = len(ports)
    full_keys = ports + states
    _param_names = tuple(p.name for p in param_specs)
    _user_fn = fn
    _PortsType = namedtuple("Ports", ports) if ports else None
    _StatesType = namedtuple("States", states) if states else None

    if len(full_keys) == 0:
        _fast_physics = lambda v, p, t: (jnp.zeros(0), jnp.zeros(0))
    else:
        # Note: 'params' argument replaces 'instance'
        def _fast_physics(vars_vec, params, t):
            signals = _PortsType(*vars_vec[:n_p]) if _PortsType else ()
            s = _StatesType(*vars_vec[n_p:]) if _StatesType else ()

            # Extract kwargs using helper (supports Dict or Object)
            kwargs = {name: _extract_param(params, name) for name in _param_names}

            if uses_time:
                f_dict, q_dict = _user_fn(signals, s, t, **kwargs)
            else:
                f_dict, q_dict = _user_fn(signals, s, **kwargs)

            f_vals = [f_dict.get(k, 0.0) for k in full_keys]
            q_vals = [q_dict.get(k, 0.0) for k in full_keys]
            return jnp.array(f_vals), jnp.array(q_vals)

    # --- Build Debug Invoker ---
    if uses_time:

        def _invoke_physics(self, signals, s, t, params):
            kwargs = {name: _extract_param(params, name) for name in _param_names}
            return _user_fn(signals, s, t, **kwargs)
    else:

        def _invoke_physics(self, signals, s, t, params):
            kwargs = {name: _extract_param(params, name) for name in _param_names}
            return _user_fn(signals, s, **kwargs)

    # Assemble
    annotations = {
        p.name: (p.annotation if p.annotation is not inspect.Parameter.empty else Any)
        for p in param_specs
    }
    defaults = {p.name: p.default for p in param_specs}

    namespace = {
        "__annotations__": annotations,
        "ports": ports,
        "states": states,
        "_fast_physics": staticmethod(_fast_physics),
        "_invoke_physics": _invoke_physics,  # Bound method for debug
        "_uses_time": uses_time,
        **defaults,
    }

    cls = type(fn.__name__, (CircuitComponent,), namespace)
    cls.__doc__ = fn.__doc__
    return cls


def component(ports=(), states=()):
    return lambda fn: _build_component(fn, ports, states, False)


def source(ports=(), states=()):
    return lambda fn: _build_component(fn, ports, states, True)
