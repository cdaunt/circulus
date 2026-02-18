"""Base class and decorators for defining JAX-compatible circuit components.

Circuit components are defined as plain Python functions decorated with
:func:`component` or :func:`source`, which compile them into
:class:`CircuitComponent` subclasses — Equinox modules whose parameters are
JAX-traceable leaves. The resulting classes expose two entry points:

- ``__call__`` — a debug-friendly instance method that accepts port voltages
  and state values as keyword arguments and returns the physics dicts directly.
- ``solver_call`` — a class method used by the transient solver that operates
  on flat JAX arrays and a parameter container, and is compatible with
  ``jax.vmap`` and ``jax.jacfwd``.

Example::

    @component(ports=("p1", "p2"))
    def Resistor(signals: Signals, s: States, R: float = 1.0):
        i = (signals.p1 - signals.p2) / R
        return {"p1": i, "p2": -i}, {}

    r = Resistor(R=100.0)
    f, q = r(p1=1.0, p2=0.0)
"""

import inspect
from collections import namedtuple
from typing import Any, ClassVar, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

PhysicsReturn = tuple[dict[str, Array], dict[str, Array]]
# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class Signals(Protocol):
    """Protocol representing the port voltage signals passed to a component's physics function.

    Attributes are accessed by port name (e.g. ``signals.p1``), backed by a
    namedtuple constructed from the component's ``ports`` declaration.
    """

    def __getattr__(self, name: str) -> Any: ...


@runtime_checkable
class States(Protocol):
    """Protocol representing the internal state variables passed to a component's physics function.

    Attributes are accessed by state name (e.g. ``s.i_L``), backed by a
    namedtuple constructed from the component's ``states`` declaration.
    """

    def __getattr__(self, name: str) -> Any: ...


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class CircuitComponent(eqx.Module):
    """Base class for all JAX-compatible circuit components.

    Subclasses are not written by hand — they are generated at import time by
    the :func:`component` and :func:`source` decorators, which inspect the
    decorated function's signature to populate the class variables and wire up
    the two physics entry points.

    Class Variables:
        ports: Ordered tuple of port names, e.g. ``("p1", "p2")``.
        states: Ordered tuple of internal state variable names,
            e.g. ``("i_L",)``. Empty for purely algebraic components.
        _uses_time: ``True`` for components decorated with :func:`source`
            whose physics function accepts a ``t`` argument.
        _VarsType_P: Namedtuple type for unpacking port voltages from a
            flat array. ``None`` if the component has no ports.
        _VarsType_S: Namedtuple type for unpacking state variables from a
            flat array. ``None`` if the component has no states.
        _n_ports: Number of ports, cached to avoid repeated ``len`` calls
            in the hot path.
        _fast_physics: Static closure over the user-defined physics function,
            compatible with ``jax.vmap`` and ``jax.jacfwd``. Signature is
            ``(vars_vec, params, t) -> (f_vec, q_vec)``.
    """

    ports: ClassVar[tuple[str, ...]] = ()
    states: ClassVar[tuple[str, ...]] = ()

    _uses_time: ClassVar[bool] = False

    _VarsType_P: ClassVar[Any] = None
    _VarsType_S: ClassVar[Any] = None
    _n_ports: ClassVar[int] = 0

    _fast_physics: ClassVar[Any] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialise namedtuple types and cached port count for each new subclass."""
        super().__init_subclass__(**kwargs)
        if cls.ports:
            cls._VarsType_P = namedtuple("Ports", cls.ports)  # noqa: PYI024
        if cls.states:
            cls._VarsType_S = namedtuple("States", cls.states) # noqa: PYI024
        cls._n_ports = len(cls.ports)

    def __call__(
        self,
        t: Any = 0.0,
        y: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluate the component physics (debug entry point).

        Accepts inputs either as a flat state vector ``y`` or as individual
        keyword arguments keyed by port and state name. A heuristic detects
        whether the first positional argument is a time value or a state vector,
        allowing the shorthand ``component(y)`` in addition to the explicit
        ``component(t=0.0, y=y)``.

        Args:
            t: Simulation time, or a state vector when called as
                ``component(y)`` without an explicit ``y`` keyword.
            y: Flat state vector of shape ``(n_ports + n_states,)``. When
                provided, port voltages are taken from ``y[:n_ports]`` and
                state values from ``y[n_ports:]``. Mutually exclusive with
                ``**kwargs``.
            **kwargs: Port voltages and state values by name, e.g.
                ``p1=1.0, p2=0.0``. Used when ``y`` is not provided.
                Missing names default to ``0.0``.

        Returns:
            A two-tuple ``(f, q)`` of dicts mapping port/state names to their
            resistive (``f``) and reactive (``q``) contributions respectively.

        """
        if y is None and not kwargs:
            is_scalar = isinstance(t, (int, float)) or (
                hasattr(t, "shape") and t.shape == ()
            )
            if not is_scalar:
                y = t
                t = 0.0

        if y is not None:
            n_p = self._n_ports
            signals = self._VarsType_P(*y[:n_p]) if self._VarsType_P else ()
            s = self._VarsType_S(*y[n_p:]) if self._VarsType_S else ()
        else:

            def _get_args(names: tuple[str, ...]) -> list[Any]:
                return [kwargs.get(name, 0.0) for name in names]

            signals = (
                self._VarsType_P(*_get_args(self.ports)) if self._VarsType_P else ()
            )
            s = self._VarsType_S(*_get_args(self.states)) if self._VarsType_S else ()

        return self._invoke_physics(signals, s, t, self)

    @classmethod
    def solver_call(
        cls,
        t: float,
        y: jax.Array,
        args: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the component physics (solver entry point).

        Thin wrapper around the static ``_fast_physics`` closure. Called by
        the transient solver inside ``jax.vmap`` across all instances in a
        component group, and differentiated via ``jax.jacfwd`` to assemble
        the system Jacobian.

        Args:
            t: Current simulation time.
            y: Flat state vector of shape ``(n_ports + n_states,)`` containing
                port voltages followed by state variable values.
            args: Parameter container for this instance. May be a dict
                ``{"R": 100.0}`` or an object (e.g. the component instance
                itself) whose attributes match the parameter names. Must not
                be a raw scalar.

        Returns:
            A two-tuple ``(f_vec, q_vec)`` of JAX arrays, each of shape
            ``(n_ports + n_states,)``, containing the resistive and reactive
            contributions for every port and state variable.

        """
        return cls._fast_physics(y, args, t)

    # -----------------------------------------------------------------------
    # Internal Dispatchers (wired up by decorator)
    # -----------------------------------------------------------------------
    def physics(self, *args: Any, **kwargs: Any) -> tuple[dict, dict]:
        """Raw physics dispatch; overridden by the decorator-generated subclass."""
        raise NotImplementedError

    def _invoke_physics(
        self,
        signals: Any,
        s: Any,
        t: float,
        params: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Trampoline used by ``__call__`` to dispatch to the user physics function.

        Overridden by the decorator-generated subclass with a closure that
        extracts named parameters from ``params`` and forwards them as keyword
        arguments to the original decorated function.

        Args:
            signals: Namedtuple of port voltages.
            s: Namedtuple of state variable values.
            t: Current simulation time.
            params: Parameter container (instance or dict).

        Returns:
            A two-tuple ``(f, q)`` of physics dicts.

        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: Parameter Extraction
# ---------------------------------------------------------------------------
def _extract_param(container: Any, name: str) -> Any:
    """Extract a named parameter from either a dict or an object.

    Args:
        container: A dict or any object with named attributes.
        name: The parameter name to look up.

    Returns:
        The parameter value.

    """
    if isinstance(container, dict):
        return container[name]
    return getattr(container, name)


# ---------------------------------------------------------------------------
# The Builder
# ---------------------------------------------------------------------------
def _build_component(  # noqa: C901
    fn: Any,
    ports: tuple[str, ...],
    states: tuple[str, ...],
    *,
    uses_time: bool,
) -> type[CircuitComponent]:
    """Compile a physics function into a :class:`CircuitComponent` subclass.

    Inspects ``fn``'s signature, validates it against the declared ``ports``
    and ``states``, performs a dry-run with default values to catch errors
    early, then builds two closures:

    - ``_fast_physics`` — a static function ``(vars_vec, params, t)`` suitable
      for ``jax.vmap`` / ``jax.jacfwd``, used by the solver.
    - ``_invoke_physics`` — a bound method used by the debug ``__call__`` path.

    Args:
        fn: The decorated physics function. Its signature must begin with
            ``(signals, s)`` for :func:`component` or ``(signals, s, t)``
            for :func:`source`, followed by any number of keyword-only
            parameters with defaults.
        ports: Ordered tuple of port names matching the netlist connections.
        states: Ordered tuple of internal state variable names.
        uses_time: ``True`` when compiling a :func:`source` component whose
            physics function accepts a time argument ``t``.

    Returns:
        A new :class:`CircuitComponent` subclass named after ``fn``.

    Raises:
        TypeError: If the function signature does not start with the required
            reserved arguments, if any parameter lacks a default value, if a
            non-source component declares a ``t`` parameter, or if the dry-run
            raises an exception.

    """
    reserved = ("signals", "s", "t") if uses_time else ("signals", "s")

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if len(params) < len(reserved):
        msg = f"Function '{fn.__name__}' must start with arguments {reserved}"
        raise TypeError(
            msg
        )
    for i, expected in enumerate(reserved):
        if params[i].name != expected:
            msg = f"Arg #{i + 1} must be '{expected}'"
            raise TypeError(msg)

    param_specs = params[len(reserved):]

    if not uses_time:
        for p in param_specs:
            if p.name == "t":
                msg = "Use @source for time-dependent components."
                raise TypeError(msg)

    for p in param_specs:
        if p.default is inspect.Parameter.empty:
            msg = f"Parameter '{p.name}' must have a default."
            raise TypeError(msg)

    _dummy_P = namedtuple("Ports", ports)(*([0.0] * len(ports))) if ports else () # noqa: PYI024
    _dummy_S = namedtuple("States", states)(*([0.0] * len(states))) if states else ()  # noqa: PYI024
    _defaults = {p.name: p.default for p in param_specs}

    try:
        if uses_time:
            fn(_dummy_P, _dummy_S, 0.0, **_defaults)
        else:
            fn(_dummy_P, _dummy_S, **_defaults)
    except Exception as exc:
        raise TypeError(f"Dry-run failed: {exc}") from exc

    n_p = len(ports)
    full_keys = ports + states
    _param_names = tuple(p.name for p in param_specs)
    _user_fn = fn
    _PortsType = namedtuple("Ports", ports) if ports else None # noqa: PYI024
    _StatesType = namedtuple("States", states) if states else None # noqa: PYI024

    if len(full_keys) == 0:
        _fast_physics = lambda v, p, t: (jnp.zeros(0), jnp.zeros(0))  # noqa: E731
    else:
        def _fast_physics(
            vars_vec: jax.Array,
            params: Any,
            t: float,
        ) -> tuple[jax.Array, jax.Array]:
            signals = _PortsType(*vars_vec[:n_p]) if _PortsType else ()
            s = _StatesType(*vars_vec[n_p:]) if _StatesType else ()
            kw = {name: _extract_param(params, name) for name in _param_names}
            if uses_time:
                f_dict, q_dict = _user_fn(signals, s, t, **kw)
            else:
                f_dict, q_dict = _user_fn(signals, s, **kw)
            f_vals = [f_dict.get(k, 0.0) for k in full_keys]
            q_vals = [q_dict.get(k, 0.0) for k in full_keys]
            return jnp.array(f_vals), jnp.array(q_vals)

    if uses_time:
        def _invoke_physics(
            self: CircuitComponent,
            signals: Any,
            s: Any,
            t: float,
            params: Any,
        ) -> tuple[dict, dict]:
            kw = {name: _extract_param(params, name) for name in _param_names}
            return _user_fn(signals, s, t, **kw)
    else:
        def _invoke_physics(
            self: CircuitComponent,
            signals: Any,
            s: Any,
            t: float,
            params: Any,
        ) -> tuple[dict, dict]:
            kw = {name: _extract_param(params, name) for name in _param_names}
            return _user_fn(signals, s, **kw)

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
        "_invoke_physics": _invoke_physics,
        "_uses_time": uses_time,
        **defaults,
    }

    cls = type(fn.__name__, (CircuitComponent,), namespace)
    cls.__doc__ = fn.__doc__
    return cls


def component(
    ports: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
) -> Any:
    """Decorator for defining a time-independent circuit component.

    Compiles the decorated physics function into a :class:`CircuitComponent`
    subclass. The function must begin with ``(signals, s)`` followed by any
    number of parameters with defaults, which become JAX-traceable Equinox
    fields on the resulting class.

    Args:
        ports: Ordered tuple of port names. Must match the connection keys
            used in the netlist.
        states: Ordered tuple of internal state variable names. State
            variables are appended to the solver's state vector after the
            node voltages.

    Returns:
        A decorator that accepts a physics function and returns a
        :class:`CircuitComponent` subclass.

    Example::

        @component(ports=("p1", "p2"))
        def Resistor(signals: Signals, s: States, R: float = 1.0):
            i = (signals.p1 - signals.p2) / R
            return {"p1": i, "p2": -i}, {}

    """
    return lambda fn: _build_component(fn, ports, states, uses_time=False)


def source(
    ports: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
) -> Any:
    """Decorator for defining a time-dependent circuit component.

    Identical to :func:`component` except the decorated physics function
    must accept ``t`` as its third argument (after ``signals`` and ``s``),
    and may use it to implement time-varying behaviour such as sinusoidal
    sources or delayed step functions.

    Args:
        ports: Ordered tuple of port names.
        states: Ordered tuple of internal state variable names.

    Returns:
        A decorator that accepts a physics function and returns a
        :class:`CircuitComponent` subclass.

    Example::

        @source(ports=("p1", "p2"), states=("i_src",))
        def VoltageSource(signals: Signals, s: States, t: float, V: float = 1.0):
            constraint = (signals.p1 - signals.p2) - V
            return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}

    """
    return lambda fn: _build_component(fn, ports, states, uses_time=True)
