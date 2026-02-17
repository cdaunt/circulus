"""Utilities for converting between S-parameters.

Utilities for converting between S-parameter and admittance representations,
and for wrapping SAX model functions as Circulus components.
"""

import inspect

import jax
import jax.numpy as jnp
from sax import get_ports, sdense

from circulus.components.base_component import Signals, States, component


@jax.jit
def s_to_y(S: jax.Array, z0: float = 1.0) -> jax.Array:
    """Convert an S-parameter matrix to an admittance (Y) matrix.

    Uses the formula ``Y = (1/z0) * (I - S) * (I + S)^-1``. Requires dense
    matrix inversion; if a component can be defined directly in terms of a
    Y-matrix it should be, to avoid the overhead of this conversion.

    Args:
        S: S-parameter matrix of shape ``(..., n, n)``.
        z0: Reference impedance in ohms. Defaults to ``1.0``.

    Returns:
        Y-matrix of the same shape and dtype as ``S``.

    """
    n = S.shape[-1]
    eye = jnp.eye(n, dtype=S.dtype)
    return (1.0 / z0) * (eye - S) @ jnp.linalg.inv(eye + S)


def sax_component(fn):  # noqa: ANN001, ANN201
    """Decorator to convert a SAX model function into a Circulus component.

    Inspects ``fn`` at decoration time to discover its port interface via a
    dry run, then wraps its S-matrix output in an admittance-based physics
    function compatible with the Circulus nodal solver.

    The conversion proceeds in three stages:

    1. **Discovery** — ``fn`` is called once with its default (or dummy)
       parameter values and :func:`sax.get_ports` extracts the sorted port
       names from the resulting S-parameter dict.
    2. **Physics wrapper** — a closure is built that calls ``fn`` at runtime,
       converts the S-dict to a dense matrix via :func:`sax.sdense`, converts
       it to an admittance matrix via :func:`s_to_y`, and returns
       ``I = Y @ V`` as a port current dict.
    3. **Component registration** — the wrapper is passed to
       :func:`~circulus.components.base_component.component` with the
       discovered ports, producing a :class:`~circulus.components.base_component.CircuitComponent`
       subclass.

    Args:
        fn: A SAX model function whose keyword arguments are scalar
            parameters and whose return value is a SAX S-parameter dict.
            All parameters must have defaults, or will be substituted with
            ``1.0`` during the dry run.

    Returns:
        A :class:`~circulus.components.base_component.CircuitComponent`
        subclass named after ``fn``.

    Raises:
        RuntimeError: If the dry run fails for any reason.

    """
    sig = inspect.signature(fn)
    defaults = {
        param.name: param.default if param.default is not inspect.Parameter.empty else 1.0
        for param in sig.parameters.values()
    }

    try:
        dummy_s_dict = fn(**defaults)
        detected_ports = get_ports(dummy_s_dict)
    except Exception as exc:
        msg = f"Failed to dry-run SAX component '{fn.__name__}': {exc}"
        raise RuntimeError(msg) from exc

    def physics_wrapper(signals: Signals, s: States, **kwargs) -> tuple[dict, dict]:  # noqa: ANN003
        s_dict = fn(**kwargs)
        s_matrix, _ = sdense(s_dict)
        y_matrix = s_to_y(s_matrix)
        v_vec = jnp.array(
            [getattr(signals, p) for p in detected_ports], dtype=jnp.complex128
        )
        i_vec = y_matrix @ v_vec
        return {p: i_vec[i] for i, p in enumerate(detected_ports)}, {}

    physics_wrapper.__name__ = fn.__name__
    physics_wrapper.__doc__ = fn.__doc__
    physics_wrapper.__signature__ = sig

    return component(ports=detected_ports)(physics_wrapper)
