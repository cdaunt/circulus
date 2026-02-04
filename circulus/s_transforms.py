import jax
import jax.numpy as jnp
import inspect
from sax import sdense, get_ports
from circulus.base_component import component, Signals, States


@jax.jit
def s_to_y(S: jax.Array, z0: float = 1.0) -> jax.Array:
    """
    Utility: Converts an S-parameter matrix to an Admittance (Y) matrix.
    Formula: Y = (1/z0) * (I - S) * (I + S)^-1

    Note: The conversion of s to y requires the use of dense matrices. If models can be defined in terms of
    y matrices alone, they should
    """
    n = S.shape[-1]
    I = jnp.eye(n, dtype=S.dtype)
    return (1.0 / z0) * (I - S) @ jnp.linalg.inv(I + S)


def sax_component(fn):
    """
    Decorator to convert a SAX model function into a Circulus component.
    
    1. Uses sax.get_ports during definition to establish the component interface.
    2. Uses sax.sdense during runtime to generate the S-matrix.
    """
    
    # -----------------------------------------------------------------------
    # 1. Introspect for default values (needed for the dry run)
    # -----------------------------------------------------------------------
    sig = inspect.signature(fn)
    defaults = {}
    for param in sig.parameters.values():
        if param.default is not inspect.Parameter.empty:
            defaults[param.name] = param.default
        else:
            defaults[param.name] = 1.0 # Dummy value for dry run

    # -----------------------------------------------------------------------
    # 2. Dry Run: Discovery
    # -----------------------------------------------------------------------
    try:
        # Run model once with dummy/default values
        dummy_s_dict = fn(**defaults)
        
        # Use SAX's native helper to get sorted port names
        detected_ports = get_ports(dummy_s_dict)
        
    except Exception as e:
        raise RuntimeError(f"Failed to dry-run SAX component '{fn.__name__}': {e}")
    
    # -----------------------------------------------------------------------
    # 3. Define the Physics Wrapper
    # -----------------------------------------------------------------------
    def physics_wrapper(signals: Signals, s: States, **kwargs):
        # A. Call the SAX model
        s_dict = fn(**kwargs)
        
        # B. Convert to Dense Matrix
        #    sdense automatically sorts ports alphabetically, matching get_ports.
        S_matrix, _ = sdense(s_dict)
        
        # C. Convert S -> Y (Admittance)
        Y_matrix = s_to_y(S_matrix)
        
        # D. Solve I = Y * V
        #    Construct voltage vector in the exact order found by get_ports
        v_vec = jnp.array([getattr(signals, p) for p in detected_ports], dtype=jnp.complex128)
        
        #    Calculate currents
        i_vec = Y_matrix @ v_vec
        
        # E. Map back to dictionary
        return {p: i_vec[i] for i, p in enumerate(detected_ports)}, {}

    # Copy metadata
    physics_wrapper.__name__ = fn.__name__
    physics_wrapper.__doc__ = fn.__doc__
    physics_wrapper.__signature__ = sig 

    # 4. Apply the Circulus component decorator
    return component(ports=detected_ports)(physics_wrapper)