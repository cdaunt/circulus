import jax
import jax.numpy as jnp
import inspect
from typing import NamedTuple, Callable, List, Dict, Any
from collections import defaultdict
import inspect
from functools import wraps, lru_cache
from natsort import natsorted
import inspect

from circulus.netlist import build_net_map

def ensure_time_signature(model_func):
    """
    Wraps a model function to ensure it accepts a 't' keyword argument.
    If the original model doesn't take 't', the wrapper swallows it.
    """
    sig = inspect.signature(model_func)
    
    if 't' in sig.parameters or '**kwargs' in str(sig):
        return model_func
    
    # Wrapper for static models
    @wraps(model_func)
    def time_aware_wrapper(vars, params, t=None):
        return model_func(vars, params)
        
    return time_aware_wrapper

class ComponentGroup(NamedTuple):
    """
    Represents a BATCH of identical components (e.g., ALL Resistors).
    Optimized for jax.vmap in the solver.
    """
    name: str                 # e.g., "resistor_model"
    physics_func: Callable    # The shared model function
    
    # BATCHED PARAMETERS: 
    # {'R': jnp.array([100.0, 200.0, ...])} - Shape (N,)
    params: Dict[str, jnp.ndarray] 
    
    # BATCHED STATE INDICES:
    # Shape (N, num_vars_per_component) - Used to gather 'v' from y0
    var_indices: jnp.ndarray 
    
    # BATCHED EQUATION INDICES:
    # Shape (N, num_eqs_per_component) - Used to scatter 'i' into residual
    eq_indices:  jnp.ndarray
    
    # FLATTENED JACOBIAN INDICES:
    # Shape (N * num_vars * num_vars,) - Concatenated for BCOO construction
    jac_rows: jnp.ndarray    
    jac_cols: jnp.ndarray

    index_map: Dict[str, int] | None = None  # Optional mapping from instance names to indices in the group



def get_model_width(func):
    """Determines the size of the 'vars' vector expected by the model."""
    sig = inspect.signature(func)
    if 'vars' not in sig.parameters:
        raise ValueError(f"{func.__name__} missing 'vars' argument")
    default_val = sig.parameters['vars'].default
    if default_val is inspect.Parameter.empty:
        raise ValueError(f"{func.__name__} 'vars' must have a default (e.g. jnp.zeros(3))")
    return len(default_val)


# --- Main Compiler ---

def merge_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merges a list of dictionaries into a single dictionary."""
    merged = {}
    for d in dict_list:
        merged.update(d)
    return merged

@lru_cache(maxsize=None)
def _get_default_params_cached(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)

    if 'params' not in sig.parameters:
        raise ValueError(f"{func.__name__} missing 'params' argument")

    default_params = sig.parameters['params'].default

    if default_params is inspect.Parameter.empty:
        raise ValueError(
            f"{func.__name__} 'params' must have a default value "
            "(e.g. {'R': 100.0})"
        )

    return default_params


def get_default_params(func: Callable) -> Dict[str, Any]:
    # Return a copy so callers can’t mutate the cache
    return dict(_get_default_params_cached(func))

def compile_netlist(
    netlist: dict, 
    models: dict[str, Callable]
) -> tuple[dict[str, ComponentGroup], int, dict[str, int]]:
    """
    Directly converts a Dictionary Netlist into Vectorized Component Groups.
    Avoids intermediate Instance objects for speed.
    """
    
    # 1. Analyze Connectivity
    port_map, num_nodes = build_net_map(netlist)
    
    # 2. Bucket Data by Component Type
    # structure: buckets[model_name] = { 'names': [], 'connections': [], 'params': {key: []} }
    buckets = defaultdict(lambda: {
        'names': [], 
        'connections': [], 
        'params': defaultdict(list)
    })
    
    instances = netlist.get("instances", {})
    
    # Single pass through the netlist dictionary
    for name, data in instances.items():
        comp_type = data.get('component')
        
        # Skip ground definition (it's implicit in the net map)
        if comp_type == 'ground' or name == 'GND':
            continue
            
        if comp_type not in models:
            raise ValueError(f"Model '{comp_type}' not found in provided models dict.")
            
        # Resolve connections to integers immediately
        # We assume standard pin naming or order implies connection order
        # If specific pin names are required, sort keys of port_map matching name
        
        # Heuristic: Find all ports in port_map that start with "InstanceName,"
        # and sort them to ensure p1, p2, p3 order.
        related_ports = [k for k in port_map.keys() if k.startswith(f"{name},")]
        related_ports = natsorted(related_ports) 
        
        if not related_ports:
            # Handle unconnected components or manual connection lists if needed
            continue
            
        conn_indices = [port_map[p] for p in related_ports]
        
        # Append to bucket
        b = buckets[comp_type]
        b['names'].append(name)
        b['connections'].append(conn_indices)

        # This is cached for performance
        default_params = get_default_params(models[comp_type])

        # Append params/fallback to defaults
        local_settings = data.get('settings', {})
        for k, v in default_params.items():
            b['params'][k].append(local_settings.get(k, v))
            

    # 3. Vectorize Buckets
    compiled_groups = {}
    
    # Track where internal variables start (after the last physical node)
    internal_var_offset = num_nodes 
    
    for comp_type, data in buckets.items():
        func = ensure_time_signature(models[comp_type])
        count = len(data['names'])
        
        if count == 0:
            continue

        # Convert connections to JAX array (N, num_ports)
        node_indices = jnp.array(data['connections'], dtype=jnp.int32)
        
        # Batch parameters. Defaults are handled at earlier stage
        batched_params = {k: jnp.array(v) for k, v in data['params'].items()}
        
        # Calculate Dimensions
        model_width = get_model_width(func)
        num_ports = node_indices.shape[1]
        num_internals = model_width - num_ports
        
        # -- Generate Internal Indices --
        if num_internals > 0:
            # Create a block of new indices: [offset, offset+1, ..., offset + N*internals]
            total_new_vars = count * num_internals
            
            # Shape (N, num_internals)
            internal_indices = jnp.arange(
                internal_var_offset, 
                internal_var_offset + total_new_vars
            ).reshape(count, num_internals)
            
            # Update global counter
            internal_var_offset += total_new_vars
            
            # Combine [Node Indices | Internal Indices]
            # Shape (N, model_width)
            var_indices = jnp.concatenate([node_indices, internal_indices], axis=1)
        else:
            var_indices = node_indices

        # -- Vectorized Jacobian Index Generation --
        # We need the Cartesian product of var_indices with itself for every instance
        # var_indices shape: (N, W)
        
        # Expand dims for broadcasting
        # rows: (N, W, 1)
        # cols: (N, 1, W)
        rows = var_indices[:, :, None]
        cols = var_indices[:, None, :]
        
        # Broadcast and flatten
        # Target shape for sparse matrix: (N * W^2, )
        jac_rows = jnp.broadcast_to(rows, (count, model_width, model_width)).flatten()
        jac_cols = jnp.broadcast_to(cols, (count, model_width, model_width)).flatten()

        index_map = {s: i for i, s in enumerate(data['names'])}

        group = ComponentGroup(
            name=comp_type,
            physics_func=func,
            params=batched_params,
            var_indices=var_indices,
            eq_indices=var_indices, # In this formulation, equations map 1-to-1 with variables
            jac_rows=jac_rows,
            jac_cols=jac_cols,
            index_map=index_map
        )
        compiled_groups[comp_type] = group

    return compiled_groups, internal_var_offset, port_map