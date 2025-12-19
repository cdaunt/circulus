import jax
import jax.numpy as jnp
import inspect
from typing import NamedTuple, Callable, List, Dict, Any
from collections import defaultdict
import inspect
from functools import wraps

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

# Data structure for a generic component in the netlist
class Instance(NamedTuple):
    name: str
    model_func: Callable
    connections: List[int] # List of Global Node Indices
    params: Dict[str, float]

# Data structure for the Solver (Compiled Block)
class CompiledBlock(NamedTuple):
    physics_func: Callable
    params: Dict[str, Any]
    
    # Map Global Vector -> Local Vector
    var_indices: jnp.ndarray 
    eq_indices:  jnp.ndarray
    
    # Map Local Jacobian -> Global Sparse Matrix
    jac_rows: jnp.ndarray
    jac_cols: jnp.ndarray

def get_model_requirements(func):
    """
    Inspects the function signature to find the expected size of 'vars'.
    """
    sig = inspect.signature(func)
    if 'vars' not in sig.parameters:
        raise ValueError(f"{func.__name__} missing 'vars' argument")
    
    default_val = sig.parameters['vars'].default
    if default_val is inspect.Parameter.empty:
        raise ValueError(f"{func.__name__} 'vars' must have a default (e.g. jnp.zeros(3))")
        
    return len(default_val)

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

def compile_netlist_vectorized(instances: List[Instance], num_nodes: int):
    """
    Compiles individual Instances into vectorized ComponentGroups.
    Groups instances by model_func to enable batch processing.
    """
    # 1. BUCKETING: Group instances by their model function
    # Key: model_func
    # Value: List of (instance, internal_start_index)
    groups = defaultdict(list)
    
    current_global_idx = num_nodes
    
    # --- Pass 1: Assign Internal Indices & Group ---
    for inst in instances:

    # 1. Normalize the function signature
        safe_func = ensure_time_signature(inst.model_func)

        total_size = get_model_requirements(safe_func)
        num_ports = len(inst.connections)
        num_internals = total_size - num_ports
        
        # Store instance with its calculated internal offset
        groups[safe_func].append((inst, current_global_idx))
        current_global_idx += num_internals

    # --- Pass 2: Stack Arrays for each Group ---
    compiled_groups = []
    
    for func, items in groups.items():
        # 'items' contains all instances sharing this physics function
        # We assume all instances of the same function have identical shapes.
        
        batch_var_indices = []
        batch_eq_indices = []
        batch_params = defaultdict(list)
        batch_jac_rows = []
        batch_jac_cols = []
        
        # Get shape info from the first item
        first_inst = items[0][0]
        total_size = get_model_requirements(func)
        
        for inst, internal_start in items:
            # 1. Resolve Indices
            node_indices = jnp.array(inst.connections)
            internal_indices = jnp.arange(internal_start, internal_start + (total_size - len(node_indices)))
            all_indices = jnp.concatenate([node_indices, internal_indices])
            
            # Store for stacking later
            batch_var_indices.append(all_indices)
            batch_eq_indices.append(all_indices) # Modify if MNA requires different eq indices
            
            # 2. Collect Parameters
            # Turn list of dicts -> dict of lists
            for key, val in inst.params.items():
                batch_params[key].append(val)
                
            # 3. Build Jacobian Coordinates (Meshgrid for this instance)
            # This creates the dense block structure for this specific instance
            r = jnp.broadcast_to(all_indices[:, None], (total_size, total_size))
            c = jnp.broadcast_to(all_indices[None, :], (total_size, total_size))
            
            batch_jac_rows.append(r.reshape(-1))
            batch_jac_cols.append(c.reshape(-1))
            
        # --- STACKING & CONCATENATION ---
        
        # Stack indices for vmap (Shape: N x vars)
        final_var_indices = jnp.stack(batch_var_indices)
        final_eq_indices = jnp.stack(batch_eq_indices)
        
        # Stack parameters into arrays (Shape: N)
        final_params = {k: jnp.array(v) for k, v in batch_params.items()}
        
        # Concatenate Jacobian indices (Shape: N*vars*vars)
        # We want one massive 1D array for the sparse matrix constructor
        final_jac_rows = jnp.concatenate(batch_jac_rows)
        final_jac_cols = jnp.concatenate(batch_jac_cols)
        
        group = ComponentGroup(
            name=func.__name__,
            physics_func=func,
            params=final_params,
            var_indices=final_var_indices,
            eq_indices=final_eq_indices,
            jac_rows=final_jac_rows,
            jac_cols=final_jac_cols
        )
        compiled_groups.append(group)

    return compiled_groups, current_global_idx