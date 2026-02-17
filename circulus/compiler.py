import jax
import jax.numpy as jnp
import inspect
from typing import NamedTuple, Callable, List, Dict, Any
from collections import defaultdict
import inspect
from functools import wraps, lru_cache
import equinox as eqx

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

class ComponentGroup(eqx.Module):
    """
    Represents a BATCH of identical components (e.g., ALL Resistors).
    Optimized for jax.vmap in the solver.
    """
    name: str = eqx.field(static=True)
    physics_func: Callable = eqx.field(static=True)
    
    # BATCHED PARAMETERS: 
    # This is the batched component instance (e.g. a Resistor where self.R is an array).
    # It acts as 'self' when passed to the physics function via vmap.
    params: Any 
    
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

    index_map: Dict[str, int] | None = eqx.field(static=True, default=None)



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



def solve_connectivity(connections: dict) -> dict:
    """
    Resolves Port-to-Port connections into a Port-to-NodeID map.
    Example: {"R1,p1": "V1,p1"} -> {"R1,p1": 1, "V1,p1": 1}
    """
    parent = {}

    def find(i):
        if i not in parent: parent[i] = i
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 1. Process all connections
    for src, targets in connections.items():
        # Ensure 'targets' is a list
        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        
        # Link Source to all Targets
        for tgt in targets:
            union(src, tgt)

    # 2. Assign Node IDs
    # We reserve ID 0 for Ground (any group containing "GND")
    groups = {}
    node_map = {}
    
    # Identify the root for "GND" if it exists
    gnd_roots = {find(k) for k in parent if "GND" in k}
    
    node_counter = 1
    
    for port in parent:
        root = find(port)
        
        if root in gnd_roots:
            node_id = 0
        else:
            if root not in groups:
                groups[root] = node_counter
                node_counter += 1
            node_id = groups[root]
            
        node_map[port] = node_id
        
    return node_map, node_counter


def compile_netlist(netlist: dict, models_map: dict):
    # --- 1. Resolve Connectivity (Using your existing function) ---
    # This returns {'R1,p1': 1, 'V1,p1': 1, 'GND,p1': 0 ...}
    port_to_node_map, num_nodes = build_net_map(netlist)
    
    # Buckets: Key = (comp_type_name, tree_structure), Value = list of instances
    # We include tree_structure to separate instances with different static fields (e.g. callables)
    buckets = defaultdict(list)
    sys_size = num_nodes 

    # --- 2. Process Instances ---
    instances = netlist.get("instances", {})
    
    for name, data in instances.items():
        comp_type = data['component']
        
        # Skip ground (it's just a marker, already handled in build_net_map)
        if comp_type == 'ground' or name == 'GND':
            continue
            
        if comp_type not in models_map:
            raise ValueError(f"Model '{comp_type}' not found for '{name}'")
            
        comp_cls = models_map[comp_type]
        settings = data.get('settings', {})
        
        # A. Create Equinox Object
        try:
            comp_obj = comp_cls(**settings)
        except TypeError as e:
            raise TypeError(f"Settings error for {name}: {e}")

        # B. Get Port Indices
        # We look up "InstanceName,PortName" in your map
        port_indices = []
        for port in comp_cls.ports:
            key = f"{name},{port}"
            
            if key in port_to_node_map:
                port_indices.append(port_to_node_map[key])
            else:
                # Error: The component has a port defined in Python (e.g. 'body')
                # but it wasn't listed in the netlist connections.
                raise ValueError(
                    f"Port '{port}' on '{name}' is unconnected.\n"
                    f"Your netlist connections must include '{key}'"
                )

        # Group by Type AND Structure (to handle static field differences)
        structure = jax.tree.structure(comp_obj)
        buckets[(comp_type, structure)].append({
            'obj': comp_obj,
            'ports': port_indices,
            'num_states': len(comp_cls.states),
            'name': name
        })

    # --- 3. Vectorize ---
    compiled_groups = {}
    
    # Helper to generate unique names for split groups
    type_counts = defaultdict(int)
    for (ctype, _) in buckets.keys():
        type_counts[ctype] += 1
    type_counters = defaultdict(int)
    
    for (comp_type, structure), items in buckets.items():
        comp_cls = models_map[comp_type]
        
        # Generate Group Name
        if type_counts[comp_type] > 1:
            idx = type_counters[comp_type]
            group_name = f"{comp_type}_{idx}"
            type_counters[comp_type] += 1
        else:
            group_name = comp_type
        
        # A. Assign Internal States
        all_var_indices = []
        for item in items:
            state_indices = []
            for _ in range(item['num_states']):
                state_indices.append(sys_size)
                sys_size += 1
            all_var_indices.append(item['ports'] + state_indices)

        # B. Batch Params
        instance_objects = [item['obj'] for item in items]
        batched_params = jax.tree.map(lambda *args: jnp.stack(args), *instance_objects)
        
        # C. Matrices
        var_indices_arr = jnp.array(all_var_indices, dtype=jnp.int32)
        width = var_indices_arr.shape[1]
        count = len(items)
        
        jac_rows = jnp.broadcast_to(var_indices_arr[:, :, None], (count, width, width))
        jac_cols = jnp.broadcast_to(var_indices_arr[:, None, :], (count, width, width))

        # Create Index Map for parameter updates
        index_map = {item['name']: i for i, item in enumerate(items)}

        compiled_groups[group_name] = ComponentGroup(
            name=group_name,
            var_indices=var_indices_arr,
            eq_indices=var_indices_arr,
            
            params=batched_params,  # This is the 'self' for the bridge
            
            # The bridge expects (instance, vars, t)
            # The solver usually does: func(params, vars, t)
            # We just need to ensure the solver argument order matches.
            physics_func=comp_cls.solver_call, 
            
            jac_rows=jac_rows,
            jac_cols=jac_cols,
            index_map=index_map
        )

    return compiled_groups, sys_size, port_to_node_map