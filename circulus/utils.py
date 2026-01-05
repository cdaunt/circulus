from typing import TYPE_CHECKING
import equinox as eqx

if TYPE_CHECKING:
    from circulus.compiler import ComponentGroup

def update_param_dict(groups_dict: dict, 
                      group_name: str, 
                      instance_name: str, 
                      param_key:str, 
                      new_value: float) -> dict[str, 'ComponentGroup']:
    """Updates a parameter for a specific instance within a component group.
    """
    
    g = groups_dict[group_name]
    
    instance_idx = g.index_map[instance_name]
    
    # Handle Equinox Component (Batched)
    batched_comp = g.params
    current_val = getattr(batched_comp, param_key)
    new_vals = current_val.at[instance_idx].set(new_value)
    
    new_batched_comp = eqx.tree_at(lambda c: getattr(c, param_key), batched_comp, new_vals)
    new_g = g._replace(params=new_batched_comp)
    
    # Return new dict (JAX helper to copy-and-modify dicts)
    return {**groups_dict, group_name: new_g}