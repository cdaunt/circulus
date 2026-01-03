
# module_a.py
from typing import TYPE_CHECKING

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
    new_vals = g.params[param_key].at[instance_idx].set(new_value)
    new_params = {**g.params, param_key: new_vals}
    new_g = g._replace(params=new_params)
    
    # Return new dict (JAX helper to copy-and-modify dicts)
    return {**groups_dict, group_name: new_g}