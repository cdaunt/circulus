

def update_param(groups, group_name, instance_idx, param_key, new_value):
    """
    Updates a parameter inside the groups list.
    Safe for JIT/VMAP because the control flow depends on static strings (g.name).
    """
    new_groups = []
    for g in groups:
        if g.name == group_name:
            updated_values = g.params[param_key].at[instance_idx].set(new_value)
            new_params = {**g.params, param_key: updated_values}
            new_groups.append(g._replace(params=new_params))
        else:
            new_groups.append(g)
        
    return new_groups