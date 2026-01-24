import jax
import jax.numpy as jnp

def _assemble_system_real(y_guess, component_groups, t1, dt):
    """Assembles Jacobian/Residual for Real systems."""
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    vals_list = []

    # Deterministic order via sorted keys
    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v_locs = y_guess[group.var_indices]
        
        # Physics & Derivatives
        def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)
        (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
        (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)
        
        # Accumulate
        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        j_eff = df_l + (dq_l / dt)
        vals_list.append(j_eff.reshape(-1))

    return total_f, total_q, jnp.concatenate(vals_list)

def _assemble_residual_only_real(y_guess, component_groups, t1, dt):
    """Assembles ONLY f and q (no Jacobian)."""
    sys_size = y_guess.shape[0]

    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v = y_guess[group.var_indices]

        def physics_at_t1(v, p): return group.physics_func(v, p, t=t1)

        # Only primal evaluation - NO JACOBIAN
        f_l, q_l = jax.vmap(physics_at_t1)(v, group.params)
        
        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)

    return total_f, total_q

def _assemble_system_complex(y_guess, component_groups, t1, dt):
    """Assembles Jacobian/Residual for Unrolled Complex systems (Block Format)."""
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]
    
    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)
    
    # Block Accumulators: [RR, RI, IR, II]
    vals_blocks = [[], [], [], []] 

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        # 1. Split Physics (Real -> Complex -> Real)
        def physics_split(vr, vi, p):
            v = vr + 1j * vi
            f, q = group.physics_func(v, p, t=t1)
            return f.real, f.imag, q.real, q.imag

        # 2. Primal & Residuals
        fr, fi, qr, qi = jax.vmap(physics_split)(v_r, v_i, group.params)
        
        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

        # 3. Jacobian (4 blocks)
        jac_res = jax.vmap(jax.jacfwd(physics_split, argnums=(0,1)))(v_r, v_i, group.params)
        ((dfr_r, dfr_i), (dfi_r, dfi_i), (dqr_r, dqr_i), (dqi_r, dqi_i)) = jac_res
        
        # J_eff = df/dv + (1/dt)*dq/dv
        vals_blocks[0].append((dfr_r + dqr_r/dt).reshape(-1)) # RR
        vals_blocks[1].append((dfr_i + dqr_i/dt).reshape(-1)) # RI
        vals_blocks[2].append((dfi_r + dqi_r/dt).reshape(-1)) # IR
        vals_blocks[3].append((dfi_i + dqi_i/dt).reshape(-1)) # II

    # Concatenate blocks in RR, RI, IR, II order to match 'init' indices
    all_vals = jnp.concatenate([jnp.concatenate(b) for b in vals_blocks])
    return total_f, total_q, all_vals

def _assemble_residual_only_complex(y_guess, component_groups, t1, dt):
    """Assembles ONLY f and q (no Jacobian)."""
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]
    
    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        def physics_split(vr, vi, p):
            v = vr + 1j * vi
            f, q = group.physics_func(v, p, t=t1)
            return f.real, f.imag, q.real, q.imag

        # Only primal evaluation - NO JACOBIAN
        fr, fi, qr, qi = jax.vmap(physics_split)(v_r, v_i, group.params)
        
        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

    return total_f, total_q


