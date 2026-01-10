import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import equinox as eqx
import diffrax
import optimistix as optx
from functools import partial
import warnings
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

# Try importing klujax for high-performance circuit solving
try:
    import klujax
    _KLU_AVAILABLE = True
except ImportError:
    _KLU_AVAILABLE = False
    klujax = None

# ==============================================================================
# 1. ASSEMBLY KERNELS (PHYSICS)
# ==============================================================================

def _coalesce_coo_host(rows, cols, vals, sys_size):
    """
    Host-side callback to coalesce COO matrix (sum duplicates) using scipy.
    This runs on CPU and is not JIT-compiled.
    
    Args:
        rows: Row indices (numpy array or jax array)
        cols: Column indices (numpy array or jax array)
        vals: Values (numpy array or jax array)
        sys_size: Matrix dimension (scalar)
    
    Returns:
        (rows_coalesced, cols_coalesced, vals_coalesced) as numpy arrays
    """
    rows_np = np.array(rows, copy=True)
    cols_np = np.array(cols, copy=True)
    vals_np = np.array(vals, copy=True)
    
    # Create scipy COO matrix - this automatically handles duplicates
    mat = scipy.sparse.coo_matrix((vals_np, (rows_np, cols_np)), shape=(sys_size, sys_size))
    
    # Explicitly sum duplicates
    mat.sum_duplicates()
    
    # Return coalesced arrays
    return mat.row.astype(rows_np.dtype), mat.col.astype(cols_np.dtype), mat.data.astype(vals_np.dtype)


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

def _compute_history(component_groups, y_c, t, num_vars):
    """Computes total charge Q at time t (Initial Condition)."""
    is_complex = jnp.iscomplexobj(y_c)
    total_q = jnp.zeros(2 * num_vars if is_complex else num_vars, dtype=jnp.float64 if is_complex else y_c.dtype)
        
    for group in component_groups.values():
        v_locs = y_c[group.var_indices]
        _, q_l = jax.vmap(lambda v, p: group.physics_func(v, p, t=t))(v_locs, group.params)
        
        if is_complex:
             total_q = total_q.at[group.eq_indices].add(q_l.real)
             total_q = total_q.at[group.eq_indices + num_vars].add(q_l.imag)
        else:
             total_q = total_q.at[group.eq_indices].add(q_l)
    return total_q

# ==============================================================================
# 2. HOST CALLBACK FOR SCIPY (CPU Direct Solve)
# ==============================================================================

def _scipy_host_solve(data, rows, cols, rhs):
    """
    Host-side callback that uses Scipy's SuperLU solver.
    This runs on CPU and automatically sums duplicate indices (COO -> CSC).
    """
    # JAX passes read-only arrays to callbacks. 
    # Scipy needs mutable arrays for in-place sorting/summing of duplicates.
    # We must explicitly copy them to mutable NumPy arrays.
    data = np.array(data, copy=True)
    rows = np.array(rows, copy=True)
    cols = np.array(cols, copy=True)
    rhs = np.array(rhs, copy=True)

    sys_size = rhs.shape[0]
    # Scipy COO matrix construction sums duplicates automatically
    mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(sys_size, sys_size))
    
    # Explicitly convert to CSR to sum duplicates and avoid SparseEfficiencyWarning
    mat = mat.tocsr()
    
    # spsolve invokes SuperLU (or UMFPACK)
    solution = scipy.sparse.linalg.spsolve(mat, rhs)
    return solution.astype(rhs.dtype)

# ==============================================================================
# 3. UNIFIED SOLVER KERNEL
# ==============================================================================

import jax.numpy as jnp

import numpy as np
import jax.numpy as jnp

def prepare_klu_settings(component_groups, num_vars, y0_sample):
    """
    Analyzes circuit topology on CPU (Numpy) to create KLU mapping.
    Run this ONCE before diffeqsolve.
    """
    # 1. Extract all row/col indices from groups
    all_rows, all_cols = [], []
    for k in sorted(component_groups.keys()):
        g = component_groups[k]
        all_rows.append(np.array(g.jac_rows).reshape(-1))
        all_cols.append(np.array(g.jac_cols).reshape(-1))
        
    static_rows = np.concatenate(all_rows)
    static_cols = np.concatenate(all_cols)

    # 2. Detect Complexity & Adjust Indices
    # Note: We assume structure matches y0_sample
    is_complex = np.iscomplexobj(y0_sample) or (y0_sample.shape[0] == 2 * num_vars)
    if not is_complex:
        # Check if parameters force complexity
        # (Simplified check: if y0 is real and params are real, assume real)
        pass 

    sys_size = num_vars
    if is_complex:
        sys_size = num_vars * 2
        r, c = static_rows, static_cols
        N = num_vars
        # Expand for Block Complex [RR, RI, IR, II]
        static_rows = np.concatenate([r, r, r+N, r+N])
        static_cols = np.concatenate([c, c+N, c, c+N])

    # 3. Add Ground Constraints
    if is_complex:
        ground_indices = np.array([0, num_vars], dtype=np.int32)
    else:
        ground_indices = np.array([0], dtype=np.int32)
        
    full_rows = np.concatenate([static_rows, ground_indices])
    full_cols = np.concatenate([static_cols, ground_indices])

    # 4. Compute Unique Mapping (The part that failed in JIT)
    # hash = row * width + col
    rc_hashes = full_rows.astype(np.int64) * sys_size + full_cols.astype(np.int64)
    unique_hashes, map_indices = np.unique(rc_hashes, return_inverse=True)
    
    unique_rows = (unique_hashes // sys_size).astype(np.int32)
    unique_cols = (unique_hashes % sys_size).astype(np.int32)
    n_unique = int(len(unique_hashes))

    # Return as JAX arrays (ready to be passed to JIT)
    return (
        jnp.array(unique_rows),
        jnp.array(unique_cols),
        jnp.array(map_indices),
        n_unique
    )

# Add 'klu_n_unique' to static_argnames
@partial(jax.jit, static_argnames=['is_complex', 'mode', 'klu_n_unique'])
def _unified_newton_step(y_guess, args, is_complex=False, mode='dense', klu_n_unique=None):
    # Unpack args as before
    (groups, t1, dt, q_prev, static_rows, static_cols, diag_mask, num_vars, klu_settings) = args
    sys_size = y_guess.shape[0]

    # ... (Assembly code remains exactly the same) ...
    if is_complex:
        total_f, total_q, all_vals = _assemble_system_complex(y_guess, groups, t1, dt)
        q_prev_flat = q_prev 
        ground_indices = [0, sys_size // 2]
    else:
        total_f, total_q, all_vals = _assemble_system_real(y_guess, groups, t1, dt)
        q_prev_flat = q_prev
        ground_indices = [0]

    residual = total_f + (total_q - q_prev_flat) / dt
    for idx in ground_indices:
        residual = residual.at[idx].add(1e9 * y_guess[idx])

    # --- Solve Strategy ---
    if mode == 'klu':
        # 1. Unpack mapping
        # CRITICAL: We ignore the 'n_unique' inside the tuple (it is Traced/Dynamic).
        # We use 'klu_n_unique' passed via kwargs (it is Static).
        (u_rows, u_cols, map_idx, _) = klu_settings 

        # 2. Build Raw Values
        g_vals = jnp.full(len(ground_indices), 1e9, dtype=all_vals.dtype)
        raw_vals = jnp.concatenate([all_vals, g_vals])

        # 3. Differentiable Coalescing
        # Use the STATIC 'klu_n_unique' here
        coalesced_vals = jax.ops.segment_sum(raw_vals, map_idx, num_segments=klu_n_unique)

        # 4. Solve
        delta = klujax.solve(u_rows, u_cols, coalesced_vals, -residual)
    
    # ... (Other modes 'dense', 'scipy', 'sparse' remain the same) ...
    elif mode == 'dense':
        J = jnp.zeros((sys_size, sys_size), dtype=residual.dtype)
        J = J.at[static_rows, static_cols].add(all_vals)
        for idx in ground_indices:
            J = J.at[idx, idx].add(1e9)
        delta = jnp.linalg.solve(J, -residual)
    # ...
    # (Copy the rest of your existing function here)
    
    # Damping
    max_change = jnp.max(jnp.abs(delta))
    voltage_limit = 0.5
    damping = jnp.minimum(1.0, voltage_limit / (max_change + 1e-9))
    
    return y_guess + (delta * damping)
# ==============================================================================
# 4. SOLVER CLASS
# ==============================================================================

class TransientSolverState(eqx.Module):
    static_rows: jax.Array
    static_cols: jax.Array
    diag_mask: jax.Array | None
    history: tuple
    is_complex_mode: bool = eqx.field(static=True)
    
    # --- New KLU Specific Fields ---
    # We store the pre-calculated mapping here.
    # arrays are dynamic tracers, the count is static.
    klu_rows: jax.Array | None = None
    klu_cols: jax.Array | None = None
    klu_map: jax.Array | None = None
    klu_n_unique: int | None = eqx.field(default=None, static=True)
    
    

class VectorizedTransientSolver(diffrax.AbstractSolver):
    mode: str = eqx.field(static=True) # 'dense', 'sparse', 'scipy', 'klu'
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms): return 1

    def init(self, terms, t0, t1, y0, args):
        # Unpack args: Expect klu_settings at the end
        # We handle both 2-element and 3-element cases for backward compatibility
        if len(args) == 3:
            component_groups, num_vars, klu_settings_input = args
        else:
            component_groups, num_vars = args
            klu_settings_input = None
        
        if self.mode == 'klu' and not _KLU_AVAILABLE:
            raise ImportError("Mode 'klu' not available.")
        
        if self.mode == 'klu' and klu_settings_input is None:
             raise ValueError
        
        if self.mode == 'klu' and not _KLU_AVAILABLE:
            raise ImportError("Mode 'klu' selected but `klujax` not installed. Please install it or use 'scipy' or 'sparse'.")

        # --- 1. Index Pre-calculation ---
        all_rows, all_cols = [], []
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            all_rows.append(g.jac_rows.reshape(-1))
            all_cols.append(g.jac_cols.reshape(-1))
            
        base_rows = jnp.concatenate(all_rows)
        base_cols = jnp.concatenate(all_cols)

        # --- 2. Detect Complexity ---
        is_complex = jnp.iscomplexobj(y0) or (y0.shape[0] == 2 * num_vars)
        if not is_complex: 
            for g in component_groups.values():
                if any(jnp.iscomplexobj(p) for p in jax.tree.leaves(g.params)):
                    is_complex = True; break

        # --- 3. Build Indices ---
        if is_complex:
            r, c = base_rows, base_cols
            N = num_vars
            static_rows = jnp.concatenate([r, r, r+N, r+N])
            static_cols = jnp.concatenate([c, c+N, c, c+N])
        else:
            static_rows, static_cols = base_rows, base_cols

        # --- 5. History Setup ---
        if jnp.iscomplexobj(y0):
            y0_flat = jnp.concatenate([y0.real, y0.imag])
        elif is_complex and y0.shape[0] == num_vars:
            y0_flat = jnp.concatenate([y0, jnp.zeros_like(y0)])
        else:
            y0_flat = y0
            
        # --- 4. Assign KLU Settings ---
        klu_rows, klu_cols, klu_map, klu_n = None, None, None, None
        
        if self.mode == 'klu':
            # JUST UNPACK. No calculation here.
            (klu_rows, klu_cols, klu_map, klu_n) = klu_settings_input

        # ... (History Setup remains the same) ...
        
        return TransientSolverState(
            static_rows=static_rows, 
            static_cols=static_cols, 
            diag_mask=(static_rows == static_cols), 
            history=(y0_flat, 1.0), 
            is_complex_mode=is_complex, # <--- Remember to keep this before KLU fields
            klu_rows=klu_rows,
            klu_cols=klu_cols,
            klu_map=klu_map,
            klu_n_unique=klu_n
        )
    
    def step(self, terms, t0, t1, y0, args, solver_state, options):
        component_groups, num_vars, klu_settings = args
        dt = t1 - t0
        y_prev_step, dt_prev = solver_state.history
        is_complex = solver_state.is_complex_mode

        # 1. Normalize State
        y0_flat = y0
        if jnp.iscomplexobj(y0): 
            y0_flat = jnp.concatenate([y0.real, y0.imag])

        # 2. Physics Prep
        y_c = y0_flat
        if is_complex:
            y_c = y0_flat[:num_vars] + 1j * y0_flat[num_vars:]

        # 3. Predictor & History
        rate = (y0_flat - y_prev_step) / (dt_prev + 1e-30)
        y_pred = y0_flat + rate * dt
        
        q_prev = _compute_history(component_groups, y_c, t0, num_vars)

        # 4. Pack KLU Settings from State
        # If mode is not 'klu', these will be None, which is fine (unused).
        klu_settings = (
            solver_state.klu_rows,
            solver_state.klu_cols,
            solver_state.klu_map,
            solver_state.klu_n_unique 
        )

        # 5. Newton Solve
        step_args = (component_groups, t1, dt, q_prev, solver_state.static_rows, 
                     solver_state.static_cols, solver_state.diag_mask, num_vars, klu_settings)
        
        # FIX: Pass 'klu_n_unique' here as a static kwarg
        solver_fn = partial(
            _unified_newton_step, 
            is_complex=is_complex, 
            mode=self.mode,
            klu_n_unique=solver_state.klu_n_unique # <--- Extracted from static field
        )

        # ... (Rest of step function remains the same) ...
        solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
        sol = optx.fixed_point(solver_fn, solver, y_pred, args=step_args, max_steps=200, throw=False)

        # 6. Result
        y_next = sol.value
        y_error = y_next - y_pred # Truncation Error

        # Update history with START of step state
        new_state = eqx.tree_at(lambda s: s.history, solver_state, (y0_flat, dt))
        
        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None
        )

        return y_next, y_error, {"y0": y0_flat, "y1": y_next}, new_state, result
    
    def func(self, terms, t0, y0, args): return terms.vf(t0, y0, args)