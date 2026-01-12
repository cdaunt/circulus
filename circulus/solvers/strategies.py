import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optimistix as optx
from abc import abstractmethod
import lineax as lx
import klujax

# ==============================================================================
# 1. ABSTRACT BASE & DC MIXIN
# ==============================================================================

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

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import lineax as lx
import optimistix as optx
from abc import abstractmethod

# ==============================================================================
# 1. ABSTRACT BASE CLASS
# ==============================================================================

class CircuitLinearSolver(lx.AbstractLinearSolver):
    """
    Base class for circuit linear solvers.
    """
    ground_indices: jax.Array

    def init(self, operator, options):
        return None

    def compute(self, state, vector, options):
        # Dispatch to specific implementation
        # operator = Jacobian values (1D array 'all_vals')
        # vector = Residual (1D array 'rhs')
        return self._solve_impl(state.operator, vector)

    def transpose(self, state, options):
        raise NotImplementedError
    
    def conj(self, state, options):
        # We generally deal with real or complex symmetric-ish systems manually.
        # Returning self is usually sufficient for custom solvers that don't support conjugate transpose.
        return self

    def allow_dependent_columns(self, operator):
        # Circuit matrices (MNA) should be full rank if grounded correctly.
        return False

    def allow_dependent_rows(self, operator):
        return False

    def _solve_impl(self, all_vals, residual):
        raise NotImplementedError

    def solve_dc(self, component_groups, num_vars, y_guess):
            """
            Helper: Performs a DC Operating Point analysis (dt=infinity).
            """

            def dc_step(y, _):
                # 1. Assemble (dt=infinity)
                total_f, _, all_vals = _assemble_system_real(y, component_groups, t1=0.0, dt=1e18)
                
                # 2. Apply ground constraints
                for idx in self.ground_indices:
                    total_f = total_f.at[idx].add(1e9 * y[idx])
                
                # 3. Solve Linear System
                # FIX: Call internal implementation directly
                sol = self._solve_impl(all_vals, -total_f)
                
                return y + sol.value

            # 4. Run Newton Loop
            solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
            sol = optx.fixed_point(dc_step, solver, y_guess, max_steps=50, throw=False)
            return sol.value

# ==============================================================================
# 2. DENSE SOLVER (O(N^3))
# ==============================================================================

class DenseSolver(CircuitLinearSolver):
    static_rows: jax.Array
    static_cols: jax.Array
    sys_size: int = eqx.field(static=True)

    def _solve_impl(self, all_vals, residual):
        # 1. Build Dense Matrix
        J = jnp.zeros((self.sys_size, self.sys_size), dtype=residual.dtype)
        J = J.at[self.static_rows, self.static_cols].add(all_vals)
        
        # 2. Apply Ground Constraints
        for idx in self.ground_indices:
            J = J.at[idx, idx].add(1e9)
            
        # 3. Solve
        x = jnp.linalg.solve(J, residual)
        
        # FIX: Added 'stats={}' which is required by Lineax
        return lx.Solution(value=x, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_circuit(cls, component_groups, num_vars, is_complex=False):
        """Factory: Builds indices on CPU."""
        all_rows, all_cols = [], []
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            all_rows.append(np.array(g.jac_rows).reshape(-1))
            all_cols.append(np.array(g.jac_cols).reshape(-1))
            
        static_rows = np.concatenate(all_rows)
        static_cols = np.concatenate(all_cols)
        
        sys_size = num_vars
        ground_idxs = np.array([0], dtype=np.int32)

        if is_complex:
            sys_size = num_vars * 2
            r, c = static_rows, static_cols
            N = num_vars
            static_rows = np.concatenate([r, r, r+N, r+N])
            static_cols = np.concatenate([c, c+N, c, c+N])
            ground_idxs = np.array([0, num_vars], dtype=np.int32)

        return cls(
            static_rows=jnp.array(static_rows),
            static_cols=jnp.array(static_cols),
            sys_size=sys_size,
            ground_indices=jnp.array(ground_idxs)
        )
    
class KLUSolver(CircuitLinearSolver):
    u_rows: jax.Array
    u_cols: jax.Array
    map_idx: jax.Array
    n_unique: int = eqx.field(static=True)

    def _solve_impl(self, all_vals, residual):
        # 1. Build Raw Values
        g_vals = jnp.full(self.ground_indices.shape[0], 1e9, dtype=all_vals.dtype)
        raw_vals = jnp.concatenate([all_vals, g_vals])

        # 2. Coalesce
        coalesced_vals = jax.ops.segment_sum(
            raw_vals, self.map_idx, num_segments=self.n_unique
        )

        # 3. Solve via KLU
        solution = klujax.solve(self.u_rows, self.u_cols, coalesced_vals, residual)
        
        # FIX: Added 'stats={}'
        return lx.Solution(value=solution, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_circuit(cls, component_groups, num_vars, is_complex=False):
        #if not _KLU_AVAILABLE: raise ImportError("klujax not installed.")
        
        all_rows, all_cols = [], []
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            all_rows.append(np.array(g.jac_rows).reshape(-1))
            all_cols.append(np.array(g.jac_cols).reshape(-1))
        
        static_rows = np.concatenate(all_rows)
        static_cols = np.concatenate(all_cols)

        sys_size = num_vars
        ground_idxs = np.array([0], dtype=np.int32)

        if is_complex:
            sys_size = num_vars * 2
            r, c = static_rows, static_cols
            N = num_vars
            static_rows = np.concatenate([r, r, r+N, r+N])
            static_cols = np.concatenate([c, c+N, c, c+N])
            ground_idxs = np.array([0, num_vars], dtype=np.int32)

        full_rows = np.concatenate([static_rows, ground_idxs])
        full_cols = np.concatenate([static_cols, ground_idxs])
        
        rc_hashes = full_rows.astype(np.int64) * sys_size + full_cols.astype(np.int64)
        unique_hashes, map_indices = np.unique(rc_hashes, return_inverse=True)
        
        u_rows = (unique_hashes // sys_size).astype(np.int32)
        u_cols = (unique_hashes % sys_size).astype(np.int32)
        n_unique = int(len(unique_hashes))

        return cls(
            u_rows=jnp.array(u_rows),
            u_cols=jnp.array(u_cols),
            map_idx=jnp.array(map_indices),
            n_unique=n_unique,
            ground_indices=jnp.array(ground_idxs)
        )

class SparseSolver(CircuitLinearSolver):
    """
    Iterative solver using BiCGSTAB with Jacobi (Diagonal) Preconditioning.
    Memory efficient O(N), but convergence depends on matrix conditioning.
    """
    static_rows: jax.Array
    static_cols: jax.Array
    diag_mask: jax.Array  # Boolean mask where rows == cols
    sys_size: int = eqx.field(static=True)
    g_leak: float = 1e-12 # Small leakage for stability

    def _solve_impl(self, all_vals, residual):
        # 1. Build Preconditioner (Diagonal Approximation)
        #    Sum all values that land on the diagonal
        diag_vals = jax.ops.segment_sum(
            all_vals * self.diag_mask, self.static_rows, num_segments=self.sys_size
        )
        
        #    Add leakage and Ground Stiffness to diagonal
        diag_vals = diag_vals + self.g_leak
        for idx in self.ground_indices:
            diag_vals = diag_vals.at[idx].add(1e9)
            
        #    Invert diagonal (Safe division)
        inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-12, 1.0, 1.0 / diag_vals)
        
        # 2. Define Linear Operator A * x
        def matvec(x):
            # Gather x values to match sparse format
            x_gathered = x[self.static_cols]
            products = all_vals * x_gathered
            
            # Sum products back into rows (Ax)
            Ax = jax.ops.segment_sum(products, self.static_rows, num_segments=self.sys_size)
            
            # Add Ground Constraints and Leakage to result
            Ax = Ax + (x * self.g_leak)
            for idx in self.ground_indices:
                Ax = Ax.at[idx].add(1e9 * x[idx])
                
            return Ax

        # 3. Solve using JAX BiCGSTAB
        #    Initial guess using the preconditioner
        delta_guess = residual * inv_diag
        
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            matvec, 
            residual, # Note: Lineax/Optx usually passes -residual, check sign in caller!
                      # If caller passes -residual (RHS), then x solves Ax = -R. Correct.
            x0=delta_guess,
            M=lambda v: inv_diag * v, # Jacobi Preconditioner
            tol=1e-5, 
            maxiter=200
        )
        
        # Return as Lineax Solution
        return lx.Solution(value=x, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_circuit(cls, component_groups, num_vars, is_complex=False):
        """Factory: Builds indices and diagonal mask on CPU."""
        # 1. Extract Indices
        all_rows, all_cols = [], []
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            all_rows.append(np.array(g.jac_rows).reshape(-1))
            all_cols.append(np.array(g.jac_cols).reshape(-1))
            
        static_rows = np.concatenate(all_rows)
        static_cols = np.concatenate(all_cols)
        
        # 2. Handle Complexity
        sys_size = num_vars
        ground_idxs = np.array([0], dtype=np.int32)

        if is_complex:
            sys_size = num_vars * 2
            r, c = static_rows, static_cols
            N = num_vars
            static_rows = np.concatenate([r, r, r+N, r+N])
            static_cols = np.concatenate([c, c+N, c, c+N])
            ground_idxs = np.array([0, num_vars], dtype=np.int32)

        # 3. Create Diagonal Mask (Where row == col)
        #    This is static, so we calculate it once here.
        diag_mask = (static_rows == static_cols)

        return cls(
            static_rows=jnp.array(static_rows),
            static_cols=jnp.array(static_cols),
            diag_mask=jnp.array(diag_mask),
            sys_size=sys_size,
            ground_indices=jnp.array(ground_idxs)
        )

class GMRESSolver(CircuitLinearSolver):
    """
    Wrapper around jax.scipy.sparse.linalg.gmres.
    """
    static_rows: jax.Array
    static_cols: jax.Array
    diag_mask: jax.Array
    sys_size: int = eqx.field(static=True)
    g_leak: float = 1e-12
    # Increased restart to 200 to capture more history for stiff DC solves
    restart: int = eqx.field(static=True, default=200) 

    def _solve_impl(self, all_vals, residual):
        # 1. Build Indefinite Preconditioner
        diag_vals = jax.ops.segment_sum(
            all_vals * self.diag_mask, self.static_rows, num_segments=self.sys_size
        )
        diag_vals = diag_vals + self.g_leak
        for idx in self.ground_indices:
            diag_vals = diag_vals.at[idx].add(1e9)
            
        inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-12, 1.0, 1.0 / diag_vals)
        
        # 2. Operator
        def matvec(x):
            x_gathered = x[self.static_cols]
            products = all_vals * x_gathered
            Ax = jax.ops.segment_sum(products, self.static_rows, num_segments=self.sys_size)
            Ax = Ax + (x * self.g_leak)
            for idx in self.ground_indices:
                Ax = Ax.at[idx].add(1e9 * x[idx])
            return Ax

        # 3. JAX GMRES
        delta_guess = residual * inv_diag
        
        x, info = jax.scipy.sparse.linalg.gmres(
            matvec, 
            residual, 
            x0=delta_guess,
            M=lambda v: inv_diag * v,
            tol=1e-8,       # Tighter tolerance for DC
            restart=self.restart,
            maxiter=1000    # Allow many iterations for stiff DC
        )
        
        # 4. Result Mapping (FIXED)
        #    Map JAX error (info > 0) to Lineax 'singular' failure
        result = jax.lax.cond(
            info == 0,
            lambda: lx.RESULTS.successful,
            lambda: lx.RESULTS.singular  # <--- FIXED: Valid Lineax code
        )
        
        return lx.Solution(value=x, result=result, state=None, stats={})

    @classmethod
    def from_circuit(cls, component_groups, num_vars, is_complex=False, restart=200):
        # ... (Indices extraction logic same as before) ...
        all_rows, all_cols = [], []
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            all_rows.append(np.array(g.jac_rows).reshape(-1))
            all_cols.append(np.array(g.jac_cols).reshape(-1))
            
        static_rows = np.concatenate(all_rows)
        static_cols = np.concatenate(all_cols)
        
        sys_size = num_vars
        ground_idxs = np.array([0], dtype=np.int32)

        if is_complex:
            sys_size = num_vars * 2
            r, c = static_rows, static_cols
            N = num_vars
            static_rows = np.concatenate([r, r, r+N, r+N])
            static_cols = np.concatenate([c, c+N, c, c+N])
            ground_idxs = np.array([0, num_vars], dtype=np.int32)

        diag_mask = (static_rows == static_cols)

        return cls(
            static_rows=jnp.array(static_rows),
            static_cols=jnp.array(static_cols),
            diag_mask=jnp.array(diag_mask),
            sys_size=sys_size,
            ground_indices=jnp.array(ground_idxs),
            restart=restart
        )