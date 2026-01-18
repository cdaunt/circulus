"""
Circuit Linear Solvers Strategy Pattern
=======================================

This module defines the linear algebra strategies used by the circuit simulator.
It leverages the `lineax` abstract base class to provide interchangeable solvers
that work seamlessly with JAX transformations (JIT, VMAP, GRAD).

Architecture
------------
The core idea is to separate the *physics assembly* (calculating Jacobian values)
from the *linear solve* (inverting the Jacobian).

Classes:
    CircuitLinearSolver: Abstract base defining the interface and common DC logic.
    DenseSolver:        Uses JAX's native dense solver (LU decomposition). Best for small circuits (N < 2000) & GPU.
    KLUSolver:          Uses the KLU sparse solver (via `klujax`). Best for large circuits on CPU.
    SparseSolver:       Uses JAX's iterative BiCGStab. Best for large transient simulations on GPU.

"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optimistix as optx
import lineax as lx
import klujax
from typing import Any, Dict, Optional, Tuple, Union

# Import physics assembly kernels (lazy import handled in methods if needed)
from circulus.solvers.assembly import _assemble_system_complex, _assemble_system_real

# ==============================================================================
# 1. ABSTRACT BASE CLASS
# ==============================================================================

class CircuitLinearSolver(lx.AbstractLinearSolver):
    """
    Abstract Base Class for all circuit linear solvers.

    This class provides the unified interface for:
    1.  Storing static matrix structure (indices, rows, cols).
    2.  Handling Real vs. Complex-Unrolled system configurations.
    3.  Providing a robust Newton-Raphson DC Operating Point solver.

    Attributes:
        ground_indices (jax.Array): Indices of nodes connected to ground (forced to 0V).
        is_complex (bool): Static flag. If True, the system is 2N x 2N (Real/Imag unrolled).
                           If False, the system is N x N (Real).
    """
    ground_indices: jax.Array
    
    # Store configuration so we don't need to pass it around
    is_complex: bool = eqx.field(static=True)
    
    # --- Lineax Interface (Required Implementation) ---
    def init(self, operator: Any, options: Any) -> Any: 
        """Initialize the solver state (No-op for stateless solvers)."""
        return None

    def compute(self, state: Any, vector: jax.Array, options: Any) -> lx.Solution: 
        """
        Main entry point for Lineax. 
        In our case, we usually call `_solve_impl` directly to avoid overhead,
        but this satisfies the API.
        """
        raise NotImplementedError("Directly call _solve_impl for internal use.")

    def transpose(self, state: Any, options: Any) -> Any: 
        raise NotImplementedError

    def conj(self, state: Any, options: Any) -> 'CircuitLinearSolver': 
        return self

    def allow_dependent_columns(self, operator: Any) -> bool: 
        return False

    def allow_dependent_rows(self, operator: Any) -> bool: 
        return False

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        """
        Internal implementation of the linear solve: Ax = b.

        Args:
            all_vals (jax.Array): Flattened array of non-zero Jacobian values.
            residual (jax.Array): The Right-Hand Side (RHS) vector 'b'.

        Returns:
            lineax.Solution: Wrapper containing the solution vector 'x'.
        """
        raise NotImplementedError

    def solve_dc(self, component_groups: Dict[str, Any], y_guess: jax.Array) -> jax.Array:
        """
        Performs a robust DC Operating Point analysis (Newton-Raphson).

        This method:
        1.  Detects if the system is Real or Complex based on `self.is_complex`.
        2.  Assembles the system with dt=infinity (to open capacitors).
        3.  Applies ground constraints (setting specific rows/cols to identity).
        4.  Solves the linear system J * delta = -Residual.
        5.  Applies voltage damping to prevent exponential overshoot.

        Args:
            component_groups (dict): The circuit components and their parameters.
            y_guess (jax.Array): Initial guess vector (Shape: [N] or [2N]).

        Returns:
            jax.Array: The converged solution vector (Flat).
        """

        def dc_step(y: jax.Array, _: Any) -> jax.Array:
            # 1. Assemble System (dt=1e18 effectively removes time-dependent terms like C*dv/dt)
            if self.is_complex:
                total_f, _, all_vals = _assemble_system_complex(y, component_groups, t1=0.0, dt=1e18)
            else:
                total_f, _, all_vals = _assemble_system_real(y, component_groups, t1=0.0, dt=1e18)
            
            # 2. Apply Ground Constraints to Residual
            #    We add a massive penalty (1e9 * V) to the residual at ground nodes.
            #    This forces the solver to drive V -> 0.
            total_f_grounded = total_f
            for idx in self.ground_indices:
                total_f_grounded = total_f_grounded.at[idx].add(1e9 * y[idx])
            
            # 3. Solve Linear System (J * delta = -R)
            sol = self._solve_impl(all_vals, -total_f_grounded)
            delta = sol.value

            # 4. Apply Voltage Limiting (Damping)
            #    Prevents the solver from taking huge steps that crash exponentials (diodes/transistors).
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, 0.5 / (max_change + 1e-9))
            
            return y + delta * damping

        # 5. Run Newton Loop (Optimistix)
        solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
        sol = optx.fixed_point(dc_step, solver, y_guess, max_steps=100, throw=False)
        return sol.value

# ==============================================================================
# 2. DENSE SOLVER (JAX Native LU)
# ==============================================================================

class DenseSolver(CircuitLinearSolver):
    """
    Solves the system using dense matrix factorization (LU).
    
    Best For:
        - Small to Medium circuits (N < 2000).
        - Wavelength sweeps (AC Analysis) on GPU.
        - Systems where VMAP parallelism is critical.
    
    Attributes:
        static_rows (jax.Array): Row indices for placing values into dense matrix.
        static_cols (jax.Array): Column indices.
        g_leak (float): Leakage conductance added to diagonal to prevent singularity.
    """
    static_rows: jax.Array
    static_cols: jax.Array
    sys_size: int = eqx.field(static=True) # Matrix Dimension (N or 2N)
    
    # Defined here to satisfy dataclass rules (must be after base fields)
    g_leak: float = 1e-9

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        # 1. Build Dense Matrix from Sparse Values
        J = jnp.zeros((self.sys_size, self.sys_size), dtype=residual.dtype)
        J = J.at[self.static_rows, self.static_cols].add(all_vals)
        
        # 2. Add Leakage (Regularization)
        #    Ensures matrix is invertible even if transistors turn off (floating nodes).
        diag_idx = jnp.arange(self.sys_size)
        J = J.at[diag_idx, diag_idx].add(self.g_leak)
        
        # 3. Apply Ground Constraints (Stiff Diagonal)
        for idx in self.ground_indices:
            J = J.at[idx, idx].add(1e9)
            
        # 4. Dense Solve (LU)
        x = jnp.linalg.solve(J, residual)
        return lx.Solution(value=x, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_circuit(cls, component_groups: Dict[str, Any], num_vars: int, is_complex: bool = False) -> 'DenseSolver':
        """Factory method to pre-calculate indices for the dense matrix."""
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
            # Expand to 2N x 2N Block Structure:
            # [ RR  RI ]
            # [ IR  II ]
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
            ground_indices=jnp.array(ground_idxs),
            is_complex=is_complex 
        )

# ==============================================================================
# 3. KLU SOLVER (CPU Sparse)
# ==============================================================================

class KLUSolver(CircuitLinearSolver):
    """
    Solves the system using the KLU sparse solver (via `klujax`).
    
    Best For:
        - Large circuits (N > 5000) running on CPU.
        - DC Operating Points of massive meshes.
        - Cases where DenseSolver runs out of memory (OOM).
    
    Note:
        Does NOT support `vmap` (batching) automatically.
    """
    u_rows: jax.Array
    u_cols: jax.Array
    map_idx: jax.Array
    n_unique: int = eqx.field(static=True)
    sys_size: int = eqx.field(static=True)
    
    g_leak: float = 1e-9

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        # 1. Prepare raw value vector including Ground and Leakage entries
        g_vals = jnp.full(self.ground_indices.shape[0], 1e9, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype) 
        
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])

        # 2. Coalesce duplicate entries (COO -> Unique COO)
        coalesced_vals = jax.ops.segment_sum(
            raw_vals, self.map_idx, num_segments=self.n_unique
        )

        # 3. Call KLU Wrapper
        solution = klujax.solve(self.u_rows, self.u_cols, coalesced_vals, residual)
        return lx.Solution(value=solution, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_circuit(cls, component_groups: Dict[str, Any], num_vars: int, is_complex: bool = False) -> 'KLUSolver':
        """Factory method to pre-hash indices for sparse coalescence."""
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

        # We must include indices for the full leakage diagonal
        leak_rows = np.arange(sys_size, dtype=np.int32)
        leak_cols = np.arange(sys_size, dtype=np.int32)

        # Combine Circuit + Ground + Leakage indices
        full_rows = np.concatenate([static_rows, ground_idxs, leak_rows])
        full_cols = np.concatenate([static_cols, ground_idxs, leak_cols])
        
        # Hashing to find unique entries for coalescence
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
            ground_indices=jnp.array(ground_idxs),
            sys_size=sys_size,
            is_complex=is_complex
        )

# ==============================================================================
# 4. SPARSE SOLVER (JAX BiCGStab Wrapper)
# ==============================================================================

class SparseSolver(CircuitLinearSolver):
    """
    Solves the system using JAX's Iterative BiCGStab solver.
    
    Best For:
        - Large Transient Simulations on GPU (uses previous step as warm start).
        - Systems where N is too large for Dense, but we need VMAP support.
    
    Attributes:
        diag_mask (jax.Array): Mask to extract diagonal elements for preconditioning.
    """
    static_rows: jax.Array
    static_cols: jax.Array
    diag_mask: jax.Array
    sys_size: int = eqx.field(static=True)
    
    g_leak: float = 1e-9

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        # 1. Build Preconditioner (Diagonal Approximation)
        #    Extract diagonal elements from the sparse entries
        diag_vals = jax.ops.segment_sum(
            all_vals * self.diag_mask, self.static_rows, num_segments=self.sys_size
        )
        #    Add Leakage & Ground stiffness to diagonal
        diag_vals = diag_vals + self.g_leak
        for idx in self.ground_indices:
            diag_vals = diag_vals.at[idx].add(1e9)
            
        #    Invert diagonal for Jacobi Preconditioner
        inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-12, 1.0, 1.0 / diag_vals)
        
        # 2. Define Linear Operator A(x)
        #    Implicitly computes A * x without forming the full matrix
        def matvec(x: jax.Array) -> jax.Array:
            x_gathered = x[self.static_cols]
            products = all_vals * x_gathered
            Ax = jax.ops.segment_sum(products, self.static_rows, num_segments=self.sys_size)
            
            # Add Leakage & Ground contributions
            Ax = Ax + (x * self.g_leak)
            for idx in self.ground_indices:
                Ax = Ax.at[idx].add(1e9 * x[idx])
            return Ax

        # 3. Solve (BiCGStab)
        delta_guess = residual * inv_diag
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            matvec, residual, x0=delta_guess,
            M=lambda v: inv_diag * v, tol=1e-5, maxiter=200
        )
        
        return lx.Solution(value=x, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_circuit(cls, component_groups: Dict[str, Any], num_vars: int, is_complex: bool = False) -> 'SparseSolver':
        """Factory method to prepare indices and diagonal mask."""
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

        # Create mask to identify diagonal elements (row == col) efficiently
        diag_mask = (static_rows == static_cols)

        return cls(
            static_rows=jnp.array(static_rows),
            static_cols=jnp.array(static_cols),
            diag_mask=jnp.array(diag_mask),
            sys_size=sys_size,
            ground_indices=jnp.array(ground_idxs),
            is_complex=is_complex
        )