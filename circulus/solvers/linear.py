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
from scipy.sparse import coo_matrix

try:
    import klujax_cpp
except ImportError:
    klujax_cpp = None

try:
    import klujax_rs  as klurs
except ImportError:
    klurs = None


# Import physics assembly kernels (lazy import handled in methods if needed)
from circulus.solvers.assembly import _assemble_system_complex, _assemble_system_real


class SymbolicHandleWrapper:
    """
    A plain Python class to manage the lifecycle of the C pointer.
    Equinox won't intercept attribute access on this class.
    """
    def __init__(self, handle, free_callable):
        self.handle = handle
        self.free_callable = free_callable


    def __del__(self):
        # Guard against globals being None during interpreter shutdown
        if klujax is None or self.handle is None:
            return
            
        try:
            # Assuming handle is a JAX array wrapping the pointer
            # We only free if it's concrete (not a Tracer)
            # Note: In standard Python usage, this will be concrete.
            self.free_callable(self.handle)
        except Exception:
            pass

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



class KLUSplitSolver(CircuitLinearSolver):
    """
    Solves the system using the KLU sparse solver (via `klujax`) with split interface.
    
    This solver performs symbolic analysis ONCE during initialization and reuses
    the symbolic handle for subsequent solves, significantly speeding up non-linear
    simulations (Newton-Raphson iterations).
    
    Best For:
        - Large circuits (N > 5000) running on CPU.
        - DC Operating Points of massive meshes.
    
    Attributes:
        Bp, Bi: CSC format indices (fixed structure).
        csc_map_idx: Mapping from raw value indices to CSC value vector.
        symbolic_handle: Pointer to the pre-computed KLU symbolic analysis.
    """
    
    u_rows: jax.Array
    u_cols: jax.Array
    map_idx: jax.Array

    n_unique: int = eqx.field(static=True)
    sys_size: int = eqx.field(static=True)
    _handle_wrapper: SymbolicHandleWrapper = eqx.field(static=True)

    g_leak: float = 1e-9

    def cleanup(self):
        del self._handle_wrapper

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
        solution = klujax.solve_with_symbol(self.u_rows, self.u_cols, coalesced_vals, residual, self._handle_wrapper.handle)
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

        symbolic = SymbolicHandleWrapper(klujax.analyze(u_rows, u_cols, sys_size),
                                         free_callable=klujax.free_symbolic)

        return cls(
            u_rows=jnp.array(u_rows),
            u_cols=jnp.array(u_cols),
            map_idx=jnp.array(map_indices),
            n_unique=n_unique,
            _handle_wrapper=symbolic,
            ground_indices=jnp.array(ground_idxs),
            sys_size=sys_size,
            is_complex=is_complex
        )
    

class KlursSplitSolver(KLUSplitSolver):

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        # 1. Prepare raw value vector including Ground and Leakage entries
        g_vals = jnp.full(self.ground_indices.shape[0], 1e9, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype) 
        
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])

        # 2. Coalesce duplicate entries (COO -> Unique COO)
        coalesced_vals = jax.ops.segment_sum(
            raw_vals, self.map_idx, num_segments=self.n_unique
        )

        # 3. Call klurs Wrapper
        #solution = klurs.solve_with_symbol(self.u_rows, self.u_cols, coalesced_vals, residual, self.symbolic_handle)
        solution = klurs.solve_with_symbol(self.u_rows, self.u_cols, coalesced_vals, residual, self._handle_wrapper.handle)
        return lx.Solution(value=solution, result=lx.RESULTS.successful, state=None, stats={})

    # def __del__(self):
    #     self.cleanup()

    # def cleanup(self):
    #     """Free the C++ symbolic handle. Call when done with the solver."""
    #     # Check if handle exists and is not None
    #     if hasattr(self, 'symbolic_handle') and self.symbolic_handle is not None:
    #         # We must be careful not to free Tracers during JIT compilation.
    #         # Tracers are instances of jax.core.Tracer.
    #         # If we are inside JIT, calling free_symbolic creates a side-effect node
    #         # which is fine (it gets DCE'd if unused), but for __del__ we want immediate host cleanup.
    #         try:
    #             # Eagerly call free if it's a concrete array
    #             if isinstance(self.symbolic_handle, jax.Array):
    #                  klurs.free_symbolic(self.symbolic_handle)
    #         except Exception:
    #             # Fallback or silence errors during shutdown
    #             pass

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

        raw_symbol = klurs.analyze(u_rows, u_cols, sys_size)
        symbolic = SymbolicHandleWrapper(raw_symbol, free_callable=klurs.free_symbolic)

        return cls(
            u_rows=jnp.array(u_rows),
            u_cols=jnp.array(u_cols),
            map_idx=jnp.array(map_indices),
            n_unique=n_unique,
            _handle_wrapper=symbolic,
            ground_indices=jnp.array(ground_idxs),
            sys_size=sys_size,
            is_complex=is_complex
        )

class KLUSplitFactorSolver(KLUSplitSolver):
    """
    Solves the system using the KLU sparse solver (via `klujax`) with split interface.
    
    This solver performs symbolic analysis ONCE during initialization and reuses
    the symbolic handle for subsequent solves, significantly speeding up non-linear
    simulations (Newton-Raphson iterations). This version of the solver is further enhanced but calculting the numeric
    part of the KLU solution only once
    
    Best For:
        - Large circuits (N > 5000) running on CPU.
        - DC Operating Points of massive meshes.
    
    Attributes:
        Bp, Bi: CSC format indices (fixed structure).
        csc_map_idx: Mapping from raw value indices to CSC value vector.
        symbolic_handle: Pointer to the pre-computed KLU symbolic analysis.
    """
    
    u_rows: jax.Array
    u_cols: jax.Array
    map_idx: jax.Array

    n_unique: int = eqx.field(static=True)
    sys_size: int = eqx.field(static=True)

    g_leak: float = 1e-9
    #numeric_handle: Optional[jax.Array] = None

    _handle_wrapper: SymbolicHandleWrapper = eqx.field(static=True)

    g_leak: float = 1e-9

    def cleanup(self):
        del self._handle_wrapper

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        """Regular solve - does full factor + solve."""
        g_vals = jnp.full(self.ground_indices.shape[0], 1e9, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype) 
        
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(
            raw_vals, self.map_idx, num_segments=self.n_unique
        )

        numeric = klujax.factor(self.u_rows, self.u_cols, coalesced_vals, self._handle_wrapper.handle)
        solution = klujax.solve_with_numeric(numeric, residual, self._handle_wrapper.handle)
        # Free the numeric handle to prevent memory leaks in the C++ backend
        klujax.free_numeric(numeric)
        return lx.Solution(value=solution.reshape(residual.shape), result=lx.RESULTS.successful, state=None, stats={})

    def solve_with_frozen_jacobian(self, residual: jax.Array, numeric: jax.Array) -> lx.Solution:
        """Solve using pre-computed numeric factorization (for frozen Jacobian Newton)."""
        solution = klujax.solve_with_numeric(numeric, residual, self._handle_wrapper.handle)
        return lx.Solution(value=solution.reshape(residual.shape), result=lx.RESULTS.successful, state=None, stats={})
    
    def factor_jacobian(self, all_vals: jax.Array) -> jax.Array:
        """Factor the Jacobian and return numeric handle."""
        g_vals = jnp.full(self.ground_indices.shape[0], 1e9, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype) 
        
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(
            raw_vals, self.map_idx, num_segments=self.n_unique
        )
        
        return klujax.factor(self.u_rows, self.u_cols, coalesced_vals, self.symbolic_handle)
    

    # def __del__(self):
    #     self.cleanup()

    # def cleanup(self):
    #     """Free the C++ symbolic handle. Call when done with the solver."""
    #     # Check if handle exists and is not None
    #     if hasattr(self, 'symbolic_handle') and self.symbolic_handle is not None:
    #         # We must be careful not to free Tracers during JIT compilation.
    #         # Tracers are instances of jax.core.Tracer.
    #         # If we are inside JIT, calling free_symbolic creates a side-effect node
    #         # which is fine (it gets DCE'd if unused), but for __del__ we want immediate host cleanup.
    #         try:
    #             # Eagerly call free if it's a concrete array
    #             if isinstance(self.symbolic_handle, jax.Array):
    #                  klujax.free_symbolic(self.symbolic_handle)
    #         except Exception:
    #             # Fallback or silence errors during shutdown
    #             pass



    
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


    # # CSC Topology (Static Structure)
    # Bp: jax.Array
    # Bi: jax.Array
    
    # # Mapping
    # csc_map_idx: jax.Array
    
    # # KLU Handle (Treated as a JAX array leaf)
    # symbolic_handle: jax.Array
    
    # # Metadata
    # n_unique: int = eqx.field(static=True)
    # sys_size: int = eqx.field(static=True)
    # is_complex: bool = eqx.field(static=True)
    # ground_indices: jax.Array
    
    # g_leak: float = 1e-9

    # def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
    #     # 1. Prepare raw value vector including Ground and Leakage entries
    #     g_vals = jnp.full(self.ground_indices.shape[0], 1e9, dtype=all_vals.dtype)
    #     l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype) 
        
    #     raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])

    #     # 2. Coalesce duplicate entries directly into CSC order (Bx)
    #     # We use csc_map_idx which maps raw -> CSC location directly.
    #     # This skips the intermediate "Unique COO" step.
    #     Bx = jax.ops.segment_sum(
    #         raw_vals, self.csc_map_idx, num_segments=self.n_unique
    #     )

    #     # 3. Call KLU Wrapper with Split Interface
    #     # Passing 'symbolic_handle' skips the expensive analysis step.
    #     # Gradients are supported via klujax's custom_vjp.
    #     solution = klujax.solve(
    #         self.Bp, 
    #         self.Bi, 
    #         Bx, 
    #         residual, 
    #         symbolic_handle=self.symbolic_handle
    #     )
        
    #     return lx.Solution(value=solution, result=lx.RESULTS.successful, state=None, stats={})

    # def cleanup(self):
    #     """Free the C++ symbolic handle. Call when done with the solver."""
    #     # Check if handle exists and is not None
    #     if hasattr(self, 'symbolic_handle') and self.symbolic_handle is not None:
    #         # We must be careful not to free Tracers during JIT compilation.
    #         # Tracers are instances of jax.core.Tracer.
    #         # If we are inside JIT, calling free_symbolic creates a side-effect node
    #         # which is fine (it gets DCE'd if unused), but for __del__ we want immediate host cleanup.
    #         try:
    #             # Eagerly call free if it's a concrete array
    #             if isinstance(self.symbolic_handle, jax.Array):
    #                  klujax.free_symbolic(self.symbolic_handle)
    #         except Exception:
    #             # Fallback or silence errors during shutdown
    #             pass

    # def __del__(self):
    #     """Ensure C++ resources are freed when the solver object is garbage collected."""
    #     self.cleanup()

    # @classmethod
    # def from_circuit(cls, component_groups: Dict[str, Any], num_vars: int, is_complex: bool = False) -> 'KLUSolver':
    #     """Factory method to pre-hash indices and perform symbolic analysis."""
    #     # --- 1. Collect All Indices (Same as before) ---
    #     all_rows, all_cols = [], []
    #     for k in sorted(component_groups.keys()):
    #         g = component_groups[k]
    #         all_rows.append(np.array(g.jac_rows).reshape(-1))
    #         all_cols.append(np.array(g.jac_cols).reshape(-1))
        
    #     static_rows = np.concatenate(all_rows)
    #     static_cols = np.concatenate(all_cols)

    #     sys_size = num_vars
    #     ground_idxs = np.array([0], dtype=np.int32)

    #     if is_complex:
    #         sys_size = num_vars * 2
    #         r, c = static_rows, static_cols
    #         N = num_vars
    #         static_rows = np.concatenate([r, r, r+N, r+N])
    #         static_cols = np.concatenate([c, c+N, c, c+N])
    #         ground_idxs = np.array([0, num_vars], dtype=np.int32)

    #     leak_rows = np.arange(sys_size, dtype=np.int32)
    #     leak_cols = np.arange(sys_size, dtype=np.int32)

    #     full_rows = np.concatenate([static_rows, ground_idxs, leak_rows])
    #     full_cols = np.concatenate([static_cols, ground_idxs, leak_cols])
        
    #     # --- 2. Hashing to find Unique Entries ---
    #     rc_hashes = full_rows.astype(np.int64) * sys_size + full_cols.astype(np.int64)
    #     unique_hashes, map_indices = np.unique(rc_hashes, return_inverse=True)
    #     n_unique = len(unique_hashes)
        
    #     # Recover COO coordinates for unique entries
    #     u_rows = (unique_hashes // sys_size).astype(np.int32)
    #     u_cols = (unique_hashes % sys_size).astype(np.int32)

    #     # --- 3. Convert Unique COO to CSC (for KLU) ---
    #     # We use Scipy to perform the sorting/compression efficiently on CPU
    #     # We use a dummy range array as data to track the permutation
    #     dummy_data = np.arange(n_unique, dtype=np.int32)
        
    #     # This effectively sorts the unique entries into CSC order (Column-Major)
    #     coo = coo_matrix((dummy_data, (u_rows, u_cols)), shape=(sys_size, sys_size))
    #     csc = coo.tocsc()
        
    #     Bp = csc.indptr.astype(np.int32)
    #     Bi = csc.indices.astype(np.int32)
        
    #     # 'csc.data' now contains the original indices from 'dummy_data' permuted into CSC order.
    #     # This tells us: Bx[k] = Unique_COO_Vals[csc.data[k]]
    #     # We want to map directly from raw inputs to Bx.
    #     # Current map: Raw -> Unique_COO (via map_indices)
    #     # We need:     Raw -> CSC_Bx
        
    #     # We compute the inverse permutation to remap the map_indices
    #     perm = csc.data
    #     inv_perm = np.zeros_like(perm)
    #     inv_perm[perm] = np.arange(len(perm))
        
    #     # New map: Raw inputs sum directly into the correct CSC slot
    #     csc_map_idx = inv_perm[map_indices]

    #     # --- 4. Perform Symbolic Analysis (Once) ---
    #     # We perform analysis now so we have the handle ready.
    #     # Note: Bp, Bi must be converted to JAX arrays for pure_callback
    #     Bp_jax = jnp.array(Bp)
    #     Bi_jax = jnp.array(Bi)
        
    #     # This call executes immediately (eagerly) since we are not in JIT yet
    #     symbolic_handle = klujax.analyze(Bp_jax, Bi_jax)

    #     return cls(
    #         Bp=Bp_jax,
    #         Bi=Bi_jax,
    #         csc_map_idx=jnp.array(csc_map_idx),
    #         symbolic_handle=symbolic_handle,
    #         n_unique=n_unique,
    #         ground_indices=jnp.array(ground_idxs),
    #         sys_size=sys_size,
    #         is_complex=is_complex
    #     )
    
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
    
backends = {'default': KLUSolver,
            'klu': KLUSolver,
            'klu_split': KLUSplitSolver,
            'dense': DenseSolver,
            'sparse': SparseSolver}


def analyze_circuit(groups: list, 
                    num_vars: int, 
                    backend: str = 'default', 
                    is_complex: bool = False) -> CircuitLinearSolver:
    """Initializes a linear solver strategy for circuit analysis.

    This function serves as a factory and wrapper to select and configure the
    appropriate numerical backend for solving the linear system of equations
    derived from a circuit's topology.

    Args:
        groups (list): A list of component groups that define the circuit's
            structure and properties.
        num_vars (int): The total number of variables in the linear system.
        backend (str, optional): The name of the solver backend to use. 
            Supported backends are 'klu', 'klu_split', 'dense', and 'sparse'.
            Defaults to 'default', which uses the 'klu' solver.
        is_complex (bool, optional): A flag indicating whether the circuit
            analysis involves complex numbers. Defaults to False.

    Returns:
        CircuitLinearSolver: An instance of a circuit linear solver strategy
        configured for the specified backend and circuit parameters.

    Raises:
        ValueError: If the specified backend is not supported.
    """
    solver_class = backends.get(backend)
    if solver_class is None:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Available backends are {list(backends.keys())}"
        )

    linear_strategy = solver_class.from_circuit(
        groups, num_vars, is_complex=is_complex
    )

    return linear_strategy




