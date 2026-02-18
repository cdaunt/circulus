"""Assembly functions for the transient circuit solver.

Provides functions for evaluating the residual vectors and effective
Jacobian of the discretised circuit equations at each Newton iteration.
Functions are provided in two variants:

- **Full assembly** (:func:`assemble_system_real`, :func:`assemble_system_complex`)
  — evaluates both the residual and the forward-mode Jacobian via
  ``jax.jacfwd``. Used once per timestep to assemble and factor the frozen
  Jacobian in :class:`~circulax.solver.FactorizedTransientSolver`.

- **Residual only** (:func:`assemble_residual_only_real`,
  :func:`assemble_residual_only_complex`) — evaluates only the primal
  residual, with no Jacobian computation. Used inside the Newton loop where
  the Jacobian has already been factored and only needs to be applied.

Each pair has a real and a complex variant. The complex variant operates on
state vectors in unrolled block format — real parts concatenated with imaginary
parts — allowing complex circuit analyses to reuse real-valued sparse linear
algebra kernels.
"""

import jax
import jax.numpy as jnp
from jax import Array


def assemble_system_real(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Assemble the residual vectors and effective Jacobian values for a real system.

    For each component group, evaluates the physics at ``t1`` and computes the
    forward-mode Jacobian via ``jax.jacfwd``. The effective Jacobian combines
    the resistive and reactive contributions as ``J_eff = df/dy + (1/dt) * dq/dy``,
    consistent with the implicit trapezoidal discretisation used by the solver.

    Components are processed in sorted key order to ensure a deterministic
    non-zero layout in the sparse Jacobian, which is required for the
    factorisation step.

    Args:
        y_guess: Current state vector of shape ``(sys_size,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Timestep duration, used to scale the reactive Jacobian block.

    Returns:
        A three-tuple ``(total_f, total_q, jac_vals)`` where:

        - **total_f** — assembled resistive residual, shape ``(sys_size,)``.
        - **total_q** — assembled reactive residual, shape ``(sys_size,)``.
        - **jac_vals** — concatenated non-zero values of the effective Jacobian
            in group-sorted order, ready to be passed to the sparse linear solver.

    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    vals_list = []

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v_locs = y_guess[group.var_indices]

        def physics_at_t1(v: Array, p: Array) -> tuple[Array, Array]:
            return group.physics_func(y=v, args=p, t=t1)

        (f_l, q_l) = jax.vmap(physics_at_t1)(v_locs, group.params)
        (df_l, dq_l) = jax.vmap(jax.jacfwd(physics_at_t1))(v_locs, group.params)

        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        j_eff = df_l + (dq_l / dt)
        vals_list.append(j_eff.reshape(-1))

    return total_f, total_q, jnp.concatenate(vals_list)


def assemble_residual_only_real(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
) -> tuple[Array, Array]:
    """Assemble the residual vectors for a real system, without computing the Jacobian.

    Cheaper than :func:`assemble_system_real` as it performs only primal
    evaluations. Used inside the frozen-Jacobian Newton loop where the
    Jacobian has already been factored and only the residual needs to be
    recomputed at each iteration.

    Args:
        y_guess: Current state vector of shape ``(sys_size,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Unused; present for signature symmetry with
            :func:`assemble_system_real` so the two functions are
            interchangeable at call sites.

    Returns:
        A two-tuple ``(total_f, total_q)`` where both arrays have shape
        ``(sys_size,)`` and ``dtype=float64``.

    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v = y_guess[group.var_indices]

        def physics_at_t1(v: Array, p: Array) -> tuple[Array, Array]:
            return group.physics_func(y=v, args=p, t=t1)

        f_l, q_l = jax.vmap(physics_at_t1)(v, group.params)

        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)

    return total_f, total_q


def assemble_system_complex(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Assemble the residual vectors and effective Jacobian values for an unrolled complex system.

    The complex state vector is stored in unrolled (block) format: the first
    half of ``y_guess`` holds the real parts of all node voltages/states, the
    second half holds the imaginary parts. This avoids JAX's limited support
    for complex-valued sparse linear solvers by keeping all arithmetic real.

    The Jacobian is split into four real blocks — RR, RI, IR, II — representing
    the partial derivatives of the real and imaginary residual components with
    respect to the real and imaginary state components respectively. The blocks
    are concatenated in RR→RI→IR→II order to match the sparsity index layout
    produced during compilation.

    Args:
        y_guess: Unrolled state vector of shape ``(2 * num_vars,)``, where
            ``y_guess[:num_vars]`` are real parts and ``y_guess[num_vars:]``
            are imaginary parts.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Timestep duration, used to scale the reactive Jacobian blocks.

    Returns:
        A three-tuple ``(total_f, total_q, jac_vals)`` where:

        - **total_f** — assembled resistive residual in unrolled format,
            shape ``(2 * num_vars,)``.
        - **total_q** — assembled reactive residual in unrolled format,
            shape ``(2 * num_vars,)``.
        - **jac_vals** — concatenated non-zero values of the four effective
            Jacobian blocks (RR, RI, IR, II) in group-sorted order.

    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]

    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)

    vals_blocks: list[list[Array]] = [[], [], [], []]

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        def physics_split(
            vr: Array, vi: Array, p: Array
        ) -> tuple[Array, Array, Array, Array]:
            v = vr + 1j * vi
            f, q = group.physics_func(y=v, args=p, t=t1)
            return f.real, f.imag, q.real, q.imag

        fr, fi, qr, qi = jax.vmap(physics_split)(v_r, v_i, group.params)

        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

        jac_res = jax.vmap(jax.jacfwd(physics_split, argnums=(0, 1)))(
            v_r, v_i, group.params
        )
        ((dfr_r, dfr_i), (dfi_r, dfi_i), (dqr_r, dqr_i), (dqi_r, dqi_i)) = jac_res

        vals_blocks[0].append((dfr_r + dqr_r / dt).reshape(-1))  # RR
        vals_blocks[1].append((dfr_i + dqr_i / dt).reshape(-1))  # RI
        vals_blocks[2].append((dfi_r + dqi_r / dt).reshape(-1))  # IR
        vals_blocks[3].append((dfi_i + dqi_i / dt).reshape(-1))  # II

    all_vals = jnp.concatenate([jnp.concatenate(b) for b in vals_blocks])
    return total_f, total_q, all_vals


def assemble_residual_only_complex(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
) -> tuple[Array, Array]:
    """Assemble the residual vectors for an unrolled complex system, without computing the Jacobian.

    The complex counterpart of :func:`assemble_residual_only_real`. The state
    vector is expected in unrolled block format (real parts followed by imaginary
    parts) matching the layout used by :func:`assemble_system_complex`.

    Args:
        y_guess: Unrolled state vector of shape ``(2 * num_vars,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Unused; present for signature symmetry with
            :func:`assemble_system_complex` so the two functions are
            interchangeable at call sites.

    Returns:
        A two-tuple ``(total_f, total_q)`` where both arrays have shape
        ``(2 * num_vars,)`` and ``dtype=float64``.

    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]

    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        def physics_split(
            vr: Array, vi: Array, p: Array
        ) -> tuple[Array, Array, Array, Array]:
            v = vr + 1j * vi
            f, q = group.physics_func(y=v, args=p, t=t1)
            return f.real, f.imag, q.real, q.imag

        fr, fi, qr, qi = jax.vmap(physics_split)(v_r, v_i, group.params)

        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

    return total_f, total_q
