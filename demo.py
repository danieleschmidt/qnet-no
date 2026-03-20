"""1D Burgers equation demo using DistributedQNO as surrogate model.

PDE: du/dt + u*du/dx = nu * d2u/dx2   on [0, 1] with periodic BC
     u(x, 0) = sin(2*pi*x)   (initial condition)

The DistributedQNO is used as a learned surrogate that maps u(x, t) -> u(x, t+dt).
This demo creates the operator, applies it to the Burgers IC, and prints solution norms.
"""

import numpy as np
from qnetno import DistributedQNO


def burgers_ic(x: np.ndarray) -> np.ndarray:
    """Standard sinusoidal initial condition for Burgers equation."""
    return np.sin(2 * np.pi * x)


def finite_diff_step(u: np.ndarray, dx: float, dt: float, nu: float) -> np.ndarray:
    """One explicit finite-difference step for Burgers equation.

    Uses upwind scheme for advection, central differences for diffusion.
    Periodic boundary conditions.

    Args:
        u: Current solution vector.
        dx: Spatial grid spacing.
        dt: Time step.
        nu: Viscosity coefficient.

    Returns:
        Updated solution vector.
    """
    # Advection (upwind): u * du/dx
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)

    # Upwind based on sign of u
    adv = np.where(
        u >= 0,
        u * (u - u_left) / dx,
        u * (u_right - u) / dx,
    )

    # Diffusion (central): nu * d2u/dx2
    diff = nu * (u_right - 2 * u + u_left) / dx ** 2

    return u + dt * (-adv + diff)


def main():
    """Run the Burgers equation demo with DistributedQNO surrogate."""
    print("=" * 60)
    print("qnet-no: Distributed QNO Burgers Equation Demo")
    print("=" * 60)

    # Spatial grid
    N = 64       # grid points
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]

    # Time parameters
    nu = 0.01    # viscosity
    dt = 0.001   # time step
    T = 0.1      # total time
    n_steps = int(T / dt)

    # Initial condition
    u0 = burgers_ic(x)
    print(f"\nInitial condition: u(x,0) = sin(2*pi*x)")
    print(f"Grid points: {N}, viscosity: nu={nu}, dt={dt}, T={T}")
    print(f"||u_0||_2 = {np.linalg.norm(u0):.6f}")

    # --- Reference finite-difference solution ---
    u_fd = u0.copy()
    for _ in range(n_steps):
        u_fd = finite_diff_step(u_fd, dx, dt, nu)
    print(f"\n[FD Reference] ||u(T={T})||_2 = {np.linalg.norm(u_fd):.6f}")

    # --- DistributedQNO surrogate ---
    print("\nInitializing DistributedQNO surrogate...")
    qno = DistributedQNO(n_nodes=2, n_qubits_per_node=4, fidelity_threshold=0.85)

    print(f"Nodes: {qno.n_nodes}, Qubits/node: {qno.n_qubits_per_node}")
    fids = qno.node_fidelities()
    for i, f in enumerate(fids):
        print(f"  Channel {i} -> {i+1} fidelity: {f:.4f} ({'QUANTUM' if f >= 0.85 else 'CLASSICAL'})")

    # Apply QNO to initial condition (as a single-step surrogate demo)
    # The QNO maps a chunk of the solution vector to expectation values
    # We repeatedly apply to demonstrate the surrogate operator
    print(f"\nApplying QNO surrogate to u(x,0)...")

    # Process the full solution in chunks for demonstration
    chunk_size = 2 ** qno.n_qubits_per_node
    n_chunks = N // chunk_size
    u_qno_out = []

    for i in range(n_chunks):
        chunk = u0[i * chunk_size:(i + 1) * chunk_size]
        out = qno.forward(chunk)
        u_qno_out.append(out)

    u_qno_flat = np.concatenate(u_qno_out)
    print(f"QNO output shape: {u_qno_flat.shape}")
    print(f"||QNO(u_0)||_2 = {np.linalg.norm(u_qno_flat):.6f}")

    # Multi-step application (just demonstrating the operator can be applied repeatedly)
    u_surr = u0[:chunk_size].copy()
    norms = [np.linalg.norm(u_surr)]
    for step in range(5):
        out = qno.forward(u_surr)
        # Rescale to maintain norm (surrogate demo, not a trained model)
        target_norm = np.linalg.norm(u_surr)
        if np.linalg.norm(out) > 1e-12:
            u_surr_new = out * (target_norm / np.linalg.norm(out))
        else:
            u_surr_new = out
        norms.append(np.linalg.norm(u_surr_new))
        u_surr = u_surr_new

    print(f"\nSurrogate solution norms over 5 steps:")
    for step, norm in enumerate(norms):
        print(f"  Step {step}: ||u||_2 = {norm:.6f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)

    return u_qno_flat


if __name__ == "__main__":
    main()
