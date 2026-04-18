"""
Solve the Gross-Pitaevskii equation for the ground state using L2 gradient flow
(imaginary time evolution):

    i ∂ψ/∂t = -∇²ψ + V(x)ψ + g|ψ|²ψ

Using imaginary time τ = it, this becomes:
    
    ∂ψ/∂τ = ∇²ψ - V(x)ψ - g|ψ|²ψ

which is the L2 gradient flow of the Gross-Pitaevskii energy functional:
    
    E[ψ] = ∫ (|∇ψ|² + V(x)|ψ|² + (g/2)|ψ|⁴) dx

The ground state is found by evolving until convergence with normalization.
"""

from firedrake import *
import numpy as np

print("=" * 70)
print("Gross-Pitaevskii Equation Ground State Solver")
print("Using L2 Gradient Flow (Imaginary Time Evolution)")
print("=" * 70)


# Mesh and function space
mesh = RectangleMesh(256, 256, 6, 6, -6, -6, diagonal ="crossed")
V = FunctionSpace(mesh, "CG", 1)  # Use quadratic elements for better accuracy

print(f"h: {assemble(CellDiameter(mesh)* dx) /assemble(1 * dx(mesh))}")

# Get spatial coordinates
x, y = SpatialCoordinate(mesh)

# External potential - harmonic trap
omega = 1.0
V_ext = 0.5 * (x**2 + y**2)
# print(f"  Harmonic trap frequency ω = {omega}")

# Alternative potentials:
# V_ext = 0.5 * (x**2 + 4*y**2)  # Anisotropic trap
# V_ext = conditional(sqrt(x**2 + y**2) < 3.0, 0.0, 1000.0)  # Hard wall box

# Physical parameters
g = [10., 100.0, 1000]  # Interaction strength (positive = repulsive)
E_ref = [0.79620688, 1.97298868, 5.99303235]

for j in range(len(g)):
    # print(f"\nPhysical parameters:")
    # print(f"  Interaction strength g = {g[j]}")

    # Define functions
    psi = Function(V, name="Wave function")
    psi_new = Function(V, name="Updated wave function")

    # Initial guess - Gaussian centered at origin
    psi.interpolate(exp(-(x**2 + y**2) / 2.0)/pi**(1/2))

    # Normalize initial condition
    def normalize(psi):
        """Normalize the wave function: ∫|ψ|² dx = 1"""
        norm_sq = assemble(psi**2 * dx)
        psi.dat.data[:] /= sqrt(norm_sq)
        return norm_sq

    normalize(psi)
    # print(f"\nInitial wave function normalized")

    # Time stepping parameters
    dt = 0.5  # Imaginary time step
    max_iter = 100
    tolerance = 1e-5

    # Define variational problem for L2 gradient flow
    # Semi-implicit scheme: (ψ_new - ψ)/dt = ∇²ψ_new - V*ψ_new - g|ψ|²ψ_new
    # where |ψ|² is evaluated at the previous step (linearization)

    u = TrialFunction(V)
    v = TestFunction(V)

    # Energy functional (for monitoring)
    def compute_energy(psi):
        """Compute the Gross-Pitaevskii energy"""
        kinetic = 0.5 * inner(grad(psi), grad(psi)) * dx
        potential = V_ext * psi**2 * dx
        interaction = 0.5 * g[j] * psi**4 * dx
        
        E_kin = assemble(kinetic)
        E_pot = assemble(potential)
        E_int = assemble(interaction)
        E_total = (E_kin + E_pot + E_int) * 0.5
        
        return E_total, E_kin, E_pot, E_int

    # Initial energy
    E0, Ek0, Ep0, Ei0 = compute_energy(psi)
    # print(f"\nInitial energy: {E0:.6f}")
    # print(f"  Kinetic:     {Ek0:.6f}")
    # print(f"  Potential:   {Ep0:.6f}")
    # print(f"  Interaction: {Ei0:.6f}")

    # print(f"\nStarting imaginary time evolution...")
    # print(f"  Time step dt = {dt}")
    # print(f"  Maximum iterations = {max_iter}")
    # print(f"  Convergence tolerance = {tolerance}")
    # print("-" * 70)

    # Store previous energy for convergence check
    bcs = [ DirichletBC(V, Constant(0.0), (1,2,3,4)) ]

    # Imaginary time evolution loop
    for n in range(max_iter):
        # Build variational form with current density |ψ|²
        # (u - ψ)/dt * v = -∇u·∇v - V*u*v - g|ψ|²*u*v
        density = psi**2  # |ψ|² from previous step
        
        a = (u * v /dt + 0.5 * inner(grad(u), grad(v)) + V_ext * u * v + g[j] * density * u * v) * dx
        L = psi * v /dt * dx
        
        # Solve for new wave function
        solve(a == L, psi_new, bcs = bcs)
        
        # Normalize
        normalize(psi_new)
        
        # Update for next iteration
        psi.assign(psi_new)
        
        # Check convergence periodically
        E_total, E_kin, E_pot, E_int = compute_energy(psi)
        
        # Compute energy change
        dE = abs(E_total - E_ref[j])
        rel_change = dE / abs(E_ref[j])
        
        
        # print(f"Iter {n+1:5d}: E = {E_total:12.8f}, dE = {dE:.3e}, ")
        
        # Check convergence
        if rel_change < tolerance:
            print("-" * 70)
            print(f"Converged after {n+1} iterations!")
            # print(f"Relative energy change: {rel_change:.3e} < {tolerance}")
            break

    # # Final results
    # print("\n" + "=" * 70)
    # print("GROUND STATE RESULTS")
    # print("=" * 70)

    # E_total, E_kin, E_pot, E_int = compute_energy(psi)

    # print(f"\nGround state energy:    E = {E_total:.8f}")
    # # Verify normalization
    # norm_sq = assemble(psi**2 * dx)
    # print(f"Normalization check: ∫|ψ|² dx = {norm_sq:.10f}")