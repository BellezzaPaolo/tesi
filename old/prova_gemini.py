from firedrake import *
import numpy as np

# 1. Setup Parameters
g_values = [10, 100, 1000]
for i in range(len(g_values)):
    g = Constant(g_values[i])  # Change this to run different cases
    dt = Constant(0.5)
    tol = 1e-5

    # Reference energies for convergence check (example values)
    # In practice, these are obtained from high-order spectral methods
    reference_energies = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235}
    E_ref = reference_energies[float(g)]

    # 2. Mesh and Function Space
    # Domain Omega = (-6, 6)^2
    L_min, L_max = -6.0, 6.0
    n_elements = 256 # Increased resolution for g=1000 accuracy
    mesh = RectangleMesh(n_elements, n_elements, L_max-L_min, L_max-L_min)#, quadrilateral=True)

    # Shift mesh to be centered at (0,0) so it spans (-6, 6)
    mesh.coordinates.dat.data[:] += [L_min, L_min]

    V = FunctionSpace(mesh, "CG", 1)

    # 3. Functions
    u = TrialFunction(V)  # Used for the bilinear form
    v = TestFunction(V)   # Used for both forms
    psi = Function(V, name="Wavefunction") # Stores solution at current n
    psi_next = Function(V)                 # Stores solution at n+1
    x, y = SpatialCoordinate(mesh)
    V_ext = 0.5 * (x**2 + y**2)

    # 4. Initial Value z0: Exact ground state for g = 0
    psi_0_expr = exp(-(x**2 + y**2) / 2.0) / sqrt(pi)
    psi.project(psi_0_expr)

    # 5. Semi-Implicit Discretization
    # We solve for 'u' (TrialFunction)
    # (u - psi)/dt = 0.5*div(grad(u)) - V_ext*u - g*|psi|^2*psi
    a = ( (u / dt) * v * dx 
        + 0.5 * dot(grad(u), grad(v)) * dx 
        + V_ext * u * v * dx
        + g * (psi**2) * u * v * dx )

    L_form = (psi / dt) * v * dx

    # 6. Solver setup
    # Now 'a' is a Bilinear form and 'L_form' is a Linear form
    problem = LinearVariationalProblem(a, L_form, psi_next)
    solver = LinearVariationalSolver(problem, solver_parameters={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg"
    })
    print(f"Starting solver for g = {float(g)}...")

    for i in range(1000):
        solver.solve()
        
        # Re-normalize
        norm = sqrt(assemble(psi_next**2 * dx))
        psi_next.assign(psi_next / norm)
        
        # Calculate Energy
        energy = assemble(0.5 * (0.5*dot(grad(psi_next), grad(psi_next)) 
                        + V_ext*psi_next**2 
                        + 0.5*g*psi_next**4) * dx)
        
        # Error criterion: |E - E_ref| / |E_ref| < TOL
        rel_error = abs(energy - E_ref) / abs(E_ref)
        
        print(f"Iter {i}: Energy = {energy:.6f}, Rel. Error = {rel_error:.2e}")
            
        if rel_error < tol:
            print(f"Converged at iteration {i} with Relative Error {rel_error:.2e}")
            break
            
        psi.assign(psi_next)
