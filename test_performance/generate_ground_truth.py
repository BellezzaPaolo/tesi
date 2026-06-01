"""Load settings JSON and build mesh, BCs, potential and initial guess.

Usage: python generate_ground_thruth.py path/to/settings.json
"""
import json
import sys
from pathlib import Path
import numpy as np
import firedrake as fd
from test_case3.pot_3 import RandomDisorderPotential


if len(sys.argv) >1:
    path = Path(sys.argv[1])
else:
    raise ValueError('Settings JSON path must be provided as argument')

settings = json.loads(path.read_text())


# Domain is of type [xmin, xmax] x [ymin, ymax]
if settings['mesh']['call'] != 'RectangleMesh':
    raise NotImplementedError('Only RectangleMesh supported by this loader')

mesh = fd.RectangleMesh(settings['mesh']['nx'], settings['mesh']['nx'], # equal number of cells in x and y
                        settings['mesh']['xmax'], # xmax
                        settings['mesh']['ymax'], # ymax
                        settings['mesh']['xmin'], # xmin
                        settings['mesh']['ymin'], # ymin
                        diagonal=settings['mesh']['diagonal']) # diagonal can be 'left', 'right' or 'crossed'

# Build the function space
W = fd.FunctionSpace(mesh, settings['function_space']['family'], settings['function_space']['degree'])

# Build BCs
if settings['bcs']['type'] != 'DirichletBC':
    raise NotImplementedError('Only DirichletBC is implemented in loader')
bcs = fd.DirichletBC(W, fd.Constant(settings['bcs']['value']), tuple(settings['bcs']['boundaries']))

# Load nonlinear parameter beta
beta = fd.Constant(settings['beta'])

x, y = fd.SpatialCoordinate(mesh)
# Build potential
if settings['potential'] == 'harmonic':
    v = 0.5 * (x**2 + y**2)

elif settings['potential'] == 'lattice':
    v = 0.5 * (x**2 + y**2) + fd.Constant(20) + fd.Constant(20) * fd.sin( 2 * fd.pi * x) * fd.sin(2 * fd.pi * y)

elif settings['potential'] == 'randomized':

    epsilon = 0.03

    nxV = int((settings['mesh']['xmax'] - settings['mesh']['xmin']) / (epsilon))
    nyV = int((settings['mesh']['ymax'] - settings['mesh']['ymin']) / (epsilon))

    meshV = fd.RectangleMesh(nxV, nyV, 
                             settings['mesh']['xmax'], settings['mesh']['ymax'], settings['mesh']['xmin'], settings['mesh']['ymin'], 
                             quadrilateral=True)
    
    # Function space for a cellwise-constant potential
    Vdg = fd.FunctionSpace(meshV, "DG", 0)   # piecewise-constant per cell

    # Create random disorder potential using the RandomDisorderPotential class
    # Using seed=21 for consistency with the original rng seed
    potential_obj = RandomDisorderPotential(number_of_cells=400, domain_size=12.0, seed=21)

    # Create the DG0 function for the potential
    V_random = fd.Function(Vdg)

    # Get cell coordinates from meshV and evaluate the potential
    coord_fn_V = fd.Function(fd.VectorFunctionSpace(meshV, 'DG', 0))
    coord_fn_V.interpolate(fd.SpatialCoordinate(meshV))
    coords_V = coord_fn_V.dat.data_ro

    # Evaluate potential at cell coordinates and assign to V_random
    V_random.dat.data[:] = potential_obj.evaluate(coords_V[:, 0], coords_V[:, 1])
    
    Wv = fd.FunctionSpace(mesh, 'DG', 0)

    v = fd.Function(Wv)

    # Get cell midpoints from the triangular mesh to evaluate V_random
    # For a DG0 function, we need one value per cell
    # Use the cell centroid coordinates
    x_dg = fd.SpatialCoordinate(mesh)
    coord_fn = fd.Function(fd.VectorFunctionSpace(mesh, 'DG', 0))
    coord_fn.interpolate(x_dg)
    coords = coord_fn.dat.data_ro

    # Step 2: Evaluate V_random at those coordinates
    vals = np.array([V_random.at((float(px), float(py)), tolerance=1e-10)
                    for px, py in coords])

    # Step 3: assign to the DG0 field on the triangular mesh
    v.dat.data[:] = vals

else:
    NameError('test must be 1, 2 or 3')


# Build initial guess
if settings['initial_guess'] == 'linear_GS':
    u0 = 1/fd.pi**(0.5) * fd.exp(-(x**2 + y**2)/2)
elif settings['initial_guess'] == 'general_TF':
    mu_TF = fd.Constant(fd.sqrt(beta / fd.pi))

    u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF - v)/beta), 0.0)
else:
    raise NotImplementedError('Unknown initial guess type. Supported types: linear_GS, general_TF')

# Define energy functional for reference computation
def energy(uh, v = v, beta = beta):
    return fd.assemble(0.5 *( 0.5 * fd.inner(fd.grad(uh), fd.grad(uh)) + (v * uh )* uh + beta/2 * uh * uh *uh * uh) * fd.dx)


u = fd.TrialFunction(W)
w = fd.TestFunction(W)
uh = fd.Function(W)
u_old = fd.Function(W)
u_old.interpolate(u0)

E = energy(u_old)

tau = fd.Constant(1.0)

toll = 1e-8

for i in range(1000):
    a = u * w * fd.dx \
        + tau * 0.5 * fd.inner(fd.grad(u),fd.grad(w)) * fd.dx \
        + tau * (v * u) * w * fd.dx \
        + tau * beta * (u_old **2 * u) * w * fd.dx
    
    rhs = u_old * w * fd.dx

    problem = fd.LinearVariationalProblem(a, rhs, uh, bcs)
    solver =  fd.LinearVariationalSolver(problem)
    solver.solve()

    uh.assign( uh / fd.norm(uh, 'L2'))

    error = fd.errornorm(uh, u_old,'L2')/fd.norm(uh,'L2')

    print(f'\rIter {i}, Error: {error:.2e}, Energy: {energy(uh)}', end="", flush=True)

    
    u_old.assign(uh)

    if error < toll:
        break

computed_energy = float(energy(uh))
settings['energy_reference'] = computed_energy

# Persist updated settings back to disk.
path.write_text(json.dumps(settings, indent=2))

print()
print()
print('Ground truth energy:', computed_energy, flush =True)
print('Saved updated JSON to:', path)