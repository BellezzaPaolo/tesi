import firedrake as fd
import numpy as np
import csv
import matplotlib.pyplot as plt
import utils
from test.pot_3 import RandomDisorderPotential
# import time
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# reproducibility
rng = np.random.default_rng(21)


def assemble_forms(u, w, v, tau, u_old, beta_c):
    # ensure numeric parameters are UFL Constants to avoid premature python-side
    # evaluation that can produce plain Python numbers instead of UFL expressions
    tau_c = fd.Constant(tau)

    a = u * w * fd.dx \
        + tau_c * 0.5 * fd.inner(fd.grad(u),fd.grad(w)) * fd.dx \
        + tau_c * (v * u) * w * fd.dx \
        + tau_c * beta_c * (u_old **2 * u) * w * fd.dx

    rhs = u_old * w * fd.dx
    return a, rhs


xmin, xmax = -6., 6.
ymin, ymax = -6., 6.

epsilon = 0.03

nxV = int((xmax - xmin) / (epsilon))
nyV = int((ymax - ymin) / (epsilon))

meshV = fd.RectangleMesh(nxV, nyV, xmax, ymax, originX=xmin, originY=ymin, quadrilateral=True)

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

h = 12 * 2**(-8)
beta = 10

nx = int((xmax - xmin) / h)
ny = int((ymax - ymin) / h)

mesh = fd.RectangleMesh(nx, ny, xmax, ymax, originX=xmin, originY=ymin, diagonal='left')
x = fd.SpatialCoordinate(mesh)

W = fd.FunctionSpace(mesh, 'CG', 1)
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

print("V_random stats: min, max, mean, unique counts:",
      float(np.min(V_random.dat.data[:])), float(np.max(V_random.dat.data[:])), float(np.mean(V_random.dat.data[:])),
      np.unique(V_random.dat.data[:]).size)

print("v (on PDE mesh) stats: min, max, mean, unique counts:",
      float(np.min(v.dat.data[:])), float(np.max(v.dat.data[:])), float(np.mean(v.dat.data[:])),
      np.unique(v.dat.data[:]).size)

# check lengths
print("n_cells source:", len(V_random.dat.data[:]), "n_cells target:", len(v.dat.data[:]))

print("meshV cellsize target approx:", (xmax-xmin)/nxV, (ymax-ymin)/nyV)
print("mesh   cellsize PDE approx:", (xmax-xmin)/nx, (ymax-ymin)/ny)
print("epsilon:", epsilon, "h (PDE):", h)


# fig, ax = plt.subplots(2,1, figsize=(5,10))
# col = fd.tripcolor(v, axes=ax[0], cmap='coolwarm')
# plt.colorbar(col)
# ax[0].set_title('Random Potential v(x)')
# ax[0].axis('equal')
# col = fd.tripcolor(V_random, axes = ax[1], cmap='coolwarm')
# plt.colorbar(col)
# ax[1].set_title('random potential')
# ax[1].axis('equal')
# plt.show()

# Plot the potential
v_func = fd.Function(W)
v_func.interpolate(v)

div_theme = LinearSegmentedColormap.from_list(
    "div_theme",
    [
        "#2c3e50",  # dark blue
        "#4a90e2",  # light blue
        "#ffffff",  # center (zero)
        "#f5b041",  # light orange
        "#e67e22"   # strong accent
    ]
)
v_func = fd.Function(W)
v_func.interpolate(v)
fig, ax = plt.subplots(1,1, figsize=(10,10))
col = fd.tripcolor(v_func, axes=ax, cmap=div_theme)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.margins(0)
ax.set_xticks([])
ax.set_yticks([])
cax = inset_axes(
    ax,
    width="100%",   # exact same width as plotting axis
    height="4%",
    loc="lower left",
    bbox_to_anchor=(0.0, -0.08, 1.0, 1.0),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
fig.colorbar(col, cax=cax, orientation='horizontal')
plt.show()

bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

tau = 10 #[1., 0.5]

# define the variational problem
u = fd.TrialFunction(W)
w = fd.TestFunction(W)

beta_c = fd.Constant(beta)

mu_TF = fd.sqrt(beta_c / fd.pi)

u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF- v)/beta_c), 0.0)
# u0 = fd.Constant(1e-13)

u_old = fd.Function(W)
u_old.interpolate(u0)


uh = fd.Function(W)

def energy(uh, v = v, beta = beta):
    return fd.assemble(0.5 *( 0.5 * fd.inner(fd.grad(uh), fd.grad(uh)) + (v * uh )* uh + beta/2 * uh * uh *uh * uh) * fd.dx)


param = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}

MaxIter = 1000
toll = 1e-12


for i in range(MaxIter):
    a, rhs = assemble_forms(u, w, v, tau, u_old, beta_c)
    problem = fd.LinearVariationalProblem(a, rhs, uh, bcs)
    solver =  fd.LinearVariationalSolver(problem, solver_parameters=param)
    solver.solve()

    uh.assign( uh / fd.norm(uh, 'L2'))

    error = fd.errornorm(uh, u_old,'L2')/fd.norm(uh,'L2')

    print(f'\rIter {i}, Error: {error:.2e}, Energy: {energy(uh):.10f}', end="", flush=True)

    
    u_old.assign(uh)

    if error < toll:
        break

e_gs = energy(uh)
lamb_gs = 2 * e_gs + beta/2 * fd.norm(uh,'L4')**4

print()
print()
print(f'Final energy estimate: {e_gs} and associated lambda: {lamb_gs} with h: {h} and beta: {beta}')


# fig, ax = plt.subplots()
# col = fd.tripcolor(uh, axes=ax, cmap = 'coolwarm')
# print(max(uh.dat.data))
# cbar = plt.colorbar(col, extendrect = 'both')
# cbar.set_ticks(np.linspace(0.0, max(uh.dat.data), 11))


# filename = './Ground_Truth_3/U_GS_b'+str(beta)+'_N'+str(nx)+'.h5'
# utils.save_uh(mesh, uh, filename)

# Plot the final solution
fig, ax = plt.subplots()
col = fd.tripcolor(uh, axes=ax, cmap='coolwarm')
plt.colorbar(col)
ax.axis('equal')
ax.axis('off')
# plt.title(f'Final Solution ($h = 12 \\times 2^{{{int(np.log2(h/12))}}}$, $\\beta = {beta}$)')
plt.show()
