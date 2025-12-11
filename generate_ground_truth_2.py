import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import utils

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

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

h = 12 * 2**(-8)
beta = 1000

nx = int((xmax-xmin)/h)
ny = int((ymax-ymin)/h)

mesh = fd.RectangleMesh(nx, ny, xmax, ymax, originX = xmin, originY = ymin, diagonal = 'right')

# function spaces
W = fd.FunctionSpace(mesh, 'CG', 1)


# Data and boundary conditions
x = fd.SpatialCoordinate(mesh)
v = 0.5 * (x[0]**2 + x[1]**2) + fd.Constant(20.0) + fd.Constant(20) * fd.sin( 2 * fd.pi * x[0]) * fd.sin(2 * fd.pi * x[1])
bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

tau = 1 #[1., 0.5]

# define the variational problem
u = fd.TrialFunction(W)
w = fd.TestFunction(W)

beta_c = fd.Constant(beta)

mu_TF = fd.sqrt(beta_c / fd.pi)

u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF- v)/beta_c), 0.0)
# u0 = fd.Constant(1.)

u_old = fd.Function(W)
u_old.interpolate(u0)

# Plot it
fig, ax = plt.subplots()
col = fd.tripcolor(fd.project(v,W), axes=ax, cmap = 'BrBG')
plt.colorbar(col)
ax.axis('equal')
plt.title('potential v(x)')

# a, rhs = assemble_forms(u, w, v, tau[0], u_old, beta[0])

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
print(f'Final energy estimate: {e_gs} and associated lambda: {lamb_gs} with h: {h} and beta: {beta}') #0.79620688 2.06380
# ground truth: 15.204825 36.708
# crossed 15.196860908481353 36.697896895581984
# left 15.204824828806638 36.70796481994021
# right 15.20482529108886 36.707967494741474
filename = './Ground_Truth_2/U_GS_b'+str(beta)+'_N'+str(nx)+'(1).h5'
# utils.save_uh(mesh, uh, filename)

# Plot it
u2 = fd.Function(W)
u2.interpolate( uh * uh)

fig, ax = plt.subplots()
col = fd.tripcolor(u2, axes=ax, cmap = 'coolwarm')
print(max(u2.dat.data))
cbar = plt.colorbar(col, extendrect = 'both')
cbar.set_ticks(np.linspace(0.0, max(u2.dat.data), 11))

ax.axis('equal')
plt.title('Solution')

plt.show()