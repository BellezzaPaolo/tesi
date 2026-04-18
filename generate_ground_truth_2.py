"""Ground-truth run for case 2: oscillatory augmented potential, beta=1000.

The script computes a highly accurate reference state with direct solves and
strict stopping tolerance, then visualizes potential and final solution.
"""

import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import old.utils as utils
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def assemble_forms(u, w, v, tau, u_old, beta_c):
    # Keep numeric parameters symbolic for UFL form assembly.
    tau_c = fd.Constant(tau)

    a = u * w * fd.dx \
        + tau_c * 0.5 * fd.inner(fd.grad(u),fd.grad(w)) * fd.dx \
        + tau_c * (v * u) * w * fd.dx \
        + tau_c * beta_c * (u_old **2 * u) * w * fd.dx

    rhs = u_old * w * fd.dx
    return a, rhs

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

# h_v = [12 * 2**(-4), 12 * 2**(-6),12 * 2**(-7), 12 * 2**(-8),12 * 2**(-9)]
# Single high-resolution mesh used for the reference quantity.
h_v = [12 * 2**(-8)]
beta = 1000

for h in h_v:
    nx = int((xmax-xmin)/h)
    ny = int((ymax-ymin)/h)

    mesh = fd.RectangleMesh(nx, ny, xmax, ymax, originX = xmin, originY = ymin, quadrilateral = True)# diagonal = 'right')

    # function spaces
    W = fd.FunctionSpace(mesh, 'CG', 1)


    # Potential for test case 2 and homogeneous Dirichlet constraints.
    x = fd.SpatialCoordinate(mesh)
    v = 0.5 * (x[0]**2 + x[1]**2) + fd.Constant(20.0) + fd.Constant(20) * fd.sin( 2 * fd.pi * x[0]) * fd.sin(2 * fd.pi * x[1])
    bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

    tau = 1 #[1., 0.5]

    # Trial/test functions and Thomas-Fermi-inspired initial state.
    u = fd.TrialFunction(W)
    w = fd.TestFunction(W)

    beta_c = fd.Constant(beta)

    mu_TF = fd.sqrt(beta_c / fd.pi)

    u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF- v)/beta_c), 0.0)
    # u0 = fd.Constant(1.)

    u_old = fd.Function(W)
    u_old.interpolate(u0)

    # a, rhs = assemble_forms(u, w, v, tau[0], u_old, beta[0])

    uh = fd.Function(W)

    def energy(uh, v = v, beta = beta):
        return fd.assemble(0.5 *( 0.5 * fd.inner(fd.grad(uh), fd.grad(uh)) + (v * uh )* uh + beta/2 * uh * uh *uh * uh) * fd.dx)


    # Direct solve configuration (MUMPS) for stable reference computations.
    param = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}

    MaxIter = 1000
    toll = 1e-12


    for i in range(MaxIter):
        # One implicit update plus L2 normalization.
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

    # Final reference energy and associated lambda estimate.
    e_gs = energy(uh)
    lamb_gs = 2 * e_gs + beta/2 * fd.norm(uh,'L4')**4

    print()
    print()
    print(f'Final energy estimate: {e_gs} and associated lambda: {lamb_gs} with h: {h} and beta: {beta}') #0.79620688 2.06380
    # ground truth: 15.204825 36.708
    # crossed 15.196860908481353 36.697896895581984
    # left 15.204824828806638 36.70796481994021
    # right 15.20482529108886 36.707967494741474
    # quad 15.197518877570118 36.69868898179708       15.257518727158924 36.77633207829467
    # filename = './Ground_Truth_2/U_GS_b'+str(beta)+'_N'+str(nx)+'Q.h5'
    # utils.save_uh(mesh, uh, filename)

# Plot potential and converged solution for inspection.
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
#plt.title(r'Potential: $V(x,y) = \frac{1}{2}(x^2 + y^2)+ 20 + 20* sin(2 \pi x) sin(2 \pi y)$')
plt.show()
# Plot it
# fig, ax = plt.subplots()
# col = fd.tripcolor(fd.project(v,W), axes=ax, cmap = 'BrBG')
# plt.colorbar(col)
# ax.axis('equal')
# plt.title('potential v(x)')
# fig.savefig("./images/plot_potential_b"+str(beta)+"_N"+str(h_v[-1])+".png")


# Plot the final solution
fig, ax = plt.subplots()
col = fd.tripcolor(uh, axes=ax, cmap='coolwarm')
plt.colorbar(col)
ax.axis('equal')
ax.axis('off')
#plt.title(f'Final Solution ($h = 12 \\times 2^{{{int(np.log2(h/12))}}}$, $\\beta = {beta}$)')
plt.show()
# Plot it
# u2 = fd.Function(W)
# u2.interpolate( uh * uh)

# fig, ax = plt.subplots()
# col = fd.tripcolor(u2, axes=ax, cmap = 'coolwarm')
# print(max(u2.dat.data))
# cbar = plt.colorbar(col, extendrect = 'both')
# cbar.set_ticks(np.linspace(0.0, max(u2.dat.data), 11))

# ax.axis('equal')
# plt.title('Solution')
# fig.savefig("./images/plot_GS_b"+str(beta)+"_N"+str(h_v[-1])+".png")

plt.show()