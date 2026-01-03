import firedrake as fd
# import numpy as np
import csv
import matplotlib.pyplot as plt
import utils
import gradients
# import time

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

# h_v = [12 * 2**(-4), 12 * 2**(-6),12 * 2**(-7), 12 * 2**(-8),12 * 2**(-9)]
h_v = [12 * 2**(-8)]#,12 * 2**(-8)]
beta_v = [10, 100, 1000]
# tau_v = [0.01, 0.005, 0.001]
tau_v = [1, 0.5]

MaxIter = 200
toll = 1e-5

filename_results = './results/test_1_no_GT.csv'#pointwise_itself.csv'


with open(filename_results, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time (quadratic)'])

E_ref = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235}

for h in h_v:
    for beta in beta_v:
        nx = int((xmax-xmin)/h)
        ny = int((ymax-ymin)/h)
        # filename_ref = './Ground_Truth_1/U_GS_b'+str(beta)+'_N'+str(nx)+'.h5'
        # mesh, u_ex = utils.load_ground_truth(filename_ref)
        mesh = fd.RectangleMesh(nx, ny, xmax, ymax, originX = xmin, originY = ymin, quadrilateral = True)# diagonal = 'crossed')
        W = fd.FunctionSpace(mesh, 'CG',1)

        # Data and boundary conditions
        x = fd.SpatialCoordinate(mesh)
        v = 0.5 * (x[0]**2 + x[1]**2)
        bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]
        u0 = 1/fd.pi**(0.5) * fd.exp(-(x[0]**2 + x[1]**2) / 2)

        problem_L2 = gradients.gradient_L2(beta, v, W, bcs, h)
        # problem_L2_e = gradients.gradient_L2_fully_expli(beta, v, W, bcs, h)
        problem_H1 = gradients.gradient_H1(beta, v, W, bcs, h)
        problem_a0 = gradients.gradient_a0(beta, v, W, bcs, h)
        problem_az = gradients.gradient_az(beta, v, W, bcs, h)

        for tau in tau_v:
            # L2 gradient
            problem_L2.assemble_problem(u0, tau, E_ref = E_ref[beta])

            res = problem_L2.minimize(MaxIter, toll)

            problem_L2.save_data(filename_results, 'L2',res)
            # problem_L2.plot_history('L2')

            if res["converged"]:
                print(f'L2 minization with h: {h}, beta: {beta}, tau: {tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            else:
                print(f'L2 minization with h: {h}, beta: {beta}, tau: {tau} did NOT converged in iterate: {res["iterate"]}')


            # #L2 gradient fully explicit
            # problem_L2_e.assemble_problem(u0, tau, u_ex, lump = False)

            # res = problem_L2_e.minimize(MaxIter, toll)

            # problem_L2_e.save_data(filename_results, 'L2_e',res)
            # problem_L2_e.plot_history('L2_e', save = True)
            
            # if res["converged"]:
            #     print(f'fully explicit L2 minization with h: {h}, beta: {beta}, tau:{tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            # else:
            #     print(f'fully explicit L2 minization with h: {h}, beta: {beta}, tau:{tau} did NOT converged in iterate: {res["iterate"]}')
            

            # H1 gradient
            problem_H1.assemble_problem(u0, tau, E_ref = E_ref[beta])

            res = problem_H1.minimize(MaxIter, toll)

            problem_H1.save_data(filename_results, 'H1',res)
            # problem_H1.plot_history('H1')

            if res["converged"]:
                print(f'H1 minization with h: {h}, beta: {beta}, tau: {tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            else:
                print(f'H1 minization with h: {h}, beta: {beta}, tau:{tau} did NOT converged in iterate: {res["iterate"]}')

            # a_0 gradient
            problem_a0.assemble_problem(u0, tau, E_ref = E_ref[beta])

            res = problem_a0.minimize(MaxIter, toll)

            problem_a0.save_data(filename_results, 'a0',res)
            # problem_a0.plot_history('a0')

            if res["converged"]:
                print(f'a_0 minization with h: {h}, beta: {beta}, tau: {tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            else:
                print(f'a_0 minization with h: {h}, beta: {beta}, tau: {tau} did NOT converged in iterate: {res["iterate"]}')

            # az gradient
            problem_az.assemble_problem(u0, tau, E_ref = E_ref[beta])

            res = problem_az.minimize(MaxIter, toll)

            problem_az.save_data(filename_results, 'az',res)
            # problem_az.plot_history('az')

            if res["converged"]:
                print(f'a_z minization with h: {h}, beta: {beta}, tau: {tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            else:
                print(f'a_z minization with h: {h}, beta: {beta}, tau: {tau} did NOT converged in iterate: {res["iterate"]}')

plt.show()