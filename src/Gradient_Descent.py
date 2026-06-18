import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from pathlib import Path
from Optimizer import Optimizer

class Gradient_Descent(Optimizer):
    '''
    This class implements the gradient descent optimization algorithm for the minimization of the energy functional.
    '''

    def __init__(self, W, bcs, beta, v, grad_type_name, type_discr):
        '''
        Constructor for the gradient descent optimizer.
        
        :param W (fd space): The function space for the problem.
        :param beta (float): The beta parameter for the problem.
        :param v (fd function): The potential function for the problem.
        :param grad_type_name (str): A string identifier for the type of gradient method used.
        :param type_discr (str): A string identifier for the type of discretization used.
        '''
        
        super().__init__(W, beta, v, 'Gradient_Descent')

        self.grad_type_name = grad_type_name
        self.type_discr = type_discr

        # attach the solver for the gradient flow
        self.solver = self.get_solver(W, bcs, beta, v, grad_type_name, type_discr)

    def compile(self, u0, E_ref, tau = None):
        '''
        Compile the forms and prepare the problem for the gradient flow minimization.

        :param u0 (fd function): initial guess for the optimization algorithm.
        :param E_ref (float): reference energy to compute the relative error.
        :param tau (float): time step for the gradient flow. If None, an adaptive time step will be used.
        '''
        super().compile(u0, E_ref)

        self.solver.assemble_problem(tau)

    def minimize(self, MaxIter, toll, verbose = True, save_history = True):
        '''
        Perform the gradient descent optimization.

        :param MaxIter (int): maximum number of iterations.
        :param toll (float): tolerance for the stopping criterion.
        :param verbose (bool): whether to print the energy and the relative error at each iteration.
        :param save_history (bool): whether to save the history of the energy and the relative error.
        '''
        converged = False
        self.MaxIter = MaxIter

        t_start = time.time()

        for i in range(MaxIter):
            # compute the gradient
            self.E = self.solver.step(self.u_old)

            # normalize
            self.solver.uh.assign(self.solver.uh / fd.norm(self.solver.uh,'L2'))

            self.uh.assign(self.solver.uh)

            # update the solution
            self.u_old.assign(self.solver.uh)

            # compute the energy and the relative error
            self.E = self.energy()
            rel_error = abs(self.E - self.E_ref) / abs(self.E_ref)

            if verbose:
                print(f"\rIteration {i}, Energy: {self.E:.6e}, Relative Error: {rel_error:.6e}, Time: {time.time() - t_start:.2f} s", end="", flush=True)

            if save_history:
                self.history_E.append(self.E)

            # check the stopping criterion
            if rel_error < toll:
                converged = True
                break


        time_tot = time.time() - t_start

        # compute the final quantities
        self.energy()
        self.compute_lambda()

        res = dict(converged = converged,
                   energy = self.E,
                   lam = self.lam,
                   iterate = i+1,
                   error = rel_error,
                   norm = fd.norm(self.uh,'L2'),
                   time_tot = time_tot,
                   mean_time = time_tot/(i+1))
        
        if verbose:
            # print('\r', end="", flush=True)
            print('\r', end="", flush=True)
            if converged:
                print(f'{self.name_optimizer} minimization using {self.solver.grad_type_name}-{self.solver.type_discr} gradient with tau = {float(self.solver.tau)} converged at iteration {i+1} at energy {self.E:.6f} and lambda {self.lam:.6f}')
            else:
                print(f'{self.name_optimizer} minimization using {self.solver.grad_type_name}-{self.solver.type_discr} gradient NOT converged in {i+1} iterate. Final relative error: {rel_error:.6e} with tau = {float(self.solver.tau)}.')

        return res


    def plot_history(self, show = False, filesave = None):
        '''
        Plot the convergence history of the minimization

        :param method_name (string): name of the gradient method used
        '''

        fig, ax = plt.subplots(2,1, figsize=(5,10))
        fig.suptitle(f'{self.name_optimizer} with {self.solver.grad_type_name}-{self.solver.type_discr}, beta = {float(self.solver.beta)}, tau = {float(self.solver.tau) }')
        ax[0].semilogy(range(1,len(self.history_E)+1), [abs(E - self.E_ref)/self.E_ref for E in self.history_E], marker='o')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Relative Error on Energy')
        ax[0].set_title('Convergence History')
        ax[0].grid(True)

        ax[1].semilogy(range(1,len(self.history_E)+1), self.history_E, marker='o')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Energy')
        ax[1].set_title('Energy History')
        ax[1].grid(True)

        if show:
            plt.show()
        if filesave is not None:
            fig.savefig(filesave)
            plt.close(fig)

        if hasattr(self.solver, 'tau_history') and self.solver.tau_history is not None:
            plt.figure()
            plt.plot(range(1,len(self.solver.tau_history)+1), self.solver.tau_history, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Tau value')
            plt.title('Tau history')
            plt.grid(True)
            if show:
                plt.show()
            if filesave is not None:
                tau_filesave = Path(filesave).with_name(Path(filesave).stem + '_tau.png')
                plt.savefig(tau_filesave)
                plt.close()

    def save_data(self, filename, res, test_name):
        '''
        Save minimization result in a csv file

        :param filename (string): path to the csv file 
        :param opt_name (string): specify which gradient has been used
        :param res (dict): dictionary that contains all the important data
        '''
        name = self.solver.grad_type_name + '_' + self.solver.type_discr
        if self.solver.adaptivity:
            name += '_ada'
            tau = float(np.mean(np.array(self.solver.tau_history)))
        else:
            tau = float(self.solver.tau)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, 12/np.sqrt(self.solver.W.dim()), test_name, tau, res["energy"], res["lam"], res["iterate"], res["error"], res["time_tot"], res["mean_time"]])
