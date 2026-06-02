from curses import error

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from Optimizer import Optimizer


class ParaFlowS(Optimizer):
    '''
    This class implements the ParaFlowS optimization algorithm for the minimization of the energy functional.
    '''

    def __init__(self, W, bcs, beta, v, fine_type_name, fine_type_discr, coarse_type_name, coarse_type_discr):
        '''
        Constructor for the ParaFlowS optimizer.

        :param W (fd space): The function space for the problem.
        :param beta (float): The beta parameter for the problem.
        :param v (fd function): The potential function for the problem.
        :param fine_type_name (str): A string identifier for the type of gradient method used on the fine propagator.
        :param fine_type_discr (str): A string identifier for the type of discretization used on the fine propagator.
        :param coarse_type_name (str): A string identifier for the type of gradient method used on the coarse propagator.
        :param coarse_type_discr (str): A string identifier for the type of discretization used on the coarse propagator.
        '''

        super().__init__(W, beta, v, 'ParaFlowS')

        # attach the solver for the fine propagators
        self.fine_type_name = fine_type_name
        self.fine_type_discr = fine_type_discr
        self.fine_solver = self.get_solver(W, bcs, beta, v, fine_type_name, fine_type_discr)

        # attach the solver for the coarse propagators
        self.coarse_type_name = coarse_type_name
        self.coarse_type_discr = coarse_type_discr
        self.coarse_solver = self.get_solver(W, bcs, beta, v, coarse_type_name, coarse_type_discr)

    def compile(self, u0, E_ref, Nf, Nc = 10, tau_f = None, tau_c = None):
        '''
        Compile the forms and prepare the problem for the ParaFlowS minimization.

        :param u0 (fd function): initial guess for the optimization algorithm.
        :param E_ref (float): reference energy to compute the relative error.
        :param Nf (int): number of fine steps to perform at each iteration of the Parareal algorithm.
        :param Nc (int): maximum number of coarse steps to perform at each iteration of the Parareal algorithm. Default is 100.
        :param tau_f (float): time step for the fine propagator. If None, an adaptive time step will be used.
        :param tau_c (float): time step for the coarse propagator. If None, an adaptive time step will be used.
        '''
        super().compile(u0, E_ref)

        self.Nf = Nf
        self.Nc = Nc

        self.fine_solver.assemble_problem(tau_f)
        self.coarse_solver.assemble_problem(tau_c)

        self.correction_uh = fd.Function(self.W)

    def minimize(self, MaxIter, toll, verbose = True, save_history = True, debug = False):
        '''
        Minimize the functional applying the step method iteratively
        
        :param MaxIter (int): maximum number of iteration allowed
        :param toll (float): tollerance of the stopping criteria
        :param verbose (bool): specify if print results or not (default True)  
        :param save_history (bool): whether to save the history of the energy and the relative error. (default True)
        '''

        converged = False
        self.MaxIter = MaxIter

        N_iter_fine = 0
        N_iter_coarse = 0


        t_start = time.time()

        for i in range(MaxIter):
            if save_history:
                self.history_fine = []
            
            # Initial coarse prediction from current iterate.
            energy_old = self.coarse_solver.step(self.u_old)

            N_iter_coarse +=1
            if debug:
                print(f'Entry error: {abs(energy_old - self.E_ref) / self.E_ref} ')

            self.fine_solver.uh.assign(self.u_old)
            alpha = 0.0
            for _ in range(self.Nf):
                # Fine phase: evolve Nf times with normalization at each step.
                self.fine_solver.uh.assign(self.fine_solver.uh / fd.norm(self.fine_solver.uh,'L2'))

                self.fine_solver.step(self.fine_solver.uh)
                energy_new = self.energy(self.fine_solver.uh/ fd.norm(self.fine_solver.uh,'L2'))

                if debug:
                    print(f'fine energy: {energy_new} fine error: {abs(energy_new - self.E_ref) / self.E_ref}')

                if save_history:
                    self.history_fine.append(energy_new)
                
                alpha += energy_new/energy_old

                energy_old = energy_new
                N_iter_fine += 1 


            # Correction direction = fine prediction - coarse prediction.
            self.correction_uh.assign(self.fine_solver.uh - self.coarse_solver.uh) # not necessary, could be done also self.fine_solver.uh.assign(self.fine_solver.uh - self.coarse_solver.uh)

            alpha /= self.Nf
            # Safety factor for the acceptance test in the coarse correction loop.
            alpha = min(1.0, alpha + 0.01)
            # print(f'Alpha value: {alpha}')

            for j in range(self.Nc):

                # Apply correction on top of current coarse state.
                self.uh.assign(self.coarse_solver.uh + self.correction_uh)

                # normalize
                self.uh.assign(self.uh / fd.norm(self.uh,'L2'))

                self.E = self.energy()
                rel_error = abs(self.E - self.E_ref) / self.E_ref

                if save_history:
                    self.history_fine.append(self.E)

                if debug:
                    print(f'    coarse iter {j}, Energy: {self.E:.10f} and error {rel_error:.6e}, energy correction: {self.energy(self.correction_uh / fd.norm(self.correction_uh, "L2"))}, old energy: {energy_old}')


                # if self.E > energy_old and j >= 1: 
                # Reject correction if energy grows too much.
                if self.E > alpha * energy_old and j >= 1:
                    if debug:
                        print('     Exiting energy grow up')
                    self.uh.assign(self.u_old)
                    break
                # Accept and terminate if target tolerance is reached.
                if rel_error < toll:
                    if debug:
                        print('     Exiting because the error is small enough')
                    break

                energy_old = self.E
                self.u_old.assign(self.uh)
                
                # Advance coarse dynamics after accepted correction.
                self.coarse_solver.step(self.u_old)
                N_iter_coarse +=1

            self.E = self.energy()

            rel_error = abs(self.E - self.E_ref) / self.E_ref
            if verbose:
                print(f'\rIter {i}, Error: {rel_error:.6e}, Energy: {self.E:.10f} and lambda: {self.compute_lambda():.6f}', end="", flush=True)
            if debug:
                print(f'Iter {i}, Error: {rel_error:.6e}, Energy: {self.E:.10f} and lambda: {self.compute_lambda():.6f}')

            if save_history:
                self.history_E.append(self.history_fine)

            if rel_error < toll:
                converged = True
                break

        time_tot = time.time() - t_start

        # compute the final quantities
        self.E = self.energy()
        self.lam = self.compute_lambda()

        res = dict(converged = converged,
                   energy = self.E,
                   lam = self.lam,
                   iterate_fine = N_iter_fine,
                   iterate_coarse = N_iter_coarse,
                   iterate = i+1,
                   error = rel_error,
                   norm = fd.norm(self.uh,'L2'),
                   time_tot = time_tot,
                   mean_time = time_tot/(i+1))
        
        if verbose:
            print('\r', end="", flush=True)

        if verbose or debug:
            if converged:
                print(f'{self.name_optimizer} minimization using coarse: {self.coarse_solver.grad_type_name}-{self.coarse_solver.type_discr} and fine: {self.fine_solver.grad_type_name}-{self.fine_solver.type_discr} gradient')
                print(f'    with tau_f = {float(self.fine_solver.tau)}, tau_c = {float(self.coarse_solver.tau)} converged in n_coarse calls: {N_iter_coarse} and N-fine calls {N_iter_fine} at energy {self.E:.6f} and lambda {self.lam:.6f}')
            else:
                print(f'{self.name_optimizer} minimization using coarse:  {self.coarse_solver.grad_type_name}-{self.coarse_solver.type_discr} and fine: {self.fine_solver.grad_type_name}-{self.fine_solver.type_discr} gradient')
                print(f'     with tau_f = {float(self.fine_solver.tau)}, tau_c = {float(self.coarse_solver.tau)} NOT converged in n_coarse calls: {N_iter_coarse} and N-fine calls {N_iter_fine}')
        
        return res


    def plot_history(self, show = False, filesave = None):
        '''
        Plot the convergence history of the minimization

        :param method_name (string): name of the gradient method used
        '''

        fig, ax = plt.subplots(1,1, figsize=(10,10))
        fig.suptitle(f'{self.name_optimizer} with fine:{self.fine_solver.name} and coarse: {self.coarse_solver.name},\n h={self.h}, beta={float(self.beta)}, tau_f={float(self.fine_solver.tau)}, tau_c={float(self.coarse_solver.tau)}')

        offset = 0
        for element in self.history_E:
            if offset == 0:
                times = np.array(range(0,self.Nf)) * float(self.fine_solver.tau) + offset
                ax.semilogy(times, [abs(element[i] - self.E_ref) / self.E_ref for i in range(self.Nf)], marker='o', linewidth = 5, markersize = 10, c = '#e67e22', label = 'Fine solver')
                #ax[1].semilogy(times, [element[i] for i in range(self.Nf)], marker='o', c = '#e67e22', label = 'Fine solver')
                offset += (self.Nf - 1) * float(self.fine_solver.tau)
                ax.semilogy(np.array(range(0,len(element) - self.Nf )) * float(self.coarse_solver.tau) + offset, [abs(element[i] - self.E_ref) / self.E_ref for i in range(self.Nf, len(element))], marker='o', linewidth = 5, markersize = 10, c = '#2c3e50', label = 'Correction')
                #ax[1].semilogy(np.array(range(0,len(element) - self.Nf )) * float(self.coarse_solver.tau) + offset, [element[i] for i in range(self.Nf, len(element))], marker='x', c = '#2c3e50', label = 'Correction')
                offset += (len(element) - self.Nf - 1) * float(self.coarse_solver.tau) + float(self.fine_solver.tau)
            else:
                times = np.array(range(0,self.Nf)) * float(self.fine_solver.tau) + offset
                ax.semilogy(times, [abs(element[i] - self.E_ref) / self.E_ref for i in range(self.Nf)], marker='o', linewidth = 5, markersize = 10, c = '#e67e22')
                #ax[1].semilogy(times, [element[i] for i in range(self.Nf)], marker='o', c = '#e67e22')
                offset += (self.Nf - 1) * float(self.fine_solver.tau)
                ax.semilogy(np.array(range(0,len(element) - self.Nf )) * float(self.coarse_solver.tau) + offset, [abs(element[i] - self.E_ref) / self.E_ref for i in range(self.Nf, len(element))], marker='o', linewidth = 5, markersize = 10, c = '#2c3e50')
                #ax[1].semilogy(np.array(range(0,len(element) - self.Nf )) * float(self.coarse_solver.tau) + offset, [element[i] for i in range(self.Nf, len(element))], marker='x', c = '#2c3e50')
                offset += (len(element) - self.Nf - 1) * float(self.coarse_solver.tau) + float(self.fine_solver.tau)
        ax.set_xlabel('Iteration', fontsize = 30)
        ax.set_ylabel('Relative Error on Energy', fontsize = 30)
        ax.set_title('Convergence History', fontsize = 30)
        ax.grid(True)
        ax.legend()

        # ax[1].set_xlabel('Iteration')
        # ax[1].set_ylabel('Energy')
        # ax[1].set_title('Energy History')
        # ax[1].grid(True)
        # ax[1].legend()

        if show:
            plt.show()
        if filesave is not None:
            fig.savefig(filesave)
            plt.close(fig)

    def save_data(self, filename, res, test_name):
        '''
        Save minimization result in a csv file

        :param filename (string): path to the csv file 
        :param opt_name (string): specify which gradient has been used
        :param res (dict): dictionary that contains all the important data
        '''
        self.gamma = 1.0

        name_f = self.fine_solver.grad_type_name + '_' + self.fine_solver.type_discr
        if self.fine_solver.adaptivity:
            name_f += '_ada'
            tau_f = float(np.mean(np.array(self.fine_solver.tau_history)))
        else:
            tau_f = float(self.fine_solver.tau)

        name_c = self.coarse_solver.grad_type_name + '_' + self.coarse_solver.type_discr
        if self.coarse_solver.adaptivity:
            name_c += '_ada'
            tau_c = float(np.mean(np.array(self.coarse_solver.tau_history)))
        else:
            tau_c = float(self.coarse_solver.tau)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name_f, name_c, self.gamma, 12/np.sqrt(self.fine_solver.W.dim()), test_name, self.Nf, self.Nc, tau_f, tau_c, res["energy"], res["lam"], res["iterate_coarse"], res["iterate_fine"], res["iterate"], res["error"], res["time_tot"], res["mean_time"]])