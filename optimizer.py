import abc
import firedrake as fd
import matplotlib.pyplot as plt
import csv
import time
from gradients import gradient_L2, gradient_L2_fully_expli, gradient_H1, gradient_a0, gradient_az

class Optimizer(abc.ABC):
    """
    General class that implements Sobolev Gradient to solve the Gross-Pitaevski equation
    
    Args:
        - beta: parameter that depends on physical properties of the particles that form the BEC
        - v: function that represents an external confining potential
        - W: function space of the solution
        - bcs: list of boundary conditions
        - h: step of spatial discretization
    """
    def __init__(self, name, beta, v, W, bcs, h):
        '''
        Constructor of the class
        
        :param name (string): name of the optimizer choosen
        :param beta (float): parameter that depends on physical properties of the particles that form the BEC
        :param v (function): function that represents an external confining potential
        :param W (fd space): function space of the solution
        :param bcs (list): list of boundary conditions
        :param h (float): step of spatial discretization
        '''

        self.name_optimizer = name

        # pde parameters
        self.beta = beta #fd.Constant(beta)
        self.v = v

        # solution space
        self.W = W
        self.u = fd.TrialFunction(self.W)
        self.w = fd.TestFunction(self.W)

        # boundary conditions
        self.bcs = bcs

        # discretization parameters
        self.h = h

        # solution related params
        self.lam = 0.
        self.E = 0.
        self.uh = fd.Function(self.W)
        self.histoy_E = []



    def energy(self, uh = None):
        '''
        Computes the energy associated to the given solution
        
        :param uh (fd.Function): solution where compute the energy, if not provided computes the solution in the point self.uh
        '''
        if uh is not None:
            E = 0.5 * fd.assemble(( 0.5 * fd.dot(fd.grad(uh), fd.grad(uh)) + self.v * uh**2 + self.beta/2 * abs(uh) **4) * fd.dx)

            return E
        else:
            self.E = fd.assemble(( 0.25 * fd.dot(fd.grad(self.uh), fd.grad(self.uh)) + 0.5 * self.v * self.uh**2 + 0.25 * self.beta * abs(self.uh) **4 )* fd.dx)

    def compute_lambda(self):
        '''
        Compute the lambda associated to the energy of self.uh
        '''

        self.lam = 2 * self.E + self.beta /2 * fd.norm(self.uh,'L4')**4
        
        return self.lam
    

    def get_solver(self, grad_type):
        '''
        '''

        if grad_type in ['L2', 'l2', 'L_2']:
            return gradient_L2(self.W, self.bcs, self.h, self.beta, self.v)

        elif grad_type in ['L2e', 'l2e', 'L_2e']:
            return gradient_L2_fully_expli(self.W, self.bcs, self.h, self.beta, self.v)

        elif grad_type in ['H1', 'h1']:
            return gradient_H1(self.W, self.bcs, self.h, self.beta, self.v)

        elif grad_type in ['a0', 'a_0']:
            return gradient_a0(self.W, self.bcs, self.h, self.beta, self.v)

        elif grad_type in ['az', 'a_z']:
            return gradient_az(self.W, self.bcs, self.h, self.beta, self.v)
            
        else:
            NotImplementedError('type of Sobolev gradient not implemented. Supported one are: L2, L2expl, H1, a0, az')
    
    def compile(self, u0, E_ref):
        '''
        Inizialize and assembles all forms related to the minimization
        
        :param u0 (fd.Function): initial guess
        :param u_ref (fd.Function): reference solution
        '''

        self.u_old = fd.Function(self.W)

        norm = fd.norm(u0,'L2')
        self.u_old.interpolate(u0/norm)
        
        self.E_ref = E_ref

        self.E = 0.
        self.lam = 0.
        self.histoy_E = []

    @abc.abstractmethod
    def minimize(self, MaxIter, toll):
        pass


    def plot_history(self, show = False, save = False):
        '''
        Plot the convergence history of the minimization

        :param method_name (string): name of the gradient method used
        '''

        fig, ax = plt.subplots(2,1, figsize=(5,10))
        fig.suptitle(f'{self.method_name}_gradient with h={self.h}, beta={self.beta }, tau={self.tau }')
        ax[0].semilogy(range(1,len(self.histoy_E)+1), [abs(E - self.E_ref)/self.E_ref for E in self.histoy_E], marker='o')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Relative Error on Energy')
        ax[0].set_title('Convergence History')
        ax[0].grid(True)

        ax[1].semilogy(range(1,len(self.histoy_E)+1), self.histoy_E, marker='o')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Energy')
        ax[1].set_title('Energy History')
        ax[1].grid(True)
        if show:
            plt.show()
        if save:
            fig.savefig("./images/plot_b"+str(self.beta )+"_N"+str(int(1/self.h))+"_tau"+str(self.tau )+"_it"+str(self.MaxIter)+"_no_lump.png")


    def save_data(self, filename, opt_name, res):
        '''
        Save minimization result in a csv file

        :param filename (string): path to the csv file 
        :param opt_name (string): specify which gradient has been used
        :param res (dict): dictionary taht contains all the important data
        '''
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([opt_name, self.h, self.beta , self.tau , res["energy"], res["lam"], res["iterate"], res["error"], res["time_tot"], res["mean_time"]])



class Gradient_Descent(Optimizer):
    def __init__(self, beta, v, W, bcs, h):
        '''
        Constructor of the class
        
        :param beta (float): parameter that depends on physical properties of the particles that form the BEC
        :param v (function): function that represents an external confining potential
        :param W (fd space): function space of the solution
        :param bcs (list): list of boundary conditions
        :param h (float): step of spatial discretization
        '''
        super().__init__('GD',beta, v, W, bcs, h)
        
    def compile(self, u0, tau, E_ref, grad_type):

        super().compile(u0, E_ref)

        self.solver = self.get_solver(grad_type)
        
        self.solver.assemble_problem(tau)


    def minimize(self, MaxIter, toll, verbose=True):
        '''
        Minimize the functional applying the step method iteratively
        
        :param MaxIter (int): maximum number of iteration allowed
        :param toll (float): tollerance of the stopping criteria
        :param verbose (bool): specify if print results or not (default True)  
        '''
        converged = False
        self.MaxIter = MaxIter


        t_start = time.time()

        for i in range(MaxIter):
            # compute new soltution
            self.solver.step(self.u_old)

            self.uh.assign(self.solver.uh)

            self.energy()

            self.histoy_E.append(self.E)

            # calculate the error
            error = abs(self.E - self.E_ref) / self.E_ref

            self.u_old.assign(self.solver.uh)

            if verbose:
                print(f'Iter {i}, Error: {error:.6e}, Energy: {self.E:.10f} and lambda: {self.compute_lambda():.6f}')

            if error <= toll:
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
                   error = error,
                   norm = fd.norm(self.uh,'L2'),
                   time_tot = time_tot,
                   mean_time = time_tot/(i+1))
        
        if verbose:
            # print('\r', end="", flush=True)
            # print('\r', end="", flush=True)
            if converged:
                print(f'{self.name_optimizer} minimization using {self.solver.name} gradient converged at iteration {i+1} at energy {self.E:.6f} and lambda {self.lam:.6f}')
            else:
                print(f'{self.name_optimizer} minimization using {self.solver.name} gradient NOT converged in {i+1} iterate')

        return res

class ParaflowS(Optimizer):
    def __init__(self, beta, v, W, bcs, h):
        '''
        Constructor of the class
        
        :param beta (float): parameter that depends on physical properties of the particles that form the BEC
        :param v (function): function that represents an external confining potential
        :param W (fd space): function space of the solution
        :param bcs (list): list of boundary conditions
        :param h (float): step of spatial discretization
        '''
        super().__init__('ParaflowS',beta, v, W, bcs, h)

    def compile(self, u0, tau, E_ref, grad_type_coarse, grad_type_fine, Nf, Ng = 100):
        '''
        Docstring for compile
        
        :param self: Description
        :param u0: Description
        :param tau: Description
        :param E_ref: Description
        :param grad_type_coarse: Description
        :param grad_type_fine: Description
        :param Nf: Description
        :param Nc: Description
        '''

        super().compile(u0, E_ref)

        self.coarse_solver = self.get_solver(grad_type_coarse)
        self.coarse_solver.assemble_problem(tau * Nf)
        self.Ng = Ng

        self.fine_solver = self.get_solver(grad_type_fine)
        self.fine_solver.assemble_problem(tau)
        self.Nf = Nf

        self.correction_uh = fd.Function(self.W)

    def minimize(self, MaxIter, toll, verbose=True):
        '''
        Minimize the functional applying the step method iteratively
        
        :param MaxIter (int): maximum number of iteration allowed
        :param toll (float): tollerance of the stopping criteria
        :param verbose (bool): specify if print results or not (default True)  
        '''

        self.MaxIter = MaxIter

        converged = False
        N_iter_fine = 0
        N_iter_coarse = 0


        t_start = time.time()

        for i in range(MaxIter):
            energy_old = self.energy(self.u_old)
            print(f'Entry error: {abs(energy_old - self.E_ref) / self.E_ref} ')

            self.coarse_solver.step(self.u_old)
            N_iter_coarse +=1

            self.fine_solver.uh = self.u_old
            for _ in range(self.Nf):
                self.fine_solver.step(self.fine_solver.uh)
                print(f'fine error: {abs(self.energy(self.fine_solver.uh)- self.E_ref) / self.E_ref}')
                N_iter_fine += 1 

            self.correction_uh.assign(self.fine_solver.uh - self.coarse_solver.uh) # not necessary, could be done also self.fine_solver.uh.assign(self.fine_solver.uh - self.coarse_solver.uh)

            for j in range(self.Ng):
                self.coarse_solver.step(self.u_old)
                N_iter_coarse +=1

                self.uh.assign(self.coarse_solver.uh + self.correction_uh)

                # normalize
                self.uh.assign(self.uh / fd.norm(self.uh,'L2'))

                print(f'    {fd.norm(self.uh)}')


                self.energy()

                error = abs(self.E - self.E_ref) / self.E_ref

                if verbose:
                    print(f'    coarse iter {j}, Energy: {self.E:.10f} and error {error:.5f}')


                if self.E > energy_old or error < toll:
                    print(f'    Exiting because energy: {self.E > energy_old} and error: {error < toll}')
                    break

                energy_old = self.E
                self.u_old.assign(self.uh)

            if verbose:
                print(f'Iter {i}, Error: {error:.6e}, Energy: {self.E:.10f} and lambda: {self.compute_lambda():.6f}')

            if error < toll:
                converged = True
                break


        t_end = time.time()

        print(f'ParaflowS ended in n_coarse calls: {N_iter_coarse} and N-fine calls {N_iter_fine}')

        
            
