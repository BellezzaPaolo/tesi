import abc
import firedrake as fd
import matplotlib.pyplot as plt
import csv
import time

class gradient(abc.ABC):
    """
    General class that implements Sobolev Gradient to solve the Gross-Pitaevski equation
    
    Args:
        - beta: parameter that depends on physical properties of the particles that form the BEC
        - v: function that represents an external confining potential
        - W: function space of the solution
        - bcs: list of boundary conditions
        - h: step of spatial discretization
    """
    def __init__(self, beta, v, W, bcs, h):
        '''
        Constructor of the class
        
        :param beta (float): parameter that depends on physical properties of the particles that form the BEC
        :param v (function): function that represents an external confining potential
        :param W (fd space): function space of the solution
        :param bcs (list): list of boundary conditions
        :param h (float): step of spatial discretization
        '''
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
            self.E = 0.5 * fd.assemble(( 0.5 * fd.dot(fd.grad(self.uh), fd.grad(self.uh)) + self.v * self.uh**2) * fd.dx + self.beta/2 * abs(self.uh) **4 * fd.dx)

    def compute_lambda(self):
        '''
        Compute the lambda associated to the energy of self.uh
        '''
        self.energy()

        self.lam = 2 * self.E + self.beta /2 * fd.norm(self.uh,'L4')**4
        
        return self.lam
    
    def assemble_problem(self, u0, tau, u_ref = None, E_ref = None):
        '''
        Inizialize and assembles all forms related to the minimization
        
        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        self.tau = fd.Constant(tau)

        self.u_old = fd.Function(self.W)

        norm = fd.norm(u0,'L2')
        self.u_old.interpolate(u0/norm)

        self.non_lin_coefficient = self.beta * abs(u0)**2
        
        if u_ref is not None and E_ref is None:
            self.E_ref = self.energy(u_ref)
        elif E_ref is not None and u_ref is None:
            self.E_ref = E_ref
        else:
            print('Warning: reference energy not known, given values of E_ref and u_ref not compatible.' )

        self.E = 0.
        self.lam = 0.
        self.histoy_E = []


    @abc.abstractmethod
    def step(self):
        '''
        Step of the minimization, based on the choosen class wil be overridden
        '''
        pass

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
            self.step()

            # normalize
            self.uh.assign(self.uh / fd.norm(self.uh,'L2'))

            self.energy()

            self.histoy_E.append(self.E)

            # calculate the error
            error = abs(self.E - self.E_ref) / self.E_ref

            self.u_old.assign(self.uh)

            if verbose:
                print(f'\rIter {i}, Error: {error:.6e}, Energy: {self.E:.10f} and lambda: {self.compute_lambda():.6f}', end="", flush=True)

            if error <= toll:
                converged = True
                break
        
        time_tot = time.time() - t_start

        # compute the final quantities
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
            print('\r', end="", flush=True)

        return res

    def plot_history(self, method_name, show = False, save = False):
        '''
        Plot the convergence history of the minimization

        :param method_name (string): name of the gradient method used'''
        fig, ax = plt.subplots(2,1, figsize=(5,10))
        fig.suptitle(f'{method_name}_gradient with h={self.h}, beta={self.beta }, tau={self.tau }')
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


class gradient_L2_fully_expli(gradient):
    def assemble_problem(self, u0, tau, u_ref = None, E_ref = None, lump = True):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        super().assemble_problem(u0, tau, u_ref, E_ref)

        if lump:
            self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs, form_compiler_parameters={"quadrature_rule": "KMV","quadrature_degree": self.W.ufl_element().degree()})

            self.solver_Mass = fd.LinearSolver(self.M)
        else:
            self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs)
            self.solver_Mass = fd.LinearSolver(self.M, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # used only to compute the LU factorization
        self.solver_Mass.solve(self.uh, fd.assemble(self.u_old * self.w * fd.dx))

        
    def step(self):
        '''
        Implements the step of L2 gradient
        '''

        intE = fd.assemble(0.5 * fd.dot(fd.grad(self.u_old), fd.grad(self.u_old)) * fd.dx + self.v * self.u_old * self.u_old * fd.dx + self.beta * abs(self.u_old)**2 * self.u_old * self.u_old * fd.dx)
        intR = fd.assemble(self.u_old * self.u_old * fd.dx)

        rhs = fd.assemble( self.u_old * self.w * fd.dx \
            - self.tau * 0.5 * fd.dot(fd.grad(self.u_old), fd.grad(self.w)) * fd.dx \
            - self.tau * self.v * self.u_old * self.w * fd.dx \
            - self.tau * self.beta * (self.u_old ** 2 * self.u_old) * self.w * fd.dx \
            + self.tau * intE/intR * self.u_old * self.w * fd.dx)
            
        self.solver_Mass.solve(self.uh, rhs)
        


class gradient_L2(gradient):
    def assemble_problem(self, u0, tau, u_ref = None, E_ref = None):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        super().assemble_problem(u0, tau, u_ref, E_ref)

        self.a = self.u * self.w * fd.dx \
            + self.tau * 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
            + self.tau * self.v * self.u * self.w * fd.dx
        
    def step(self):
        '''
        Implements the step of L2 gradient
        '''
        rhs = self.u_old * self.w * fd.dx 
        
        problem = fd.LinearVariationalProblem(self.a + self.tau * self.beta * (self.u_old **2 * self.u) * self.w * fd.dx,
                                                rhs,
                                                self.uh,
                                                self.bcs)
        solver =  fd.LinearVariationalSolver(problem)#, solver_parameters=param)
        solver.solve()


class gradient_H1(gradient):
    def assemble_problem(self, u0, tau, u_ref = None, E_ref = None):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''

        super().assemble_problem(u0, tau, u_ref, E_ref)

        self.A = fd.assemble(0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx, bcs = self.bcs)

        # assemble the solver for the Riesz rappresentation and gradient
        self.R_u = fd.Function(self.W)
        self.gradE = fd.Function(self.W)
        self.solver_Stiffnes = fd.LinearSolver(self.A, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # used only to compute the LU factorization
        self.solver_Stiffnes.solve(self.R_u, fd.assemble(self.u_old * self.w * fd.dx))

        # # assemble the solver for the gradient
        # self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs)
        # self.solver_Mass = fd.LinearSolver(self.M, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    def step(self):
        '''
        Implements a step of the gradient H1
        '''
        # compute the Riesz projection
        rhs_R = fd.assemble(self.u_old * self.w * fd.dx)

        self.solver_Stiffnes.solve(self.R_u, rhs_R)
        
        # compute the gradient
        rhs_E = fd.assemble(0.5 * fd.dot(fd.grad(self.u_old), fd.grad(self.w)) * fd.dx \
                + self.v * self.u_old * self.w * fd.dx \
                + self.beta * abs(self.u_old) **2 * self.u_old * self.w * fd.dx)

        self.solver_Stiffnes.solve(self.gradE, rhs_E)

        # compute the solution

        intE = fd.assemble(self.gradE * self.u_old * fd.dx)
        intR = fd.assemble(self.R_u * self.u_old * fd.dx)

        # rhs_u = fd.assemble(self.u_old * self.w * fd.dx \
        #         - self.tau * self.gradE * self.w * fd.dx \
        #         + self.tau * intE/intR * self.R_u * self.w * fd.dx)
        
        # self.solver_Mass.solve(self.uh, rhs_u)
        self.uh.assign(self.u_old - self.tau * self.gradE + self.tau * intE/intR * self.R_u)

            
class gradient_a0(gradient):
    def assemble_problem(self, u0, tau, u_ref = None, E_ref = None):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        super().assemble_problem(u0, tau, u_ref, E_ref)

        # assemble the solver for Riesz projections
        self.R_u = fd.Function(self.W)
        self.R_u2u = fd.Function(self.W)

        self.A = fd.assemble(0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx\
                                + self.v * self.u * self.w * fd.dx, bcs = self.bcs)

        self.solver_Stiffness = fd.LinearSolver(self.A, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # used only to compute the LU factorization
        self.solver_Stiffness.solve(self.R_u, fd.assemble(self.u_old * self.w * fd.dx))

        # assemble the solver for the solution
        # self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs)

        # self.solver_Mass = fd.LinearSolver(self.M, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})


    def step(self):
        '''
        Impelements one step of the a_0 gradient 
        '''
        # compute reisz prjections
        rhs_Ru = fd.assemble(self.u_old * self.w * fd.dx)
        self.solver_Stiffness.solve(self.R_u, rhs_Ru)

        rhs_Ru2u = fd.assemble(self.beta * abs(self.u_old) ** 2 * self.u_old * self.w * fd.dx)

        self.solver_Stiffness.solve(self.R_u2u,rhs_Ru2u)

        # compute the solution
        intE = fd.assemble((self.u_old + self.R_u2u) * self.u_old * fd.dx)
        intR = fd.assemble(self.R_u * self.u_old * fd.dx)

        # rhs_u = fd.assemble(self.u_old * self.w * fd.dx \
        #         - self.tau * (self.u_old + self.R_u2u) * self.w * fd.dx \
        #         + self.tau * intE/intR * self.R_u * self.w * fd.dx)
        
        # self.solver_Mass.solve(self.uh, rhs_u)
        self.uh.assign((1 - self.tau) * self.u_old  - self.tau * self.R_u2u + self.tau * intE/intR * self.R_u)

class gradient_az(gradient):
    def assemble_problem(self, u0, tau, u_ref = None, E_ref = None):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        super().assemble_problem(u0, tau, u_ref, E_ref)

        # initialize the forms for the Riesz solver
        self.R_u = fd.Function(self.W)

        self.a = 0.5 * fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
                + self.v * self.u * self.w * fd.dx

        # assemble ths solver for the solution
        # self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs)

        # self.solver_Mass = fd.LinearSolver(self.M, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        
    
    def step(self):
        '''
        Implements one step of the a_z gradient
        '''
        # compute Riesz
        rhs_R = self.u_old * self.w * fd.dx

        # for i in range(self.u_old.dat.data.shape[0]):
        #     self.u2.dat.data[i] = abs(self.u_old.dat.data[i]) ** 2
        self.non_lin_coefficient = self.beta * abs(self.u_old)**2

        problem_R = fd.LinearVariationalProblem(self.a + self.non_lin_coefficient * self.u * self.w * fd.dx,
                                                rhs_R,
                                                self.R_u,
                                                self.bcs)
        solver_R = fd.LinearVariationalSolver(problem_R)
        solver_R.solve()

        # compute solution
        intE = fd.assemble(self.u_old * self.u_old * fd.dx)
        intR = fd.assemble(self.R_u * self.u_old * fd.dx)

        # rhs_u = fd.assemble((1 - self.tau) * self.u_old * self.w * fd.dx \
        #         + self.tau * intE/intR * self.R_u * self.w * fd.dx)
        
        # self.solver_Mass.solve(self.uh, rhs_u)

        self.uh.assign((1 - self.tau) * self.u_old + self.tau * intE/intR * self.R_u)