import abc
import firedrake as fd

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
        self.beta = fd.Constant(beta)
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


    def energy(self, uh = None):
        '''
        Computes the energy associated to the given solution
        
        :param uh (fd.Function): solution where compute the energy, if not provided computes the solution in the point self.uh
        '''
        if uh is not None:
            E = 0.5 * fd.assemble(( 0.5 * fd.dot(fd.grad(uh), fd.grad(uh)) + self.v * uh**2 + self.beta/2 * abs(uh) **4) * fd.dx)

            return E
        else:
            self.E = 0.5 * fd.assemble(( 0.5 * fd.dot(fd.grad(self.uh), fd.grad(self.uh)) + self.v * self.uh**2 + self.beta/2 * abs(self.uh) **4) * fd.dx)

    def compute_lambda(self):
        '''
        Compute the lambda associated to the energy of self.uh
        '''
        self.energy()

        self.lam = 2 * self.E + self.beta.values()[0]/2 * fd.norm(self.uh,'L4')**4
        
        return self.lam
    
    def assemble_problem(self, u0, tau, u_ref):
        '''
        Inizialize and assembles all forms related to the minimization
        
        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        self.tau = fd.Constant(tau)

        self.u_old = fd.Function(self.W)
        self.u_old.interpolate(u0)
        
        self.E_ref = self.energy(u_ref)

        self.E = 0.
        self.lam = 0.


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

        for i in range(MaxIter):
            # compute new soltution
            self.step()

            # normalize
            self.uh.assign(self.uh / fd.norm(self.uh,'L2'))

            self.energy()

            # calculate the error
            error = abs(self.E - self.E_ref) / self.E_ref

            self.u_old.assign(self.uh)

            if verbose:
                print(f'\rIter {i}, Error: {error:.6e}, Energy: {self.E:.10f}', end="", flush=True)

            if error <= toll:
                converged = True
                break

        # compute the final quantities
        self.compute_lambda()

        res = dict(converged = converged,
                   energy = self.E,
                   lam = self.lam,
                   iterate = i,
                   error = error,
                   norm = fd.norm(self.uh,'L2'))
        
        if verbose:
            print('\r', end="", flush=True)

        return res

class gradient_L2(gradient):
    def assemble_problem(self, u0, tau, u_ref):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        super().assemble_problem(u0, tau, u_ref)

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
    def assemble_problem(self, u0, tau, u_ref):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''

        super().assemble_problem(u0, tau, u_ref)

        self.a = 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx

        self.m = self.u * self.w * fd.dx

        self.R_u = fd.Function(self.W)
        self.gradE = fd.Function(self.W)

    def step(self):
        '''
        Implements a step of the gradient H1
        '''
        # compute the Riesz projection
        rhs_R = self.u_old * self.w * fd.dx 

        problem_R = fd.LinearVariationalProblem(self.a, rhs_R, self.R_u, self.bcs)
        solver_R = fd.LinearVariationalSolver(problem_R)#, solver_parameters = param)
        solver_R.solve()
        
        # compute the gradient
        rhs_E = 0.5 * fd.dot(fd.grad(self.u_old), fd.grad(self.w)) * fd.dx \
                + self.v * self.u_old * self.w * fd.dx \
                + self.beta * abs(self.u_old) **2 * self.u_old * self.w * fd.dx
        
        problem_E = fd.LinearVariationalProblem(self.a, rhs_E, self.gradE, self.bcs)
        solver_E =  fd.LinearVariationalSolver(problem_E)#, solver_parameters=param)
        solver_E.solve()

        # compute the solution

        intE = fd.assemble(self.gradE * self.u_old * fd.dx)
        intR = fd.assemble(self.R_u * self.u_old * fd.dx)

        rhs_u = self.u_old * self.w * fd.dx \
                - self.tau * self.gradE * self.w * fd.dx \
                + self.tau * intE/intR * self.R_u * self.w * fd.dx
        
        problem_u = fd.LinearVariationalProblem(self.a, rhs_u, self.uh, self.bcs)
        solver_u = fd.LinearVariationalSolver(problem_u)
        solver_u.solve()

            
class gradient_a0(gradient):
    def assemble_problem(self, u0, tau, u_ref):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        super().assemble_problem(u0, tau, u_ref)

        self.a = 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx\
                + self.v * self.u * self.w * fd.dx
        
        self.m = self.u * self.w * fd.dx

        self.R_u = fd.Function(self.W)
        self.R_u2u = fd.Function(self.W)

    def step(self):
        '''
        Impelements one step of the a_0 gradient 
        '''
        # compute reisz prjections
        rhs_Ru = self.u_old * self.w * fd.dx

        problem_Ru = fd.LinearVariationalProblem(self.a, rhs_Ru, self.R_u, self.bcs)
        solver_Ru = fd.LinearVariationalSolver(problem_Ru)
        solver_Ru.solve()

        rhs_Ru2u = self.beta * abs(self.u_old) ** 2 * self.u_old * self.w * fd.dx

        problem_Ru2u = fd.LinearVariationalProblem(self.a, rhs_Ru2u, self.R_u2u, self.bcs)
        solver_Ru2u = fd.LinearVariationalSolver(problem_Ru2u)
        solver_Ru2u.solve()

        # compute the solution
        intE = fd.assemble((self.u_old + self.R_u2u) * self.u_old * fd.dx)
        intR = fd.assemble(self.R_u * self.u_old * fd.dx)

        rhs_u = self.u_old * self.w * fd.dx \
                - self.tau * (self.u_old + self.R_u2u) * self.w * fd.dx \
                + self.tau * intE/intR * self.R_u * self.w * fd.dx
        
        problem_u = fd.LinearVariationalProblem(self.m, rhs_u, self.uh, self.bcs)
        solver_u = fd.LinearVariationalSolver(problem_u)
        solver_u.solve()


class gradient_az(gradient):
    def assemble_problem(self, u0, tau, u_ref):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        super().assemble_problem(u0, tau, u_ref)

        self.a = 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
                + self.v * self.u * self.w * fd.dx
        
        self.m = self.u * self.w * fd.dx

        self.R_u = fd.Function(self.W)
    
    def step(self):
        '''
        Implements one step of the a_z gradient
        '''
        # compute Riesz
        rhs_R = self.u_old * self.w * fd.dx

        problem_R = fd.LinearVariationalProblem(self.a + self.beta * abs(self.u_old)**2 * self.u * self.w * fd.dx,
                                                rhs_R,
                                                self.R_u,
                                                self.bcs)
        solver_R = fd.LinearVariationalSolver(problem_R)
        solver_R.solve()

        # compute solution
        intE = fd.assemble(self.u_old * self.u_old * fd.dx)
        intR = fd.assemble(self.R_u * self.u_old * fd.dx)

        rhs_u = self.u_old * self.w * fd.dx \
                - self.tau * self.u_old * self.w * fd.dx\
                + self.tau * intE/intR * self.R_u * self.w * fd.dx
        
        problem_u = fd.LinearVariationalProblem(self.m, rhs_u, self.uh, self.bcs)
        solver_u = fd.LinearVariationalSolver(problem_u)
        solver_u.solve()