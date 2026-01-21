import abc
import firedrake as fd


class gradient(abc.ABC):
    """
    General class that implements Sobolev Gradient to solve the Gross-Pitaevski equation
    
    Args:
        - W: function space of the solution
        - bcs: list of boundary conditions
        - u: Trial function
        - w: Test function
        - uh: solution
        - h: step of spatial discretization
    """
    def __init__(self, W, bcs, h, beta, v, name):
        '''
        Constructor of the class
        
        :param W (fd space): function space of the solution
        :param bcs (list): list of boundary conditions
        :param h (float): step of spatial discretization
        '''
        self.name = name

        # solution space
        self.W = W
        self.u = fd.TrialFunction(self.W)
        self.w = fd.TestFunction(self.W)
        self.uh = fd.Function(self.W)

        # boundary conditions
        self.bcs = bcs

        # discretization parameters
        self.h = h

        # parametrs of the problem
        self.beta = beta
        self.v = v

    @abc.abstractmethod
    def assemble_problem(self):
        '''
        Inizialize and assembles all forms related to the minimization
        '''
        pass

    @abc.abstractmethod
    def step(self, u_old):
        '''
        Step of the minimization that starts from u_old

        :param u_old (fd.Function): solution at the previous time step
        '''
        pass

class gradient_L2_fully_expli(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v,'L2 explicit')

    def assemble_problem(self, tau, lump = False):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        :param lump (bool): decides if the system is solved with lumping or not
        '''

        self.tau = fd.Constant(tau)

        # assemble the mass matrix with or without lumping
        if lump:
            self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs, form_compiler_parameters={"quadrature_rule": "KMV","quadrature_degree": self.W.ufl_element().degree()})

            self.solver_Mass = fd.LinearSolver(self.M)
        else:
            self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs)
            # NOTE: the default solver of firedrake is lu so this parameter are useless
            self.solver_Mass = fd.LinearSolver(self.M) # solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # one solution of the linear system used only to compute the LU factorization
        self.solver_Mass.solve(self.uh, fd.assemble( self.w * fd.dx))

        
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-14:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # intE = ∫ 0.5 ∇u∇u + Vu^2 + β |u|^2 uudx
        intE = fd.assemble(0.5 * fd.dot(fd.grad(u_old), fd.grad(u_old)) * fd.dx + self.v * u_old * u_old * fd.dx + self.beta * abs(u_old)**2 * u_old * u_old * fd.dx)
        # intR = ∫ u^2 dx
        intR = fd.assemble(u_old * u_old * fd.dx)

        rhs = fd.assemble( u_old * self.w * fd.dx \
            - self.tau * 0.5 * fd.dot(fd.grad(u_old), fd.grad(self.w)) * fd.dx \
            - self.tau * self.v * u_old * self.w * fd.dx \
            - self.tau * self.beta * (u_old ** 2 * u_old) * self.w * fd.dx \
            + self.tau * intE/intR * u_old * self.w * fd.dx)
        
        # M u^n+1 = Mu^n - τ ( 0.5 * A + v M + β N(u^2) )u^n + τ intE/intR * Mu^n
        self.solver_Mass.solve(self.uh, rhs)

        


class gradient_L2(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v, 'L2')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''

        self.tau = fd.Constant(tau)

        self.a = self.u * self.w * fd.dx \
            + self.tau * 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
            + self.tau * self.v * self.u * self.w * fd.dx
        
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-14:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        rhs = u_old * self.w * fd.dx 
        
        problem = fd.LinearVariationalProblem(self.a + self.tau * self.beta * (u_old **2 * self.u) * self.w * fd.dx,
                                                rhs,
                                                self.uh,
                                                self.bcs)
        solver =  fd.LinearVariationalSolver(problem)#, solver_parameters=param)
        solver.solve()


class gradient_L2_P(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v, 'L2_P')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''

        self.tau = fd.Constant(tau)

        self.a = self.u * self.w * fd.dx \
            + self.tau * 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
            + self.tau * self.v * self.u * self.w * fd.dx
        
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-14:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        shift = (fd.assemble((0.5 * fd.dot(fd.grad(u_old), fd.grad(u_old)) + self.v * u_old * u_old + self.beta * abs(u_old)**2 * u_old * u_old)* fd.dx))/fd.assemble(u_old * u_old * fd.dx)
        rhs = (1 + self.tau * shift) * u_old * self.w * fd.dx
        
        problem = fd.LinearVariationalProblem(self.a + self.tau * self.beta * (u_old **2 * self.u) * self.w * fd.dx,
                                                rhs,
                                                self.uh,
                                                self.bcs)
        solver =  fd.LinearVariationalSolver(problem)#, solver_parameters=param)
        solver.solve()


class gradient_H1(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v,'H1')

    def assemble_problem(self,tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''

        self.tau = fd.Constant(tau)

        self.A = fd.assemble(0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx, bcs = self.bcs)

        # assemble the solver for the Riesz rappresentation and gradient
        self.R_u = fd.Function(self.W)
        self.gradE = fd.Function(self.W)
        self.solver_Stiffnes = fd.LinearSolver(self.A)#, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # used only to compute the LU factorization
        self.solver_Stiffnes.solve(self.R_u, fd.assemble( self.w * fd.dx))

    def step(self, u_old):
        '''
        Implements a step of the gradient H1
        '''
        if fd.norm(u_old, 'L2')-1 >1e-14:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # compute the Riesz projection
        rhs_R = fd.assemble(u_old * self.w * fd.dx)

        self.solver_Stiffnes.solve(self.R_u, rhs_R)
        
        # compute the gradient
        rhs_E = fd.assemble(0.5 * fd.dot(fd.grad(u_old), fd.grad(self.w)) * fd.dx \
                + self.v * u_old * self.w * fd.dx \
                + self.beta * abs(u_old) **2 * u_old * self.w * fd.dx)

        self.solver_Stiffnes.solve(self.gradE, rhs_E)

        # compute the solution

        intE = fd.assemble(self.gradE * u_old * fd.dx)
        intR = fd.assemble(self.R_u * u_old * fd.dx)

        self.uh.assign(u_old - self.tau * self.gradE + self.tau * intE/intR * self.R_u)


            
class gradient_a0(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v,'a0')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        self.tau = fd.Constant(tau)

        # assemble the solver for Riesz projections
        self.R_u = fd.Function(self.W)
        self.R_u2u = fd.Function(self.W)

        self.A = fd.assemble(0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx\
                                + self.v * self.u * self.w * fd.dx, bcs = self.bcs)

        self.solver_Stiffness = fd.LinearSolver(self.A)#, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly', 'ksp_rtol': 1e-05, 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps', 'pc_factor_mat_mumps_icntl_14': 200, "ksp_view": None})

        # used only to compute the LU factorization
        self.solver_Stiffness.solve(self.R_u, fd.assemble( self.w * fd.dx))



    def step(self, u_old):
        '''
        Impelements one step of the a_0 gradient 
        '''
        if fd.norm(u_old, 'L2')-1 >1e-14:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # compute reisz prjections
        rhs_Ru = fd.assemble(u_old * self.w * fd.dx)
        self.solver_Stiffness.solve(self.R_u, rhs_Ru)

        rhs_Ru2u = fd.assemble(self.beta * abs(u_old) ** 2 * u_old * self.w * fd.dx)

        self.solver_Stiffness.solve(self.R_u2u,rhs_Ru2u)

        # compute the solution
        intE = fd.assemble((u_old + self.R_u2u) * u_old * fd.dx)
        intR = fd.assemble(self.R_u * u_old * fd.dx)

        self.uh.assign((1 - self.tau) * u_old  - self.tau * self.R_u2u + self.tau * intE/intR * self.R_u)


class gradient_az(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v,'az')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        self.tau = fd.Constant(tau)

        # initialize the forms for the Riesz solver
        self.R_u = fd.Function(self.W)

        self.a = 0.5 * fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
                + self.v * self.u * self.w * fd.dx
        
    
    def step(self, u_old):
        '''
        Implements one step of the a_z gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-14:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # compute Riesz
        rhs_R = fd.inner(u_old , self.w) * fd.dx

        # from petsc4py import PETSc
        # fd.assemble(self.a + self.beta * abs(u_old)**2 * self.u * self.w * fd.dx, bcs = self.bcs).M.handle.view(viewer=PETSc.Viewer.STDOUT_WORLD)

        problem_R = fd.LinearVariationalProblem(self.a + self.beta * abs(u_old)**2 * self.u * self.w * fd.dx,
                                                rhs_R,
                                                self.R_u,
                                                self.bcs)
        solver_R = fd.LinearVariationalSolver(problem_R)#, solver_parameters={"ksp_view": None})
        solver_R.solve()

        # compute solution
        # intE = fd.assemble(u_old * u_old * fd.dx) # should be 1 in exact aritmetic because it's simply the norm of hte previous uh
        intR = fd.assemble(self.R_u * u_old * fd.dx)

        self.uh.assign((1 - self.tau) * u_old + self.tau * 1/intR * self.R_u)
