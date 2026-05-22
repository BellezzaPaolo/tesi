'''
This module implements the L2 gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient

class Gradient_L2(Gradient):
    '''
    This class implements the L2 gradient of the cost functional.

    X = L^2(Omega) with its standard inner product, so:

    R_{L^2}(z) = z
    P_{z,L^2}(v) = v - (z,v)_{L^2}/(z,z)_{L^2}z
    '''
    
    def __init__(self, W,  bcs, beta, v, type_discr):
        super().__init__(W,  bcs, beta, v, 'L2', type_discr)

    def compute_Reiz_representative(self, u_old):
        '''
        For the L2 gradient, the Riesz representative is simply the function itself.
        '''
        Warning("The Riesz representative for the L2 gradient is the function itself, so this method is not needed.")

        return u_old
    
    def compute_gradient(self):
        raise KeyError("The L2 gradient is not implemented because it's exactly the application of the Gross-Pitaevskii operator to the current solution.")

    def step(self, u_old):
        
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        alpha0 = fd.assemble(0.5 * fd.inner(fd.grad(u_old), fd.grad(u_old)) * fd.dx \
                            + self.v * u_old * u_old * fd.dx)
        beta0 = fd.assemble(self.beta * 0.5 * (u_old)**4 * fd.dx)
        gamma0 = fd.assemble(u_old**2 * fd.dx)

        return 0.5 * (alpha0/gamma0**2 + beta0/gamma0**4)


class Gradient_L2_explicit(Gradient_L2):
    '''
    This class implements the L2 gradient by discretizing the gradient flow with Forward Euler:
    
    ∫ z^{n+1} v dx = ∫ z^n v dx - τ ∫ 0.5 ∇z^n ∇v + Vz^n v + β |z^n|^2 z^n v dx + τ (∫ 0.5 ∇u∇u + Vu^2 + β |u|^2 u u dx)/(|| z^n ||_{L^2}^2) ∫ z^n v dx
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'explicit')

    def assemble_problem(self, tau = None, lump = False):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        :param lump (bool): decides if the system is solved with lumping or not
        '''
        if tau is None:
            raise ValueError("The time step tau must be provided for the L2 explicit gradient")
        
        self.tau = fd.Constant(tau)

        # Assemble mass matrix (optionally mass-lumped) used at each explicit step.
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
        E_old = super().step(u_old)

        # Evaluate energy-like scaling terms used by the explicit update.
        # intE = ∫ 0.5 ∇u∇u + Vu^2 + β |u|^2 u u dx
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

        return E_old


class Gradient_L2_semimplicit(Gradient_L2):
    '''
    This class implements the L2 gradient by discretizing the gradient flow with a semi-implicit scheme:
    
    ∫ z^{n+1} v dx + τ ∫ 0.5 ∇z^{n+1} ∇v + Vz^{n+1} v + β |z^n|^2 z^{n+1} v dx = ∫ z^n v dx + τ (∫ 0.5 ∇u∇u + Vu^2 + β |u|^2 u u dx)/(|| z^n ||_{L^2}^2) ∫ z^n v dx
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'semimplicit')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        if tau is None:
            raise ValueError("The time step tau must be provided for the L2 gradient semimplivit")
        
        self.tau = fd.Constant(tau)

        self.a = self.u * self.w * fd.dx \
            + self.tau * 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
            + self.tau * self.v * self.u * self.w * fd.dx
        
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        E_old = super().step(u_old)

        # Dynamic scalar shift that comes from the projection term.
        shift = (fd.assemble((0.5 * fd.dot(fd.grad(u_old), fd.grad(u_old)) + self.v * u_old * u_old + self.beta * abs(u_old)**2 * u_old * u_old)* fd.dx))/fd.assemble(u_old * u_old * fd.dx)
        rhs = (1 + self.tau * shift) * u_old * self.w * fd.dx
        
        problem = fd.LinearVariationalProblem(self.a + self.tau * self.beta * (u_old **2 * self.u) * self.w * fd.dx,
                                                rhs,
                                                self.uh,
                                                self.bcs)
        solver =  fd.LinearVariationalSolver(problem)#, solver_parameters=param)
        solver.solve()

        return E_old


# this is the original implementation of the L2 gradient semiimplicit, 
# where the projection term can be hidden in the normalization step but ParaFlowS requires to make it explicit.
class Gradient_L2_no_projection(Gradient_L2):
    def __init__(self, W, bcs, h, beta, v):
        UserWarning("The original implementation of the L2 gradient with the projection term is not needed, since the projection is performed in the normalization step. The class gradient_L2_semimplicit should be used instead.")
        super().__init__(W, bcs, h, beta, v, 'semimplicit')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''

        if tau is None:
            raise ValueError("The time step tau must be provided for the L2 gradient")
        
        self.tau = fd.Constant(tau)

        # Linear part fixed across iterations; nonlinear term is added in step().
        self.a = self.u * self.w * fd.dx \
            + self.tau * 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
            + self.tau * self.v * self.u * self.w * fd.dx
        
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        alpha0 = fd.assemble(0.5 * fd.inner(fd.grad(u_old), fd.grad(u_old)) * fd.dx \
                            + self.v * u_old * u_old * fd.dx)
        beta0 = fd.assemble(self.beta * 0.5 * (u_old)**4 * fd.dx)
        gamma0 = fd.assemble(u_old**2 * fd.dx)

        E_old = 0.5 * (alpha0/gamma0**2 + beta0/gamma0**4)

        rhs = u_old * self.w * fd.dx 
        
        problem = fd.LinearVariationalProblem(self.a + self.tau * self.beta * (u_old **2 * self.u) * self.w * fd.dx,
                                                rhs,
                                                self.uh,
                                                self.bcs)
        solver =  fd.LinearVariationalSolver(problem)#, solver_parameters=param)
        solver.solve()
        
        return E_old