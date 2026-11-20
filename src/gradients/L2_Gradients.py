'''
This module implements the L2 gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient


try:
    from line_profiler import profile
except NameError:
    def profile(function):
        return function


class Gradient_L2(Gradient):
    '''
    This class implements the L2 gradient of the cost functional.

    X = L^2(Omega) with its standard inner product, so:

    R_{L^2}(z) = z
    P_{z,L^2}(v) = v - (z,v)_{L^2}/(z,z)_{L^2}z
    '''
    
    def __init__(self, W,  bcs, beta, v, type_discr):
        super().__init__(W,  bcs, beta, v, 'L2', type_discr)

    @profile
    def compute_Reiz_representative(self, u_old):
        '''
        For the L2 gradient, the Riesz representative is simply the function itself.
        '''
        Warning("The Riesz representative for the L2 gradient is the function itself, so this method is not needed.")

        return u_old
    
    @profile
    def compute_gradient(self):
        raise KeyError("The L2 gradient is not implemented because it's exactly the application of the Gross-Pitaevskii operator to the current solution.")

# class Gradient_L2_explicit(Gradient_L2):
#     '''
#     This class implements the L2 gradient by discretizing the gradient flow with Forward Euler:
    
#     ∫ z^{n+1} v dx = ∫ z^n v dx - τ ∫ 0.5 ∇z^n ∇v + Vz^n v + β |z^n|^2 z^n v dx + τ (∫ 0.5 ∇u∇u + Vu^2 + β |u|^2 u u dx)/(|| z^n ||_{L^2}^2) ∫ z^n v dx
#     '''
#     def __init__(self, W, bcs, beta, v):
#         super().__init__(W, bcs, beta, v, 'explicit')

#     @profile
#     def assemble_problem(self, tau = None, lump = False):
#         '''
#         Allocate and assembles forms and minimization quantities

#         :param tau (float): time step
#         :param lump (bool): decides if the system is solved with lumping or not
#         '''
#         # Assemble mass matrix (optionally mass-lumped) used at each explicit step.
#         if lump:
#             self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs, form_compiler_parameters={"quadrature_rule": "KMV","quadrature_degree": self.W.ufl_element().degree()})

#         else:
#             self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs)
        
#         # NOTE: the default solver of firedrake is lu so this parameter are useless
#         self.solver_Mass = fd.LinearSolver(self.M) # solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

#         self.u_current = fd.Function(self.W)
#         self.u_current.dat.data[1] = 1.0 # firedrake initializes at 0.0, change a random value in order to not have /0.0
#         self.shift_term = fd.Function(self.W)
#         self.rhs =  self.u_current * self.w * fd.dx \
#             - self.tau * 0.5 * fd.dot(fd.grad(self.u_current), fd.grad(self.w)) * fd.dx \
#             - self.tau * self.v * self.u_current * self.w * fd.dx \
#             - self.tau * self.beta * (self.u_current ** 2 * self.u_current) * self.w * fd.dx \
#             + self.tau * fd.assemble(0.5 * fd.dot(fd.grad(self.u_current), fd.grad(self.u_current)) * fd.dx \
#                                      + self.v * self.u_current * self.u_current * fd.dx \
#                                      + self.beta * self.u_current**2 * self.u_current * self.u_current * fd.dx)\
#                         / fd.assemble(self.u_current * self.u_current * fd.dx) * self.u_current * self.w * fd.dx 
        
#         self.Rhs = fd.assemble(self.rhs)

#         # one solution of the linear system used only to compute the LU factorization
#         self.solver_Mass.solve(self.uh, self.Rhs)

#         if tau is None:
#             self.adaptivity = True
#             self.tau = 1.0 # dummy value, the actual time step will be computed at each step.
#             self.tau_history = []
#         else:
#             self.adaptivity = False
#             self.tau = tau

#         self.proj_term = fd.Function(self.W)

#         super().assemble_problem(compute_intE = True, compute_energy = True, adaptivity = self.adaptivity)

#         # warm up with dummy inputs
#         _ = self.compute_integrals(self.u_current, self.R_u, self.gradE, self.proj_term)

#     @profile
#     def step(self, u_old):
#         '''
#         Implements the step of L2 gradient
#         '''
#         super().step(u_old)

#         self.u_current.dat.data[:] = u_old.dat.data_ro[:]

#         fd.assemble(self.rhs, tensor = self.Rhs)
        
#         # M u^n+1 = Mu^n - τ ( 0.5 * A + v M + β N(u^2) )u^n + τ E(u)/∫ u^2 dx * Mu^n
#         self.solver_Mass.solve(self.uh, self.Rhs)


class Gradient_L2_semimplicit(Gradient_L2):
    '''
    This class implements the L2 gradient by discretizing the gradient flow with a semi-implicit scheme:
    
    ∫ z^{n+1} v dx + τ ∫ 0.5 ∇z^{n+1} ∇v + Vz^{n+1} v + β |z^n|^2 z^{n+1} v dx = ∫ z^n v dx + τ (∫ 0.5 ∇u∇u + Vu^2 + β |u|^2 u u dx)/(|| z^n ||_{L^2}^2) ∫ z^n v dx
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'semimplicit')

    @profile
    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        if tau is None:
            raise ValueError("The time step tau must be provided for the L2 gradient semimplivit")
        
        self.tau = fd.Constant(tau)

        self.nonlinear_coefficient = fd.Function(self.W)

        self.s = self.u * self.w * fd.dx \
            + self.tau * 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
            + self.tau * self.v * self.u * self.w * fd.dx \
            + self.tau * self.nonlinear_coefficient * self.u * self.w * fd.dx

        self.S = fd.assemble(self.s, bcs = self.bcs)

        self.Solver_System = fd.LinearSolver(self.S)

        self.Rhs = fd.assemble( self.w * fd.dx)
        
        super().assemble_problem(compute_intE = False, compute_energy = True, adaptivity = False)

        # warm up with dummy inputs
        _ = self.compute_integrals(self.uh.dat.data_ro)


    @profile
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        self.compute_integrals(u_old.dat.data_ro)

        fd.assemble((1 + self.tau * (self.integrals[1] +  0.5 * self.beta * self.integrals[2]) / self.integrals[0]) * u_old * self.w * fd.dx, tensor = self.Rhs)

        self.nonlinear_coefficient.dat.data[:] = self.beta * u_old.dat.data_ro[:]**2
        
        fd.assemble(self.s, bcs=self.bcs, tensor=self.S)

        self.Solver_System.solve(self.uh, self.Rhs)


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