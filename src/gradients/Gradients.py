r'''
To solve the GPE:
-\Delta z + V z+ \beta |z|^2 z = \lambda z

use the gradient flow that minimizes the energy functional:
E(z) = \int |\nabla z|^2 + V |z|^2 + \frac{\beta}{2} |z|^4 dx

The Projected gradient flow is given by:
Select X such that $H_0^1(D) \sub X \sub L^2(D)$, denote by $\nabla_X E(z) : H_0^1(D) \rightarrow X$ the Riesz rapresentative of $E'(z)$:

(\nabla_X E(z),v)_X = <E'(z),v> \qquad \forall v \in X

Impose the constrain $\|z\| = 1$ and define $T_{z,X} = \{v \in X | (v,z)_{L^2}= 0\}$:

(P_{z,X}(v),\psi)_X = (v, \psi)_X \qquad \forall v \in T_{z,X}

and in term of Riesz rapresentative:

P_{z,X}(v) = v - \frac{(z,v)_{L^2}}{(z,R_X(z))_{L^2}} R_X(z)

where $R_X(z) \in X$ is the Reisz rapresentative of $z$ in $X$. So the final gradient flow equation is:

z'(t) = - (P_{z,X} \circ \nabla_X E)(z(t))

This class implements the abstract class and then in different files the different choices of X will be taken into account.
'''

import abc
import firedrake as fd

class Gradient(abc.ABC):
    """Abstract base class for gradient computation."""

    def __init__(self, W, bcs, beta, v, grad_type_name, type_discr):
        '''
        Constructor for the Gradient class.
        
        :param W (fd space): The function space for the problem.
        :param bcs (list): The boundary conditions.
        :param beta (float): The interaction strength parameter.
        :param v (fd function): The potential function.
        :param grad_type_name (str): A string identifier for the type of gradient method used.
        :param type_discr (str): A string identifier for the type of discretization used.
        '''

        # Construct the functional space and the functions in that space
        self.W = W
        self.u = fd.TrialFunction(W)
        self.w = fd.TestFunction(W)
        self.uh = fd.Function(W)

        # Boundary conditions
        self.bcs = bcs

        # Gross-Pitaevskii parameters
        self.beta = beta
        self.v = v

        # Name of the gradient type, used for logging and saving results
        self.grad_type_name = grad_type_name
        self.type_discr = type_discr
        
        # Boolean to decide if the time step is adapted during the gradient flow or not.
        # if the gradient flow could support it the __init__ method of the child class should set it to True 
        # and the assemble_problem method should be able to handle the case tau = None.

        self.adaptivity = False

    def golden_search(self, func, a = 0.01, b = 2.0, tol = 1e-5):
        """
        Golden-section search
        to find the minimum of f on [a,b]

        * f: a strictly unimodal function on [a,b]

        Example:
        >>> def f(x): return (x - 2) ** 2
        >>> x = gss(f, 1, 5)
        >>> print(f"{x:.5f}")
        2.00000

        """
        # Golden-section search over scalar step size interval [a, b].
        invphi = (fd.sqrt(5) - 1) / 2  # 1 / phi

        # x = np.linspace(0, 10, 100)
        # fig, ax = plt.subplots()
        # ax.plot(x, func(x))
        # plt.legend()
        # plt.show()

        while b - a > tol:
            c = b - (b - a) * invphi
            d = a + (b - a) * invphi
            if func(c) < func(d):
                b = d
            else:  # func(c) > func(d) to find the maximum
                a = c
        return (b + a) / 2
    
    @abc.abstractmethod
    def assemble_problem(self):
        '''
        Abstract method to assemble the variational problem for the gradient computation.
        This method must be implemented by any subclass of Gradient.
        '''
        pass

    @abc.abstractmethod
    def compute_Reiz_representative(self):
        '''
        Abstract method to compute the Riesz representative of the current solution.
        This method must be implemented by any subclass of Gradient.
        '''
        pass

    @abc.abstractmethod
    def compute_gradient(self):
        '''
        Abstract method to compute the projected gradient.
        This method must be implemented by any subclass of Gradient.
        '''
        pass

    @abc.abstractmethod
    def step(self, u_old):
        '''
        Abstract method to perform a single gradient step.
        This method must be implemented by any subclass of Gradient.
        '''
        pass