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
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(fastmath=True, parallel=True)
def _compute_intR(u, R, cell_nodes, areas, output):
    intR = 0.0

    n_cells = cell_nodes.shape[0]
    for c in prange(n_cells):
        i0, i1, i2 = cell_nodes[c, 0], cell_nodes[c, 1], cell_nodes[c, 2]
        u1, u2, u3 = u[i0], u[i1], u[i2]
        r1, r2, r3 = R[i0], R[i1], R[i2]

        area = areas[c]

        Pu = u1 + u2 + u3

        w12 = area / 12.0
        Pr = r1 + r2 + r3
        dr1, dr2, dr3 = u1 * r1, u2 * r2, u3 * r3
        intR += w12 * (Pu * Pr + dr1 + dr2 + dr3)

    output[0] = intR

@njit(fastmath=True, parallel=True)
def _compute_intR_intE(u, R, gradE, cell_nodes, areas, output):
    intR = 0.0
    intE = 0.0

    n_cells = cell_nodes.shape[0]
    for c in prange(n_cells):
        i0, i1, i2 = cell_nodes[c, 0], cell_nodes[c, 1], cell_nodes[c, 2]
        u1, u2, u3 = u[i0], u[i1], u[i2]
        r1, r2, r3 = R[i0], R[i1], R[i2]
        e1, e2, e3 = gradE[i0], gradE[i1], gradE[i2]

        area = areas[c]

        Pu = u1 + u2 + u3

        w12 = area / 12.0
        Pr = r1 + r2 + r3
        dr1, dr2, dr3 = u1 * r1, u2 * r2, u3 * r3
        intR += w12 * (Pu * Pr + dr1 + dr2 + dr3)

        Pe = e1 + e2 + e3
        de1, de2, de3 = u1 * e1, u2 * e2, u3 * e3
        intE += w12 * (Pu * Pe + de1 + de2 + de3)

    output[0] = intR
    output[1] = intE

@njit(fastmath=True, parallel=True)
def _compute_energy_intR(u, cell_nodes, areas, grad_lambda, M_v, output):
    intR = 0.0
    a0 = b0 = 0.0

    n_cells = cell_nodes.shape[0]
    for c in prange(n_cells):
        i0, i1, i2 = cell_nodes[c, 0], cell_nodes[c, 1], cell_nodes[c, 2]
        u1, u2, u3 = u[i0], u[i1], u[i2]

        area = areas[c]
        u1_2, u2_2, u3_2 = u1 * u1, u2 * u2, u3 * u3

        Pu = u1 + u2 + u3

        w12 = area / 12.0
        dr1, dr2, dr3 = u1 * u1, u2 * u2, u3 * u3
        intR += w12 * (Pu * Pu + dr1 + dr2 + dr3)

        gl1x, gl1y = grad_lambda[c, 0, 0], grad_lambda[c, 0, 1]
        gl2x, gl2y = grad_lambda[c, 1, 0], grad_lambda[c, 1, 1]
        gl3x, gl3y = grad_lambda[c, 2, 0], grad_lambda[c, 2, 1]

        gux = u1 * gl1x + u2 * gl2x + u3 * gl3x
        guy = u1 * gl1y + u2 * gl2y + u3 * gl3y

        mv11, mv22, mv33 = M_v[c, 0], M_v[c, 1], M_v[c, 2]
        mv12, mv23, mv31 = M_v[c, 3], M_v[c, 4], M_v[c, 5]

        vu2 = (mv11 * u1_2 + mv22 * u2_2 + mv33 * u3_2
                + 2.0 * (mv12 * u1 * u2 + mv23 * u2 * u3 + mv31 * u3 * u1))

        a0 += 0.5 * area * (gux * gux + guy * guy) + vu2

        w15 = area / 15.0
        b0 += w15 * (
            u1_2 * u1_2 + u2_2 * u2_2 + u3_2 * u3_2
            + u1_2 * u1 * (u2 + u3) + u2_2 * u2 * (u1 + u3) + u3_2 * u3 * (u1 + u2)
            + u1_2 * u2_2 + u2_2 * u3_2 + u3_2 * u1_2
            + u1 * u2 * u3 * Pu
        )

    output[0] = intR
    output[1] = a0
    output[2] = b0



@njit(fastmath=True, parallel=True)
def _compute_integrals_adaptivity(u, proj_term, cell_nodes, areas, grad_lambda, M_v, output):
    a0 = b0 = g0 = 0.0
    a1 = a2 = 0.0
    b1 = b2 = b3 = b4 = 0.0
    g1 = g2 = 0.0

    n_cells = cell_nodes.shape[0]
    for c in prange(n_cells):
        i0, i1, i2 = cell_nodes[c, 0], cell_nodes[c, 1], cell_nodes[c, 2]
        u1, u2, u3 = u[i0], u[i1], u[i2]
        p1, p2, p3 = proj_term[i0], proj_term[i1], proj_term[i2]

        area = areas[c]
        u1_2, u2_2, u3_2 = u1 * u1, u2 * u2, u3 * u3

        Pu = u1 + u2 + u3
        Qu = u1_2 + u2_2 + u3_2
        Su = Pu * Qu + u1 * u2 * u3

        w12 = area / 12.0
        p1_2, p2_2, p3_2 = p1 * p1, p2 * p2, p3 * p3
        Pp = p1 + p2 + p3
        Qp = p1_2 + p2_2 + p3_2
        Sp = Pp * Qp + p1 * p2 * p3

        gl1x, gl1y = grad_lambda[c, 0, 0], grad_lambda[c, 0, 1]
        gl2x, gl2y = grad_lambda[c, 1, 0], grad_lambda[c, 1, 1]
        gl3x, gl3y = grad_lambda[c, 2, 0], grad_lambda[c, 2, 1]

        gux = u1 * gl1x + u2 * gl2x + u3 * gl3x
        guy = u1 * gl1y + u2 * gl2y + u3 * gl3y

        mv11, mv22, mv33 = M_v[c, 0], M_v[c, 1], M_v[c, 2]
        mv12, mv23, mv31 = M_v[c, 3], M_v[c, 4], M_v[c, 5]

        vu2 = (mv11 * u1_2 + mv22 * u2_2 + mv33 * u3_2
                + 2.0 * (mv12 * u1 * u2 + mv23 * u2 * u3 + mv31 * u3 * u1))

        a0 += 0.5 * area * (gux * gux + guy * guy) + vu2

        w15 = area / 15.0
        b0 += w15 * (
            u1_2 * u1_2 + u2_2 * u2_2 + u3_2 * u3_2
            + u1_2 * u1 * (u2 + u3) + u2_2 * u2 * (u1 + u3) + u3_2 * u3 * (u1 + u2)
            + u1_2 * u2_2 + u2_2 * u3_2 + u3_2 * u1_2
            + u1 * u2 * u3 * Pu
        )
        g0 += w12 * (Pu * Pu + Qu)

        d1, d2, d3 = u1 * p1, u2 * p2, u3 * p3
        m12 = u1 * p2 + u2 * p1
        m23 = u2 * p3 + u3 * p2
        m31 = u3 * p1 + u1 * p3

        gpx = p1 * gl1x + p2 * gl2x + p3 * gl3x
        gpy = p1 * gl1y + p2 * gl2y + p3 * gl3y

        vup = (mv11 * d1 + mv22 * d2 + mv33 * d3
                + mv12 * m12 + mv23 * m23 + mv31 * m31)
        vp2 = (mv11 * p1_2 + mv22 * p2_2 + mv33 * p3_2
                + 2.0 * (mv12 * p1 * p2 + mv23 * p2 * p3 + mv31 * p3 * p1))

        a1 += 0.5 * area * (gux * gpx + guy * gpy) + vup
        a2 += 0.5 * area * (gpx * gpx + gpy * gpy) + vp2

        w60 = area / 60.0
        w90 = area / 90.0
        cu1 = Su + u1 * (2.0 * u1 * Pu + Qu + u2 * u3)
        cu2 = Su + u2 * (2.0 * u2 * Pu + Qu + u1 * u3)
        cu3 = Su + u3 * (2.0 * u3 * Pu + Qu + u1 * u2)
        b1 += w60 * (p1 * cu1 + p2 * cu2 + p3 * cu3)

        b2 += w90 * (
            6.0 * (d1 * d1 + d2 * d2 + d3 * d3)
            + 2.0 * (d1 * d2 + d2 * d3 + d3 * d1)
            + 3.0 * (m12 * (d1 + d2) + m23 * (d2 + d3) + m31 * (d3 + d1))
            + (d1 * m23 + d2 * m31 + d3 * m12)
            + (m12 * m12 + m23 * m23 + m31 * m31)
            + (m12 * m23 + m23 * m31 + m31 * m12)
        )

        cp1 = Sp + p1 * (2.0 * p1 * Pp + Qp + p2 * p3)
        cp2 = Sp + p2 * (2.0 * p2 * Pp + Qp + p1 * p3)
        cp3 = Sp + p3 * (2.0 * p3 * Pp + Qp + p1 * p2)
        b3 += w60 * (u1 * cp1 + u2 * cp2 + u3 * cp3)

        b4 += w15 * (
            p1_2 * p1_2 + p2_2 * p2_2 + p3_2 * p3_2
            + p1_2 * p1 * (p2 + p3) + p2_2 * p2 * (p1 + p3) + p3_2 * p3 * (p1 + p2)
            + p1_2 * p2_2 + p2_2 * p3_2 + p3_2 * p1_2
            + p1 * p2 * p3 * Pp
        )

        g1 += w12 * (Pu * Pp + d1 + d2 + d3)
        g2 += w12 * (Pp * Pp + Qp)

    output[0] = a0
    output[1] = a1
    output[2] = a2
    output[3] = b0
    output[4] = b1
    output[5] = b2
    output[6] = b3
    output[7] = b4
    output[8] = g0
    output[9] = g1
    output[10] = g2


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

    def golden_search(self, func, a = 0.01, b = 10.0, tol = 1e-5, plot = False):
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
        invphi = (np.sqrt(5) - 1) / 2  # 1 / phi

        while b - a > tol:
            c = b - (b - a) * invphi
            d = a + (b - a) * invphi
            if func(c) < func(d):
                b = d
            else:  # func(c) > func(d) to find the maximum
                a = c

        if plot:
            x = np.linspace(a - 0.3, b + 1, 100)
            fig, ax = plt.subplots()
            ax.plot(x, func(x))
            plt.plot((b + a) / 2, func((b + a) / 2), 'ro', label='Minimum')
            plt.legend()
            plt.show()

        # print(f'new energy {func((a+b)/2)/2}')

        return (b + a) / 2
    
    @abc.abstractmethod
    def assemble_problem(self, compute_intE, compute_energy, adaptivity):
        '''
        Abstract method to assemble the variational problem for the gradient computation.
        This method must be implemented by any subclass of Gradient.
        '''
        self.cell_nodes = np.ascontiguousarray(self.W.cell_node_list)  # Shape: (num_cells, 3)

        coords = self.W.mesh().coordinates.dat.data_ro[self.cell_nodes]  # Shape: (num_cells, 3, 2)

        x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
        x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
        x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

        signed_area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        self.areas = np.ascontiguousarray(np.abs(signed_area))
        if adaptivity or compute_energy:
            inv_2A = 1.0 / (2.0 * signed_area)

            # Precompute barycentric basis gradients: shape (num_cells, 3, 2)
            self.grad_lambda = np.empty((self.cell_nodes.shape[0], 3, 2), dtype=np.float64)
            self.grad_lambda[:, 0, 0] = (y2 - y3) * inv_2A
            self.grad_lambda[:, 0, 1] = (x3 - x2) * inv_2A
            self.grad_lambda[:, 1, 0] = (y3 - y1) * inv_2A
            self.grad_lambda[:, 1, 1] = (x1 - x3) * inv_2A
            self.grad_lambda[:, 2, 0] = (y1 - y2) * inv_2A
            self.grad_lambda[:, 2, 1] = (x2 - x1) * inv_2A

            # Precompute exact local potential mass matrix M_v for v(x,y) = 0.5*(x^2 + y^2)
            # Stored as 6 symmetric entries per cell: [M11, M22, M33, M12, M23, M31]
            v11 = 0.5 * (x1 * x1 + y1 * y1)
            v22 = 0.5 * (x2 * x2 + y2 * y2)
            v33 = 0.5 * (x3 * x3 + y3 * y3)
            v12 = x1 * x2 + y1 * y2
            v23 = x2 * x3 + y2 * y3
            v31 = x3 * x1 + y3 * y1

            w180 = self.areas / 180.0
            self.M_v = np.empty((self.cell_nodes.shape[0], 6), dtype=np.float64)
            self.M_v[:, 0] = w180 * (12.0 * v11 + 2.0 * v22 + 2.0 * v33 + 3.0 * v12 + v23 + 3.0 * v31)  # M11
            self.M_v[:, 1] = w180 * (2.0 * v11 + 12.0 * v22 + 2.0 * v33 + 3.0 * v12 + 3.0 * v23 + v31)  # M22
            self.M_v[:, 2] = w180 * (2.0 * v11 + 2.0 * v22 + 12.0 * v33 + v12 + 3.0 * v23 + 3.0 * v31)  # M33
            self.M_v[:, 3] = w180 * (3.0 * v11 + 3.0 * v22 + v33 + 2.0 * v12 + v23 + v31)               # M12
            self.M_v[:, 4] = w180 * (v11 + 3.0 * v22 + 3.0 * v33 + v12 + 2.0 * v23 + v31)               # M23
            self.M_v[:, 5] = w180 * (3.0 * v11 + v22 + 3.0 * v33 + v12 + v23 + 2.0 * v31)               # M31


        if compute_energy:
            self.integrals = np.zeros(3, dtype=float)
            cell_nodes = self.cell_nodes
            areas = self.areas
            grad_lambda = self.grad_lambda
            M_v = self.M_v
            output = self.integrals

            def compute_integrals(u, cell_nodes=cell_nodes, areas=areas,
                                  grad_lambda=grad_lambda, M_v=M_v,
                                  output=output):
                return _compute_energy_intR(
                    u, cell_nodes, areas, grad_lambda, M_v, output)

        elif compute_intE:
            self.integrals = np.zeros(2, dtype=float)
            cell_nodes = self.cell_nodes
            areas = self.areas
            output = self.integrals

            def compute_integrals(u, R, gradE, cell_nodes=cell_nodes,
                                  areas=areas, output=output):
                return _compute_intR_intE(
                    u, R, gradE, cell_nodes, areas, output)

        else:
            raise ValueError("Impossible, you need to compute some integral in order to project the solution")
            # self.integrals = np.zeros(1, dtype=float)
            # cell_nodes = self.cell_nodes
            # areas = self.areas
            # grad_lambda = self.grad_lambda
            # M_v = self.M_v
            # output = self.integrals

            # def compute_integrals(u, R, gradE = None, cell_nodes=cell_nodes,
            #                       areas=areas, grad_lambda=grad_lambda,
            #                       M_v=M_v, output=output):
            #     return _compute_intR(
            #         u, R, cell_nodes, areas, grad_lambda, M_v, output)

        self.compute_integrals = compute_integrals

        if adaptivity:
            self.integrals_adaptivity = np.zeros(11, dtype=float)
            cell_nodes = self.cell_nodes
            areas = self.areas
            grad_lambda = self.grad_lambda
            M_v = self.M_v
            output = self.integrals_adaptivity

            def compute_integrals_adaptivity(u, proj_term,
                                  cell_nodes=cell_nodes, areas=areas,
                                  grad_lambda=grad_lambda, M_v=M_v,
                                  output=output):
                return _compute_integrals_adaptivity(
                    u, proj_term,
                    cell_nodes, areas, grad_lambda, M_v, output)
            
            self.compute_integral_adaptivity = compute_integrals_adaptivity


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