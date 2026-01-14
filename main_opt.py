import firedrake as fd
from optimizer import Gradient_Descent, ParaflowS


nx = 256
beta_v = [1000]#,100,1000]
tau_v = [0.5, 0.25] #[1, 0.5]
MaxIter = 500
toll = 1e-5
E_ref = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235}
methods = ['L2', 'H1', 'a0', 'az']
Nf_v = [1, 2, 5, 10]

mesh = fd.RectangleMesh(nx, nx, 6, 6, -6, -6, diagonal = 'left')
W = fd.FunctionSpace(mesh,'CG',1)

x,y = fd.SpatialCoordinate(mesh)
v = 0.5 * (x**2 + y**2)
bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4))]
u0 = 1/fd.pi**(0.5) * fd.exp(-(x**2 + y**2)/2)

import sys

orig_stdout = sys.stdout

for beta in beta_v:
    optim = ParaflowS(beta,v,W, bcs, 12 * 2**(-8))
    optim_GD = Gradient_Descent(beta,v,W, bcs, 12 * 2**(-8))

    for tau in tau_v:
        for name in methods:
            optim_GD.compile(u0, tau, E_ref[beta],grad_type = name)

            filename = './log/GD'+ name + '_tau' + str(tau) + '.txt'
            f = open(filename, 'w')
            sys.stdout = f
            optim_GD.minimize(MaxIter, toll)

            for Nf in Nf_v:

                optim.compile(u0, tau, E_ref[beta],grad_type_coarse=name, grad_type_fine = name, Nf = Nf)
                
                filename = './log/PF'+ name + '_tau' + str(tau) + '_Nf'+ str(Nf)+ '.txt'
                f = open(filename, 'w')
                sys.stdout = f
                optim.minimize(MaxIter, toll)



sys.stdout = orig_stdout
f.close()
