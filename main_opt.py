import firedrake as fd
from optimizer import Gradient_Descent, ParaflowS


nx = 256
beta_v = [1000]#,100,1000]
tau_v = [1]#, 0.5]
MaxIter = 100
toll = 1e-5
E_ref = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235}
methods = ['L2', 'H1', 'a0', 'az']

mesh = fd.RectangleMesh(nx, nx, 6, 6, -6, -6, diagonal = 'left')
W = fd.FunctionSpace(mesh,'CG',1)

x,y = fd.SpatialCoordinate(mesh)
v = 0.5 * (x**2 + y**2)
bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4))]
u0 = 1/fd.pi**(0.5) * fd.exp(-(x**2 + y**2)/2)
# Source - https://stackoverflow.com/a
# Posted by Gringo Suave, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-13, License - CC BY-SA 4.0

import sys

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

for beta in beta_v:
    optim = ParaflowS(beta,v,W, bcs, 12 * 2**(-8))
    # optim = Gradient_Descent(beta,v,W, bcs, 12 * 2**(-8))

    for tau in tau_v:
        #for name in methods:

            optim.compile(u0, tau, E_ref[beta],grad_type_coarse='L2', grad_type_fine = 'L2', Nf = 1)
            # optim.compile(u0, 1, E_ref[beta],grad_type = 'az')

            optim.minimize(MaxIter, toll)



sys.stdout = orig_stdout
f.close()
