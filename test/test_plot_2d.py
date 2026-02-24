import firedrake as fd
import sys
import csv
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
from pot_3 import RandomDisorderPotential

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizer import Gradient_Descent, ParaflowS

is_save_CSV = True
is_save_log = False
is_save_plot = False

folder = Path('~/Desktop/tesi/tesi/test').expanduser()

nx = 256
# beta_v = [1000]#,100,1000]
MaxIter = 1000
toll = 1e-5
test = '1' # '1', '2' or '3'

methods_coarse = ['L2_P']# ['L2_P', 'az']
methods_fine = ['L2_P', 'az']
Nf_v = [4]#[2, 3, 4, 5, 6]#, 10]
Ng_v = [100]#, 20]#, 100]

if test == '1':
    tau_v ={'az': [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], 'L2_P': [0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1, 1.5, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100]} #L2_P
    # tau_v = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]#az#1, 0.5, 0.25, 0.1, 0.05, 0.025] #[1, 0.5]
    #tau_v = [1.2,1.3,1.5,1.7,1.9]#az
    E_ref = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235}
    beta = 1000

    case_folder = folder / 'case_test1'
    case_folder.mkdir(parents=True, exist_ok=True)
    filename_results_GD = str(case_folder / 'GD.csv')
    # filename_results_GD = str(case_folder / 'GD_conv.csv')
    # filename_results_PF = str(case_folder / 'PFm_dt.csv')
    filename_results_PF = str(case_folder / 'PFalpha_dtxNf.csv')
elif test == '2':
    tau_v = {'az':list(np.linspace(0.05, 1.7,19)), 'L2_P':[0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1, 1.5, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100]} #L2_P

    beta = 1000
    # tau_v = list(np.linspace(0.05, 1.7,19)) #az
    E_ref = {1000: 15.204825}

    case_folder = folder / 'case_test2'
    case_folder.mkdir(parents=True, exist_ok=True)
    filename_results_GD = str(case_folder / 'GD.csv')
    filename_results_PF = str(case_folder / 'PFalpha_dt.csv')
elif test == '3':
    tau_v = {'az':list(np.linspace(0.5, 2.0, 15)), 'L2_P': [0.01, 0.05, 0.1, 0.5, 0.8, 1, 1.5, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100]} #L2_P
    beta = 10
    # tau_v = list(np.linspace(0.5, 2.0, 15)) #az
    E_ref = {10: 4.602621438437267}

    case_folder = folder / 'case_test3'
    case_folder.mkdir(parents=True, exist_ok=True)
    # filename_results_GD = str(case_folder / 'GD.csv')
    filename_results_PF = str(case_folder / 'PFalpha_dtxNf.csv')
else:
    NameError('test must be 1, 2 or 3')

mesh = fd.RectangleMesh(nx, nx, 6, 6, -6, -6, diagonal = 'left')
W = fd.FunctionSpace(mesh,'CG',1)

x,y = fd.SpatialCoordinate(mesh)
if test == '1':
    v = 0.5 * (x**2 + y**2)
    u0 = 1/fd.pi**(0.5) * fd.exp(-(x**2 + y**2)/2)
elif test == '2':
    v = 0.5 * (x**2 + y**2) + fd.Constant(20) + fd.Constant(20) * fd.sin( 2 * fd.pi * x) * fd.sin(2 * fd.pi * y)

    mu_TF = fd.Constant(fd.sqrt(beta / fd.pi))

    u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF - v)/beta), 0.0)
elif test == '3':

    xmin, xmax = -6., 6.
    ymin, ymax = -6., 6.

    epsilon = 0.03

    nxV = int((xmax - xmin) / (epsilon))
    nyV = int((ymax - ymin) / (epsilon))

    meshV = fd.RectangleMesh(nxV, nyV, xmax, ymax, originX=xmin, originY=ymin, quadrilateral=True)
    # Function space for a cellwise-constant potential
    Vdg = fd.FunctionSpace(meshV, "DG", 0)   # piecewise-constant per cell

    # Create random disorder potential using the RandomDisorderPotential class
    # Using seed=21 for consistency with the original rng seed
    potential_obj = RandomDisorderPotential(number_of_cells=400, domain_size=12.0, seed=21)

    # Create the DG0 function for the potential
    V_random = fd.Function(Vdg)

    # Get cell coordinates from meshV and evaluate the potential
    coord_fn_V = fd.Function(fd.VectorFunctionSpace(meshV, 'DG', 0))
    coord_fn_V.interpolate(fd.SpatialCoordinate(meshV))
    coords_V = coord_fn_V.dat.data_ro

    # Evaluate potential at cell coordinates and assign to V_random
    V_random.dat.data[:] = potential_obj.evaluate(coords_V[:, 0], coords_V[:, 1])
    
    Wv = fd.FunctionSpace(mesh, 'DG', 0)

    v = fd.Function(Wv)

    # Get cell midpoints from the triangular mesh to evaluate V_random
    # For a DG0 function, we need one value per cell
    # Use the cell centroid coordinates
    x_dg = fd.SpatialCoordinate(mesh)
    coord_fn = fd.Function(fd.VectorFunctionSpace(mesh, 'DG', 0))
    coord_fn.interpolate(x_dg)
    coords = coord_fn.dat.data_ro

    # Step 2: Evaluate V_random at those coordinates
    vals = np.array([V_random.at((float(px), float(py)), tolerance=1e-10)
                    for px, py in coords])

    # Step 3: assign to the DG0 field on the triangular mesh
    v.dat.data[:] = vals

    mu_TF = fd.Constant(fd.sqrt(beta / fd.pi))

    u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF - v)/beta), 0.0)

else:
    NameError('test must be 1, 2 or 3')

bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4))]

if is_save_CSV:
    # with open(filename_results_GD, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])

    with open(filename_results_PF, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['fine_operator', 'coarse_operator', 'h', 'beta', 'N_fine', 'N_coarse', 'tau_fine', 'tau_coarse', 'energy', 'lambda', 'iterate_coarse', 'iterate_fine', 'iterate', 'error', 'total_time', 'mean_time'])

orig_stdout = sys.stdout
t_start = time.time()


optim = ParaflowS(beta,v,W, bcs, 12 * 2**(-8))
optim_GD = Gradient_Descent(beta,v,W, bcs, 12 * 2**(-8))

done = True
for name_fine in methods_fine:
    # if name_fine == 'az':
    #      done = False

    # if not(done):
        for tau in tqdm(tau_v[name_fine]):
            # optim_GD.compile(u0, tau, E_ref[beta],grad_type = name_fine)

            # filename = 'GD'+ name_fine + '_tau' + str(tau)
            # if is_save_log:
            #     f = open(str(folder / (filename + '.log')), 'w')
            #     sys.stdout = f
            # res = optim_GD.minimize(MaxIter, toll, is_save_log)
            # if is_save_plot:
            #     optim_GD.plot_history(filesave = str(folder / (filename + '.png')))

            # if is_save_CSV:
            #     optim_GD.save_data(filename_results_GD, res)

            # if is_save_log:
            #     f.close()

            #for name_coarse in methods_coarse:
            name_coarse = name_fine
            for Nf in Nf_v:
                for Ng in Ng_v:

                    optim.compile(u0, tau, tau *Nf, E_ref[beta],grad_type_coarse=name_coarse, grad_type_fine = name_fine, Nf = Nf, Ng = Ng)

                    filename = 'PF'+ name_fine + '_' + name_coarse + '_tau' + str(tau) + '_Nf'+ str(Nf)+ '_Ng' + str(Ng)
                    if is_save_log:
                        f = open(str(folder / (filename + '.log')), 'w')
                        sys.stdout = f
                    res = optim.minimize(100, toll, is_save_log)
                    if is_save_plot:
                        optim.plot_history(filesave = str(folder / (filename + '.png')))
                    if is_save_CSV:
                        optim.save_data(filename_results_PF, res)
                    if is_save_log:
                        f.close()
                    
            sys.stdout = orig_stdout

            print(f'Done {name_fine} minimization for tau = {tau} and beta = {beta} in {time.time() - t_start} seconds.')