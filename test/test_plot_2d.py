import firedrake as fd
import sys
import csv
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizer import Gradient_Descent, ParaflowS

is_save_CSV = True
is_save_log = False
is_save_plot = False

folder = Path('~/Desktop/tesi/tesi/test').expanduser()

nx = 256
beta_v = [1000]#,100,1000]
MaxIter = 1000
toll = 1e-5
test = '1' # '1', '2' or '3'

methods_coarse = ['L2_P', 'az']
methods_fine = ['az']#['L2_P', 'az']
Nf_v = [2, 4, 5, 6]#, 10]
Ng_v = [10, 20, 100]

if test == '1':
    tau_v = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]#1, 0.5, 0.25, 0.1, 0.05, 0.025] #[1, 0.5]
    E_ref = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235}
    case_folder = folder / 'case_test1'
    case_folder.mkdir(parents=True, exist_ok=True)
    filename_results_GD = str(case_folder / 'GD.csv')
    filename_results_PF = str(case_folder / 'PF_dtxNf_2.csv')
elif test == '2':
    {# elif test == '2':
# #     tau_v = 
# L2 minization with h: 0.046875, beta: 1000, tau:0.0025 did NOT converged in iterate: 100


# L2 minization with h: 0.046875, beta: 1000, tau:0.01320487975126251 converged to energy: 15.20496807604604 with lambda: 36.72672163088761 at the iterate: 88


# L2 minization with h: 0.046875, beta: 1000, tau:0.06974753969812106 converged to energy: 15.204965503866429 with lambda: 36.726214997561286 at the iterate: 40


# L2 minization with h: 0.046875, beta: 1000, tau:0.1 converged to energy: 15.204955679756143 with lambda: 36.72541988435078 at the iterate: 37


# L2 minization with h: 0.046875, beta: 1000, tau:0.3684031498640387 converged to energy: 15.204961776074994 with lambda: 36.725772054503096 at the iterate: 31


# L2 minization with h: 0.046875, beta: 1000, tau:1.9458877175763893 converged to energy: 15.204970124685158 with lambda: 36.72634696616739 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:5.3578947368421055 converged to energy: 15.204962002996442 with lambda: 36.725740445059586 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:10.278085328021946 converged to energy: 15.204959842521504 with lambda: 36.72557674930644 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:10.615789473684211 converged to energy: 15.204959768111808 with lambda: 36.725571093196265 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:15.873684210526317 converged to energy: 15.204959019598771 with lambda: 36.72551412758915 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:21.131578947368425 converged to energy: 15.204958644679188 with lambda: 36.725485547397675 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:26.38947368421053 converged to energy: 15.204958419515519 with lambda: 36.72546836801225 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:31.647368421052636 converged to energy: 15.204958269318077 with lambda: 36.725456902033834 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:36.905263157894744 converged to energy: 15.204958161991701 with lambda: 36.72544870564982 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:42.16315789473685 converged to energy: 15.204958081472657 with lambda: 36.72544255485589 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:47.42105263157895 converged to energy: 15.204958018833016 with lambda: 36.725437768828435 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:52.678947368421056 converged to energy: 15.204957968712206 with lambda: 36.72543393869582 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:54.28835233189815 converged to energy: 15.2049579553137 with lambda: 36.72543291469746 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:57.93684210526317 converged to energy: 15.204957927698647 with lambda: 36.725430804084574 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:63.19473684210527 converged to energy: 15.204957893516792 with lambda: 36.725428191314975 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:68.45263157894738 converged to energy: 15.2049578645909 with lambda: 36.725425980085554 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:73.71052631578948 converged to energy: 15.20495783979499 with lambda: 36.725424084437805 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:78.96842105263158 converged to energy: 15.204957818303466 with lambda: 36.72542244130956 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:84.22631578947369 converged to energy: 15.204957799497677 with lambda: 36.72542100339887 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:89.48421052631579 converged to energy: 15.204957782903003 with lambda: 36.7254197345183 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:94.7421052631579 converged to energy: 15.20495776815161 with lambda: 36.72541860651853 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:100.0 converged to energy: 15.204957754952654 with lambda: 36.725417597167684 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:286.7484657747549 converged to energy: 15.204957600128475 with lambda: 36.72540575468715 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:1514.5916037658599 converged to energy: 15.20495753295899 with lambda: 36.725400615194054 at the iterate: 29


# L2 minization with h: 0.046875, beta: 1000, tau:8000.0 converged to energy: 15.204957520244959 with lambda: 36.72539964233727 at the iterate: 29
    }
    
    tau_v = list(np.linspace(0.05, 1.7,19))
    E_ref = {1000: 15.204825}
    case_folder = folder / 'case_test2'
    case_folder.mkdir(parents=True, exist_ok=True)
    filename_results_GD = str(case_folder / 'GD.csv')
    filename_results_PF = str(case_folder / 'PF_dtxNf_2x3.csv')
# elif test == '3':
#     tau_v = 
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

    mu_TF = fd.Constant(fd.sqrt(beta_v[0] / fd.pi))

    u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF - v)/beta_v[0]), 0.0)
# elif test == '3':
#     tau_v = 
else:
    NameError('test must be 1, 2 or 3')

bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4))]

if is_save_CSV:
    with open(filename_results_GD, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])

    with open(filename_results_PF, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['fine_operator', 'coarse_operator', 'h', 'beta', 'N_fine', 'N_coarse', 'tau_fine', 'tau_coarse', 'energy', 'lambda', 'iterate_coarse', 'iterate_fine', 'iterate', 'error', 'total_time', 'mean_time'])

orig_stdout = sys.stdout
t_start = time.time()

only_for_print_time = 0

for beta in beta_v:
    optim = ParaflowS(beta,v,W, bcs, 12 * 2**(-8))
    optim_GD = Gradient_Descent(beta,v,W, bcs, 12 * 2**(-8))

    for tau in tqdm(tau_v):
        for name_fine in methods_fine:
            optim_GD.compile(u0, tau, E_ref[beta],grad_type = name_fine)

            filename = 'GD'+ name_fine + '_tau' + str(tau)
            if is_save_log:
                f = open(str(folder / (filename + '.log')), 'w')
                sys.stdout = f
            res = optim_GD.minimize(MaxIter, toll, is_save_log)
            if is_save_plot:
                optim_GD.plot_history(filesave = str(folder / (filename + '.png')))

            if is_save_CSV:
                optim_GD.save_data(filename_results_GD, res)

            if is_save_log:
                f.close()

            #for name_coarse in methods_coarse:
            name_coarse = name_fine
            for Nf in Nf_v:
                for Ng in Ng_v:

                    optim.compile(u0, tau, tau * Nf / 2, E_ref[beta],grad_type_coarse=name_coarse, grad_type_fine = name_fine, Nf = Nf, Ng = Ng)

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