import numpy as np
import matplotlib.pyplot as plt

def Paraflow(f, g, x0, MaxIter, toll, lr, Nf, val):
    x = x0
    x_coarse = x0

    x_old = x0

    history_f = []
    history_points = []
    history_fine = []
    history_points.append(x0)
    history_f.append(f(x0))
    history_fine.append(True)
    for i in range(MaxIter):
        x_coarse = x_old - lr*Nf * g(x_old)

        for j in range(Nf):
            x_old = x_old - lr * g(x_old)
            history_points.append(x_old)
            history_f.append(abs(f(x_old) - val))
            history_fine.append(True)

        # print(f'xcoarse: {x_coarse}, xold: {x_old}')
        correction = x_old - x_coarse
        error_old = abs(f(x_old) - val)
        # print(f'Outer Iter {i}, point {x_old} and error: {error_old}, correction norm: {np.linalg.norm(correction)}')
        figure, ax = plt.subplots(  1,1, figsize = (12,12))
        first = True
        if i == 0:
            for l, fine in enumerate(history_fine):
                if l == 0:
                    ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#e67e22', label = 'Fine solver')
                elif fine:
                    ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#e67e22')#, label = 'Fine solver')
                else:
                    if first:
                        ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#2c3e50', label = 'Correction')
                        first = False
                    else:
                        ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#2c3e50')#, label = 'Correction')
            ax.set_yscale('log')
            ax.set_xlim(-0.5,55)
            ax.set_ylim(3e-5, 40)
            ax.set_xlabel('Iteration', fontsize = 30)
            ax.set_ylabel('Objective function value', fontsize = 30)
            ax.legend(loc = 'upper right')
            plt.grid(True, alpha=0.5, linestyle='-')
            plt.show()

        for k in range(10):
            x_coarse = x_old - lr * Nf * g(x_old)

            x = x_coarse + correction
            # print(f'    Inner Iter {k}, point {x}, xoarse {x_coarse} and correction {correction}')

            error = abs(f(x) - val)
            

            print(f'    Iter {k}, point {x} and error: {error}')

            history_points.append(x_old)
            history_f.append(error)
            history_fine.append(False)
            if i == 0:
                figure, ax = plt.subplots(  1,1, figsize = (12,12))
                first = True
                for l, fine in enumerate(history_fine):
                    if l == 0:
                        ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#e67e22', label = 'Fine solver')
                    elif fine:
                        ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#e67e22')#, label = 'Fine solver')
                    else:
                        if first:
                            ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#2c3e50', label = 'Correction')
                            first = False
                        else:
                            ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#2c3e50')#, label = 'Correction')
                ax.set_yscale('log')
                ax.set_xlim(-0.5,55)
                ax.set_ylim(3e-5, 40)
                ax.set_xlabel('Iteration', fontsize = 30)
                ax.set_ylabel('Objective function value', fontsize = 30)
                ax.legend(loc = 'upper right')
                plt.grid(True, alpha=0.5, linestyle='-')
                plt.show()
            if error > error_old:
                x = x_old
                break

            x_old = x.copy()

            error_old = error

        print(f'Outer Iter {i}, point {x_old} and error: {error_old}')

        if i >0:
            for l, fine in enumerate(history_fine):
                if l == 0:
                    ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#e67e22', label = 'Fine solver')
                elif fine:
                    ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#e67e22')#, label = 'Fine solver')
                else:
                    if first:
                        ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#2c3e50', label = 'Correction')
                        first = False
                    else:
                        ax.plot(l, history_f[l], marker='o', linestyle='-', linewidth = 5, markersize = 10, c = '#2c3e50')#, label = 'Correction')
            ax.set_yscale('log')
            ax.set_xlim(-0.5,55)
            ax.set_ylim(3e-5, 40)
            ax.set_xlabel('Iteration', fontsize = 30)
            ax.set_ylabel('Objective function value', fontsize = 30)
            ax.legend(loc = 'upper right')
            plt.grid(True, alpha=0.5, linestyle='-')
            plt.show()
        if error_old < toll:
            break

    return history_f, history_points, history_fine

def GD(f, g, x0, MaxIter, toll, lr):
    x = x0
    history_f = []
    history_points_GD = []
    history_points_GD.append(x0)
    history_f.append(f(x0))
    for i in range(MaxIter):
        x = x - lr * g(x)
        error = abs(f(x))
        history_points_GD.append(x)
        history_f.append(error)
        # print(f'Iter {i}, point {x} and error: {error}')
        if error < toll:
            break
    return history_f, history_points_GD


f = lambda x: x[0]**4 + x[0] + x[1]**2
g = lambda x: np.array([4*x[0]**3 + 1, 2*x[1]])

x0 = np.array([-1., 5.])
lr = 0.05
Nf = 6
val = -3. / 4.**(4./3.)


history_f, history_points, bool_fine = Paraflow(f = f, g = g, x0 = x0, MaxIter = 100, toll = 10**-4, lr = lr, Nf = Nf,val = val)
history_f_gd, history_points_gd = GD(f = f, g = g, x0 = x0, MaxIter = 1000, toll = 10**-3, lr = lr)

figure, ax = plt.subplots(  1,2, figsize = (12,5))
# ax[0].plot(history_f, marker = 'o', label = 'ParaFlow')
# ax[0].plot(history_f_gd, marker = 'x', label = 'Gradient Descent')
# ax[0].set_yscale('log')
# ax[0].set_xlabel('Iteration')
# ax[0].set_ylabel('Objective function value')
# ax[0].legend()
x_plot1 = np.linspace(-1.5,2.5,10)
y_plot1 = np.linspace(-1.5,2.5,10)
X, Y = np.meshgrid(x_plot1, y_plot1)
Z1 = g([X,Y])

cs = ax[0].quiver(X, Y, Z1[0], Z1[1], scale = 100)
figure.colorbar(cs, ax = ax[0], label = 'Objective function value')
ax[0].plot([p[0] for p in history_points_gd], [p[1] for p in history_points_gd],marker = 'o', label = 'Gradient Descent')

ax[0].plot([p[0] for p in history_points], [p[1] for p in history_points], label = 'ParaFlow')
for i, fine in enumerate(bool_fine):
    if fine:
        ax[0].plot(history_points[i][0], history_points[i][1], marker = 'x', color = 'red')
    else:
        ax[0].plot(history_points[i][0], history_points[i][1], marker = 'o', color = 'white')

        
x_plot = np.linspace(-1.5,2.5,100)
y_plot = np.linspace(-1.5,2.5,100)
X, Y = np.meshgrid(x_plot, y_plot)
Z = f([X,Y])

cs = ax[1].contourf(X, Y, Z, levels = 50, cmap = 'viridis')
figure.colorbar(cs, ax = ax[1], label = 'Objective function value')
ax[1].plot([p[0] for p in history_points_gd], [p[1] for p in history_points_gd],marker = 'o', label = 'Gradient Descent')

ax[1].plot([p[0] for p in history_points], [p[1] for p in history_points], label = 'ParaFlow')
for i, fine in enumerate(bool_fine):
    if fine:
        ax[1].plot(history_points[i][0], history_points[i][1], marker = 'x', color = 'red')
    else:
        ax[1].plot(history_points[i][0], history_points[i][1], marker = 'o', color = 'white')
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')
ax[1].legend()
plt.show()