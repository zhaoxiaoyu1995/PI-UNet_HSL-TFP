import matplotlib
import matplotlib.pyplot as plt
import numpy as np

TOL = 1e-14


def visualize_heatmap(x, y, heat_list, heat_pre_list, epoch):
    plt.figure(figsize=(18, 25))
    num = len(heat_list)
    for i in range(num):
        plt.subplot(num, 3, i * 3 + 1)
        plt.contourf(x, y, heat_list[i], levels=50, cmap=matplotlib.cm.coolwarm)
        plt.colorbar()
        plt.title('True')
        plt.subplot(num, 3, i * 3 + 2)
        plt.contourf(x, y, heat_pre_list[i], levels=50, cmap=matplotlib.cm.coolwarm)
        plt.colorbar()
        plt.title('Prediction')
        plt.subplot(num, 3, i * 3 + 3)
        plt.contourf(x, y, heat_pre_list[i] - heat_list[i], levels=50, cmap=matplotlib.cm.coolwarm)
        plt.colorbar()
        plt.title('Error')
    plt.savefig('figure/epoch' + str(epoch) + '_pre.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def plotBC(ax, bcs, h):
    for line in bcs:
        (lx, ly), (rx, ry) = line
        delta_x = rx - lx
        delta_y = ry - ly
        assert delta_x < TOL or delta_y < TOL, 'check the boundary conditions'
        if delta_x > TOL:
            x = np.arange(lx, rx, h)
            y = np.ones_like(x) * ly
            ax.plot(x, y, '-o', color='blue')
        else:
            y = np.arange(ly, ry, h)
            x = np.ones_like(y) * lx
            ax.plot(x, y, '-o', color='blue')
    return ax


def plotMesh(ax, x, y, width=0.05):
    [ny, nx] = x.shape
    for j in range(0, nx):
        ax.plot(x[:, j], y[:, j], color='black', linewidth=width)
    for i in range(0, ny):
        ax.plot(x[i, :], y[i, :], color='black', linewidth=width)
    return ax


def setAxisLabel(ax, type):
    if type == 'p':
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
    elif type == 'r':
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')
    else:
        raise ValueError('The axis type only can be reference or physical')