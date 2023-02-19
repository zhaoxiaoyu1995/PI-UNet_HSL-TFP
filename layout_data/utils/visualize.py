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
