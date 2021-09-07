import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


load_path1 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_1_28.202131271362305.mat'
load_path2 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_reflect_1_0.03061039000749588.mat'
data1 = sio.loadmat(load_path1)
data2 = sio.loadmat(load_path2)
plt.figure(figsize=(26, 5))

x = np.linspace(0, 0.1, 200)
y = np.linspace(0, 0.1, 200)

plt.subplot(1, 4, 1)
plt.contourf(x, y, data1['u'], levels=50, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('FDM Computation', fontsize=14)

plt.subplot(1, 4, 2)
plt.contourf(x, y, data2['u_pre'], levels=50, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('Off-grad of $T(x_i,y_j)$', fontsize=14)

plt.subplot(1, 4, 3)
plt.contourf(x, y, data1['u_pre'], levels=50, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('On-grad of $T(x_i,y_j)$', fontsize=14)


import seaborn as sb
sb.set_style('whitegrid')
import pandas as pd


data = pd.read_csv('data/on_off_grad.csv')
data = data.to_numpy()
plt.subplot(1, 4, 4)
plt.plot(data[:, 0], data[:, 1], linewidth=2.0, label='on-grad')
plt.plot(data[:, 0], data[:, 2], linewidth=2.0, label='off-grad')

plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.legend()
plt.yscale('log')
# plt.show()
plt.savefig('on_off_grad.pdf', bbox_inches='tight')