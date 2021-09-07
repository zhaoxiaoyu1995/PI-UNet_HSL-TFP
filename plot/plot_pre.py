import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# Plot layout and temperature field of simple cases
# load_path1 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/pre/simple/simple_18_0.0049846298061311245.mat'
# load_path2 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/pre/simple/simple_3_0.0038084937259554863.mat'
# load_path1 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/pre/complex/complex_10_0.009652662090957165.mat'
# load_path2 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/pre/complex/complex_20_0.013872256502509117.mat'
load_path1 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_1_28.202131271362305.mat'
load_path2 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_reflect_1_0.03061039000749588.mat'
data1 = sio.loadmat(load_path1)
data2 = sio.loadmat(load_path2)
plt.figure(figsize=(25, 10))

x = np.linspace(0, 0.1, 200)
y = np.linspace(0, 0.1, 200)

plt.subplot(2, 4, 1)
plt.imshow(data1['layout'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('Layout', fontsize=14)

plt.subplot(2, 4, 2)
plt.contourf(x, y, data1['u'], levels=100, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('FDM Computation', fontsize=14)

plt.subplot(2, 4, 3)
plt.contourf(x, y, data1['u_pre'], levels=100, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('PI-UNet Prediction', fontsize=14)

plt.subplot(2, 4, 4)
plt.contourf(x, y, data1['u_pre'] - data1['u'], levels=40, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('Error', fontsize=14)

plt.subplot(2, 4, 5)
plt.imshow(data2['layout'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('Layout', fontsize=14)

plt.subplot(2, 4, 6)
plt.contourf(x, y, data2['u'], levels=100, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('FDM Computation', fontsize=14)

plt.subplot(2, 4, 7)
plt.contourf(x, y, data2['u_pre'], levels=100, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('PI-UNet Prediction', fontsize=14)

plt.subplot(2, 4, 8)
plt.contourf(x, y, data2['u_pre'] - data2['u'], levels=40, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('Error', fontsize=14)

plt.show()
# plt.savefig('case1_pre.pdf', bbox_inches='tight', pad_inches=0)