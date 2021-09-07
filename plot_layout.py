import scipy.io as sio
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# Plot layout and temperature field of simple cases
load_path = '/mnt/zhaoxiaoyu/data/layout_data/ul/200x200_0.045_0.055/simple_component/FDM/train/Example8001.mat'
data = sio.loadmat(load_path)
plt.imshow(data['F'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.show()
# plt.savefig('simple_layout.pdf', bbox_inches='tight', pad_inches=0)
plt.close()
x = np.linspace(0, 0.1, 200)
y = np.linspace(0, 0.1, 200)
plt.contourf(x, y, data['F'], levels=100, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.show()
# plt.savefig('simple_u.pdf', bbox_inches='tight', pad_inches=0)
plt.close()


# Plot layout and temperature field of complex cases
load_path = '/mnt/zhaoxiaoyu/data/layout_data/ul/200x200_0.045_0.055/complex_component/FDM/train/Example8002.mat'
data = sio.loadmat(load_path)
plt.imshow(data['F'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.show()
# plt.savefig('complex_layout.pdf', bbox_inches='tight', pad_inches=0)
plt.close()
x = np.linspace(0, 0.1, 200)
y = np.linspace(0, 0.1, 200)
plt.contourf(x, y, data['F'], levels=100, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.show()
# plt.savefig('complex_u.pdf', bbox_inches='tight', pad_inches=0)