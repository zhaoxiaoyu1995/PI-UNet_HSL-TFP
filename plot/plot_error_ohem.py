import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


load_path1 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_1_0.18042431771755219.mat'
load_path2 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_reflect_1_0.03061039000749588.mat'
data1 = sio.loadmat(load_path1)
data2 = sio.loadmat(load_path2)
# plt.figure(figsize=(10, 5))

x = np.linspace(0, 0.1, 200)
y = np.linspace(0, 0.1, 200)

plt.imshow(data1['loss'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('Layout', fontsize=12)

plt.show()
# plt.savefig('padding_distribution.pdf', bbox_inches='tight')