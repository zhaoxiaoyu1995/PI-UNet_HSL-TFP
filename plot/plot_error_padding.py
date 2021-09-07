import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# Plot layout and temperature field of simple cases
# load_path1 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_zeros_1_0.018095433712005615.mat'
# load_path2 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_reflect_10_0.009652662090957165.mat'
load_path1 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_zeros_10_0.06379574537277222.mat'
load_path2 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/complex_reflect_10_0.009652662090957165.mat'
load_path3 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/simple_zeros_0_0.07433939725160599.mat'
load_path4 = '/mnt/data3/zhaoxiaoyu/layout-data-master/plot/data/simple_reflect_0_0.0071641081012785435.mat'
data1 = sio.loadmat(load_path1)
data2 = sio.loadmat(load_path2)
data3 = sio.loadmat(load_path3)
data4 = sio.loadmat(load_path4)
plt.figure(figsize=(20, 8))

x = np.linspace(0, 0.1, 200)
y = np.linspace(0, 0.1, 200)

plt.subplot(2, 4, 1)
plt.imshow(data3['layout'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('Layout', fontsize=12)

plt.subplot(2, 4, 2)
plt.imshow(data3['loss'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.clim(0.0, 0.003)
plt.title('Zeros Padding', fontsize=12)

plt.subplot(2, 4, 3)
plt.imshow(data4['loss'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.clim(0.0, 0.003)
plt.title('Reflect Padding', fontsize=12)

plt.subplot(2, 4, 4)
data3_list = np.concatenate([
    data3['loss'][:3, :].reshape(-1, 1),
    data3['loss'][-3:, :].reshape(-1, 1),
    data3['loss'][3:-3, :3].reshape(-1, 1),
    data3['loss'][3:-3, -3:].reshape(-1, 1),
], axis=0)
data4_list = np.concatenate([
    data4['loss'][:3, :].reshape(-1, 1),
    data4['loss'][-3:, :].reshape(-1, 1),
    data4['loss'][3:-3, :3].reshape(-1, 1),
    data4['loss'][3:-3, -3:].reshape(-1, 1),
], axis=0)
plt.hist(data3_list, np.arange(0, 0.0025, 0.0025 / 16), alpha=0.75, label='zeros')
plt.hist(data4_list, np.arange(0, 0.0025, 0.0025 / 16), alpha=0.75, label='reflect')
plt.legend()
plt.title('Distribution of loss values in boundary', fontsize=12)

plt.subplot(2, 4, 5)
plt.imshow(data1['layout'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.title('Layout', fontsize=12)

plt.subplot(2, 4, 6)
plt.imshow(data1['loss'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.clim(0.0, 0.006)
plt.title('Zeros Padding', fontsize=12)

plt.subplot(2, 4, 7)
plt.imshow(data2['loss'], cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.clim(0.0, 0.006)
plt.title('Reflect Padding', fontsize=12)

plt.subplot(2, 4, 8)
data1_list = np.concatenate([
    data1['loss'][:3, :].reshape(-1, 1),
    data1['loss'][-3:, :].reshape(-1, 1),
    data1['loss'][3:-3, :3].reshape(-1, 1),
    data1['loss'][3:-3, -3:].reshape(-1, 1),
], axis=0)
data2_list = np.concatenate([
    data2['loss'][:3, :].reshape(-1, 1),
    data2['loss'][-3:, :].reshape(-1, 1),
    data2['loss'][3:-3, :3].reshape(-1, 1),
    data2['loss'][3:-3, -3:].reshape(-1, 1),
], axis=0)
plt.hist(data1_list, np.arange(0, 0.0035, 0.0035 / 16), alpha=0.75, label='zeros')
plt.hist(data2_list, np.arange(0, 0.0035, 0.0035 / 16), alpha=0.75, label='reflect')
plt.legend()
plt.title('Distribution of loss values in boundary', fontsize=12)

# plt.show()
plt.savefig('padding_distribution.pdf', bbox_inches='tight')