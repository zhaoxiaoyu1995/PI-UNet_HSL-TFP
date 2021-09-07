# -*- coding: utf-8 -*-
# @Time    : 2021/7/28 11:05
# @Author  : zhaoxiaoyu
# @File    : plot_batch.py
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set_style('whitegrid')


data = pd.read_excel('data/ohem.xlsx', engine='openpyxl')
data = data.to_numpy()
plt.plot(data[:, 0], data[:, 1], linewidth=2.0, marker='x', label='P-OHEM')
plt.plot(data[:, 0], data[:, 2], linewidth=2.0, marker='*', label='L1')
plt.plot(data[:, 0], data[:, 3], linewidth=2.0, marker='+', label='MSE')

# plt.title('Case 1')
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.legend()
plt.yscale('log')
# plt.show()
plt.savefig('simple_ohem.pdf', bbox_inches='tight')

plt.close()
plt.plot(data[:, 0], data[:, 1 + 3], linewidth=2.0, marker='x', label='P-OHEM')
plt.plot(data[:, 0], data[:, 2 + 3], linewidth=2.0, marker='*', label='L1')
plt.plot(data[:, 0], data[:, 3 + 3], linewidth=2.0, marker='+', label='MSE')

# plt.title('Case 2')
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.legend()
plt.yscale('log')
# plt.show()
plt.savefig('complex_ohem.pdf', bbox_inches='tight')

