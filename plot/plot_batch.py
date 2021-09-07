# -*- coding: utf-8 -*-
# @Time    : 2021/7/28 11:05
# @Author  : zhaoxiaoyu
# @File    : plot_batch.py
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set_style('whitegrid')


data = pd.read_excel('data/batch.xlsx', engine='openpyxl')
data = data.to_numpy()
plt.plot(data[:, 0], data[:, 1], linewidth=2.0, marker='x', label='1')
plt.plot(data[:, 0], data[:, 2], linewidth=2.0, marker='*', label='4')
plt.plot(data[:, 0], data[:, 3], linewidth=2.0, marker='+', label='8')
plt.plot(data[:, 0], data[:, 4], linewidth=2.0, marker='.', label='16')

plt.title('Case 1')
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.legend()
plt.yscale('log')
# plt.show()
plt.savefig('simple_batch.pdf', bbox_inches='tight')

plt.close()
plt.plot(data[:, 0], data[:, 1 + 4], linewidth=2.0, marker='x', label='1')
plt.plot(data[:, 0], data[:, 2 + 4], linewidth=2.0, marker='*', label='4')
plt.plot(data[:, 0], data[:, 3 + 4], linewidth=2.0, marker='+', label='8')
plt.plot(data[:, 0], data[:, 4 + 4], linewidth=2.0, marker='.', label='16')

plt.title('Case 2')
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.legend()
plt.yscale('log')
# plt.show()
plt.savefig('complex_batch.pdf', bbox_inches='tight')

