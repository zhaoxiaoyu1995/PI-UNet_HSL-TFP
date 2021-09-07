# -*- coding: utf-8 -*-
# @Time    : 2021/7/28 11:05
# @Author  : zhaoxiaoyu
# @File    : plot_batch.py
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set_style('whitegrid')


data = pd.read_csv('data/on_off_grad.csv')
data = data.to_numpy()
plt.plot(data[:, 0], data[:, 1], linewidth=2.0, label='on-grad')
plt.plot(data[:, 0], data[:, 2], linewidth=2.0, label='off-grad')

plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.legend()
plt.yscale('log')
# plt.show()
plt.savefig('on_off_grad.pdf', bbox_inches='tight')
