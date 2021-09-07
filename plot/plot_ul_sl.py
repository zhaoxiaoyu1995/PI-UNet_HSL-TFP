# -*- coding: utf-8 -*-
# @Time    : 2021/8/26 14:39
# @Author  : zhaoxiaoyu
# @File    : plot_padding.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style('whitegrid')

# Case 1
data = pd.read_csv('data/case1_ul_sl.csv')
plt.figure(figsize=(10, 5))
sns.catplot(x="values", y="metrics", hue="method", kind="bar", data=data, legend_out=False)

plt.xlabel("")
plt.ylabel("")
# plt.show()
plt.savefig('case1_ul_sl.pdf', bbox_inches='tight')

# Case 2
data = pd.read_csv('data/case2_ul_sl.csv')
ax = sns.catplot(x="values", y="metrics", hue="method", kind="bar", data=data, legend_out=False)

plt.xlabel("")
plt.ylabel("")
ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3', r"$\vdots$", '1.8', '1.9', '2.0', ' '])
# plt.show()
plt.savefig('case2_ul_sl.pdf', bbox_inches='tight')