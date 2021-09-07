# -*- coding: utf-8 -*-
# @Time    : 2021/8/26 14:39
# @Author  : zhaoxiaoyu
# @File    : plot_padding.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style('whitegrid')

# Case 1
data = pd.read_csv('data/padding_case1.csv')
sns.catplot(x="Index", y="Values", hue="Padding Mode", kind="bar", data=data, legend_out=False)

plt.xlabel("")
plt.ylabel("")
plt.show()
# plt.savefig('case1_padding.pdf', bbox_inches='tight')

# Case 2
data = pd.read_csv('data/padding_case2.csv')
sns.catplot(x="Index", y="Values", hue="Padding Mode", kind="bar", data=data, legend_out=False)

plt.xlabel("")
plt.ylabel("")
plt.show()
# plt.savefig('case2_padding.pdf', bbox_inches='tight')