"""第17讲 拓展：客户聚类与解释
业务问题：算法分群能否提供比人工RFM更多的线索？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

from scipy.cluster.vq import kmeans2
# %% 先缩小长尾，再标准化，防止金额单位主导距离
rfm=rfm_table();features=['recency_days','frequency','monetary_cents']
raw=np.log1p(rfm[features].to_numpy(dtype=float));mu=raw.mean(axis=0);sd=raw.std(axis=0,ddof=1)
assert (sd>0).all();Z=(raw-mu)/sd
best=None
for seed in range(10):
    centers,label=kmeans2(Z,3,iter=50,minit='++',rng=np.random.default_rng(100+seed),missing='raise')
    inertia=float(((Z-centers[label])**2).sum())
    if best is None or inertia<best[0]:best=(inertia,centers,label)
rfm['cluster']=best[2]
profile=rfm.groupby('cluster',as_index=False).agg(n=('customer_id','size'),
    median_recency=('recency_days','median'),median_frequency=('frequency','median'),median_value_cents=('monetary_cents','median'))
show(profile);save_table(profile,17,'cluster_profile');save_table(rfm,17,'cluster_customers')
fig,ax=plt.subplots(figsize=(7,4))
for g,color in zip(sorted(rfm.cluster.unique()),[BLUE,TEAL,ORANGE]):
    s=rfm.loc[rfm.cluster.eq(g)];ax.scatter(s.recency_days,s.monetary_cents/100,s=10,alpha=.5,label=f'Cluster {g}',color=color)
ax.set(title='Exploratory customer clusters',xlabel='Recency days',ylabel='Observed net value CNY');ax.legend(fontsize=8)
save_figure(fig,17,'clusters')
print('多初始值降低局部最优风险；k=3仅为教学选择，不能据此宣称存在3种真实客户本质类型。')
