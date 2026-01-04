import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# 将2D3D信息和原ECFP指纹整合 生成新的输入

p_ecfp = r"D:\work2\DiM2-DRP-main\data\GDSC_ECFP.csv"
p_2d   = r"D:\work2\DiM2-DRP-main\data\drug_features_2D_30.csv"
p_3d   = r"D:\work2\DiM2-DRP-main\data\drug_features_3D_30.csv"

ecfp = pd.read_csv(p_ecfp)
f2d  = pd.read_csv(p_2d)
f3d  = pd.read_csv(p_3d)

# 仅保留 2D/3D 的数值特征列：我们主脚本导出的列名是 f2d_00..f2d_29 / f3d_00..f3d_29
# 如果你的列名不同，可改成用 select_dtypes(include=['number']) 再排除前两列。
keep2d = ['drug_idx','drug_name'] + [c for c in f2d.columns if c.startswith('f2d_')]
keep3d = ['drug_idx','drug_name'] + [c for c in f3d.columns if c.startswith('f3d_')]
f2d = f2d[keep2d].copy()
f3d = f3d[keep3d].copy()

# 用 drug_idx 连接，避免名字不一致
m = ecfp.merge(f2d, on=['drug_idx','drug_name'], how='left') \
        .merge(f3d, on=['drug_idx','drug_name'], how='left')

# 2D/3D 归一化到 [0,1]（ECFP 本来就是0/1不用动）
scaler = MinMaxScaler()
two_three_cols = [c for c in m.columns if c.startswith('f2d_') or c.startswith('f3d_')]
m[two_three_cols] = scaler.fit_transform(m[two_three_cols])

# 严格保证：前两列是 id+name，后面都是数值
feat_cols = m.columns[2:]
m[feat_cols] = m[feat_cols].apply(pd.to_numeric, errors='coerce')  # 非数值→NaN
m[feat_cols] = m[feat_cols].fillna(0.0).astype(np.float32)

out = r"D:\work2\DiM2-DRP-main\data\GDSC_ECFP_2D3D.csv"
m.to_csv(out, index=False)
print("saved:", out, m.shape)

