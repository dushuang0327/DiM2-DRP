import os
import random
import numpy as np
import json
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
import pandas as pd

def _build_cancer_maps(cancer_label_csv, cancer_label_mapping_json=None):

    if cancer_label_csv is None or not os.path.exists(cancer_label_csv):
        return {}, {}

    cdf = pd.read_csv(cancer_label_csv)

    cls2id = {}
    if cancer_label_mapping_json and os.path.exists(cancer_label_mapping_json):
        with open(cancer_label_mapping_json, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        # 允许两种键：'class_name' 或 'cancer_label'
        keys = list(mapping.keys())
        sample_key = keys[0] if keys else None
        if sample_key is not None:
            # {name->id} 或 {id_str->id_int}
            # 统一转 dict[str,int]
            try:
                cls2id = {str(k): int(v) for k, v in mapping.items()}
            except Exception:
                # 兜底
                unique_classes = sorted(set(map(str, mapping.keys())))
                cls2id = {c: i for i, c in enumerate(unique_classes)}

    if not cls2id:
        if 'cancer_label' in cdf.columns:  # 文本类名
            classes = sorted(cdf['cancer_label'].astype(str).unique().tolist())
            cls2id = {c: i for i, c in enumerate(classes)}
        elif 'cancer_id' in cdf.columns:
            ids = sorted(cdf['cancer_id'].dropna().astype(int).unique().tolist())
            cls2id = {str(i): int(i) for i in ids}
        elif 'cancer_label_id' in cdf.columns:
            ids = sorted(cdf['cancer_label_id'].dropna().astype(int).unique().tolist())
            cls2id = {str(i): int(i) for i in ids}
        else:
            # 没找到可用列，就返回空映射
            return {}, {}

    cell_idx2cancer = {}
    if 'cell_idx' in cdf.columns:
        if 'cancer_label' in cdf.columns:
            for _, r in cdf.iterrows():
                cell_idx2cancer[int(r['cell_idx'])] = cls2id.get(str(r['cancer_label']), -1)
        elif 'cancer_id' in cdf.columns:
            for _, r in cdf.iterrows():
                cell_idx2cancer[int(r['cell_idx'])] = int(r['cancer_id'])
        elif 'cancer_label_id' in cdf.columns:
            for _, r in cdf.iterrows():
                cell_idx2cancer[int(r['cell_idx'])] = int(r['cancer_label_id'])
    return cls2id, cell_idx2cancer


def get_target_class_weights(self):
    # 统计各类别频次（忽略 -1）
    from collections import Counter
    labels = [v for v in self.drug_idx2target.values() if v >= 0]
    if len(labels) == 0 or self.num_target_classes == 0:
        return None
    cnt = Counter(labels)
    # 逆频率 → 越少的类权重越大
    weights = [1.0 / cnt.get(i, 1) for i in range(self.num_target_classes)]
    # 归一化到均值=1，避免过大
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    import torch
    return torch.tensor(weights, dtype=torch.float32)

def build_drug_target_map(drug_smiles_csv):
    import pandas as pd
    df = pd.read_csv(drug_smiles_csv)
    # 只保留有有效 target 的条目
    df = df[['drug_idx','target']].copy()
    df['target'] = df['target'].astype(str).str.strip()
    valid = df[df['target'].str.upper()!='N/A'].copy()
    classes = sorted(valid['target'].unique().tolist())
    cls2id = {c:i for i,c in enumerate(classes)}
    # 对于无标签（N/A）的药物，标注为 -1
    id_map = dict(df.apply(lambda r: (int(r['drug_idx']),
                                      cls2id[r['target']] if str(r['target']).upper()!='N/A' else -1), axis=1).values)
    return cls2id, id_map




class NPweightingDataSet(Dataset):
    def __init__(self,
                 response_fpath,
                 drug_input,
                 exprs_input,
                 mut_input,
                 response_type='IC50',
                 drug_smiles_csv=None,
                 cancer_label_csv=None,                 # <--- 新增
                 cancer_label_mapping_json=None):       # <--- 新增
        ...
        # 现有：target 映射（药物靶点）
        self.target_cls2id, self.drug_idx2target = ({}, {})
        if drug_smiles_csv is not None:
            self.target_cls2id, self.drug_idx2target = build_drug_target_map(drug_smiles_csv)
        self.response_type = response_type
        # 新增：癌种标签映射（基于 cell_idx）
        self.cancer_cls2id, self.cell_idx2cancer = _build_cancer_maps(
            cancer_label_csv, cancer_label_mapping_json
        )
        self.response_df = pd.read_csv(response_fpath, sep=',', header=0)
        self.drug_fp_df = pd.read_csv(drug_input, sep=',', header=0)
        self.cell_exprs_df = pd.read_csv(exprs_input, sep=',', header=0)
        self.mut_score_df = pd.read_csv(mut_input, sep=',', header=0)

        # 可选：保证索引列为 int，避免后续 key 查找类型不一致
        for df in (self.response_df, self.drug_fp_df, self.cell_exprs_df, self.mut_score_df):
            for key in ('cell_idx', 'drug_idx'):
                if key in df.columns:
                    df[key] = pd.to_numeric(df[key], errors='coerce').astype('Int64')

    # === 在 NPweightingDataSet 类内新增属性 ===
    @property
    def num_target_classes(self):
        return len(self.target_cls2id)

    @property
    def num_cancer_classes(self):
        return len(self.cancer_cls2id)

    def __len__(self):
        return self.response_df.shape[0]

# === 在 __getitem__ 里补充 cancer_label，并把返回的 tuple 扩展为五元组 ===
    def __getitem__(self, index):
        response_sample = self.response_df.iloc[index]
        cell_idx = response_sample['cell_idx']
        drug_idx = response_sample['drug_idx']

        response_val = torch.tensor(response_sample[self.response_type]).float()
        cell_vec = torch.from_numpy(
            np.array(self.cell_exprs_df[self.cell_exprs_df['cell_idx'] == cell_idx].iloc[:, 3:].astype(float))
        ).float().squeeze(0)

        drug_vec = torch.from_numpy(
            np.array(self.drug_fp_df[self.drug_fp_df['drug_idx'] == drug_idx].iloc[:, 2:])).float().squeeze(0)
        mut_score = torch.from_numpy(
            np.array(self.mut_score_df[self.mut_score_df['cell_idx'] == cell_idx].iloc[:, 2:])).float().squeeze(0)

        # 已有：药物靶点标签（没有则 -1）
        if self.drug_idx2target:
            target_label = torch.tensor(self.drug_idx2target.get(int(drug_idx), -1), dtype=torch.long)
        else:
            target_label = torch.tensor(-1, dtype=torch.long)

        # 新增：癌种标签（cell_idx -> label_id，没找到则 -1）
        cancer_label = torch.tensor(self.cell_idx2cancer.get(int(cell_idx), -1), dtype=torch.long)

        # 返回： (inputs_with_labels, regression_y)
        return (drug_vec, cell_vec, mut_score, target_label, cancer_label), response_val



