# 生成 2D 与 3D（增强版：非键边 + 多构象 + 清洗/容错）
import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

from drug_3D_features import mol_to_3d_from_smiles
from drug_gnn import GNN_3DModel, drug_2d_encoder

# 关闭 RDKit 冗余输出（可选）
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass


# =========================
#  数据清洗与判定工具
# =========================
def clean_smiles(s: str) -> str:
    """
    - 去首尾空白
    - 解析失败则原样返回（由外层 try/except 跳过）
    - 去盐：仅保留最大片段
    - 解质子：尽量规范化
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    s = (s or "").strip()
    if not s:
        return s
    m = Chem.MolFromSmiles(s)
    if m is None:
        return s
    chooser = rdMolStandardize.LargestFragmentChooser()
    m = chooser.choose(m)
    try:
        uncharger = rdMolStandardize.Uncharger()
        m = uncharger.uncharge(m)
    except Exception:
        pass
    return Chem.MolToSmiles(m, isomericSmiles=True)


def has_metal_atom(s: str) -> bool:
    """粗略判定是否包含金属元素（常见配合物容易导致 3D 构型失败）"""
    from rdkit import Chem
    metals = {
        'Li','Na','K','Rb','Cs','Mg','Ca','Sr','Ba','Zn','Cu','Ni','Co','Fe','Mn',
        'Cr','V','Ti','Sc','Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd','Hg','Al',
        'Ga','In','Sn','Sb','Te','Pt','Au','Pb','Bi','As','Se'
    }
    m = Chem.MolFromSmiles(s)
    if m is None:
        return False
    return any(a.GetSymbol() in metals for a in m.GetAtoms())


# =========================
#  PyG 工具
# =========================
def _ensure_batch_attr_for_pyg(data_obj: Data):
    """确保 PyG Data 上存在 batch 属性（单分子 -> 全 0）"""
    if getattr(data_obj, 'batch', None) is None:
        num = data_obj.x.shape[0]
        device = data_obj.x.device if hasattr(data_obj.x, "device") else None
        data_obj.batch = torch.zeros(num, dtype=torch.long, device=device)
    return data_obj


def _build_nonbond_edges(pos: torch.Tensor, edge_index_bond: torch.Tensor,
                         k: int = 8, cutoff: float = 5.0):
    """
    基于 3D 坐标构造非键原子对边（kNN + 距离阈值）。
    返回: (edge_index_nb[2,E], edge_attr_nb[E,1])；若无可用边，返回 (None, None)。
    """
    n = pos.size(0)
    if n < 2:
        return None, None

    # 距离矩阵并屏蔽对角
    dmat = torch.cdist(pos, pos)
    dmat.fill_diagonal_(float('inf'))

    # 屏蔽成键对
    if edge_index_bond is not None and edge_index_bond.numel() > 0:
        i, j = edge_index_bond
        dmat[i, j] = float('inf')
        dmat[j, i] = float('inf')

    # 自适应 k，避免当 n-1 < k 时 topk 越界
    k_eff = min(k, max(1, n - 1))
    _, idx = torch.topk(-dmat, k_eff, dim=1)  # 取最近的 k_eff 个
    rows = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(idx)
    dsel = dmat[rows, idx]
    keep = torch.isfinite(dsel) & (dsel < cutoff)

    rows = rows[keep]
    cols = idx[keep]
    if rows.numel() == 0:
        return None, None

    dist = dsel[keep].unsqueeze(1)  # [E,1]
    ei = torch.stack([rows, cols], dim=0).long()
    # 双向
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    dist = torch.cat([dist, dist], dim=0)
    return ei, dist


def _pack_geo_mol(drug_atom: Data, drug_bond: Data, k_nb=8, cutoff=5.0):
    """
    将 atom/bond 两个图对象打包成 GeoGNN 期望的命名空间对象，
    并补上非键原子对（edge_index_nb/edge_attr_nb）。
    """
    from types import SimpleNamespace
    mol = SimpleNamespace()
    mol.x_atom = drug_atom.x
    mol.edge_index_atom = drug_atom.edge_index
    mol.edge_attr_atom = drug_atom.edge_attr
    mol.x_atom_batch = drug_atom.batch

    mol.x_bond = drug_bond.x
    mol.edge_index_bond = drug_bond.edge_index
    mol.edge_attr_bond = drug_bond.edge_attr
    mol.x_bond_batch = torch.zeros(mol.x_bond.size(0), dtype=torch.long, device=mol.x_bond.device)

    # 非键边（需要 drug_atom.pos）
    if hasattr(drug_atom, "pos") and drug_atom.pos is not None:
        ei_nb, da_nb = _build_nonbond_edges(drug_atom.pos, drug_atom.edge_index, k=k_nb, cutoff=cutoff)
        if ei_nb is not None:
            mol.edge_index_nb = ei_nb.to(drug_atom.x.device)
            mol.edge_attr_nb = da_nb.to(drug_atom.x.device)
        else:
            mol.edge_index_nb = None
            mol.edge_attr_nb = None
    else:
        mol.edge_index_nb = None
        mol.edge_attr_nb = None

    return mol


# =========================
#  模型构建
# =========================
def build_models(device: torch.device):
    """
    两个编码器都输出 30 维：
      - 2D：drug_2d_encoder(embed_dim=30)
      - 3D：GNN_3DModel(embed_dim=30, readout='mean')
    """
    enc2d_cfg = dict(embed_dim=30, dropout_rate=0.1, layer_num=3, readout='mean')
    enc3d_cfg = dict(embed_dim=30, dropout_rate=0.1, layer_num=3, readout='mean')

    enc2d = drug_2d_encoder(enc2d_cfg).to(device).eval()
    enc3d = GNN_3DModel(enc3d_cfg).to(device).eval()
    return enc2d, enc3d


# =========================
#  单个 SMILES 编码
# =========================
@torch.no_grad()
def encode_one(smiles: str, enc2d, enc3d, device,
               k_nb: int = 8, cutoff: float = 5.0, n_conf: int = 1):
    """
    从 SMILES 生成 2D 和 (增强版) 3D 特征:
      - 2D: drug_2d_encoder(drug_atom) -> [1, 30]
      - 3D: GNN_3DModel(mol 或 mol_list) -> graph_repr -> [1, 30]
      - 非键距离: 由 _pack_geo_mol() 构建 edge_index_nb / edge_attr_nb
      - 多构象: n_conf>1 时生成多个构象并做均值（forward 已支持 list/tuple）
      - 金属分子：保留 2D，3D 置零
    """
    # 清洗
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        raise ValueError("Empty SMILES")
    s = clean_smiles(smiles)

    # 金属配合物：仅 2D，3D 置零
    if has_metal_atom(s):
        drug_atom, drug_bond = mol_to_3d_from_smiles(s)  # 若此处也失败，会被外层 try/except 捕获
        if (drug_atom is None) or (drug_bond is None):
            raise ValueError("Metal molecule graph build failed")
        drug_atom = _ensure_batch_attr_for_pyg(drug_atom).to(device)
        f2d = enc2d(drug_atom).squeeze(0).detach().cpu().numpy()
        f3d = np.zeros((30,), dtype=np.float32)
        return f2d, f3d

    # 非金属：正常流程
    if n_conf <= 1:
        drug_atom, drug_bond = mol_to_3d_from_smiles(s)

        if (drug_atom is None) or (drug_bond is None):
            raise ValueError("mol_to_3d_from_smiles returned None")
        if getattr(drug_atom, "x", None) is None or getattr(drug_bond, "x", None) is None:
            raise ValueError("Graph build failed: missing x")
        if getattr(drug_atom, "pos", None) is None:
            raise ValueError("drug_atom.pos missing (3D coordinates)")

        # 2D
        drug_atom = _ensure_batch_attr_for_pyg(drug_atom).to(device)
        f2d = enc2d(drug_atom).squeeze(0).detach().cpu().numpy()

        # 3D（增强版：含非键边）
        drug_bond = _ensure_batch_attr_for_pyg(drug_bond).to(device)
        mol = _pack_geo_mol(drug_atom, drug_bond, k_nb=k_nb, cutoff=cutoff)
        if getattr(mol, "x_atom", None) is None:
            raise ValueError("Packed mol missing x_atom")

        _, _, graph_repr = enc3d(mol)  # enc3d = GNN_3DModel(...)
        f3d = graph_repr.squeeze(0).detach().cpu().numpy()
        return f2d, f3d

    # 多构象
    f2d = None
    mol_list = []
    for seed in range(int(n_conf)):
        try:
            drug_atom, drug_bond = mol_to_3d_from_smiles(s, seed=seed)
        except TypeError:
            drug_atom, drug_bond = mol_to_3d_from_smiles(s)

        if (drug_atom is None) or (drug_bond is None):
            raise ValueError("mol_to_3d_from_smiles returned None (multi-conf)")
        if getattr(drug_atom, "x", None) is None or getattr(drug_bond, "x", None) is None:
            raise ValueError("Graph build failed: missing x (multi-conf)")
        if getattr(drug_atom, "pos", None) is None:
            raise ValueError("drug_atom.pos missing (multi-conf)")

        drug_atom = _ensure_batch_attr_for_pyg(drug_atom).to(device)
        drug_bond = _ensure_batch_attr_for_pyg(drug_bond).to(device)

        if f2d is None:
            f2d = enc2d(drug_atom).squeeze(0).detach().cpu().numpy()  # 2D 只算一次

        mol_list.append(_pack_geo_mol(drug_atom, drug_bond, k_nb=k_nb, cutoff=cutoff))

    _, _, graph_repr = enc3d(mol_list)
    f3d = graph_repr.squeeze(0).detach().cpu().numpy()
    return f2d, f3d


# =========================
#  主流程
# =========================
def main(
    csv_path=r"D:\work2\DiM2-DRP-main\data\GDSC_drug_smiles.csv",
    out_dir=r"D:\work2\DiM2-DRP-main\data",
    dedup_key="drug_idx",          
    k_nb=12, cutoff=6.0, n_conf=5  
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ---- Quizartinib SMILES 快修（原 CSV 第 302 行括号异常）----
    if "drug_name" in df.columns and "smiles" in df.columns:
        mask = (df['drug_name'] == 'Quizartinib') & (df['smiles'].astype(str).str.startswith('(C)C1', na=False))
        if mask.any():
            df.loc[mask, 'smiles'] = df.loc[mask, 'smiles'].astype(str).str.replace(r'^\(C\)C1', 'CC1', regex=True)

    # 列检查
    need_cols = ["drug_idx", "drug_name", "smiles", "target"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"输入CSV缺少列: {c}")
    df = df[need_cols].copy()

    # 去重（同一药物可能多条记录）
    if dedup_key in df.columns:
        df_uni = df.dropna(subset=["smiles"]).drop_duplicates(subset=[dedup_key]).reset_index(drop=True)
    else:
        df_uni = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc2d, enc3d = build_models(device)

    recs_2d, recs_3d = [], []
    for _, row in tqdm(df_uni.iterrows(), total=len(df_uni), desc="Encoding drugs (2D + GNN_3D 3D)"):
        meta = {k: row[k] for k in ["drug_idx", "drug_name", "smiles", "target"]}

        # 清洗 SMILES
        smiles = clean_smiles(str(meta["smiles"]))
        if not smiles:
            print(f"[WARN] {meta['drug_idx']} 空SMILES，已跳过")
            continue

        try:
            f2d, f3d = encode_one(
                smiles, enc2d, enc3d, device,
                k_nb=k_nb, cutoff=cutoff, n_conf=n_conf
            )
        except Exception as e:
            print(f"[WARN] 处理 {meta['drug_idx']} 失败：{e}")
            continue

        recs_2d.append({**meta, **{f"f2d_{i:02d}": float(f2d[i]) for i in range(30)}})
        recs_3d.append({**meta, **{f"f3d_{i:02d}": float(f3d[i]) for i in range(30)}})

    out_2d = pd.DataFrame(recs_2d)
    out_3d = pd.DataFrame(recs_3d)

    p2d = os.path.join(out_dir, "drug_features_2D_30.csv")
    p3d = os.path.join(out_dir, "drug_features_3D_30.csv")
    out_2d.to_csv(p2d, index=False)
    out_3d.to_csv(p3d, index=False)

    merged = out_2d.merge(out_3d, on=["drug_idx", "drug_name", "smiles", "target"], how="inner")
    p60 = os.path.join(out_dir, "drug_features_2D3D_60.csv")
    merged.to_csv(p60, index=False)
    try:
        X3 = merged.filter(like="f3d_").values
        print("[Check] ||3D|| mean:", float(np.linalg.norm(X3, axis=1).mean()), " var:", float(X3.var()))
    except Exception:
        pass

    print(f"[OK] 保存完成：\n  2D: {p2d}\n  3D: {p3d}\n  2D+3D: {p60}")


if __name__ == "__main__":
    main()
