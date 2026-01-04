# # === Fixed === #
# import os
# import random
# import argparse
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore")
# from collections import defaultdict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from sklearn.metrics import mean_squared_error
# from lifelines.utils import concordance_index
# from scipy.stats import pearsonr,spearmanr
# import scipy.stats as stats
#
# import sys
# sys.path.append('..')
# from copy import deepcopy
# from dataset import NPweightingDataSet
# from utils import *
# from trainers import logging, train, validate, test
#
#
# class CrossBlock(nn.Module):
#     """单层：pre-LN → Cross-Attn → 残差 → FFN → 残差"""
#     def __init__(self, dim, num_heads=4, attn_dropout=0.1, ffn_ratio=2.0, ffn_dropout=0.1):
#         super().__init__()
#         self.q_norm = nn.LayerNorm(dim)
#         self.kv_norm = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
#         self.attn_drop = nn.Dropout(attn_dropout)
#
#         hidden = int(ffn_ratio * dim)
#         self.ffn_norm = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, hidden),
#             nn.GELU(),
#             nn.Dropout(ffn_dropout),
#             nn.Linear(hidden, dim),
#             nn.Dropout(ffn_dropout),
#         )
#
#     def forward(self, q_seq, kv_seq):
#         # pre-LN
#         q = self.q_norm(q_seq)
#         kv = self.kv_norm(kv_seq)
#
#         # cross-attn
#         out, _ = self.attn(query=q, key=kv, value=kv)
#         q_seq = q_seq + self.attn_drop(out)
#
#         # FFN
#         q_seq = q_seq + self.ffn(self.ffn_norm(q_seq))
#         return q_seq
#
#
# class BiCrossAttentionFusion(nn.Module):
#     """
#     最强版：支持 CLS pooling + learnable position embedding + 多层堆叠
#     - 输入：drug (B,30)，gene (B,68=mut30+pathway38)
#     - 输出：drug_fused (B,30)，gene_fused (B,68)
#     """
#     def __init__(
#         self,
#         dim_drug: int,
#         dim_gene: int,
#         attn_dim: int = 64,
#         num_heads: int = 4,
#         n_layers: int = 2,
#         attn_dropout: float = 0.1,
#         ffn_ratio: float = 2.0,
#         ffn_dropout: float = 0.1,
#         use_pos_emb: bool = True,
#         use_cls: bool = True,
#     ):
#         super().__init__()
#         self.dim_drug, self.dim_gene = dim_drug, dim_gene
#         self.attn_dim = attn_dim
#         self.use_pos_emb = use_pos_emb
#         self.use_cls = use_cls
#
#         # --- token embedding（保留你对 gene 的“mut / pathway”拆分方式，更强表达） ---
#         self.drug_pre_norm = nn.LayerNorm(1)
#         self.gene_pre_norm = nn.LayerNorm(1)
#
#         self.drug_embed = nn.Linear(1, attn_dim)     # 30 标量 → 30×D
#         self.mut_embed  = nn.Linear(1, attn_dim)     # 30 标量 → 30×D
#         self.pathway_embed1 = nn.Linear(1, 8)        # 38 标量 → 38×8 → 38×D
#         self.pathway_embed2 = nn.Linear(8, attn_dim)
#
#         # --- CLS token（可学习） ---
#         if use_cls:
#             self.cls_drug = nn.Parameter(torch.zeros(1, 1, attn_dim))
#             self.cls_gene = nn.Parameter(torch.zeros(1, 1, attn_dim))
#             nn.init.trunc_normal_(self.cls_drug, std=0.02)
#             nn.init.trunc_normal_(self.cls_gene, std=0.02)
#
#         # --- 可学习位置编码（长度+1 因为加了 CLS） ---
#         if use_pos_emb:
#             self.pos_drug = nn.Parameter(torch.zeros(1, (dim_drug + (1 if use_cls else 0)), attn_dim))
#             self.pos_gene = nn.Parameter(torch.zeros(1, (dim_gene + (1 if use_cls else 0)), attn_dim))
#             nn.init.trunc_normal_(self.pos_drug, std=0.02)
#             nn.init.trunc_normal_(self.pos_gene, std=0.02)
#
#         # --- 双向 Cross 堆叠：drug←gene 与 gene←drug 各一套，每层交替增强 ---
#         self.blocks_d_from_g = nn.ModuleList([
#             CrossBlock(attn_dim, num_heads, attn_dropout, ffn_ratio, ffn_dropout) for _ in range(n_layers)
#         ])
#         self.blocks_g_from_d = nn.ModuleList([
#             CrossBlock(attn_dim, num_heads, attn_dropout, ffn_ratio, ffn_dropout) for _ in range(n_layers)
#         ])
#
#         # --- 池化后映射回原维度（用 CLS 作为全局表示；若不启用 CLS 则用 mean pooling） ---
#         self.out_drug = nn.Linear(attn_dim, dim_drug)
#         self.out_gene = nn.Linear(attn_dim, dim_gene)
#
#         # --- 残差 + LayerNorm + Dropout(0.1)（回原空间后再做一次 Transformer 风格收尾） ---
#         self.final_norm_drug = nn.LayerNorm(dim_drug)
#         self.final_norm_gene = nn.LayerNorm(dim_gene)
#         self.final_drop = nn.Dropout(0.1)
#
#         self.final_ffn_drug = nn.Sequential(
#             nn.Linear(dim_drug, 2 * dim_drug), nn.GELU(), nn.Linear(2 * dim_drug, dim_drug), nn.Dropout(0.1)
#         )
#         self.final_ffn_gene = nn.Sequential(
#             nn.Linear(dim_gene, 2 * dim_gene), nn.GELU(), nn.Linear(2 * dim_gene, dim_gene), nn.Dropout(0.1)
#         )
#
#     def _embed_drug(self, drug):
#         # (B,30) -> (B,30,1) -> preLN -> (B,30,D)
#         x = self.drug_pre_norm(drug.unsqueeze(2))
#         x = self.drug_embed(x)
#         if self.use_cls:
#             cls = self.cls_drug.expand(x.size(0), -1, -1)  # (B,1,D)
#             x = torch.cat([cls, x], dim=1)                 # (B,31,D)
#         if self.use_pos_emb:
#             x = x + self.pos_drug[:, :x.size(1), :]
#         return x  # (B,30(+1),D)
#
#     def _embed_gene(self, gene):
#         # gene = [mut(30) | pathway(38)]
#         mut  = self.gene_pre_norm(gene[:, :30].unsqueeze(2))
#         path = self.gene_pre_norm(gene[:, 30:].unsqueeze(2))
#
#         mut_e  = self.mut_embed(mut)                                     # (B,30,D)
#         path_e = self.pathway_embed2(F.relu(self.pathway_embed1(path)))  # (B,38,D)
#         x = torch.cat([mut_e, path_e], dim=1)                            # (B,68,D)
#
#         if self.use_cls:
#             cls = self.cls_gene.expand(x.size(0), -1, -1)                # (B,1,D)
#             x = torch.cat([cls, x], dim=1)                               # (B,69,D)
#         if self.use_pos_emb:
#             x = x + self.pos_gene[:, :x.size(1), :]
#         return x  # (B,68(+1),D)
#
#     def _pool(self, seq, use_cls: bool):
#         if use_cls:
#             return seq[:, 0, :]   # 取 CLS
#         else:
#             return seq.mean(dim=1)
#
#     def forward(self, drug, gene):
#         # ---- 1) token 化 + 位置编码 + 可学习 CLS ----
#         d = self._embed_drug(drug)   # (B,30/31,D)
#         g = self._embed_gene(gene)   # (B,68/69,D)
#
#         # ---- 2) 堆叠的双向 cross-attn（交替更新）----
#         for d_blk, g_blk in zip(self.blocks_d_from_g, self.blocks_g_from_d):
#             d = d_blk(d, g)  # drug 被 gene 引导
#             g = g_blk(g, d)  # gene 被 drug 引导
#
#         # ---- 3) 池化到全局向量（默认用 CLS）----
#         d_vec = self._pool(d, self.use_cls)  # (B,D)
#         g_vec = self._pool(g, self.use_cls)  # (B,D)
#
#         # ---- 4) 回原维度 + 最终残差/归一化/丢弃 + FFN ----
#         d_out = self.out_drug(d_vec)                     # (B,30)
#         g_out = self.out_gene(g_vec)                     # (B,68)
#
#         d_out = self.final_norm_drug(drug + self.final_drop(d_out))
#         d_out = d_out + self.final_ffn_drug(d_out)
#
#         g_out = self.final_norm_gene(gene + self.final_drop(g_out))
#         g_out = g_out + self.final_ffn_gene(g_out)
#         return d_out, g_out
#
#
# # ====== Random Seed Initialization ====== #
# def seed_everything(seed = 3078):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
# seed_everything()
#
# def createFolder(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print ('Error: Creating directory. ' +  directory)
#
#
# # ====== Model Definition ====== #
# class DeepAutoencoderThreeHiddenLayers(nn.Module):
#     def __init__(self, input_dim, hidden_dims, code_dim, activation_func=nn.ReLU,
#                  code_activation=True, dropout=False, dropout_rate=0.5):
#         super(DeepAutoencoderThreeHiddenLayers, self).__init__()
#         # Establish encoder
#         modules = []
#         modules.append(nn.Linear(input_dim, hidden_dims[0]))
#         modules.append(activation_func())
#         if dropout:
#             modules.append(nn.Dropout(dropout_rate))
#
#
#
#         for input_size, output_size in zip(hidden_dims, hidden_dims[1:]):
#
#             modules.append(nn.Linear(input_size, output_size))
#             modules.append(activation_func())
#             if dropout:
#                 modules.append(nn.Dropout(dropout_rate))
#
#         modules.append(nn.Linear(hidden_dims[-1], code_dim))
#         if code_activation:
#             modules.append(activation_func())
#         self.encoder = nn.Sequential(*modules)
#
#         # Establish decoder
#         modules = []
#
#         modules.append(nn.Linear(code_dim, hidden_dims[-1]))
#         modules.append(activation_func())
#         if dropout:
#             modules.append(nn.Dropout(dropout_rate))
#
#         for input_size, output_size in zip(hidden_dims[::-1], hidden_dims[-2::-1]):
#             modules.append(nn.Linear(input_size, output_size))
#             modules.append(activation_func())
#             if dropout:
#                 modules.append(nn.Dropout(dropout_rate))
#         modules.append(nn.Linear(hidden_dims[0], input_dim))
#         # modules.append(nn.Sigmoid())
#         self.decoder = nn.Sequential(*modules)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         code = x
#         x = self.decoder(x)
#         return code, x
#
#
# class ForwardNetworkTwoHiddenLayers(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, activation_func=nn.ReLU,
#                 out_activation=None):
#         super(ForwardNetworkTwoHiddenLayers, self).__init__()
#
#         self.layers = nn.Sequential(
#              nn.Linear(input_dim, hidden_dim1),
#              nn.BatchNorm1d(hidden_dim1),
#              activation_func(),
#              nn.Linear(hidden_dim1, hidden_dim2),
#              nn.BatchNorm1d(hidden_dim2),
#              activation_func(),
#              nn.Linear(hidden_dim2, 1))
#
#         self.out_activation = out_activation
#
#
#     def forward(self, x):
#         if self.out_activation:
#             return self.out_activation(self.layers(x))
#         else:
#             return self.layers(x)
#
# class DEERS_Concat(torch.nn.Module):
#     def __init__(self, drug_autoencoder, mut_line_autoencoder,
#                  forward_network,
#                  use_cross_attn: bool = True,
#                  attn_dim: int = 64,
#                  attn_heads: int = 4,
#                  attn_dropout: float = 0.1,
#                  num_target_classes: int = 0,
#                  num_cancer_classes: int = 0,
#                  ):     # <--- 新增
#         super(DEERS_Concat, self).__init__()
#         self.drug_autoencoder = drug_autoencoder
#         self.mut_line_autoencoder = mut_line_autoencoder
#         self.forward_network = forward_network
#         self.fusion_norm = nn.LayerNorm(30 + 68)
#         self.fusion_dropout = nn.Dropout(0.1)
#
#         self.use_cross_attn = use_cross_attn
#         if self.use_cross_attn:
#             # self.cross_fusion = BiCrossAttentionFusion(
#             #     dim_drug=30, dim_gene=30+38,
#             #     attn_dim=attn_dim, num_heads=attn_heads, dropout=attn_dropout
#             # )
#             self.cross_fusion = BiCrossAttentionFusion(
#                 dim_drug=30, dim_gene=30 + 38,
#                 attn_dim=64, num_heads=4,
#                 n_layers=2,  # 建议2或3
#                 attn_dropout=0.1, ffn_dropout=0.1,
#                 use_pos_emb=True, use_cls=True
#             )
#
#         # 现有：靶点分类头（基于 drug_code）
#         self.num_target_classes = num_target_classes
#         if num_target_classes and num_target_classes > 0:
#             self.target_head = nn.Sequential(
#                 nn.Linear(30, 64), nn.ReLU(), nn.Dropout(0.1),
#                 nn.Linear(64, num_target_classes)
#             )
#         else:
#             self.target_head = None
#
#         # 新增：癌种分类头（基于基因侧表示）
#         self.num_cancer_classes = num_cancer_classes
#         if num_cancer_classes and num_cancer_classes > 0:
#             self.cancer_head = nn.Sequential(
#                 nn.Linear(30+38, 128), nn.ReLU(), nn.Dropout(0.1),
#                 nn.Linear(128, num_cancer_classes)
#             )
#         else:
#             self.cancer_head = None
#
#     def forward(self, drug_features, mut_features, cell_features):
#         drug_code, drug_reconstruction = self.drug_autoencoder(drug_features)  # (B,30)
#         mut_code,  mut_reconstruction  = self.mut_line_autoencoder(mut_features)  # (B,30)
#
#         if not self.use_cross_attn:
#             gene_repr = torch.cat((mut_code, cell_features), dim=1)  # (B,68)
#             x = torch.cat((drug_code, mut_code, cell_features), dim=1)  # (B,98)
#             y_hat = self.forward_network(x)
#
#             target_logits = self.target_head(drug_code) if self.target_head is not None else None
#             cancer_logits = self.cancer_head(gene_repr) if self.cancer_head is not None else None
#             return y_hat, drug_reconstruction, mut_reconstruction, target_logits, cancer_logits
#
#         # 开启双向交叉注意力
#         gene_repr = torch.cat((mut_code, cell_features), dim=1)      # (B,68)
#         drug_fused, gene_fused = self.cross_fusion(drug_code, gene_repr)  # 维度仍是 (B,30) 和 (B,68)
#
#         x = torch.cat((drug_fused, gene_fused), dim=1)
#         x = self.fusion_norm(x)
#         x = self.fusion_dropout(x)
#         y_hat = self.forward_network(x)
#
#         target_logits = self.target_head(drug_fused) if self.target_head is not None else None
#         cancer_logits = self.cancer_head(gene_fused) if self.cancer_head is not None else None
#         return y_hat, drug_reconstruction, mut_reconstruction, target_logits, cancer_logits
#
#
#
# class MergedLoss(nn.Module):
#     def __init__(self, y_loss_weight=1., drug_reconstruction_loss_weight=0.1, mut_reconstruction_loss_weight=0.2):
#         super(MergedLoss, self).__init__()
#         self.y_loss_weight = y_loss_weight
#         self.drug_reconstruction_loss_weight = drug_reconstruction_loss_weight
#         self.mut_reconstruction_loss_weight = mut_reconstruction_loss_weight
#         self.output_criterion = nn.MSELoss()
#         # self.reconstruction_criterion = nn.BCELoss()
#         self.reconstruction_criterion = nn.MSELoss()
#
#     def forward(self, pred_y, drug_reconstruction, mut_reconstruction,drug_input,mut_input, true_y):
#         output_loss = self.output_criterion(pred_y, true_y)
#         drug_reconstruction_loss = self.reconstruction_criterion(drug_reconstruction, drug_input)
#         mut_reconstruction_loss = self.reconstruction_criterion(mut_reconstruction, mut_input)
#         return output_loss, drug_reconstruction_loss,mut_reconstruction_loss
#
# class DrugMultiEncoderWithRecon(nn.Module):
#     """
#     三路并行压缩：ECFP→20维，2D→5维，3D→5维；再拼成30维。
#     返回 (code_30, recon_full) 以兼容原来的重构损失。
#     """
#     def __init__(self, ecfp_dim: int, f2d_dim: int = 30, f3d_dim: int = 30,
#                  ecfp_code_dim: int = 20, f2d_code_dim: int = 5, f3d_code_dim: int = 5,
#                  dropout: float = 0.2):
#         super().__init__()
#         self.ecfp_dim, self.f2d_dim, self.f3d_dim = ecfp_dim, f2d_dim, f3d_dim
#         # ECFP 分支（大一点容量）
#         self.ecfp_enc = nn.Sequential(
#             nn.Linear(ecfp_dim, 300), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(300, 100), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(100, ecfp_code_dim)
#         )
#         self.ecfp_dec = nn.Sequential(
#             nn.Linear(ecfp_code_dim, 100), nn.ReLU(),
#             nn.Linear(100, 300), nn.ReLU(),
#             nn.Linear(300, ecfp_dim), nn.Sigmoid()
#         )
#         # 2D 分支
#         self.f2d_enc = nn.Sequential(
#             nn.LayerNorm(f2d_dim),
#             nn.Linear(f2d_dim, 32), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(32, f2d_code_dim)
#         )
#         self.f2d_dec = nn.Sequential(
#             nn.Linear(f2d_code_dim, 32), nn.ReLU(),
#             nn.Linear(32, f2d_dim), nn.Sigmoid()
#         )
#         # 3D 分支
#         self.f3d_enc = nn.Sequential(
#             nn.LayerNorm(f3d_dim),
#             nn.Linear(f3d_dim, 32), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(32, f3d_code_dim)
#         )
#         self.f3d_dec = nn.Sequential(
#             nn.Linear(f3d_code_dim, 32), nn.ReLU(),
#             nn.Linear(32, f3d_dim), nn.Sigmoid()
#         )
#         self.out_dim = ecfp_code_dim + f2d_code_dim + f3d_code_dim  # 30
#
#     def forward(self, drug_features):
#         # 切三段：ECFP | 2D | 3D
#         ecfp = drug_features[:, :self.ecfp_dim]
#         f2d  = drug_features[:, self.ecfp_dim:self.ecfp_dim+self.f2d_dim]
#         f3d  = drug_features[:, self.ecfp_dim+self.f2d_dim:]
#         # 编码
#         ecfp_code = self.ecfp_enc(ecfp)
#         f2d_code  = self.f2d_enc(f2d)
#         f3d_code  = self.f3d_enc(f3d)
#         code = torch.cat([ecfp_code, f2d_code, f3d_code], dim=1)  # (B,30)
#         # 重构（兼容原重构损失）
#         ecfp_rec = self.ecfp_dec(ecfp_code)
#         f2d_rec  = self.f2d_dec(f2d_code)
#         f3d_rec  = self.f3d_dec(f3d_code)
#         recon = torch.cat([ecfp_rec, f2d_rec, f3d_rec], dim=1)    # (B, in_dim)
#         return code, recon
#
# class MergedLossWithTarget(nn.Module):
#     def __init__(self, y_loss_weight=1.,
#                  drug_reconstruction_loss_weight=0.1,
#                  mut_reconstruction_loss_weight=0.2,
#                  target_loss_weight=0.5,
#                  ce_weight=None,
#                  label_smoothing=0.05,
#                  cancer_loss_weight=0.0,
#                  cancer_ce_weight=None,
#                  cancer_label_smoothing=0.0):
#         super().__init__()
#         self.y_loss_weight = y_loss_weight
#         self.drug_reconstruction_loss_weight = drug_reconstruction_loss_weight
#         self.mut_reconstruction_loss_weight  = mut_reconstruction_loss_weight
#         self.target_loss_weight = target_loss_weight
#         self.output_criterion = nn.MSELoss()
#         # self.reconstruction_criterion = nn.BCELoss
#         self.reconstruction_criterion = nn.MSELoss()
#         self.ce_weight = ce_weight
#         self.label_smoothing = label_smoothing
#         self.cancer_loss_weight = cancer_loss_weight
#         self.cancer_ce_weight = cancer_ce_weight
#         self.cancer_label_smoothing = cancer_label_smoothing
#
#
#     def forward(self, pred_y, drug_reconstruction, mut_reconstruction,
#                 drug_input, mut_input, true_y,
#                 target_logits=None, target_labels=None,
#                 cancer_logits=None, cancer_labels=None):
#
#         output_loss = self.output_criterion(pred_y, true_y)
#         drug_rec_loss = self.reconstruction_criterion(drug_reconstruction, drug_input)
#         mut_rec_loss  = self.reconstruction_criterion(mut_reconstruction,  mut_input)
#
#         cls_loss = torch.tensor(0.0, device=pred_y.device)
#         if (target_logits is not None) and (target_labels is not None):
#             mask = (target_labels >= 0)
#             if mask.any():
#                 weight = self.ce_weight.to(pred_y.device) if self.ce_weight is not None else None
#                 ce = F.cross_entropy(
#                     target_logits[mask], target_labels[mask],
#                     weight=weight, label_smoothing=self.label_smoothing, reduction='mean'
#                 )
#                 cls_loss = ce
#
#         cancer_loss = torch.tensor(0.0, device=pred_y.device)
#         if (cancer_logits is not None) and (cancer_labels is not None):
#             cmask = (cancer_labels >= 0)
#             if cmask.any():
#                 cweight = self.cancer_ce_weight.to(pred_y.device) if self.cancer_ce_weight is not None else None
#                 cce = F.cross_entropy(
#                     cancer_logits[cmask], cancer_labels[cmask],
#                     weight=cweight, label_smoothing=self.cancer_label_smoothing, reduction='mean'
#                 )
#                 cancer_loss = cce
#
#         total = (self.y_loss_weight * output_loss
#                  + self.drug_reconstruction_loss_weight * drug_rec_loss
#                  + self.mut_reconstruction_loss_weight * mut_rec_loss
#                  + self.target_loss_weight * cls_loss
#                  + self.cancer_loss_weight * cancer_loss)
#
#         return total, output_loss, drug_rec_loss, mut_rec_loss, cls_loss, cancer_loss
#
#
# === Fixed === #
import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
import scipy.stats as stats

import sys

sys.path.append('..')
from copy import deepcopy
from dataset import NPweightingDataSet
from utils import *
from trainers import logging, train, validate, test


class CrossBlock(nn.Module):
    """单层：pre-LN → Cross-Attn → 残差 → FFN → 残差"""

    def __init__(self, dim, num_heads=4, attn_dropout=0.1, ffn_ratio=2.0, ffn_dropout=0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_drop = nn.Dropout(attn_dropout)

        hidden = int(ffn_ratio * dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(self, q_seq, kv_seq):
        # pre-LN
        q = self.q_norm(q_seq)
        kv = self.kv_norm(kv_seq)

        # cross-attn
        out, attn = self.attn(query=q, key=kv, value=kv, need_weights=True, average_attn_weights=False)
        q_seq = q_seq + self.attn_drop(out)

        # FFN
        q_seq = q_seq + self.ffn(self.ffn_norm(q_seq))
        return q_seq, attn  # 新增 attn


class BiCrossAttentionFusion(nn.Module):
    """
    最强版：支持 CLS pooling + learnable position embedding + 多层堆叠
    - 输入：drug (B,30)，gene (B,68=mut30+pathway38)
    - 输出：drug_fused (B,30)，gene_fused (B,68)
    """

    def __init__(
            self,
            dim_drug: int,
            dim_gene: int,
            attn_dim: int = 64,
            num_heads: int = 4,
            n_layers: int = 2,
            attn_dropout: float = 0.1,
            ffn_ratio: float = 2.0,
            ffn_dropout: float = 0.1,
            use_pos_emb: bool = True,
            use_cls: bool = True,
    ):
        super().__init__()
        self.dim_drug, self.dim_gene = dim_drug, dim_gene
        self.attn_dim = attn_dim
        self.use_pos_emb = use_pos_emb
        self.use_cls = use_cls

        # --- token embedding（保留你对 gene 的“mut / pathway”拆分方式，更强表达） ---
        self.drug_pre_norm = nn.LayerNorm(1)
        self.gene_pre_norm = nn.LayerNorm(1)

        self.drug_embed = nn.Linear(1, attn_dim)  # 30 标量 → 30×D
        self.mut_embed = nn.Linear(1, attn_dim)  # 30 标量 → 30×D
        self.pathway_embed1 = nn.Linear(1, 8)  # 38 标量 → 38×8 → 38×D
        self.pathway_embed2 = nn.Linear(8, attn_dim)

        # --- CLS token（可学习） ---
        if use_cls:
            self.cls_drug = nn.Parameter(torch.zeros(1, 1, attn_dim))
            self.cls_gene = nn.Parameter(torch.zeros(1, 1, attn_dim))
            nn.init.trunc_normal_(self.cls_drug, std=0.02)
            nn.init.trunc_normal_(self.cls_gene, std=0.02)

        # --- 可学习位置编码（长度+1 因为加了 CLS） ---
        if use_pos_emb:
            self.pos_drug = nn.Parameter(torch.zeros(1, (dim_drug + (1 if use_cls else 0)), attn_dim))
            self.pos_gene = nn.Parameter(torch.zeros(1, (dim_gene + (1 if use_cls else 0)), attn_dim))
            nn.init.trunc_normal_(self.pos_drug, std=0.02)
            nn.init.trunc_normal_(self.pos_gene, std=0.02)

        # --- 双向 Cross 堆叠：drug←gene 与 gene←drug 各一套，每层交替增强 ---
        self.blocks_d_from_g = nn.ModuleList([
            CrossBlock(attn_dim, num_heads, attn_dropout, ffn_ratio, ffn_dropout) for _ in range(n_layers)
        ])
        self.blocks_g_from_d = nn.ModuleList([
            CrossBlock(attn_dim, num_heads, attn_dropout, ffn_ratio, ffn_dropout) for _ in range(n_layers)
        ])

        # --- 池化后映射回原维度（用 CLS 作为全局表示；若不启用 CLS 则用 mean pooling） ---
        self.out_drug = nn.Linear(attn_dim, dim_drug)
        self.out_gene = nn.Linear(attn_dim, dim_gene)

        # --- 残差 + LayerNorm + Dropout(0.1)（回原空间后再做一次 Transformer 风格收尾） ---
        self.final_norm_drug = nn.LayerNorm(dim_drug)
        self.final_norm_gene = nn.LayerNorm(dim_gene)
        self.final_drop = nn.Dropout(0.1)

        self.final_ffn_drug = nn.Sequential(
            nn.Linear(dim_drug, 2 * dim_drug), nn.GELU(), nn.Linear(2 * dim_drug, dim_drug), nn.Dropout(0.1)
        )
        self.final_ffn_gene = nn.Sequential(
            nn.Linear(dim_gene, 2 * dim_gene), nn.GELU(), nn.Linear(2 * dim_gene, dim_gene), nn.Dropout(0.1)
        )

    def _embed_drug(self, drug):
        # (B,30) -> (B,30,1) -> preLN -> (B,30,D)
        x = self.drug_pre_norm(drug.unsqueeze(2))
        x = self.drug_embed(x)
        if self.use_cls:
            cls = self.cls_drug.expand(x.size(0), -1, -1)  # (B,1,D)
            x = torch.cat([cls, x], dim=1)  # (B,31,D)
        if self.use_pos_emb:
            x = x + self.pos_drug[:, :x.size(1), :]
        return x  # (B,30(+1),D)

    def _embed_gene(self, gene):
        # gene = [mut(30) | pathway(38)]
        mut = self.gene_pre_norm(gene[:, :30].unsqueeze(2))
        path = self.gene_pre_norm(gene[:, 30:].unsqueeze(2))

        mut_e = self.mut_embed(mut)  # (B,30,D)
        path_e = self.pathway_embed2(F.relu(self.pathway_embed1(path)))  # (B,38,D)
        x = torch.cat([mut_e, path_e], dim=1)  # (B,68,D)

        if self.use_cls:
            cls = self.cls_gene.expand(x.size(0), -1, -1)  # (B,1,D)
            x = torch.cat([cls, x], dim=1)  # (B,69,D)
        if self.use_pos_emb:
            x = x + self.pos_gene[:, :x.size(1), :]
        return x  # (B,68(+1),D)

    def _pool(self, seq, use_cls: bool):
        if use_cls:
            return seq[:, 0, :]  # 取 CLS
        else:
            return seq.mean(dim=1)

    def forward(self, drug, gene):
        # ---- 1) token 化 + 位置编码 + 可学习 CLS ----
        d = self._embed_drug(drug)  # (B,30/31,D)
        g = self._embed_gene(gene)  # (B,68/69,D)

        # ---- 2) 堆叠的双向 cross-attn（交替更新）----
        attn_dg_list = []
        attn_gd_list = []

        for d_blk, g_blk in zip(self.blocks_d_from_g, self.blocks_g_from_d):
            d, attn_dg = d_blk(d, g)  # drug 被 gene 影响
            g, attn_gd = g_blk(g, d)  # gene 被 drug 影响
            attn_dg_list.append(attn_dg)
            attn_gd_list.append(attn_gd)

        # ---- 3) 池化到全局向量（默认用 CLS）----
        d_vec = self._pool(d, self.use_cls)  # (B,D)
        g_vec = self._pool(g, self.use_cls)  # (B,D)

        # ---- 4) 回原维度 + 最终残差/归一化/丢弃 + FFN ----
        d_out = self.out_drug(d_vec)  # (B,30)
        g_out = self.out_gene(g_vec)  # (B,68)

        d_out = self.final_norm_drug(drug + self.final_drop(d_out))
        d_out = d_out + self.final_ffn_drug(d_out)

        g_out = self.final_norm_gene(gene + self.final_drop(g_out))
        g_out = g_out + self.final_ffn_gene(g_out)
        return d_out, g_out, d_vec, g_vec, attn_dg_list, attn_gd_list


# ====== Random Seed Initialization ====== #
def seed_everything(seed=3078):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# ====== Model Definition ====== #
class DeepAutoencoderThreeHiddenLayers(nn.Module):
    def __init__(self, input_dim, hidden_dims, code_dim, activation_func=nn.ReLU,
                 code_activation=True, dropout=False, dropout_rate=0.5):
        super(DeepAutoencoderThreeHiddenLayers, self).__init__()
        # Establish encoder
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dims[0]))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))

        for input_size, output_size in zip(hidden_dims, hidden_dims[1:]):

            modules.append(nn.Linear(input_size, output_size))
            modules.append(activation_func())
            if dropout:
                modules.append(nn.Dropout(dropout_rate))

        modules.append(nn.Linear(hidden_dims[-1], code_dim))
        if code_activation:
            modules.append(activation_func())
        self.encoder = nn.Sequential(*modules)

        # Establish decoder
        modules = []

        modules.append(nn.Linear(code_dim, hidden_dims[-1]))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))

        for input_size, output_size in zip(hidden_dims[::-1], hidden_dims[-2::-1]):
            modules.append(nn.Linear(input_size, output_size))
            modules.append(activation_func())
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dims[0], input_dim))
        # modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder(x)
        code = x
        x = self.decoder(x)
        return code, x


class ForwardNetworkTwoHiddenLayers(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, activation_func=nn.ReLU,
                 out_activation=None):
        super(ForwardNetworkTwoHiddenLayers, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            activation_func(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            activation_func(),
            nn.Linear(hidden_dim2, 1))

        self.out_activation = out_activation

    def forward(self, x):
        if self.out_activation:
            return self.out_activation(self.layers(x))
        else:
            return self.layers(x)


class DEERS_Concat(torch.nn.Module):
    def __init__(self, drug_autoencoder, mut_line_autoencoder,
                 forward_network,
                 use_cross_attn: bool = True,
                 attn_dim: int = 64,
                 attn_heads: int = 4,
                 attn_dropout: float = 0.1,
                 num_target_classes: int = 0,
                 num_cancer_classes: int = 0,
                 ):  # <--- 新增
        super(DEERS_Concat, self).__init__()
        self.drug_autoencoder = drug_autoencoder
        self.mut_line_autoencoder = mut_line_autoencoder
        self.forward_network = forward_network
        self.fusion_norm = nn.LayerNorm(30 + 68)
        self.fusion_dropout = nn.Dropout(0.1)

        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            self.cross_fusion = BiCrossAttentionFusion(
                dim_drug=30, dim_gene=30 + 38,
                attn_dim=64, num_heads=4,
                n_layers=2,  # 建议2或3
                attn_dropout=0.1, ffn_dropout=0.1,
                use_pos_emb=True, use_cls=True
            )
            # ---- 对比学习用投影头 ----
            proj_dim = 128
            self.proj_head = nn.Sequential(
                nn.Linear(attn_dim, attn_dim),
                nn.BatchNorm1d(attn_dim),
                nn.ReLU(inplace=True),
                nn.Linear(attn_dim, proj_dim)
            )

        # 现有：靶点分类头（基于 drug_code）
        self.num_target_classes = num_target_classes
        if num_target_classes and num_target_classes > 0:
            self.target_head = nn.Sequential(
                nn.Linear(30, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, num_target_classes)
            )
        else:
            self.target_head = None

        # 新增：癌种分类头（基于基因侧表示）
        self.num_cancer_classes = num_cancer_classes
        if num_cancer_classes and num_cancer_classes > 0:
            self.cancer_head = nn.Sequential(
                nn.Linear(30 + 38, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, num_cancer_classes)
            )
        else:
            self.cancer_head = None

    def forward(self, drug_features, mut_features, cell_features):
        drug_code, drug_reconstruction = self.drug_autoencoder(drug_features)  # (B,30)
        mut_code, mut_reconstruction = self.mut_line_autoencoder(mut_features)  # (B,30)

        if not self.use_cross_attn:
            gene_repr = torch.cat((mut_code, cell_features), dim=1)  # (B,68)
            x = torch.cat((drug_code, mut_code, cell_features), dim=1)  # (B,98)
            y_hat = self.forward_network(x)

            target_logits = self.target_head(drug_code) if self.target_head is not None else None
            cancer_logits = self.cancer_head(gene_repr) if self.cancer_head is not None else None
            return y_hat, drug_reconstruction, mut_reconstruction, target_logits, cancer_logits

        # 开启双向交叉注意力
        gene_repr = torch.cat((mut_code, cell_features), dim=1)  # (B,68)
        drug_fused, gene_fused, drug_lat, gene_lat, attn_dg, attn_gd = self.cross_fusion(drug_code, gene_repr)
        drug_proj = self.proj_head(drug_lat)
        gene_proj = self.proj_head(gene_lat)

        x = torch.cat((drug_fused, gene_fused), dim=1)
        x = self.fusion_norm(x)
        x = self.fusion_dropout(x)
        y_hat = self.forward_network(x)

        target_logits = self.target_head(drug_fused) if self.target_head is not None else None
        cancer_logits = self.cancer_head(gene_fused) if self.cancer_head is not None else None
        return (
            y_hat,
            drug_reconstruction, mut_reconstruction,
            target_logits, cancer_logits,
            drug_lat, gene_lat,
            drug_proj, gene_proj,
            attn_dg, attn_gd
        )


class MergedLoss(nn.Module):
    def __init__(self, y_loss_weight=1., drug_reconstruction_loss_weight=0.1, mut_reconstruction_loss_weight=0.2):
        super(MergedLoss, self).__init__()
        self.y_loss_weight = y_loss_weight
        self.drug_reconstruction_loss_weight = drug_reconstruction_loss_weight
        self.mut_reconstruction_loss_weight = mut_reconstruction_loss_weight
        self.output_criterion = nn.MSELoss()
        # self.reconstruction_criterion = nn.BCELoss()
        self.reconstruction_criterion = nn.MSELoss()

    def forward(self, pred_y, drug_reconstruction, mut_reconstruction, drug_input, mut_input, true_y):
        output_loss = self.output_criterion(pred_y, true_y)
        drug_reconstruction_loss = self.reconstruction_criterion(drug_reconstruction, drug_input)
        mut_reconstruction_loss = self.reconstruction_criterion(mut_reconstruction, mut_input)
        return output_loss, drug_reconstruction_loss, mut_reconstruction_loss


class DrugMultiEncoderWithRecon(nn.Module):
    """
    三路并行压缩：ECFP→20维，2D→5维，3D→5维；再拼成30维。
    返回 (code_30, recon_full) 以兼容原来的重构损失。
    """

    def __init__(self, ecfp_dim: int, f2d_dim: int = 30, f3d_dim: int = 30,
                 ecfp_code_dim: int = 20, f2d_code_dim: int = 5, f3d_code_dim: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.ecfp_dim, self.f2d_dim, self.f3d_dim = ecfp_dim, f2d_dim, f3d_dim
        # ECFP 分支（大一点容量）
        self.ecfp_enc = nn.Sequential(
            nn.Linear(ecfp_dim, 300), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(300, 100), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(100, ecfp_code_dim)
        )
        self.ecfp_dec = nn.Sequential(
            nn.Linear(ecfp_code_dim, 100), nn.ReLU(),
            nn.Linear(100, 300), nn.ReLU(),
            nn.Linear(300, ecfp_dim), nn.Sigmoid()
        )
        # 2D 分支
        self.f2d_enc = nn.Sequential(
            nn.LayerNorm(f2d_dim),
            nn.Linear(f2d_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, f2d_code_dim)
        )
        self.f2d_dec = nn.Sequential(
            nn.Linear(f2d_code_dim, 32), nn.ReLU(),
            nn.Linear(32, f2d_dim), nn.Sigmoid()
        )
        # 3D 分支
        self.f3d_enc = nn.Sequential(
            nn.LayerNorm(f3d_dim),
            nn.Linear(f3d_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, f3d_code_dim)
        )
        self.f3d_dec = nn.Sequential(
            nn.Linear(f3d_code_dim, 32), nn.ReLU(),
            nn.Linear(32, f3d_dim), nn.Sigmoid()
        )
        self.out_dim = ecfp_code_dim + f2d_code_dim + f3d_code_dim  # 30

    def forward(self, drug_features):
        # 切三段：ECFP | 2D | 3D
        ecfp = drug_features[:, :self.ecfp_dim]
        f2d = drug_features[:, self.ecfp_dim:self.ecfp_dim + self.f2d_dim]
        f3d = drug_features[:, self.ecfp_dim + self.f2d_dim:]
        # 编码
        ecfp_code = self.ecfp_enc(ecfp)
        f2d_code = self.f2d_enc(f2d)
        f3d_code = self.f3d_enc(f3d)
        code = torch.cat([ecfp_code, f2d_code, f3d_code], dim=1)  # (B,30)
        # 重构（兼容原重构损失）
        ecfp_rec = self.ecfp_dec(ecfp_code)
        f2d_rec = self.f2d_dec(f2d_code)
        f3d_rec = self.f3d_dec(f3d_code)
        recon = torch.cat([ecfp_rec, f2d_rec, f3d_rec], dim=1)  # (B, in_dim)
        return code, recon


class MergedLossWithTarget(nn.Module):
    def __init__(self, y_loss_weight=1.,
                 drug_reconstruction_loss_weight=0.1,
                 mut_reconstruction_loss_weight=0.2,
                 target_loss_weight=0.5,
                 ce_weight=None,
                 label_smoothing=0.05,
                 cancer_loss_weight=0.0,
                 cancer_ce_weight=None,
                 cancer_label_smoothing=0.0):
        super().__init__()
        self.y_loss_weight = y_loss_weight
        self.drug_reconstruction_loss_weight = drug_reconstruction_loss_weight
        self.mut_reconstruction_loss_weight = mut_reconstruction_loss_weight
        self.target_loss_weight = target_loss_weight
        self.output_criterion = nn.MSELoss()
        # self.reconstruction_criterion = nn.BCELoss
        self.reconstruction_criterion = nn.MSELoss()
        self.ce_weight = ce_weight
        self.label_smoothing = label_smoothing
        self.cancer_loss_weight = cancer_loss_weight
        self.cancer_ce_weight = cancer_ce_weight
        self.cancer_label_smoothing = cancer_label_smoothing

    def forward(self, pred_y, drug_reconstruction, mut_reconstruction,
                drug_input, mut_input, true_y,
                target_logits=None, target_labels=None,
                cancer_logits=None, cancer_labels=None,
                contrast_loss=None, contrast_weight=0.05):

        output_loss = self.output_criterion(pred_y, true_y)
        drug_rec_loss = self.reconstruction_criterion(drug_reconstruction, drug_input)
        mut_rec_loss = self.reconstruction_criterion(mut_reconstruction, mut_input)
        cls_loss = torch.tensor(0.0, device=pred_y.device)
        if (target_logits is not None) and (target_labels is not None):
            mask = (target_labels >= 0)
            if mask.any():
                weight = self.ce_weight.to(pred_y.device) if self.ce_weight is not None else None
                ce = F.cross_entropy(
                    target_logits[mask], target_labels[mask],
                    weight=weight, label_smoothing=self.label_smoothing, reduction='mean'
                )
                cls_loss = ce

        cancer_loss = torch.tensor(0.0, device=pred_y.device)
        if (cancer_logits is not None) and (cancer_labels is not None):
            cmask = (cancer_labels >= 0)
            if cmask.any():
                cweight = self.cancer_ce_weight.to(pred_y.device) if self.cancer_ce_weight is not None else None
                cce = F.cross_entropy(
                    cancer_logits[cmask], cancer_labels[cmask],
                    weight=cweight, label_smoothing=self.cancer_label_smoothing, reduction='mean'
                )
                cancer_loss = cce

        total = (
                self.y_loss_weight * output_loss
                + self.drug_reconstruction_loss_weight * drug_rec_loss
                + self.mut_reconstruction_loss_weight * mut_rec_loss
                + self.target_loss_weight * cls_loss
                + self.cancer_loss_weight * cancer_loss
        )
        if contrast_loss is not None:
            total = total + contrast_weight * contrast_loss

        return total, output_loss, drug_rec_loss, mut_rec_loss, cls_loss, cancer_loss


