import os, time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr

# ================= Memory Queue =================
class MemoryQueue:
    def __init__(self, dim, K=4096, device='cuda'):
        self.K = K
        self.device = device
        self.queue = torch.randn(K, dim, device=device)
        self.queue = F.normalize(self.queue, dim=1)
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, feats):
        feats = feats.detach()
        B = feats.size(0)

        if B >= self.K:
            self.queue = feats[-self.K:]
            self.ptr = 0
            return

        end = self.ptr + B
        if end <= self.K:
            self.queue[self.ptr:end] = feats
        else:
            first = self.K - self.ptr
            self.queue[self.ptr:] = feats[:first]
            self.queue[:B-first] = feats[first:]

        self.ptr = (self.ptr + B) % self.K


# --------------------------------------------------------
# Contrastive losses
# --------------------------------------------------------
def _attn_sample_weights(attn_dg_list):
    """
    attn_dg_list: list of layers, each (B, heads, Lq, Lk)
    输出样本级权重 (B,)
    """
    # stack over layers: L × B × H × Lq × Lk
    A = torch.stack(attn_dg_list, dim=0).mean(dim=(0,2))  # → (B, Lq, Lk)

    # CLS → gene attention
    cls2gene = A[:, 0, 1:]             # (B, Lk-1)

    w = cls2gene.mean(dim=1)           # (B,)
    w = w / (w.mean().detach() + 1e-8) # normalize
    return w.detach()


def attn_weighted_contrastive_loss(drug_lat, gene_lat, attn_dg_list, tau=0.1):
    """
    drug_lat/gene_lat: (B, D)
    attn_dg_list: list of attention maps
    """
    drug_f = F.normalize(drug_lat, p=2, dim=1)
    gene_f = F.normalize(gene_lat, p=2, dim=1)

    sim = torch.mm(drug_f, gene_f.t()) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    per_sample = F.cross_entropy(sim, labels, reduction='none')  # (B,)
    w = _attn_sample_weights(attn_dg_list).to(per_sample.device)

    return (w * per_sample).mean()

def attn_weighted_contrastive_loss_queue(
    q, k_pos, queue, attn_list, tau=0.1
):
    """
    q:      (B, D) query
    k_pos:  (B, D) positive key
    queue:  (K, D) negative keys
    """
    q = F.normalize(q, dim=1)
    k_pos = F.normalize(k_pos, dim=1)
    queue = F.normalize(queue, dim=1)

    # 正样本
    l_pos = torch.sum(q * k_pos, dim=1, keepdim=True)   # (B,1)
    # 负样本（来自 memory queue）
    l_neg = torch.mm(q, queue.t())                      # (B,K)

    logits = torch.cat([l_pos, l_neg], dim=1) / tau
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    per_sample = F.cross_entropy(logits, labels, reduction='none')
    w = _attn_sample_weights(attn_list).to(per_sample.device)

    return (w * per_sample).mean()


# --------------------------------------------------------
# EMA
# --------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


# --------------------------------------------------------
# logging
# --------------------------------------------------------
def logging(msg, outdir, log_fpath):
    fpath = os.path.join(outdir, log_fpath)
    os.makedirs(outdir, exist_ok=True)
    with open(fpath, 'a', encoding='utf-8') as fw:
        fw.write(f"{msg}\n")


# --------------------------------------------------------
# MC dropout utils
# --------------------------------------------------------
@torch.no_grad()
def _mc_dropout_pred(model, drug_input, mut_input, cell_input, n: int):
    model.eval()

    # turn on dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds = []
    for _ in range(n):
        y_hat, *_ = model(drug_input, mut_input, cell_input)
        preds.append(y_hat)

    # turn off dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    return torch.stack(preds, dim=0).mean(0)


# --------------------------------------------------------
# train
# --------------------------------------------------------
def train(model, epoch, train_loader, optimizer, loss_fn, device, ema: EMA = None, tau=0.1, mem_queue=None):
    base_w = 0.05
    list_train_batch_loss = []
    list_train_batch_out  = []
    list_train_batch_true = []

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        drug_input, cell_input, mut_input, target_label, cancer_label = \
            tuple(d.to(device) for d in data)
        true_y = target.unsqueeze(1).to(device)

        optimizer.zero_grad()

        # ---------------- forward ----------------
        pred_y, drug_rec, mut_rec, target_logits, cancer_logits, \
            drug_lat, gene_lat, drug_proj, gene_proj, attn_dg, attn_gd = model(drug_input, mut_input, cell_input)

        # ---------------- contrastive loss ----------------
        if mem_queue is None:
            # 退化为原来的 in-batch InfoNCE
            loss_dg = attn_weighted_contrastive_loss(drug_proj, gene_proj, attn_dg, tau=tau)
            loss_gd = attn_weighted_contrastive_loss(gene_proj, drug_proj, attn_gd, tau=tau)
        else:
            # 使用 memory queue
            loss_dg = attn_weighted_contrastive_loss_queue(
                drug_proj, gene_proj, mem_queue.queue, attn_dg, tau=tau
            )
            loss_gd = attn_weighted_contrastive_loss_queue(
                gene_proj, drug_proj, mem_queue.queue, attn_gd, tau=tau
            )

        contrast_loss = 0.5 * (loss_dg + loss_gd)

        # ---------- global attention confidence ----------
        with torch.no_grad():
            # attn_dg: list of (B, heads, Lq, Lk)
            A = torch.stack(attn_dg, dim=0).mean()
            attn_confidence = A.clamp(min=0.5, max=2.0)

        # ---------------- main loss ----------------
        total_loss, output_loss, drug_rec_loss, mut_rec_loss, cls_loss, cancer_loss = loss_fn(
            pred_y, drug_rec, mut_rec,
            drug_input, mut_input, true_y,
            target_logits=target_logits, target_labels=target_label,
            cancer_logits=cancer_logits, cancer_labels=cancer_label,
            contrast_loss=contrast_loss, contrast_weight = base_w * attn_confidence.detach()

        )

        # ---------------- backward ----------------
        total_loss.backward()
        optimizer.step()
        if mem_queue is not None:
            mem_queue.enqueue(drug_proj)

        if ema is not None:
            ema.update(model)

        list_train_batch_out.extend(pred_y.detach().cpu().numpy())
        list_train_batch_true.extend(true_y.detach().cpu().numpy())
        list_train_batch_loss.append(total_loss.detach().cpu().numpy())

    return model, list_train_batch_loss, list_train_batch_out, list_train_batch_true


# --------------------------------------------------------
# validate
# --------------------------------------------------------
@torch.no_grad()
def validate(model, valid_loader, loss_fn, device, ema: EMA = None, tau=0.1):
    base_w = 0.05
    list_val_batch_loss = []
    list_val_batch_out  = []
    list_val_batch_true = []

    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    for batch_idx, (data, target) in enumerate(valid_loader):
        drug_input, cell_input, mut_input, target_label, cancer_label = \
            tuple(d.to(device) for d in data)
        true_y = target.unsqueeze(1).to(device)

        pred_y, drug_rec, mut_rec, target_logits, cancer_logits, \
            drug_lat, gene_lat, drug_proj, gene_proj, attn_dg, attn_gd = model(drug_input, mut_input, cell_input)

        # -------- contrastive loss --------
        loss_dg = attn_weighted_contrastive_loss(drug_proj, gene_proj, attn_dg, tau=tau)
        loss_gd = attn_weighted_contrastive_loss(gene_proj, drug_proj, attn_gd, tau=tau)
        contrast_loss = 0.5 * (loss_dg + loss_gd)
        # ---------- global attention confidence ----------
        with torch.no_grad():
            # attn_dg: list of (B, heads, Lq, Lk)
            A = torch.stack(attn_dg, dim=0).mean()
            attn_confidence = A.clamp(min=0.5, max=2.0)

        # -------- main loss (contrast_loss included) --------
        total_loss, *_ = loss_fn(
            pred_y, drug_rec, mut_rec,
            drug_input, mut_input, true_y,
            target_logits=target_logits, target_labels=target_label,
            cancer_logits=cancer_logits, cancer_labels=cancer_label,
            contrast_loss=contrast_loss, contrast_weight = base_w * attn_confidence.detach()
        )

        list_val_batch_out.extend(pred_y.detach().cpu().numpy())
        list_val_batch_true.extend(true_y.detach().cpu().numpy())
        list_val_batch_loss.append(total_loss.detach().cpu().numpy())

    if ema is not None:
        ema.restore(model)

    return list_val_batch_loss, list_val_batch_out, list_val_batch_true


# --------------------------------------------------------
# test
# --------------------------------------------------------
@torch.no_grad()
def test(model, test_loader, loss_fn, device, ema: EMA = None, mc_dropout: int = 0):

    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    total_loss_sum = 0.0
    list_test_loss, list_test_output_loss = [], []
    list_test_out, list_test_true = [], []

    for batch_idx, (data, target) in enumerate(test_loader):

        drug_input, cell_input, mut_input, target_label, cancer_label = \
            tuple(d.to(device) for d in data)
        true_y = target.unsqueeze(1).to(device)

        if mc_dropout and mc_dropout > 0:
            pred_y = _mc_dropout_pred(model, drug_input, mut_input, cell_input, n=mc_dropout)
            y_hat_once, drug_rec, mut_rec, target_logits, cancer_logits, *_ = model(drug_input, mut_input, cell_input)

            total_loss, output_loss, *_ = loss_fn(
                pred_y, drug_rec, mut_rec, drug_input, mut_input, true_y,
                target_logits=target_logits, target_labels=target_label,
                cancer_logits=cancer_logits, cancer_labels=cancer_label
            )
        else:
            y_hat, drug_rec, mut_rec, target_logits, cancer_logits, *_ = model(drug_input, mut_input, cell_input)
            pred_y = y_hat

            total_loss, output_loss, *_ = loss_fn(
                pred_y, drug_rec, mut_rec, drug_input, mut_input, true_y,
                target_logits=target_logits, target_labels=target_label,
                cancer_logits=cancer_logits, cancer_labels=cancer_label
            )

        total_loss_sum += float(total_loss.item())
        list_test_loss.append(float(total_loss.item()))
        list_test_output_loss.append(float(output_loss.item()))
        list_test_out.append(pred_y.detach().cpu())
        list_test_true.append(true_y.detach().cpu())

    # summary
    y_pred = torch.cat(list_test_out, dim=0).numpy().reshape(-1)
    y_true = torch.cat(list_test_true, dim=0).numpy().reshape(-1)

    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    pcc = float(np.corrcoef(y_pred, y_true)[0, 1])
    spearman = float(spearmanr(y_pred, y_true).correlation)
    ci = float(concordance_index(y_true, y_pred))
    test_loss_avg = total_loss_sum / max(1, len(test_loader))

    if ema is not None:
        ema.restore(model)

    return test_loss_avg, rmse, pcc, spearman, ci, list_test_loss, list_test_output_loss, list_test_out, list_test_true
