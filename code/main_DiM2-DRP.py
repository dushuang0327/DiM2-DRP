import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings("ignore")
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import scipy.stats as stats
import sys
sys.path.append('..')
from copy import deepcopy
from dataset import NPweightingDataSet, get_target_class_weights
from utils import *
from models import *
from trainers import train, validate, test, EMA, logging, MemoryQueue




# ====== Argument Parsing ====== #
parser = argparse.ArgumentParser()
parser.add_argument('--WORKDIR_PATH', type=str, default="D:/work2/DiM2-DRP-main")
parser.add_argument('--inputdir', type=str, default="D:/work2/DiM2-DRP-main/data")
parser.add_argument('--mut_input_fpath', type=str, default="D:/work2/DiM2-DRP-main/data/GDSC_mutation_input.csv")
parser.add_argument('--drug_input_fpath', type=str, default="D:/work2/DiM2-DRP-main/data/GDSC_ECFP_2D3D.csv")
parser.add_argument('--exprs_input_fpath', type=str, default="D:/work2/DiM2-DRP-main/data/GDSC_ssgsea_input.csv")
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--model_name', type=str, default='my')
parser.add_argument('--split_type', type=str, default='borh', choices=['cell','drug','borh','mix'])
parser.add_argument('--response_type', type=str, default='IC50', choices=['IC50', 'AUC'])
parser.add_argument('--drug_smiles_fpath', type=str, default="D:/work2/DiM2-DRP-main/data/GDSC_drug_smiles.csv")
parser.add_argument('--cancer_label_mapping_fpath', type=str,default="D:/work2/DiM2-DRP-main/data/cancer_label_mapping.json")
parser.add_argument('--cancer_class_weights_fpath', type=str,default="D:/work2/DiM2-DRP-main/data/cancer_class_weights.csv")
parser.add_argument('--cellline_cancer_labels_fpath', type=str,default="D:/work2/DiM2-DRP-main/data/cellline_cancer_labels.csv")



# === Train setting === #
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--testset_yes', type=bool, default=True)
parser.add_argument('--use_cross_attn', type=bool, default=True,help="是否启用基因-药物双向多头交叉注意力 (默认开启)")
parser.add_argument("--attn_dim", type=int, default=64)
parser.add_argument("--attn_heads", type=int, default=4)
parser.add_argument("--attn_dropout", type=float, default=0.1)
parser.add_argument('--mc_dropout', type=int, default=0, help='#forward passes at test time (0=off, e.g. 16)')
parser.add_argument('--target_loss_weight', type=float, default=0.4)
parser.add_argument('--cancer_loss_weight', type=float, default=0.1)
parser.add_argument('--drug_reconstruction_loss_weight', type=float, default=0.05)
parser.add_argument('--mut_reconstruction_loss_weight',  type=float, default=0.10)
parser.add_argument('--cancer_label_smoothing', type=float, default=0.05)




args = parser.parse_known_args()[0]

# === Model Setting === #
args.code_dim = 30
args.drug_hidden_dims = [300, 100]
args.mut_hidden_dims = [300, 100]
args.code_dropout = True
args.code_dropout_rate = 0.2
args.forward_net_hidden_dim1 = 98
args.forward_net_hidden_dim2 = 98
args.forward_net_out_act = None
args.y_loss_weight=1.

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print('torch version: ', torch.__version__)
print(device)

def experiment(args, dataset_partition, model, loss_fn, device):

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    proj_dim = 128  # models.py 里 projection head 的维度
    mem_queue = MemoryQueue(dim=proj_dim, K=4096, device=device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)

    # ====== Cross Validation Best Performance Dict ====== #
    best_performances = {}
    best_performances['best_epoch'] = 0
    best_performances['best_train_loss'] = float('inf')
    best_performances['best_train_corr'] = 0.0
    best_performances['best_valid_loss'] = float('inf')
    best_performances['best_valid_corr'] = 0.0
    # ==================================================== #

    list_epoch = []
    list_train_epoch_loss = []
    list_epoch_rmse = []
    list_epoch_corr = []
    list_epoch_spearman = []
    list_epoch_ci = []

    list_val_epoch_loss = []
    list_val_epoch_rmse = []
    list_val_epoch_corr = []
    list_val_spearman = []
    list_val_ci = []

    counter = 0

    counter = 0
    # 初始化 best 记录（新增 best_valid_rmse，用它做比较）
    best_performances.setdefault('best_valid_rmse', float('inf'))
    best_performances.setdefault('best_epoch', -1)
    best_performances.setdefault('best_train_loss', None)
    best_performances.setdefault('best_train_corr', None)
    best_performances.setdefault('best_valid_loss', None)
    best_performances.setdefault('best_valid_corr', None)

    model_max = model  # 若从未触发“更优”，测试阶段也有可用权重

    for epoch in range(args.epochs):
        list_epoch.append(epoch)
        # ====== TRAIN Epoch ====== #
        model, list_train_batch_loss, list_train_batch_out, list_train_batch_true = \
            train(model, epoch, train_loader, optimizer, loss_fn, device,
                  ema=ema, mem_queue=mem_queue)

        epoch_train_rmse = np.sqrt(mean_squared_error(np.array(list_train_batch_out).squeeze(1),
                                                      np.array(list_train_batch_true).squeeze(1)))
        epoch_train_corr, _p = pearsonr(np.array(list_train_batch_out).squeeze(1),
                                        np.array(list_train_batch_true).squeeze(1))
        epoch_train_spearman, _p = spearmanr(np.array(list_train_batch_out).squeeze(1),
                                             np.array(list_train_batch_true).squeeze(1))
        epoch_train_ci = concordance_index(np.array(list_train_batch_out).squeeze(1),
                                           np.array(list_train_batch_true).squeeze(1))
        train_epoch_loss = sum(list_train_batch_loss) / len(list_train_batch_loss)

        list_train_epoch_loss.append(train_epoch_loss)
        list_epoch_rmse.append(epoch_train_rmse)
        list_epoch_corr.append(epoch_train_corr)
        list_epoch_spearman.append(epoch_train_spearman)
        list_epoch_ci.append(epoch_train_ci)

        # ====== VALID Epoch ====== #
        # 用 EMA 权重评估（validate 内部会自动 apply/restore）
        list_val_batch_loss, list_val_batch_out, list_val_batch_true = \
            validate(model, valid_loader, loss_fn, device, ema=ema)

        epoch_val_rmse = np.sqrt(mean_squared_error(np.array(list_val_batch_out).squeeze(1),
                                                    np.array(list_val_batch_true).squeeze(1)))
        epoch_val_corr, _p = pearsonr(np.array(list_val_batch_out).squeeze(1),
                                      np.array(list_val_batch_true).squeeze(1))
        epoch_val_spearman, _p = spearmanr(np.array(list_val_batch_out).squeeze(1),
                                           np.array(list_val_batch_true).squeeze(1))
        epoch_val_ci = concordance_index(np.array(list_val_batch_out).squeeze(1),
                                         np.array(list_val_batch_true).squeeze(1))

        val_epoch_loss = sum(list_val_batch_loss) / len(list_val_batch_loss)
        list_val_epoch_loss.append(val_epoch_loss)
        list_val_epoch_rmse.append(epoch_val_rmse)
        list_val_epoch_corr.append(epoch_val_corr)
        list_val_spearman.append(epoch_val_spearman)
        list_val_ci.append(epoch_val_ci)

        # ====== 以 验证RMSE 作为早停&保存标准（而不是 total loss） ====== #
        improved = epoch_val_rmse < best_performances['best_valid_rmse'] - 1e-6
        if improved:
            best_performances['best_epoch'] = epoch
            best_performances['best_train_loss'] = train_epoch_loss
            best_performances['best_train_corr'] = epoch_train_corr
            best_performances['best_valid_loss'] = val_epoch_loss
            best_performances['best_valid_corr'] = epoch_val_corr
            best_performances['best_valid_rmse'] = epoch_val_rmse

            # 用 EMA 权重保存最佳（临时套用 -> 保存 -> 还原）
            ema.apply_shadow(model)
            torch.save(model.state_dict(), best_ckpt_path)
            # 同步 model_max（用已套用 EMA 的权重拷贝）
            model_max = deepcopy(model)
            ema.restore(model)

            counter = 0
        else:
            counter += 1
            logging(f'Early Stopping counter: {counter} out of {args.patience}', args.outdir, args.exp_name + '.log')
            if counter == args.patience:
                break

        logging(
            f'Epoch: {epoch:02d}, '
            f'Train loss: {list_train_epoch_loss[-1]:.4f}, rmse: {epoch_train_rmse:.4f}, corr: {epoch_train_corr:.4f}, '
            f'Valid loss: {list_val_epoch_loss[-1]:.4f}, rmse: {epoch_val_rmse:.4f}, pcc: {epoch_val_corr:.4f}',
            args.outdir, args.exp_name + '.log'
        )

        # 学习率调度也改用 RMSE 驱动（比 total loss 更贴近主任务）
        scheduler.step(epoch_val_rmse)

    if args.testset_yes:
        # （可选更稳）从文件加载最佳 EMA 权重
        if os.path.exists(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
            model_max = model

        # 用 EMA + MC Dropout 测试
        test_loss, test_rmse, test_corr, test_spearman, test_ci, list_test_loss, list_test_output_loss, list_test_out, list_test_true = \
            test(model_max, test_loader, loss_fn, device, ema=ema, mc_dropout=32)

        logging(
            f"Test:\tLoss: {test_loss}\tRMSE: {test_rmse}\tCORR: {test_corr}\tSPEARMAN: {test_spearman}\tCI: {test_ci}",
            args.outdir, f'{args.exp_name}_test.log')

        response_df = test_set.response_df.copy()
        response_df['test_loss'] = list_test_loss
        response_df['output_loss'] = list_test_output_loss
        response_df['test_pred'] = torch.cat(list_test_out, dim=0).cpu().numpy().reshape(-1)
        response_df['test_true'] = torch.cat(list_test_true, dim=0).cpu().numpy().reshape(-1)

        filename = os.path.join(args.outdir, f'{args.exp_name}_test.csv')
        response_df.to_csv(filename, sep=',', header=True, index=False)

    # ====== Add Result to Dictionary ====== #

    result = {}
    result['train_losses'] = list_train_epoch_loss
    result['val_losses'] = list_val_epoch_loss
    result['train_accs'] = list_epoch_corr
    result['val_accs'] = list_val_epoch_corr
    result['train_acc'] = epoch_train_corr
    result['val_acc'] = epoch_val_corr
    if args.testset_yes:
        result['test_acc'] = test_corr

    filename = os.path.join(args.outdir, f'{args.exp_name}_best_performances.json')
    with open(filename, 'w') as f:
        def _to_py(v):
            # numpy 标量
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            # numpy 数组
            if isinstance(v, np.ndarray):
                return v.tolist()
            # torch 张量（仅支持标量或向量；你这里放进去的都是标量）
            if torch.is_tensor(v):
                return v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
            return v

        safe = {k: _to_py(v) for k, v in best_performances.items()}
        json.dump(safe, f, ensure_ascii=False, indent=2)

    return vars(args), result, best_performances, model_max

# ====== Experiment  ====== #
total_results = defaultdict(list)
best_best_epoch = 0
best_best_train_loss = 99.
best_best_train_metric = 0
best_best_valid_loss = 99.
best_best_valid_metric = 0

if __name__ == '__main__':

    args.exp_name = f'{args.model_name}_{args.split_type}'
    args.outdir = os.path.join(args.WORKDIR_PATH, 'Results', args.exp_name)

    createFolder(args.outdir)

    # =============== #
    # === Dataset === #
    # =============== #

    train_set = NPweightingDataSet(
        response_fpath=os.path.join(args.inputdir, f'GDSC_train_IC50_by_{args.split_type}_cv00.csv'),
        drug_input=args.drug_input_fpath,
        exprs_input=args.exprs_input_fpath,
        mut_input=args.mut_input_fpath,
        response_type=args.response_type,
        drug_smiles_csv=args.drug_smiles_fpath,
        cancer_label_csv=args.cellline_cancer_labels_fpath,  # <---
        cancer_label_mapping_json=args.cancer_label_mapping_fpath  # <---
    )

    # num_cancer_classes
    num_cancer_classes = 0
    if os.path.exists(args.cancer_label_mapping_fpath):
        with open(args.cancer_label_mapping_fpath, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        num_cancer_classes = int(len(mapping))

    # cancer class weights
    cancer_weights_t = None
    if os.path.exists(args.cancer_class_weights_fpath):
        wdf = pd.read_csv(args.cancer_class_weights_fpath)
        # 保证按照 label id 排序
        wdf = wdf.sort_values('cancer_label')
        cancer_weights_t = torch.tensor(wdf['class_weight'].values, dtype=torch.float32)

    valid_set = NPweightingDataSet(
        response_fpath=os.path.join(args.inputdir, f'GDSC_valid_IC50_by_{args.split_type}_cv00.csv'),
        drug_input=args.drug_input_fpath,
        exprs_input=args.exprs_input_fpath,
        mut_input=args.mut_input_fpath,
        response_type=args.response_type,
        drug_smiles_csv=args.drug_smiles_fpath,
        cancer_label_csv=args.cellline_cancer_labels_fpath,  # <---
        cancer_label_mapping_json=args.cancer_label_mapping_fpath  # <---
    )

    test_set = NPweightingDataSet(
        response_fpath=os.path.join(args.inputdir, f'GDSC_test_IC50_by_{args.split_type}_cv00.csv'),
        drug_input=args.drug_input_fpath,
        exprs_input=args.exprs_input_fpath,
        mut_input=args.mut_input_fpath,
        response_type=args.response_type,
        drug_smiles_csv=args.drug_smiles_fpath,
        cancer_label_csv=args.cellline_cancer_labels_fpath,  # <---
        cancer_label_mapping_json=args.cancer_label_mapping_fpath  # <---
    )

    # === input === #
    args.drug_in_dim = pd.read_csv(args.drug_input_fpath, sep=',', header=0).shape[1] - 2
    args.cell_in_dim = pd.read_csv(args.exprs_input_fpath, sep=',', header=0).shape[1] - 2
    args.mut_score_dim = pd.read_csv(args.mut_input_fpath, sep=',', header=0).shape[1] - 2

    # === Data Set/Loader === #
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,pin_memory = True,num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False,pin_memory = True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False,pin_memory = True,num_workers=0)

    dataset_partition = {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader' : test_loader
    }

    # ============= #
    # === Model === #
    # ============= #

    # drug_autoencoder = DeepAutoencoderThreeHiddenLayers(input_dim=args.drug_in_dim,
    #                                 hidden_dims=args.drug_hidden_dims,
    #                                 code_dim=args.code_dim,
    #                                 code_activation=True, dropout=args.code_dropout, dropout_rate=args.code_dropout_rate)
    ecfp_dim = args.drug_in_dim - 60
    assert ecfp_dim > 0, "drug_in_dim 计算异常：应为 ECFP + 2D(30) + 3D(30)"

    drug_autoencoder = DrugMultiEncoderWithRecon(
        ecfp_dim=ecfp_dim, f2d_dim=30, f3d_dim=30,
        ecfp_code_dim=20, f2d_code_dim=5, f3d_code_dim=5,  # 20+5+5=30
        dropout=0.2
    ).to(device)

    mut_autoencoder = DeepAutoencoderThreeHiddenLayers(
        input_dim=args.mut_score_dim,
        hidden_dims=args.mut_hidden_dims,
        code_dim=args.code_dim,
        code_activation=True,
        dropout=args.code_dropout,
        dropout_rate=args.code_dropout_rate
    )

    # Forward network
    net = ForwardNetworkTwoHiddenLayers((2 * args.code_dim +38),
                                            args.forward_net_hidden_dim1,
                                            args.forward_net_hidden_dim2,
                                            out_activation=args.forward_net_out_act)

    # === 由数据集得到 num_cancer_classes，然后传给模型 ===
    num_target_classes = train_set.num_target_classes  # 可能为 0
    num_cancer_classes = train_set.num_cancer_classes  # <--- 新增

    model = DEERS_Concat(
        drug_autoencoder=drug_autoencoder,
        mut_line_autoencoder=mut_autoencoder,
        forward_network=net,
        use_cross_attn=args.use_cross_attn,
        attn_dim=args.attn_dim,
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
        num_target_classes=num_target_classes,
        num_cancer_classes=num_cancer_classes  # <--- 新增
    ).to(device)

    ema = EMA(model, decay=0.9995)
    best_ckpt_path = os.path.join(args.outdir, args.exp_name + ".best.pth")

    class_weights = get_target_class_weights(train_set)  # 可能返回 None

    # === 读取癌种权重 CSV（你已写好占位），并传入 loss ===
    # 确保 cancer_weights_t 是 torch.float32 的一维张量，顺序按 label id 排好
    if os.path.exists(args.cancer_class_weights_fpath):
        wdf = pd.read_csv(args.cancer_class_weights_fpath)
        wdf = wdf.sort_values('cancer_label')  # 或 'label_id'，与 CSV 列名一致
        cancer_weights_t = torch.tensor(wdf['class_weight'].values, dtype=torch.float32)
    else:
        cancer_weights_t = None

    loss_fn = MergedLossWithTarget(
        y_loss_weight=args.y_loss_weight,
        drug_reconstruction_loss_weight=args.drug_reconstruction_loss_weight,
        mut_reconstruction_loss_weight=args.mut_reconstruction_loss_weight,
        target_loss_weight=args.target_loss_weight,
        ce_weight=class_weights,
        label_smoothing=0.05,
        cancer_loss_weight=args.cancer_loss_weight,
        cancer_ce_weight=cancer_weights_t,
        cancer_label_smoothing=args.cancer_label_smoothing
    )

    # =============== #
    # ===== Run ===== #
    # =============== #
    setting, result, best_performances, model_max = experiment(args, dataset_partition, model, loss_fn, device)
    save_exp_result(setting, result, args.outdir)

    if best_performances['best_valid_corr'] >= best_best_valid_metric:
        best_best_epoch = best_performances['best_epoch']
        best_best_train_loss = best_performances['best_train_loss']
        best_best_train_metric = best_performances['best_train_corr']
        best_best_valid_loss = best_performances['best_valid_loss']
        best_best_valid_metric = best_performances['best_valid_corr']
        best_setting = setting
        best_result = result

    total_results['best_epoch'].append(best_performances['best_epoch'])
    total_results['best_train_loss'].append(best_performances['best_train_loss'])
    total_results['best_train_corr'].append(best_performances['best_train_corr'])
    total_results['best_valid_loss'].append(best_performances['best_valid_loss'])
    total_results['best_valid_corr'].append(best_performances['best_valid_corr'])

print(f'Best Train Loss: {best_best_train_loss:.4f}')
print(f'Best Train Corr: {best_best_train_metric:.4f}')
print(f'Best Valid Loss: {best_best_valid_loss:.4f}')
print(f'Best Valid Corr: {best_best_valid_metric:.4f}')

