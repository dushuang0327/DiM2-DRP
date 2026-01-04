import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

# === Task-specific === #
from copy import deepcopy
from dataset import NPweightingDataSet
from utils import *
from trainers import logging, train, validate, test
from models import *


dr = pd.read_csv("D:\\work2\\DiM2-DRP-main\\data/GDSC_ECFP_2D3D.csv", sep=',', header=0)
mut_score_df = pd.read_csv("D:\\work2\\DiM2-DRP-main\\data\\GDSC_mutation_input.csv", sep=',', header=0)
cell_exprs_df = pd.read_csv("/data/GDSC_ssgsea_input.csv", sep=',', header=0)
samples_train = pd.read_csv("D:\\work2\\DiM2-DRP-main\\data\\GDSC_IC50_by_both.csv", sep=',', header=0)

model = torch.load("D:\\work2\\DiM2-DRP-main\\model\\my_both__cv00.model")
drug_autoencoder=model.drug_autoencoder
input_tensor = torch.from_numpy(dr.iloc[0:, 2:].values.astype('float64')).float()
dr_codes, dr_recs = drug_autoencoder(input_tensor)
my_array = dr_codes.detach().numpy()
my_array = pd.DataFrame(my_array, index=dr['drug_name'].values)

my_array.to_csv(r'D:\work2\DiM2-DRP-main\data\drug_dim.csv', index=True)

model = torch.load("D:\\work2\\DiM2-DRP-main\\model\\my_both__cv00.model")

samples  = samples_train.copy()

cell_idx = samples['cell_idx']
drug_idx = samples['drug_idx']

drug_input = torch.from_numpy(dr.loc[drug_idx.values].iloc[:, 2:].values.astype('float64')).float()
cl_input1 = torch.from_numpy(mut_score_df.loc[cell_idx.values].iloc[:, 2:].values.astype('float64')).float()
cl_input2 = torch.from_numpy(cell_exprs_df.loc[cell_idx.values].iloc[:, 2:].values.astype('float64')).float()

drug_autoencoder=model.drug_autoencoder
dr_codes, dr_recs = drug_autoencoder(drug_input)

mut_line_autoencoder=model.mut_line_autoencoder
mut_codes, mut_recs = mut_line_autoencoder(cl_input1)

forward_input = torch.cat((dr_codes, mut_codes, cl_input2),axis = 1)
forward_input = forward_input.detach().numpy()
forward_input = pd.DataFrame(forward_input)
forward_input = pd.concat([samples_train[['cell_name', 'drug_name']], forward_input], axis=1)

forward_input.to_csv(r'D:\work2\DiM2-DRP-main\data\forward_input.csv', index=False)

