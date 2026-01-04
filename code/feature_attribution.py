import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# ====== DEERS: Model Definition ====== #
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
        modules.append(nn.Sigmoid())
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
                 forward_network):
        super(DEERS_Concat, self).__init__()
        self.drug_autoencoder = drug_autoencoder
        self.mut_line_autoencoder = mut_line_autoencoder
        self.forward_network = forward_network

    def forward(self, drug_features, mut_features, cell_features):
        drug_code, drug_reconstruction = self.drug_autoencoder(drug_features)
        mut_code, mut_reconstruction = self.mut_line_autoencoder(mut_features)
        x = torch.cat((drug_code, mut_code, cell_features), axis=1)  ####
        return self.forward_network(x), drug_reconstruction, mut_reconstruction


class MergedLoss(nn.Module):
    def __init__(self, y_loss_weight=1., drug_reconstruction_loss_weight=0.1, mut_reconstruction_loss_weight=0.2):
        super(MergedLoss, self).__init__()
        self.y_loss_weight = y_loss_weight
        self.drug_reconstruction_loss_weight = drug_reconstruction_loss_weight
        self.mut_reconstruction_loss_weight = mut_reconstruction_loss_weight
        self.output_criterion = nn.MSELoss()
        self.reconstruction_criterion = nn.BCELoss()

    def forward(self, pred_y, drug_reconstruction, mut_reconstruction, drug_input, mut_input, true_y):
        output_loss = self.output_criterion(pred_y, true_y)
        drug_reconstruction_loss = self.reconstruction_criterion(drug_reconstruction, drug_input)
        mut_reconstruction_loss = self.reconstruction_criterion(mut_reconstruction, mut_input)
        return output_loss, drug_reconstruction_loss, mut_reconstruction_loss


dr = pd.read_csv("C:/Users/DELL/Desktop/work2/DiM2-DRP-main/data/GDSC_ECFP_2D3D.csv", sep=',', header=0)
mut_score_df = pd.read_csv("C:/Users/DELL/Desktop/work2/DiM2-DRP-main/data/GDSC_mutation_input.csv", sep=',', header=0)
cell_exprs_df = pd.read_csv("C:/Users/DELL/Desktop/work2/DiM2-DRP-main/data/GDSC_ssgsea_input.csv", sep=',', header=0)
samples_train = pd.read_csv("C:/Users/DELL/Desktop/work2/DiM2-DRP-main/data/GDSC_IC50_by_both.csv", sep=',', header=0)

model = torch.load("C:/Users/DELL/Desktop/work2/DiM2-DRP-main/model/my_both__cv00.model")
drug_autoencoder = model.drug_autoencoder
mut_autoencoder = model.mut_line_autoencoder

drug_mut_exp_attributions_list = []
for i in range(1, int(samples_train.copy().shape[0] / 50)):
    samples = samples_train.copy()[(i - 1) * 50:i * 50]

    cell_idx = samples['cell_idx']
    drug_idx = samples['drug_idx']

    drug_input = torch.from_numpy(dr.loc[drug_idx.values].iloc[:, 2:].values.astype('float64')).float()
    cl_input1 = torch.from_numpy(mut_score_df.loc[cell_idx.values].iloc[:, 2:].values.astype('float64')).float()
    cl_input2 = torch.from_numpy(cell_exprs_df.loc[cell_idx.values].iloc[:, 2:].values.astype('float64')).float()
    mut_codes, mut_recs = mut_autoencoder(cl_input1)

    dr_codes, dr_recs = drug_autoencoder(drug_input)

    forward_input = torch.cat((dr_codes, mut_codes, cl_input2), axis=1)


    def net(forward_input):
        return model.forward_network(forward_input)


    ig = IntegratedGradients(net)

    forward_input.requires_grad_()
    forward_input.shape

    attributions, delta = ig.attribute(forward_input, return_convergence_delta=True)
    drug_mut_exp_attributions = attributions
    drug_mut_exp_attributions = drug_mut_exp_attributions.cpu().detach().numpy()
    drug_mut_exp_attributions = np.mean(drug_mut_exp_attributions, axis=0)
    drug_mut_exp_attributions_list.append(drug_mut_exp_attributions)

drug = np.array(drug_mut_exp_attributions_list)
drug_mut_exp_attributions = np.mean(drug, axis=0)

df = pd.DataFrame(drug_mut_exp_attributions[1:])
csv_file_path = 'C:/Users/DELL/Desktop/work2/DiM2-DRP-main/data/drug_mut_exp_attributions.csv'
df.to_csv(csv_file_path, index=False)
