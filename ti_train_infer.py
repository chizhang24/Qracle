#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import h5py
import numpy as np
import networkx as nx
import re
from torch_geometric.loader import DataLoader

from qracle.datasets import SpinDataset
import matplotlib.pyplot as plt
import h5py
import os
import json
import torch

import pennylane as qml

from qracle.models import DGNN, GIN


from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool

import torch.nn as nn

import torch.optim as optim
import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


train_file = 'data/ti_train.h5'
valid_file = 'data/ti_valid.h5'
root = './data/transverse_ising'
train_dataset = SpinDataset(root=root, file_name=train_file, model_name='ti')
valid_dataset = SpinDataset(root=root, file_name=valid_file, model_name='ti')


# In[7]:


n_epochs = 600
batch_size = 32
myloader = DataLoader(train_dataset, batch_size=batch_size)
youloader = DataLoader(valid_dataset, batch_size=batch_size)

for data in myloader:
    print(data.edge_index.shape)


# In[8]:


model = DGNN(n_x=train_dataset.num_node_features, n_y=train_dataset.num_classes, dim_h=128)

model2 = GIN(n_x=train_dataset.num_node_features, n_y=train_dataset.num_classes, dim_h=128)

model = model.to(device)
model2 = model2.to(device)



loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

n_epochs = 600
batch_size = 32
myloader = DataLoader(train_dataset, batch_size=batch_size)
youloader = DataLoader(valid_dataset, batch_size=batch_size)

for epoch in range(n_epochs):
    model.train()
    mse_list = []
    for data in myloader:
        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse_list.append(float(loss))
    print(f"model epoch {epoch} mse {sum(mse_list) / len(mse_list)}")

  

    model2.train()
    mse_list = []
    for data in myloader:
        data = data.to(device)

        out = model2(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        mse_list.append(float(loss))
    print(f"model2 epoch {epoch} mse {sum(mse_list) / len(mse_list)}")

model.eval()
mse_list = []
for data in youloader:
    data = data.to(device)
    data.x = data.x.to(torch.float32)
    data.y = data.y.to(torch.float32)
    out = model(data.x, data.edge_index, data.batch)
    loss = loss_fn(out, data.y)
    mse_list.append(float(loss))
print(sum(mse_list) / len(mse_list))


model2.eval()
mse_list = []
for data in youloader:
    data = data.to(device)
    data.x = data.x.to(torch.float32)
    data.y = data.y.to(torch.float32)
    out = model2(data.x, data.edge_index, data.batch)
    loss = loss_fn(out, data.y)
    mse_list.append(float(loss))
print(sum(mse_list) / len(mse_list))


# In[9]:


infer_loader = DataLoader(valid_dataset, batch_size=1)
label_shape = valid_dataset.y_shape

dgnn_model_param = []
gin_model_param = []
for data in infer_loader:
    data = data.to(device)
    data.x = data.x.to(torch.float32)
    out = model(data.x, data.edge_index, data.batch)

    out_2 = model2(data.x, data.edge_index, data.batch)
    out_r = out.reshape(label_shape)

    out_2_r = out_2.reshape(label_shape)
    dgnn_model_param.append(out_r.detach().cpu().numpy())
    gin_model_param.append(out_2_r.detach().cpu().numpy())


with h5py.File(valid_file, 'a') as f:
  if 'dgnn_model_param' not in f:
    f.create_group('dgnn_model_param')
  else:
    del f['dgnn_model_param']
    f.create_group('dgnn_model_param')

  if 'gin_model_param' not in f:
    f.create_group('gin_model_param')
  else:
    del f['gin_model_param']
    f.create_group('gin_model_param')


  dgnn_model_param_group = f['dgnn_model_param']
  gin_model_param_group = f['gin_model_param']
  refer_group = f['loss_history']
  index = 0
  for key in refer_group.keys():
    dgnn_p = dgnn_model_param[index]
    gin_p = gin_model_param[index]
    index = index + 1
    dgnn_model_param_group.create_dataset(key, data=dgnn_p)
    gin_model_param_group.create_dataset(key, data=gin_p)


# In[14]:


from qracle.ansatz import CZRXRYLayer

n_layers = 1



old_init = []
dgnn_init = []
gin_init = []
diff_init = []
with h5py.File(valid_file, 'r') as f:
    
    dgnn_model_param_group = f['dgnn_model_param']
    gin_model_param_group = f['gin_model_param']
    loss_history_group = f['loss_history']
    diff_model_param_group = f['diff_model_param']
    n_qubits_group = f['n_qubits']
    coupling_group = f['j_h']


    for key in dgnn_model_param_group.keys():
        # dgnn_model_param = torch.from_numpy(dgnn_model_param_group[key][:])
        # gin_model_param = torch.from_numpy(gin_model_param_group[key][:]) 

        j = coupling_group[key][...][0]
        h = coupling_group[key][...][1]
        n_qubits = n_qubits_group[key][...].item()
        n_cells = [4, 2]
        # dgnn_model_param = dgnn_model_param.permute(1, 2, 0)
        dgnn_model_param = dgnn_model_param_group[key][...]
        dgnn_model_param = dgnn_model_param.reshape((n_layers, n_qubits, 2))
        dgnn_model_param = torch.from_numpy(dgnn_model_param)


        gin_model_param = gin_model_param_group[key][...]
        gin_model_param = gin_model_param.reshape((n_layers, n_qubits, 2))
        gin_model_param = torch.from_numpy(gin_model_param)


        diff_model_param = diff_model_param_group[key][...]
        diff_model_param = diff_model_param.reshape((n_layers, n_qubits, 2))
        diff_model_param = torch.from_numpy(diff_model_param)

        ti_h = qml.spin.transverse_ising('rectangle', n_cells, coupling=j, h=h)
        dev = qml.device('lightning.qubit', wires=n_qubits)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def vqe_circuit(h, params, n_qubits, n_layers):
            for i in range(n_layers):
                CZRXRYLayer(params[i], n_qubits)
            return qml.expval(h)
        
        dgnn_loss = vqe_circuit(ti_h, dgnn_model_param, n_qubits, n_layers)
        gin_loss = vqe_circuit(ti_h, gin_model_param, n_qubits, n_layers)
        diff_loss = vqe_circuit(ti_h, diff_model_param, n_qubits, n_layers)
        old_loss = loss_history_group[key][...][0]

        dgnn_init.append(dgnn_loss)
        gin_init.append(gin_loss)
        old_init.append(old_loss)
        diff_init.append(diff_loss)


# In[15]:


from qracle.utils import get_gnn_results


# In[16]:


get_gnn_results(old_init, diff_init)
get_gnn_results(old_init, dgnn_init)
get_gnn_results(old_init, gin_init)


# In[13]:


torch.save(model, 'saves/ti/dgnn_model.pth')
torch.save(model2, 'saves/ti/gin_model.pth')



