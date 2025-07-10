#!/usr/bin/env python
# coding: utf-8

# In[8]:


import joblib
import h5py
import numpy as np
import networkx as nx
import re
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import h5py
import os
import json
import torch

import pennylane as qml




import torch.nn.functional as F




from qracle.models import DGNN, GCN, GIN
from qracle.datasets import DDataset
import torch.nn as nn

import torch.optim as optim
import tqdm


# In[6]:




# In[9]:


train_file = 'data/heisenberg_xyz_train.h5'
valid_file = 'data/heisenberg_xyz_valid.h5'

root = './data/'

train_dataset = DDataset(root=root, file_name=train_file)
test_dataset = DDataset(root=root, file_name=valid_file)


# In[11]:


model = DGNN(n_x=train_dataset.num_node_features, n_y=train_dataset.num_classes, dim_h=128)
model1 = GCN(n_x=train_dataset.num_node_features, n_y=train_dataset.num_classes, dim_h=128)
model2 = GIN(n_x=train_dataset.num_node_features, n_y=train_dataset.num_classes, dim_h=128)

loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

n_epochs = 600
batch_size = 32
myloader = DataLoader(train_dataset, batch_size=batch_size)
youloader = DataLoader(test_dataset, batch_size=batch_size)

for epoch in range(n_epochs):
    model.train()
    mse_list = []
    for data in myloader:
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse_list.append(float(loss))
    print(f"model epoch {epoch} mse {sum(mse_list) / len(mse_list)}")

    # model1.train()
    # mse_list = []
    # for data in myloader:
    #    out = model1(data.x, data.edge_index, data.batch)
    #    loss = loss_fn(out, data.y)
    #    optimizer1.zero_grad()
    #    loss.backward()
    #    optimizer1.step()
    #    mse_list.append(float(loss))
    # print(f"model1 epoch {epoch} mse {sum(mse_list) / len(mse_list)}")

    model2.train()
    mse_list = []
    for data in myloader:
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
    out = model(data.x, data.edge_index, data.batch)
    loss = loss_fn(out, data.y)
    mse_list.append(float(loss))
print(sum(mse_list) / len(mse_list))

# model1.eval()
# mse_list = []
# for data in youloader:
#    out = model1(data.x, data.edge_index, data.batch)
#    loss = loss_fn(out, data.y)
#    mse_list.append(float(loss))
# print(sum(mse_list) / len(mse_list))

model2.eval()
mse_list = []
for data in youloader:
    out = model2(data.x, data.edge_index, data.batch)
    loss = loss_fn(out, data.y)
    mse_list.append(float(loss))
print(sum(mse_list) / len(mse_list))


torch.save(model, './saves/dgnn_model.pth')
torch.save(model2, './saves/gin_model.pth')


# In[27]:


dgnn_model_param = []
gin_model_param = []



for key, data in test_dataset.pairs:
      out = model(data.x, data.edge_index, torch.tensor([0]*16, dtype=torch.int64))
      out_2 = model2(data.x, data.edge_index, torch.tensor([0]*16, dtype=torch.int64))
      out_r = out.reshape(test_dataset.label_shape)
      out_2_r = out_2.reshape(test_dataset.label_shape)
      dgnn_model_param.append((key, out_r.detach().numpy()))
      gin_model_param.append((key, out_2_r.detach().numpy()))


# In[28]:


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
  for key, dgnn_p in dgnn_model_param:
    dgnn_model_param_group.create_dataset(key, data=dgnn_p)

  for key, gin_p in gin_model_param:
    gin_model_param_group.create_dataset(key, data=gin_p)

