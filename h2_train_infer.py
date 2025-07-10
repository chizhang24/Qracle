#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qracle.datasets import DDataset
from qracle.models import DGNN, GIN
from torch_geometric.loader import DataLoader


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import h5py


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




train_file = 'data/h2_train.h5'
valid_file = 'data/h2_valid.h5'
       


# In[2]:


train_dataset = DDataset(root='./data/h2_small/', file_name=train_file, molecule_name='h2')
test_dataset = DDataset(root='./data/h2_small/', file_name=valid_file, molecule_name='h2')

test_dataset.num_node_features


# In[3]:


model = DGNN(n_x=train_dataset.num_node_features, n_y=train_dataset.num_classes, dim_h=128)
model = model.to(device)
model2 = GIN(n_x=train_dataset.num_node_features, n_y=train_dataset.num_classes, dim_h=128)
model2 = model2.to(device)
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

n_epochs = 8300
batch_size = 32
myloader = DataLoader(train_dataset, batch_size=batch_size)
youloader = DataLoader(test_dataset, batch_size=batch_size)

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
    out = model(data.x, data.edge_index, data.batch)
    loss = loss_fn(out, data.y)
    mse_list.append(float(loss))
print(sum(mse_list) / len(mse_list))



model2.eval()
mse_list = []
for data in youloader:
    data = data.to(device)
    out = model2(data.x, data.edge_index, data.batch)
    loss = loss_fn(out, data.y)
    mse_list.append(float(loss))
print(sum(mse_list) / len(mse_list))


# torch.save(model, 'saves/h2/dgnn_model.pth')
# torch.save(model2, 'saves/h2/gin_model.pth')


# In[4]:


infer_loader = DataLoader(test_dataset, batch_size = 1)
label_shape = test_dataset.y_shape
dgnn_model_param = []
gin_model_param = []
for data in infer_loader:
    data = data.to(device)
    out = model(data.x, data.edge_index, data.batch)

    out_2 = model2(data.x, data.edge_index, data.batch)
    out_r = out.reshape(label_shape)

    out_2_r = out_2.reshape(label_shape)
    dgnn_model_param.append(out_r.detach().cpu().numpy())
    gin_model_param.append(out_2_r.detach().cpu().numpy())


# In[5]:


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


# In[ ]:





# In[10]:


from qracle.hamil import generate_h2_hamiltonian

import pennylane as qml
import matplotlib.pyplot as plt
n_layers = 2
n_qubits = 4



old_init = []
dgnn_init = []
gin_init = []
diff_init = []

with h5py.File(valid_file, 'r') as f:
    
    dgnn_model_param_group = f['dgnn_model_param']
    gin_model_param_group = f['gin_model_param']
    loss_history_group = f['loss_history']
    diff_model_param_group = f['diff_model_param']
    bond_len_group = f['bond_length']


    for key in dgnn_model_param_group.keys():
        dgnn_model_param = torch.from_numpy(dgnn_model_param_group[key][:])
        gin_model_param = torch.from_numpy(gin_model_param_group[key][:]) 
        diff_model_param = torch.from_numpy(diff_model_param_group[key][:])

        bond_len = bond_len_group[key][...].item()

        dgnn_model_param = dgnn_model_param.permute(1, 2, 0)

        gin_model_param = gin_model_param.permute(1, 2, 0)
        diff_model_param = diff_model_param.permute(1, 2, 0)

        hamil, _, _ = generate_h2_hamiltonian(bond_len)

        dev = qml.device('lightning.qubit', wires = n_qubits)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        assert shape == dgnn_model_param.shape
        assert shape == gin_model_param.shape
        assert shape == diff_model_param.shape

        @qml.qnode(dev)
        def hamiltonian_exp_val(h, params):
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return qml.expval(h)
        

        dgnn_loss = hamiltonian_exp_val(hamil, dgnn_model_param)
        gin_loss = hamiltonian_exp_val(hamil, gin_model_param)
        diff_loss = hamiltonian_exp_val(hamil, diff_model_param)

        loss_history = loss_history_group[key][...]

        dgnn_init.append(dgnn_loss.item())
        gin_init.append(gin_loss.item())
        old_init.append(loss_history[0])
        diff_init.append(diff_loss.item())


# In[11]:


def get_gnn_results(old_init, new_init):
    old_init = np.array(old_init)
    new_init = np.array(new_init)




    improved_init = old_init >= new_init

    worsened_init = old_init < new_init

    improved_cnt = np.sum(improved_init)
    worsened_cnt = np.sum(worsened_init)
    print(f"Improved initial loss: {improved_cnt} / {len(old_init)}")
    print(f"Worsened initial loss: {worsened_cnt} / {len(old_init)}")
    categories = ['lower than random init', 'higher  than random init']
    counts = [improved_cnt, worsened_cnt]  # Given data

    # Create the bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=['green', 'orange'])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, str(height),
                ha='center', va='bottom', fontsize=12, fontweight='bold')


    # Add labels and title
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.title(f"Classification Outcomes Distribution out of {len(old_init)} circuits")

    # Show the plot
    plt.show()


# In[12]:


get_gnn_results(old_init, diff_init)
get_gnn_results(old_init, dgnn_init)
get_gnn_results(old_init, gin_init)


# In[9]:


torch.save(model, 'saves/h2_small/dgnn_model.pth')
torch.save(model2, 'saves/h2_small/gin_model.pth')

