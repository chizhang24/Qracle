#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from pennylane import numpy as np
import torch 
import torch.multiprocessing as mp 
import h5py 
from tqdm import tqdm
import math
import json 
import argparse


# In[2]:


def CZRXRYLayer(params, n_qubits):
    for i in range(n_qubits -1):
        qml.CZ(wires = [i, i+1])
    qml.CZ(wires = [n_qubits - 1, 0])

    for i in range(n_qubits):
        qml.RX(params[i, 0], wires = i)
        qml.RY(params[i, 1], wires = i)


# In[3]:


def valid_transverse_ising(key, j, h, init_params,  cell_wid = 2, cell_len = 4, n_layers = 1, n_steps = 500,  lr = 1e-2, model_name = 'dgnn'):
    n_cells = [cell_len, cell_wid]
    n_qubits = cell_wid * cell_len
    ti_h = qml.spin.transverse_ising('rectangle', n_cells, coupling = j, h = h)

    dev = qml.device('lightning.qubit', wires=n_qubits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def vqe_circuit(h, params, n_qubits, n_layers):
        for i in range(n_layers):
            CZRXRYLayer(params[i], n_qubits)
        return qml.expval(h)
  

    print(f'converting {model_name} to tensor')
    params = torch.tensor(init_params, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([params], lr=lr)

    loss_history = np.zeros(n_steps) 

    for i in tqdm(range(n_steps), desc=f'Optimizing Fermi Hubbard {key}'):
        optimizer.zero_grad()
        loss = vqe_circuit(ti_h, params, n_qubits, n_layers)
        loss.backward()
        optimizer.step()
        loss_history[i] = loss.item()


    best_params = params.detach().cpu().numpy()
    best_params = best_params.reshape((1, 2, n_qubits))
    n_cells = np.array(n_cells)
    return key, j, h , n_cells, n_qubits,  loss_history, best_params


# In[ ]:


def valid_in_batches(h5_file, cell_wid=2, cell_len= 4,  n_layers=1, n_steps=500, lr=1e-2, batch_size = 10, n_workders = 8, model_name = 'dgnn'):
    with h5py.File(h5_file, 'a') as f:
        if model_name == 'dgnn':
            new_loss_history_group = f.create_group('dgnn_loss_history')
            opt_model_param_group = f['dgnn_model_param']
        elif model_name == 'diff':
            new_loss_history_group = f.create_group('diff_loss_history')
            opt_model_param_group = f['diff_model_param']
        elif model_name == 'gin':
            new_loss_history_group = f.create_group('gin_loss_history')
            opt_model_param_group = f['gin_model_param']

        # key, j, h, cell_wid = 2, cell_len = 4, n_layers = 1, n_steps = 500,  lr = 1e-2)
        j_h_group = f['j_h']

        keys = list(opt_model_param_group.keys())

        n_qubits_group = f['n_qubits']

        n_samples = len(keys)
        for i in tqdm(range(0, n_samples, batch_size), desc='Processing in batches'):
            batch_tasks = []
            for k in range(batch_size):

                key = keys[i + k]

                j = j_h_group[key][...][0]
                h = j_h_group[key][...][1]
                opt_model_param = opt_model_param_group[key][...]

                n_qubits = n_qubits_group[key][...].item()

                init_params = opt_model_param.reshape((n_layers, n_qubits, 2))


                batch_tasks.append((key, j, h, init_params, cell_wid, cell_len, n_layers, n_steps, lr, model_name))
                # key, t, U, n_layers = 1, n_qubits = 8, n_steps = 500,  lr = 1e-2
            assert len(batch_tasks) == batch_size, "Batch size mismatch"

            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=n_workders) as pool:
                results = pool.starmap(valid_transverse_ising, batch_tasks)
                # key, j, h , n_cells, n_qubits,  loss_history, best_params
            for key, _, _, _, _,  new_loss_history, _ in results:
                new_loss_history_group.create_dataset(key, data=new_loss_history)

            
            f.flush()

            print(f"Batch {i//batch_size + 1} processed and written to file.")
    
    f.close()



# In[ ]:


if __name__ == '__main__':


    # h5_file, cell_wid=2, cell_len= 4,  n_layers=1, n_steps=500, lr=1e-2, batch_size = 10, n_samples = 2000, n_workders = 8
    parser = argparse.ArgumentParser(description='Validate Transverse Ising data')
    parser.add_argument('--cell_wid', type=int, default=2, help='Width of the cell')
    parser.add_argument('--cell_len', type=int, default=4, help='Length of the cell')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_steps', type=int, default=600, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
  
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--h5_file', type=str, default='data/ti_valid.h5', help='Path to HDF5 file')

    parser.add_argument('--model_name', type=str, default='dgnn', help='Model name')
    args = parser.parse_args()

    valid_in_batches(args.h5_file, args.cell_wid, args.cell_len,  args.n_layers, args.n_steps, args.lr, args.batch_size, args.n_workers, args.model_name)

    print(f"Data validation complete. Data saved to {args.h5_file}")

