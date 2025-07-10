#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pennylane as qml
from pennylane import numpy as np
import torch 
import torch.multiprocessing as mp 
import h5py 
from tqdm import tqdm
import math
import json 
import argparse
from qracle.ansatz import CZRXRYLayer

# In[4]:




# In[ ]:


def valid_fermi_hubbard(key, t, U, init_params,  n_layers = 1, n_qubits = 8, n_steps = 500,  lr = 1e-2, model_name = 'dgnn'):
    n_cells = [n_qubits//2]

    fermi_hubbard_h = qml.spin.fermi_hubbard('chain', n_cells, hopping=t, coulomb=U, mapping='jordan_wigner')

    dev = qml.device('lightning.qubit', wires=n_qubits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def vqe_circuit(h, params, n_qubits, n_layers):
        for i in range(n_layers):
            CZRXRYLayer(params[i], n_qubits)
        return qml.expval(h)
    
    def hf_expectation(h, params,  n_qubits, n_layers):
        num_electrons = n_qubits // 2
        hf_state = np.array([1] * num_electrons + [0] * (n_qubits - num_electrons))
        qml.BasisState(hf_state, wires=range(n_qubits))
        for i in range(n_layers):
            CZRXRYLayer(params[i], n_qubits)
            
        return qml.expval(h)


    params = torch.tensor(init_params, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([params], lr=lr)

    loss_history = np.zeros(n_steps) 

    if model_name in {'dgnn', 'diff', 'gin'}:
        for i in tqdm(range(n_steps), desc=f'Optimizing Fermi Hubbard {key}'):
            optimizer.zero_grad()
            loss = vqe_circuit(fermi_hubbard_h, params, n_qubits, n_layers)
            loss.backward()
            optimizer.step()
            loss_history[i] = loss.item()
    elif model_name == 'hf':
        for i in tqdm(range(n_steps), desc=f'Optimizing Fermi Hubbard {key}'):
            optimizer.zero_grad()
            loss = hf_expectation(fermi_hubbard_h, params, n_qubits, n_layers)
            loss.backward()
            optimizer.step()
            loss_history[i] = loss.item()


    best_params = params.detach().cpu().numpy()
    best_params = best_params.reshape((1, 2, n_qubits))
    return key, n_qubits, t, U , loss_history, best_params


# In[ ]:


def valid_in_batches(h5_file, n_qubits, n_layers=1, n_steps=500, lr=1e-2, batch_size = 10, n_workders = 8, model_name = 'dgnn'):
    with h5py.File(h5_file, 'a') as f:
        if model_name == 'dgnn':
            new_loss_history_group = f.require_group('dgnn_loss_history')
            opt_model_param_group = f.require_group('dgnn_model_param')
        elif model_name == 'diff':
            new_loss_history_group = f.require_group('diff_loss_history')
            opt_model_param_group = f.require_group('diff_model_param')
        elif model_name == 'gin':
            new_loss_history_group = f.require_group('gin_loss_history')
            opt_model_param_group = f.require_group('gin_model_param')
        elif model_name == 'hf':
            new_loss_history_group = f.require_group('hf_loss_history')
            opt_model_param_group = f.require_group('hf_model_param')


        t_U_group = f['t_U']

        keys = list(opt_model_param_group.keys())

        n_qubits_group = f.require_group('n_qubits')

        n_samples = len(keys)

        for i in tqdm(range(0, n_samples, batch_size), desc='Processing in batches'):
            batch_tasks = []
            for j in range(batch_size):
                key = keys[i+j]
                t = t_U_group[key][...][0]
                U = t_U_group[key][...][1]

                opt_model_param = opt_model_param_group[key][...]
                n_qubits = n_qubits_group[key][...].item()
                init_params = opt_model_param.reshape(n_layers, n_qubits, 2)

                batch_tasks.append((key, t, U, init_params, n_layers, n_qubits, n_steps, lr, model_name))
                # key, t, U, n_layers = 1, n_qubits = 8, n_steps = 500,  lr = 1e-2
            assert len(batch_tasks) == batch_size, "Batch size mismatch"

            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=n_workders) as pool:
                results = pool.starmap(valid_fermi_hubbard, batch_tasks)

            for key, _, _, _, new_loss_history, _ in results:
                if key in new_loss_history_group:
                    del new_loss_history_group[key]
                else:
                    new_loss_history_group.create_dataset(key, data=new_loss_history)

            f.flush()

            print(f"Batch {i//batch_size + 1} processed and written to file.")
            
        f.close()


# In[ ]:


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Validate Fermi Hubbard data')
    parser.add_argument('--n_qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_steps', type=int, default=600, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
   
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--h5_file', type=str, default='data/fh_valid.h5', help='Path to HDF5 file')
    parser.add_argument('--model_name', type=str, default='dgnn', help='Model name')
    args = parser.parse_args()


    valid_in_batches(args.h5_file, args.n_qubits, args.n_layers, args.n_steps, args.lr, args.batch_size, args.n_workers, args.model_name)

    print(f"Data validtation complete. Data saved to {args.h5_file}")

