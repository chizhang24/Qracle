#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pennylane as qml 
from pennylane import numpy as np 
import torch 
from tqdm import tqdm
import h5py
import torch.multiprocessing as mp 

from qracle import generate_h2_hamiltonian

import argparse


# In[ ]:


def valid_molecule_hamiltonian(key, bond_length, opt_params, n_layers = 1, n_steps = 600, lr = 1e-3, molecule = 'h2', model_name = 'dgnn'):
    if molecule == 'h2':
        h, n_qubits, coordinates = generate_h2_hamiltonian(bond_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = qml.device('lightning.qubit', wires=n_qubits)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def hamiltonian_exp_val(h, params):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return qml.expval(h)
    
    params = torch.tensor(opt_params, requires_grad=True, device=device)


    assert params.shape == shape, f"Expected params shape {shape}, but got {params.shape}"
    optimizer = torch.optim.Adam([params], lr=lr)
    loss_history = np.zeros(n_steps)

    for i in tqdm(range(n_steps), desc=f'Optimizing {molecule} Hamiltonian using {model_name} initialization'):
        optimizer.zero_grad()
        loss = hamiltonian_exp_val(h, params)

        loss.backward()
        optimizer.step()
        loss_history[i] = loss.item()

    return key, loss_history 


# In[ ]:


# key, bond_length, opt_params, n_layers = 1, n_steps = 600, lr = 1e-3, molecule = 'h2'

def valid_in_batches(valid_file_path, n_layers = 1, n_steps = 600, lr = 1e-3, batch_size = 10, n_workers = 8, molecule = 'h2', model_name = 'dgnn'):

    with h5py.File(valid_file_path, 'a') as f:
        if model_name == 'dgnn':
            new_loss_history_group = f.require_group('dgnn_loss_history')
            opt_model_param_group = f['dgnn_model_param']
        elif model_name == 'diff':
            new_loss_history_group = f.require_group('diff_loss_history')
            opt_model_param_group = f['diff_model_param']
        elif model_name == 'gin':
            new_loss_history_group = f.require_group('gin_loss_history')
            opt_model_param_group = f['gin_model_param']

        bond_length_group = f['bond_length']
        # n_qubits_group = f['n_qubits']
        keys = list(opt_model_param_group.keys())

        n_samples = len(keys)

        for i in tqdm(range(0, n_samples, batch_size), desc=f'Validating using {model_name} initialization in batches'):
            batch_tasks = []

            if i + batch_size > n_samples:
                batch_size = n_samples - i
            # Ensure batch size is correct
            for j in range(batch_size):

                key = keys[i+j]

                bond_length = bond_length_group[key][...].item()
                opt_model_param = opt_model_param_group[key][...]
                # n_qubits = n_qubits_group[key][...].item()
                opt_model_param = np.transpose(opt_model_param, (1, 2, 0))  # Shape: (4, 5, 6)

                batch_tasks.append((key, bond_length, opt_model_param, n_layers, n_steps, lr, molecule, model_name))
            assert len(batch_tasks) == batch_size, "Batch size mismatch"

            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=n_workers) as pool:
                results = pool.starmap(valid_molecule_hamiltonian, batch_tasks)

            for key, loss_history in results:
                new_loss_history_group.create_dataset(key, data=loss_history)

            f.flush()

            print(f"Batch {i//batch_size + 1} processed and written to file.")

        f.close()


# In[ ]:


if __name__ == '__main__':
# valid_file_path, n_layers = 1, n_steps = 600, lr = 1e-3, batch_size = 10, n_workers = 8, molecule = 'h2', model_name = 'dgnn'

    parser = argparse.ArgumentParser(description='Validate molecule hamiltonian data')

    parser.add_argument('--valid_file_path', type=str, default='data/h2_valid.h5', help='Path to the validation h5 file')

    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers for the quantum circuit')
    parser.add_argument('--n_steps', type=int, default=600, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for validation')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of worker processes for parallel computation')
    parser.add_argument('--molecule', type=str, default='h2', help='Molecule type (default: h2)')
    parser.add_argument('--model_name', type=str, default='dgnn', help='Model name for initialization (default: dgnn)')
    args = parser.parse_args()

    valid_in_batches(args.valid_file_path,
                     n_layers=args.n_layers,
                     n_steps=args.n_steps,
                     lr=args.lr,
                     batch_size=args.batch_size,
                     n_workers=args.n_workers,
                     molecule=args.molecule,
                     model_name=args.model_name)
    print("Validation completed and saved to file.")

