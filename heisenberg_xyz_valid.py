#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pennylane as qml
from pennylane import numpy as np
import torch 
import torch.multiprocessing as mp 
import h5py 
from tqdm import tqdm
import math
import json 


# In[ ]:


def CZRXRYLayer(params, n_qubits):
    for i in range(n_qubits -1):
        qml.CZ(wires = [i, i+1])
    qml.CZ(wires = [n_qubits - 1, 0])

    for i in range(n_qubits):
        qml.RX(params[i, 0], wires = i)
        qml.RY(params[i, 1], wires = i)





# In[ ]:


def get_coupling_const(n_coupling=3):
    coupling_const = []
    for i in range(n_coupling):
        j = np.random.choice(np.arange(-3.0, 3.1, 0.1))
        coupling_const.append(j)
    return np.array(coupling_const)


# In[ ]:





# In[ ]:


def valid_heisenberg_xyz(key, n_qubits, J, init_params,  n_layers=1, n_steps = 500, lr = 1e-2, model_name = 'dgnn'): 
    n_cells = [n_qubits]
    heisenberg_xyz_h = qml.spin.heisenberg('chain', n_cells, coupling=J)

    dev = qml.device('lightning.qubit', wires = n_qubits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def vqe_circuit(h, params, n_qubits, n_layers):
        for i in range(n_layers):
            CZRXRYLayer(params[i], n_qubits)
        return qml.expval(h)
  
    

    print(f'converting {model_name} params to tensor')
    params = torch.tensor(init_params, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([params], lr=lr)


    loss_history = np.zeros(n_steps)
    for i in tqdm(range(n_steps), desc=f'Optimizing Heisenberg XYZ model {key}'):
        optimizer.zero_grad()
        loss = vqe_circuit(heisenberg_xyz_h, params, n_qubits, n_layers)

        loss.backward()
        optimizer.step()
        loss_history[i] = loss.item()
        # if i % 10 == 0:
        #     print('Step', i, 'Loss', loss.item())
    


    best_params = params.cpu().detach().numpy()
    print('best params shape', best_params.shape)
    return key, n_qubits, J, loss_history, best_params.reshape(1, 2, n_qubits)


# In[ ]:


def valid_in_batches(h5_file_path, n_qubits, n_layers=1, n_steps=500, lr = 1e-2, batch_size = 10, n_workers = 8, model_name = 'dgnn'):
     
    with h5py.File(h5_file_path, 'a') as f:
        # new_loss_history_group = f.require_group('new_loss_history')
        if model_name == 'dgnn':
            new_loss_history_group = f.require_group('dgnn_loss_history')

            opt_model_param_group = f['dgnn_model_param']
        elif model_name == 'diff':
            new_loss_history_group = f.require_group('diff_loss_history')
            opt_model_param_group = f['diff_model_param']
        elif model_name == 'gin':
            new_loss_history_group = f.require_group('gin_loss_history')
            opt_model_param_group = f['gin_model_param']


        coupling_const_group = f['coupling_const']

        keys = list(opt_model_param_group.keys())
        n_qubits_group = f['n_qubits']

        n_samples = len(keys)

        for i in tqdm(range(0, n_samples, batch_size), desc='Processing in batches'):
            

            batch_tasks = []

            #key, n_qubits, J, n_layers=1, n_steps = 500, lr = 1e-2, init_params = None
            for j in range(batch_size):
                

                key = keys[i + j] 

                opt_model_param = opt_model_param_group[key][...]

                n_qubits  = n_qubits_group[key][...].item()

                init_params = opt_model_param.reshape(n_layers, n_qubits, 2)

                J = coupling_const_group[key][...]


                batch_tasks.append((key, n_qubits, J, init_params, n_layers, n_steps, lr, model_name))


            assert len(batch_tasks) == batch_size , 'batch size mismatch'

            mp.set_start_method("spawn", force=True)
            with mp.Pool(processes=n_workers) as pool:
                results = pool.starmap(valid_heisenberg_xyz, batch_tasks)



            # Write results to HDF5 file
            for k, _, _, new_loss_history, _ in results:
                new_loss_history_group.create_dataset(k, data=new_loss_history)
                

            f.flush()
            print(f'Batch {i // batch_size + 1} written to file and flushed.')

        f.close()






        # for key, loss_history, params in results:
        #     print(f'{key} final loss: {loss_history[-1]}')


# In[ ]:


if __name__ == "__main__":

    with open('HeisenbergXYZConfig.json', 'r') as f:
        config = json.load(f)
    
    n_workers = config['valid_params']['n_workers']  # Number of parallel processes
    batch_size = config['valid_params']['batch_size']  # Number of samples per batch
    valid_file_path = config['valid_params']['h5_file_path']  # Path to save the data
    n_layers = config['valid_params']['n_layers']  
    n_qubits = config['valid_params']['n_qubits']
    n_steps = config['valid_params']['n_steps']
    lr = config['valid_params']['lr']
    model_name = config['valid_params']['model_name']
  
    valid_in_batches(valid_file_path, n_qubits, n_layers=n_layers, n_steps=n_steps, lr=lr,  batch_size=batch_size, n_workers=n_workers, model_name=model_name)


    # process_heisenberg_xyz('sample_0', n_qubits, get_coupling_const(), n_layers=n_layers, n_steps=n_steps, lr=lr, init_params = init_params)

    # process_in_batches(h5_file_path, n_qubits, n_layers=n_layers, n_steps=n_steps, lr=lr, init_params = init_params,  batch_size=batch_size, n_samples=n_samples, n_workers=n_workers)

    # Use multiprocessing Pool to parallelize VQE training
    # mp.set_start_method("spawn", force=True)
    # with mp.Pool(processes=num_processes) as pool:
    #     results = pool.map(run_heisenberg_xyz, range(num_processes))  # Each process gets a different seed

    # print("Final Energies from Parallel VQE Runs:", results)


# In[ ]:




