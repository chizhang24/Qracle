
import joblib
import h5py
import numpy as np
import networkx as nx
import re
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx
import matplotlib.pyplot as plt
import os
import json
import torch

import pennylane as qml
from pennylane.ops.op_math.sum import Sum



from tqdm import tqdm

class DDataset(InMemoryDataset):
  def __init__(self, root,  file_name=None, transform=None, pre_transform=None, pre_filter=None, molecule_name=None):
    self.filename = file_name
    self.molecule_name = molecule_name

    super(DDataset, self).__init__(root, transform, pre_transform, pre_filter)
    self.load(self.processed_paths[0])
    #self.data, self.slices = torch.load(self.processed_paths[0])

    self.y_shape = self.get_label_shape()


  @property
  def raw_file_names(self):
    return [self.filename]

  @property
  def processed_file_names(self):
    return [self.filename + '.pt']


  def get_h2_hamil_matrix(self, bond_length):
      symbols = ["H", "H"]

      # Hydrogen atoms along the x-axis at ± bond_length/2
      coordinates = np.array([
          [-bond_length / 2, 0.0, 0.0],  # Hydrogen 1
          [bond_length / 2, 0.0, 0.0]    # Hydrogen 2
      ])

      # Generate the Hamiltonian using PennyLane's quantum chemistry module
      h, _ = qml.qchem.molecular_hamiltonian(
          symbols, coordinates,
          charge=0, 
          mult=1, 
          basis="sto-3g"
      )

      h_mat = h.eigendecomposition['eigval']


      return np.diag(h_mat)
  


  def get_hehp_hamil_matrix(self, bond_length):
      symbols = ['He', 'H']
        
        
      coordinates = np.array([[bond_length / 3, 0.0, 0.0], [-2 * bond_length / 3, 0.0, 0.0]]) 

      # Generate the Hamiltonian using PennyLane's quantum chemistry module
      h, _ = qml.qchem.molecular_hamiltonian(
          symbols, coordinates,
          charge=1, 
          mult=1, 
          basis="sto-3g"
      )

      h_mat = h.eigendecomposition['eigval']


      return np.diag(h_mat)

  def get_label_shape(self):
    with h5py.File(self.filename, 'r') as f:
      x_group = f['model_param']
      for key, param in x_group.items():
        if key is not None:
          x_shape = param.shape
          break
      f.close()

    return x_shape


  def process(self):
    data_list = []
    bond_len_list = []
    label_list = []

    n_qubits_list = []
    with h5py.File(self.filename, "r") as f:
      bond_len_group = f["bond_length"]
      for key in bond_len_group.keys():
        bond_len_list.append(bond_len_group[key][...].item())



      x_group = f["model_param"]
      for key in x_group.keys():
        label_list.append(x_group[key][...])

      n_qubits_group = f["n_qubits"]
      for key in n_qubits_group.keys():
        n_qubits = int(n_qubits_group[key][()])
        n_qubits_list.append(n_qubits)

      f.close()

    index = 0
    for bond_len in bond_len_list:
      if self.molecule_name == 'h2':
        h_matrix = self.get_h2_hamil_matrix(bond_len)
      elif self.molecule_name == 'hehp':
        h_matrix = self.get_hehp_hamil_matrix(bond_len)

      row, col = np.nonzero(h_matrix)

      weights = h_matrix[row, col]

      num_nodes, _ = h_matrix.shape

      edge_index = torch.tensor([row, col], dtype=torch.long)
      edge_weight = torch.tensor(weights, dtype=torch.float32)
    
      edge_attr = edge_weight.unsqueeze(-1)  # Convert to [num_edges, 1] for GINConv/GATConv
      y = torch.tensor([label_list[index].flatten()], dtype=torch.float32)
      if self.molecule_name == 'h2':
        x = torch.tensor([[i, bond_len, -bond_len / 2,0, 0,  bond_len / 2, 0, 0, n_qubits_list[index]]for i in range(num_nodes)], dtype=torch.float32)
      elif self.molecule_name == 'hehp':
        x = torch.tensor([[i, bond_len, bond_len / 3, 0, 0, -2*bond_len / 3, 0, 0, n_qubits_list[index]]for i in range(num_nodes)], dtype=torch.float32)
      data = Data(edge_index=edge_index,edge_weight=edge_weight, edge_attr=edge_attr, x=x, y=y)
      data_list.append(data)
      index = index + 1

    self.save(data_list, self.processed_paths[0])









class SpinDataset(InMemoryDataset):
  def __init__(self, root,  file_name=None, transform=None, pre_transform=None, pre_filter=None, model_name=None):
    self.filename = file_name
    self.model_name = model_name

    super(SpinDataset, self).__init__(root, transform, pre_transform, pre_filter)
    self.load(self.processed_paths[0])
    #self.data, self.slices = torch.load(self.processed_paths[0])

    self.y_shape = self.get_label_shape()


  @property
  def raw_file_names(self):
    return [self.filename]

  @property
  def processed_file_names(self):
    return [self.filename + '.pt']



  def get_fh_hamil_matrix(self, n_qubits, t, U):
    n_cells = [n_qubits//2]
    h = qml.spin.fermi_hubbard('chain', n_cells, hopping=t, coulomb = U, mapping= 'jordan_wigner')
    h_mat = h.eigendecomposition['eigval']


    return np.diag(h_mat)

  def get_ti_hamil_matrix(self, j, h, cell_len = 4, cell_wid = 2):
    n_cells = [cell_len, cell_wid]
    ti_h = qml.spin.transverse_ising('rectangle', n_cells, coupling = j, h = h)
    if type(ti_h).__name__ == 'SProd':
      ti_h = Sum(ti_h)
              
    h_mat = ti_h.eigendecomposition['eigval']

    return np.diag(h_mat)


  def get_label_shape(self):
    with h5py.File(self.filename, 'r') as f:
      x_group = f['model_param']
      for key, param in x_group.items():
        if key is not None:
          x_shape = param.shape
          break
      f.close()

    return x_shape


  def process(self):
    data_list = []
    coupling_list = []
    label_list = []

    n_qubits_list = []

    with h5py.File(self.filename, "r") as f:
      if self.model_name == 'fh':
        coupling_group = f["t_U"]
        for key in coupling_group.keys():
          coupling_list.append(coupling_group[key][...])
      elif self.model_name == 'ti':
        coupling_group = f['j_h']
        for key in coupling_group.keys():
          coupling_list.append(coupling_group[key][...])
      
      x_group = f["model_param"]
      for key in x_group.keys():
        label_list.append(x_group[key][...])

      n_qubits_group = f["n_qubits"]
      for key in n_qubits_group.keys():
        n_qubits = int(n_qubits_group[key][()])
        n_qubits_list.append(n_qubits)

      f.close()

    index = 0
    for coupling in tqdm(coupling_list, desc="Processing hamiltonians"):
      if self.model_name == 'fh':
        t = coupling[0]
        U = coupling[1]
        n_qubits = n_qubits_list[index]
        h_matrix = self.get_fh_hamil_matrix(n_qubits, t, U)

      elif self.model_name == 'ti':
        j = coupling[0]
        h = coupling[1]
        cell_len = 4 
        cell_wid = 2
        n_qubits = cell_len * cell_wid
        h_matrix = self.get_ti_hamil_matrix(j, h, cell_len, cell_wid)

      row, col = np.nonzero(h_matrix)

      weights = h_matrix[row, col]

      num_nodes, _ = h_matrix.shape

      edge_index = torch.tensor([row, col], dtype=torch.long)
      edge_weight = torch.tensor(weights, dtype=torch.float32)
    
      edge_attr = edge_weight.unsqueeze(-1)  # Convert to [num_edges, 1] for GINConv/GATConv
      y = torch.tensor([label_list[index].flatten()], dtype=torch.float32)

      if self.model_name == 'fh':
        x = torch.tensor([[i, n_qubits, t, U]for i in range(num_nodes)], dtype=torch.float32)
      elif self.model_name == 'ti':
        x = torch.tensor([[i, cell_len, cell_wid, n_qubits, j, h]for i in range(num_nodes)], dtype=torch.float32)
      data = Data(edge_index=edge_index,edge_weight=edge_weight, edge_attr=edge_attr, x=x, y=y)
      data_list.append(data)
      index = index + 1

    self.save(data_list, self.processed_paths[0])



