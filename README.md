This repo contains the full implementation of the paper (Qracle: A Graph-Neural-Network-based Parameter Initializer for Variational Quantum Eigensolvers)[https://arxiv.org/abs/2505.01236] (Accepted by IEEE QCE25).








## Environment Configuration

create a `pip` venv 

```bash 
python3 -m venv ~/venvs/env_qracle # replace '~/venvs/env_qracle' by your actual venv path
```

activate the venv `env_qracle` 

```bash
source ~/venvs/env_qracle/bin/activate
```


inside `env_qracle`, first install `PyTorch` using `pip`, according to the  [official `PyTorch` instruction](https://pytorch.org/get-started/locally/) and your cuda version (mine is `CUDA 12.8`). 


```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install `PyG` (PyTorch Geometric) and necessary dependencies according to the [official instruction](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and your `torch` and `CUDA` version. 

```bash
pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html #install torch_geometric independencies
```


Then install `pennylane` 

```bash
pip install pennylane --upgrade
```


then install other necessary packages 
```bash
pip install joblib
pip install h5py
pip install matplotlib
pip install pandas
pip install seaborn
```

## Usage

Train `Qracle` on the Fermi-Hubbard dataset 

```bash 
python fh_train_infer.py
```
During training, `PyG` will store processed dataset of graphs into `./pyg_data/fh`

Validate the result of training

```bash 
python fh_valid.py --h5_file ./data/fh_valid.h5
```
can adjust other parameters as described in `fh_valid.py`. 

## Cite As 

If you use this work in your research, please cite:

@article{zhang2025qracle,
  title={Qracle: A Graph-Neural-Network-based Parameter Initializer for Variational Quantum Eigensolvers},
  author={Zhang, Chi and Jiang, Lei and Chen, Fan},
  arxiv={arXiv:2505.01236},
  year={2025}
}