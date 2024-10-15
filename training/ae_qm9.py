"""

Experimental Implementation of a simple Equivariant GNN


"""

from typing import Union, Tuple, Optional
import numpy as np
from torch.nn import Module
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch
from torch_geometric.typing import OptTensor
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.inits import reset

from giae.sn.metrics import PermutationMatrixPenalty

from torch_scatter import scatter

from giae.se3.utils import get_rotation_matrix_from_two_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
sys.path.append('./training/models/')
from giae_model import Model

import torch
from torch.utils.data import Dataset

from torch_geometric.utils import remove_self_loops
try:
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError:
    from torch_geometric.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset

from pytorch_lightning import LightningDataModule
import math

class DataModule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, num_eval_samples, train_samples=100):
        super().__init__()
        self.dataset = dataset
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_eval_samples = num_eval_samples

    def setup(self, stage=None):
        pass

    def train_dataloader(self, shuffle=False):
        dataset = self.dataset(root='./datasets/qm9-2.4.0/')
        data_list = []
        for i, data in enumerate(dataset):
            if i < self.train_samples:
                data_list.append(data)
            else:
                break
        dataloader = DataLoader(
            dataset=data_list,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,   # True
            shuffle=shuffle
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset(root='./datasets/qm9-2.4.0/')
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,   # True
        )
        return dataloader

if __name__ == '__main__':
    from torch_geometric.datasets import QM9
    datamodule = DataModule(dataset=QM9,
                            batch_size=32,
                            num_workers=0,
                            train_samples=5000,
                            num_eval_samples=1000)

    datamodule.setup()
    loader = datamodule.train_dataloader()
    model = Model(hidden_dim=256, emb_dim=32, num_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=0.995,
        )
    perm_loss = PermutationMatrixPenalty()
    
    epochs=1000
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            pos_out, perm, vout, rot = model(data, hard=False)
            loss = torch.pow(data.pos - pos_out, 2)
            loss = scatter(loss, data.batch, dim=0, reduce="add")
            mse_loss = loss.mean()
            ploss = perm_loss(perm)
            loss = mse_loss + 0.025 * ploss
            loss.backward()
            optimizer.step()
            print(f" Epoch: {epoch}/{epochs}, Step {i}/{len(loader)}, Loss: {loss.item():.4f}")

                    # Accumulate total loss and total number of samples
            total_loss += mse_loss.item() * data.num_graphs
            total_samples += data.num_graphs
        epoch_rmse = np.sqrt(total_loss / total_samples)
        lr_scheduler.step()
        print(f"Epoch {epoch}: RMSE: {epoch_rmse:.4f}")

    model = model.to("cpu")
    torch.save(model.state_dict(), './training/models/giae_model.pth')
