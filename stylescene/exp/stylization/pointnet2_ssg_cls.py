import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms





class PointNet2ClassificationSSG(pl.LightningModule):
  def __init__(self):
    super().__init__()

    self._build_model()

  def _build_model(self):
    self.SA_modules = nn.ModuleList()
    self.SA_modules.append(
      PointnetSAModule(
        npoint=4096,
        radius=0.05,
        nsample=64,
        mlp=[256, 128],
        use_xyz=True, 
      )
    )
    self.SA_modules.append(
      PointnetSAModule(
        npoint=2048,
        radius=0.1,
        nsample=64,
        mlp=[128, 64],
        use_xyz=True, 
      )
    )
    self.SA_modules.append(
      PointnetSAModule(
        npoint=1024,
        radius=0.2,
        nsample=64,
        mlp=[64, 32],
        use_xyz=True, 
      )
    )
    


  def forward(self, xyz, features):
    r"""
      Forward pass of the network

      Parameters
      ----------
      pointcloud: Variable(torch.cuda.FloatTensor)
        (B, N, 3 + input_channels) tensor
        Point cloud to run predicts on
        Each point in the point-cloud MUST
        be formated as (x, y, z, features...)
    """
    for module in self.SA_modules:
      xyz, features = module(xyz, features) 
    return features


