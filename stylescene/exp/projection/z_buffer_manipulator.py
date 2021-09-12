# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from pytorch3d.structures import Pointclouds

EPS = 1e-2


def get_splatter(
  name, depth_values, opt=None, size=256, C=64, points_per_pixel=8
):
  if name == "xyblending":
    from projection.z_buffer_layers import RasterizePointsXYsBlending

    return RasterizePointsXYsBlending(
      C,
      learn_feature=opt.learn_default_feature,
      radius=opt.radius,
      size=size,
      points_per_pixel=points_per_pixel,
      opts=opt,
    )
  else:
    raise NotImplementedError()


