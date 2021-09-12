import numpy as np
import PIL
import logging
import sys
import glob
sys.path.append("../")
import co
import ext

import cv2
import imageio
from pathlib import Path
import config
import os.path
import torch
import random


def load(p, height=None, width=None):
  if p.suffix == ".npy":
    return np.load(p)
  elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
    im = PIL.Image.open(p)
    im = np.array(im)
    if (
      height is not None
      and width is not None
      and (im.shape[0] != height or im.shape[1] != width)
    ):
      raise Exception("invalid size of image")
    im = (im.astype(np.float32) / 255) * 2 - 1
    im = im.transpose(2, 0, 1)
    return im
  else:
    raise Exception("invalid suffix")


def load_resize_test(p, height=None, width=None):
  if p.suffix == ".npy":
    return np.load(p)
  elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
    im = PIL.Image.open(p).convert('RGB')
    im = im.resize((224,224)) 
    im = np.array(im)
    if (
      height is not None
      and width is not None
      and (im.shape[0] != height or im.shape[1] != width)
    ):
      raise Exception("invalid size of image")
    im = (im.astype(np.float32) / 255)
    if len(im.shape) == 2: 
      im = np.stack((im,im,im), 2)
    im = im.transpose(2, 0, 1)
    return im
  else:
    raise Exception("invalid suffix")



def load_resize_train(p, height=None, width=None):
  if p.suffix == ".npy":
    return np.load(p)
  elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
    try:
      im = PIL.Image.open(p).convert('RGB')
      im = im.resize((496,288)) 
      im = np.array(im)
    except:
      im = np.zeros((496,288,3))

    if (
      height is not None
      and width is not None
      and (im.shape[0] != height or im.shape[1] != width)
    ):
      raise Exception("invalid size of image")
    im = (im.astype(np.float32) / 255)
    if len(im.shape) == 2: 
      im = np.stack((im,im,im), 2)
    im = im.transpose(2, 0, 1)
    return im
  else:
    raise Exception("invalid suffix")


class Dataset(co.mytorch.BaseDataset):
  def __init__(
    self,
    *,
    name,
    tgt_im_paths,
    tgt_dm_paths,
    tgt_Ks,
    tgt_Rs,
    tgt_ts,
    tgt_counts,
    src_im_paths,
    src_dm_paths,
    src_Ks,
    src_Rs,
    src_ts,
    im_size=None,
    pad_width=None,
    patch=None,
    n_nbs=5,
    nbs_mode="sample",
    bwd_depth_thresh=0.1,
    invalid_depth_to_inf=True,
    **kwargs,
  ):
    super().__init__(name=name, **kwargs)

    self.tgt_im_paths = tgt_im_paths
    self.tgt_dm_paths = tgt_dm_paths
    self.tgt_Ks = tgt_Ks
    self.tgt_Rs = tgt_Rs
    self.tgt_ts = tgt_ts
    self.tgt_counts = tgt_counts

    self.src_im_paths = src_im_paths
    self.src_dm_paths = src_dm_paths
    self.src_Ks = src_Ks
    self.src_Rs = src_Rs
    self.src_ts = src_ts

    self.im_size = im_size
    self.pad_width = pad_width
    self.patch = patch
    self.n_nbs = n_nbs
    self.nbs_mode = nbs_mode
    self.bwd_depth_thresh = bwd_depth_thresh
    self.invalid_depth_to_inf = invalid_depth_to_inf

    tmp = np.load(tgt_dm_paths[0])
    self.height, self.width = tmp.shape
    del tmp

    n_tgt_im_paths = len(tgt_im_paths) if tgt_im_paths else 0
    shape_tgt_im = (
      self.load_pad(tgt_im_paths[0]).shape if tgt_im_paths else None
    )
    logging.info(
      f"  #tgt_im_paths={n_tgt_im_paths}, #tgt_counts={tgt_counts.shape}, tgt_im={shape_tgt_im}, tgt_dm={self.load_pad(tgt_dm_paths[0]).shape}"
    )

    path = str(tgt_dm_paths[0]).replace("dm_00000000.npy","").replace("long","pw_0.25")
    points = np.load(os.path.join(path,"points.npy"))
    emb  = np.load(os.path.join(path, "r31.npy"))
    H1, W1 = points.shape[1:3]
    H2, W2 = emb.shape[2:4]
    y = [int(round(y)) for y in np.array(list(range(H2))) / (H2 - 1) * (H1 - 1)]
    x = [int(round(x)) for x in np.array(list(range(W2))) / (W2 - 1) * (W1 - 1)]
    points = np.take(points, y, axis=1)
    points = np.take(points, x, axis=2)
    emb = emb.transpose(1,0,2,3)
    self.points = points.reshape(-1, 3)
    self.feats = emb.reshape(256, -1)
    del points
    del emb
    
    if config.Train:
      skip = int(self.points.shape[0] // 100000)
      self.points = self.points[::skip, :].astype(np.float32)
      self.feats = self.feats[:, ::skip].astype(np.float32)
      self.style = sorted([os.path.join(config.styleroot,"train/%s" % s.strip()) for s in open("list.txt").readlines()])
    else:
      self.points = self.points.astype(np.float32)
      self.feats = self.feats.astype(np.float32)
      ind = [int(ii / 1800000 * self.points.shape[0]) for ii in list(range(1800000))]
      self.points = self.points[ind, :]
      self.feats = self.feats[:, ind]
      self.style = [os.path.join(config.styleroot,"style120/%d.jpg") % i for i in config.Test_style]
    print("point size: ", self.points.shape)

  def pad(self, im):
    if self.im_size is not None:
      shape = [s for s in im.shape]
      shape[-2] = self.im_size[0]
      shape[-1] = self.im_size[1]
      im_p = np.zeros(shape, dtype=im.dtype)
      sh = min(im_p.shape[-2], im.shape[-2])
      sw = min(im_p.shape[-1], im.shape[-1])
      im_p[..., :sh, :sw] = im[..., :sh, :sw]
      im = im_p
    if self.pad_width is not None:
      h, w = im.shape[-2:]
      mh = h % self.pad_width
      ph = 0 if mh == 0 else self.pad_width - mh
      mw = w % self.pad_width
      pw = 0 if mw == 0 else self.pad_width - mw
      shape = [s for s in im.shape]
      shape[-2] += ph
      shape[-1] += pw
      im_p = np.zeros(shape, dtype=im.dtype)
      im_p[..., :h, :w] = im
      im = im_p
    return im

  def load_pad(self, p):
    im = load(p)
    return self.pad(im)

  def base_len(self):
    return len(self.tgt_dm_paths)

  def base_getitem(self, idx, rng):
    ret = {}
    if config.Train:
      sidx = np.random.randint(0,len(self.style))
      ret["style"] = load_resize_train(Path(self.style[sidx]))
    else:
      ret["style"] = np.array([load_resize_test(Path(self.style[sidx])) for sidx in range(len(self.style))])
  
    m = max(np.abs(np.max(self.points)), np.abs(np.min(self.points)))
    nbs = [idx]
    ret["points"] = self.points
    ret["points2"] = self.points / m
    ret["feats"] = self.feats
  
    tgt_dm = load(self.tgt_dm_paths[idx])
    tgt_dm = self.pad(tgt_dm)
    tgt_K = self.tgt_Ks[idx]
    tgt_R = self.tgt_Rs[idx]
    tgt_t = self.tgt_ts[idx]
    src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
    src_dms = self.pad(src_dms)
    src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
    src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
    src_ts = np.array([self.src_ts[ii] for ii in nbs])
    patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)
    
    if ret["points"].shape[0] < 1000000:
      sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
      tgt_dm,
      tgt_K,
      tgt_R,
      tgt_t,
      src_dms,
      src_Ks,
      src_Rs,
      src_ts,
      patch,
      self.bwd_depth_thresh,
      self.invalid_depth_to_inf,
      ret["points"]
      )
    else:
      sampling_maps = []
      for pt in range(0,ret["points"].shape[0], 1000000):
        sampling_map, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
        tgt_dm,
        tgt_K,
        tgt_R,
        tgt_t,
        src_dms,
        src_Ks,
        src_Rs,
        src_ts,
        patch,
        self.bwd_depth_thresh,
        self.invalid_depth_to_inf,
        ret["points"][pt:pt+1000000,:]
        )
        sampling_maps.append(sampling_map)
      sampling_maps = np.concatenate(sampling_maps, 0)
    
    ret["points"] = sampling_maps
    tgt_height = min(tgt_dm.shape[0], patch[1]) - patch[0]
    tgt_width = min(tgt_dm.shape[1], patch[3]) - patch[2]
    ret["src"] = np.zeros((3, tgt_height, tgt_width), dtype=np.float32)
    ret["tgt"] = load(self.tgt_im_paths[idx])
  
    return ret

