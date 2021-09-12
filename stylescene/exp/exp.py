import torch
import numpy as np
import sys
import logging
from pathlib import Path
import PIL
import dataset
import modules

sys.path.append("../")
import co
import ext
import config
import imageio

def get_np(est):
  est = torch.clamp(est, -1, 1)
  est = 255 * (est + 1) / 2
  est = est.type(torch.uint8)
  est = torch.squeeze(est).permute(1,2,0).cpu().numpy()
  return est


class Worker(co.mytorch.Worker):
  def __init__(
    self,
    train_dsets,
    eval_dsets="",
    train_n_nbs=1,
    train_nbs_mode="argmax",
    train_scale=1,
    train_patch=192,
    eval_n_nbs=1,
    eval_scale=-1,
    n_train_iters=100000,
    num_workers=1,
    **kwargs,
  ):
    super().__init__(
      n_train_iters=n_train_iters,
      num_workers=num_workers,
      train_device=config.train_device,
      eval_device=config.eval_device,
      **kwargs,
    )

    self.train_dsets = train_dsets
    self.eval_dsets = eval_dsets
    self.train_n_nbs = train_n_nbs
    self.train_nbs_mode = train_nbs_mode
    self.train_scale = train_scale
    self.train_patch = train_patch
    self.eval_n_nbs = eval_n_nbs
    self.eval_scale = train_scale if eval_scale <= 0 else eval_scale
    self.bwd_depth_thresh = 0.01
    self.invalid_depth_to_inf = True

    self.train_loss = modules.VGGPerceptualLoss()
    if config.lpips_root:
      self.eval_loss = modules.LPIPS()
    else:
      self.eval_loss = self.train_loss

  def get_pw_dataset(
    self,
    *,
    name,
    ibr_dir,
    im_size,
    patch,
    pad_width,
    n_nbs,
    nbs_mode,
    train,
    tgt_ind=None,
    n_max_sources=-1,
  ):


    logging.info(f"  create dataset for {name}")
  
    if not config.Train:
      ibr_dir = Path(str(ibr_dir).replace("pw_0.25","long"))
      im_paths = sorted(ibr_dir.glob(f"dm_*.png"))
      im_paths += sorted(ibr_dir.glob(f"dm_*.jpg"))
      im_paths += sorted(ibr_dir.glob(f"dm_*.jpeg"))
    else:
      im_paths = sorted(ibr_dir.glob(f"im_*.png"))
      im_paths += sorted(ibr_dir.glob(f"im_*.jpg"))
      im_paths += sorted(ibr_dir.glob(f"im_*.jpeg"))
      
      
    dm_paths = sorted(ibr_dir.glob("dm_*.npy"))
    count_paths = sorted(ibr_dir.glob("count_*.npy"))
    counts = []
    for count_path in count_paths:
      counts.append(np.load(count_path))
    counts = np.array(counts)
    Ks = np.load(ibr_dir / "Ks.npy")
    Rs = np.load(ibr_dir / "Rs.npy")
    ts = np.load(ibr_dir / "ts.npy")

    tgt_ind = np.arange(len(im_paths))
    src_ind = np.arange(len(im_paths))

    if not config.Train:
      counts = np.zeros((len(im_paths),len(im_paths)))
    else:
      counts = counts[tgt_ind]
      counts = counts[:, src_ind]

    dset = dataset.Dataset(
      name=name,
      tgt_im_paths=[im_paths[idx] for idx in tgt_ind],
      tgt_dm_paths=[dm_paths[idx] for idx in tgt_ind],
      tgt_Ks=Ks[tgt_ind],
      tgt_Rs=Rs[tgt_ind],
      tgt_ts=ts[tgt_ind],
      tgt_counts=counts,
      src_im_paths=[im_paths[idx] for idx in src_ind],
      src_dm_paths=[dm_paths[idx] for idx in src_ind],
      src_Ks=Ks[src_ind],
      src_Rs=Rs[src_ind],
      src_ts=ts[src_ind],
      im_size=im_size,
      pad_width=pad_width,
      patch=patch,
      n_nbs=n_nbs,
      nbs_mode=nbs_mode,
      bwd_depth_thresh=self.bwd_depth_thresh,
      invalid_depth_to_inf=self.invalid_depth_to_inf,
      train=train,
    )
    return dset


  def get_track_dataset(
    self,
    name,
    src_ibr_dir,
    tgt_ibr_dir,
    n_nbs,
    im_size=None,
    pad_width=16,
    patch=None,
    nbs_mode="argmax",
    train=False,
  ):
    logging.info(f"  create dataset for {name}")

    src_im_paths = sorted(src_ibr_dir.glob(f"im_*.png"))
    src_im_paths += sorted(src_ibr_dir.glob(f"im_*.jpg"))
    src_im_paths += sorted(src_ibr_dir.glob(f"im_*.jpeg"))
    src_dm_paths = sorted(src_ibr_dir.glob("dm_*.npy"))
    src_Ks = np.load(src_ibr_dir / "Ks.npy")
    src_Rs = np.load(src_ibr_dir / "Rs.npy")
    src_ts = np.load(src_ibr_dir / "ts.npy")

    tgt_im_paths = sorted(tgt_ibr_dir.glob(f"im_*.png"))
    tgt_im_paths += sorted(tgt_ibr_dir.glob(f"im_*.jpg"))
    tgt_im_paths += sorted(tgt_ibr_dir.glob(f"im_*.jpeg"))
    if len(tgt_im_paths) == 0:
      tgt_im_paths = None
    tgt_dm_paths = sorted(tgt_ibr_dir.glob("dm_*.npy"))
    count_paths = sorted(tgt_ibr_dir.glob("count_*.npy"))
    counts = []
    for count_path in count_paths:
      counts.append(np.load(count_path))
    counts = np.array(counts)
    tgt_Ks = np.load(tgt_ibr_dir / "Ks.npy")
    tgt_Rs = np.load(tgt_ibr_dir / "Rs.npy")
    tgt_ts = np.load(tgt_ibr_dir / "ts.npy")

    dset = dataset.Dataset(
      name=name,
      tgt_im_paths=tgt_im_paths,
      tgt_dm_paths=tgt_dm_paths,
      tgt_Ks=tgt_Ks,
      tgt_Rs=tgt_Rs,
      tgt_ts=tgt_ts,
      tgt_counts=counts,
      src_im_paths=src_im_paths,
      src_dm_paths=src_dm_paths,
      src_Ks=src_Ks,
      src_Rs=src_Rs,
      src_ts=src_ts,
      im_size=im_size,
      pad_width=pad_width,
      patch=patch,
      n_nbs=n_nbs,
      nbs_mode=nbs_mode,
      bwd_depth_thresh=self.bwd_depth_thresh,
      invalid_depth_to_inf=self.invalid_depth_to_inf,
      train=train,
    )
    return dset

  def get_train_set_tat(self, dset):
    dense_dir = config.tat_root / dset / "dense"
    ibr_dir = dense_dir / f"ibr3d_pw_{self.train_scale:.2f}"
    dset = self.get_pw_dataset(
      name=f'tat_{dset.replace("/", "_")}',
      ibr_dir=ibr_dir,
      im_size=None,
      pad_width=16,
      patch=(self.train_patch, self.train_patch),
      n_nbs=self.train_n_nbs,
      nbs_mode=self.train_nbs_mode,
      train=True,
    )
    return dset

  def get_train_set(self):
    logging.info("Create train datasets")
    dsets = co.mytorch.MultiDataset(name="train")
    if "tat" in self.train_dsets:
      for dset in config.tat_train_sets:
        dsets.append(self.get_train_set_tat(dset))
    return dsets

  def get_eval_set_tat(self, dset, mode):
    dense_dir = config.tat_root / dset / "dense"
    ibr_dir = dense_dir / f"ibr3d_pw_{self.eval_scale:.2f}"
    if mode == "all":
      tgt_ind = None
    elif mode == "subseq":
      tgt_ind = config.tat_eval_tracks[dset]
    else:
      raise Exception("invalid mode for get_eval_set_tat")
    dset = self.get_pw_dataset(
      name=f'tat_{mode}_{dset.replace("/", "_")}',
      ibr_dir=ibr_dir,
      im_size=None,
      pad_width=16,
      patch=None,
      n_nbs=self.eval_n_nbs,
      nbs_mode="argmax",
      tgt_ind=tgt_ind,
      train=False,
    )
    return dset
    


  def get_eval_sets(self):
    logging.info("Create eval datasets")
    eval_sets = []
    if "tat" in self.eval_dsets:
      for dset in config.tat_eval_sets:
        dset = self.get_eval_set_tat(dset, "all")
        eval_sets.append(dset)
    for dset in self.eval_dsets:
      if dset.startswith("tat-scene-"):
        dset = dset[len("tat-scene-") :]
        dset = self.get_eval_set_tat(dset, "all")
        eval_sets.append(dset)
    if "tat-subseq" in self.eval_dsets:
      for dset in config.tat_eval_sets:
        dset = self.get_eval_set_tat(dset, "subseq")
        eval_sets.append(dset)
    for dset in eval_sets:
      dset.logging_rate = 1
      dset.vis_ind = np.arange(len(dset))
    return eval_sets


  def copy_data(self, data, device, train):
    self.data = {}
    for k, v in data.items():
      self.data[k] = v.to(device).requires_grad_(requires_grad=False)

  def net_forward(self, net, train, iter):
    return net(**self.data)

  def loss_forward(self, output, train, iter):
    errs = {}
    tgt = self.data["tgt"]
    est = output["out"]
    est = est[..., : tgt.shape[-2], : tgt.shape[-1]]

    if train:
      style = output["style"]
      for lidx, loss in enumerate(self.train_loss(est, tgt, style)):
        errs[f"rgb{lidx}"] = loss  
        if iter % 100 == 0:      
          I1 = get_np(est)
          I2 = get_np(tgt)
          I = np.concatenate((I1,I2), 0)
          imageio.imsave("log/out%06d.png" % iter, I)
    
    else:
      est = torch.clamp(est, -1, 1)
      est = 255 * (est + 1) / 2
      est = est.type(torch.uint8)
      est = est.type(torch.float32)
      est = (est / 255 * 2) - 1

    output["out"] = est      
    return errs


  def callback_eval_start(self, **kwargs):
    self.metric = None

  def im_to2np(self, im):
    im = im.detach().to("cpu").numpy()
    im = (np.clip(im, -1, 1) + 1) / 2
    
    if len(im.shape)== 4:
      im = im.transpose(0, 2, 3, 1)
    else:
      im = im.transpose(0, 1, 3, 4, 2)
    
    return im


  def callback_eval_add(self, **kwargs):
    output = kwargs["output"]
    batch_idx = kwargs["batch_idx"]
    iter = kwargs["iter"]
    eval_set = kwargs["eval_set"]
    eval_set_name = eval_set.name.replace("/", "_")
    eval_set_name = f"{eval_set_name}_{self.eval_scale}"
    

    # write debug images
    out_dir = self.exp_out_root / f"{eval_set_name}_n{self.eval_n_nbs}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not config.Train:
      ta = self.im_to2np(self.data["tgt"])
      es = self.im_to2np(output["out"])

      for b in range(ta.shape[0]):
        bidx = batch_idx * ta.shape[0] + b
        out_im = (255 * es[b]).astype(np.uint8)
        for c in range(out_im.shape[0]):
          out = PIL.Image.fromarray(out_im[c])
          out.save(out_dir / f"{c:02d}_s{bidx:04d}_es.jpg")



  def callback_eval_stop(self, **kwargs):
    eval_set = kwargs["eval_set"]
    iter = kwargs["iter"]
    mean_loss = kwargs["mean_loss"]
    eval_set_name = eval_set.name.replace("/", "_")
    eval_set_name = f"{eval_set_name}_{self.eval_scale}"
    method = self.experiment_name + f"_n{self.eval_n_nbs}"


if __name__ == "__main__":
  parser = co.mytorch.get_parser()
  parser.add_argument("--net", type=str, required=True)
  parser.add_argument("--train-dsets", nargs="+", type=str, default=["tat"])
  parser.add_argument(
    "--eval-dsets", nargs="+", type=str, default=["tat", "tat-subseq"]
  )
  parser.add_argument("--train-n-nbs", type=int, default=5)
  parser.add_argument("--train-scale", type=float, default=0.25)
  parser.add_argument("--train-patch", type=int, default=192)
  parser.add_argument("--eval-n-nbs", type=int, default=5)
  parser.add_argument("--eval-scale", type=float, default=-1)
  parser.add_argument("--log-debug", type=str, nargs="*", default=[])
  args = parser.parse_args()

  experiment_name = f"{'+'.join(args.train_dsets)}_nbs{args.train_n_nbs}_s{args.train_scale}_p{args.train_patch}_{args.net}"

  worker = Worker(
    experiments_root=args.experiments_root,
    experiment_name=experiment_name,
    train_dsets=args.train_dsets,
    eval_dsets=args.eval_dsets,
    train_n_nbs=args.train_n_nbs,
    train_scale=args.train_scale,
    train_patch=args.train_patch,
    eval_n_nbs=args.eval_n_nbs,
    eval_scale=args.eval_scale,
  )
  worker.log_debug = args.log_debug
  worker.save_frequency = co.mytorch.Frequency(hours=1)
  worker.eval_frequency = co.mytorch.Frequency(hours=1)
  worker.train_batch_size = 1
  worker.eval_batch_size = 1
  worker.train_batch_acc_steps = 1

  worker_objects = co.mytorch.WorkerObjects(
    optim_f=lambda net: torch.optim.Adam(net.matrix.parameters(), lr=1e-4)
  )

  if args.net == "fixed_vgg16unet3_unet4.64.3":
    worker_objects.net_f = lambda: modules.get_fixed_net(
      enc_net="vgg16unet3", dec_net="unet4.64.3", n_views=4
    )
    worker.train_n_nbs = 4
    worker.eval_n_nbs = 4
  else:
    raise Exception("invalid net in exp.py")

  worker.do(args, worker_objects)
