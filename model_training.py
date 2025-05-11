
"""Faster training loader – **integrity‑checked** version
========================================================
This file supersedes the legacy pickle‑per‑sample loader.  It
*builds once* a contiguous `.pt` shard per split and re‑uses it on
subsequent runs.  It now mirrors the **exact filename conventions** of
the original Deezer code so paths match out‑of‑the‑box.

Filename conventions we replicate
---------------------------------
* **Training** pickles live in `{master}/{emb_ver}/train/` and are named
  `x_train_{i}.pkl` & `y_train_{i}.pkl`.
* **Validation** pickles live in `{master}/{emb_ver}/validation/` and are
  `x_{i}.pkl` & `y_listened_songs_{i}.pkl`.

Other assumptions
-----------------
* Each *x* example is a 1‑D tensor and must become length
  `config['input_dim']`.
* Each target *y* is 1‑D and must become `config['embeddings_dim']`.
* Shorter vectors are zero‑padded; longer ones truncated.
"""
import os, glob, pickle, time, random
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from options import config           # unchanged
from model   import RegressionTripleHidden  # unchanged
# from model import EmbeddingRegressor128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_length(vec: torch.Tensor, length: int) -> torch.Tensor:
    """Ensure a 1‑D tensor has **exactly** `length` elements."""
    if vec.numel() == length:
        return vec.view(-1)
    if vec.numel() > length:
        return vec[:length].view(-1)
    pad = vec.new_zeros(length - vec.numel())
    return torch.cat([vec.view(-1), pad])


def _glob_pair_paths(split_dir: str, train: bool) -> List[tuple[str,str]]:
    """Return aligned lists of (x_path, y_path) following Deezer naming."""
    if train:
        x_files = sorted(glob.glob(os.path.join(split_dir, "x_train_*.pkl")))
        pairs = []
        for x_fp in x_files:
            idx = os.path.splitext(os.path.basename(x_fp))[0].split("_")[-1]
            y_fp = os.path.join(split_dir, f"y_train_{idx}.pkl")
            pairs.append((x_fp, y_fp))
        return pairs
    # validation
    x_files = sorted(glob.glob(os.path.join(split_dir, "x_*.pkl")))
    pairs = []
    for x_fp in x_files:
        idx = os.path.splitext(os.path.basename(x_fp))[0].split("_")[-1]
        y_fp = os.path.join(split_dir, f"y_listened_songs_{idx}.pkl")
        pairs.append((x_fp, y_fp))
    return pairs


def _build_shard(split_dir: str, emb_ver: str, train: bool) -> Dict[str,torch.Tensor]:
    name = "train" if train else "validation"
    shard_path = os.path.join(split_dir, f"{emb_ver}_{name}_shard.pt")
    print(f"⋯ Building shard {shard_path}")
    if os.path.exists(shard_path):
        print(f"⋯ Loading shard {shard_path}")
        return torch.load(shard_path, map_location="cpu", mmap=True)

    pairs = _glob_pair_paths(split_dir, train)
    print(f"⋯ Packing {len(pairs)} pickle pairs from {split_dir} → {shard_path}")
    xs, ys = [], []
    for x_fp, y_fp in pairs:
        with open(x_fp, "rb") as fx, open(y_fp, "rb") as fy:
            xs.append(_fit_length(torch.tensor(pickle.load(fx)), config["input_dim"]))
            ys.append(_fit_length(torch.tensor(pickle.load(fy)), config["embeddings_dim"]))
    shard = {"x": torch.stack(xs), "y": torch.stack(ys)}
    torch.save(shard, shard_path)
    return shard

# ---------------------------------------------------------------------------
# Training entry
# ---------------------------------------------------------------------------

def training(dataset_path: str, master_path: str, *, embeddings_version: str = "svd", save_model: bool = True, model_filename: str | None = None):
    # hyper‑params
    use_cuda   = config["use_cuda"]
    device     = torch.device(config["device_number"]) if use_cuda else torch.device("cpu")
    n_epochs   = config["nb_epochs"]
    bs         = config["batch_size"]
    lr         = config["learning_rate"]
    reg        = config["reg_param"]
    dropout    = config["drop_out"]
    eval_every = config["eval_every"]

    if model_filename is None:
        model_filename = f"regression_{embeddings_version}"
    model_path = os.path.join(master_path, model_filename + ".pt")

    # dirs
    master_path =  "/home/mickaelz/Recommendation systems/project/deezer"
    print(f"Master path: {master_path}")

    train_dir = os.path.join(master_path, embeddings_version, "train")
    val_dir   = os.path.join(master_path, embeddings_version, "validation")

    train_shard = _build_shard(train_dir, embeddings_version, train=True)
    val_shard   = _build_shard(val_dir,   embeddings_version, train=False)

    train_loader = DataLoader(TensorDataset(train_shard["x"], train_shard["y"]), bs, shuffle=True, num_workers=4, pin_memory=use_cuda)
    val_loader   = DataLoader(TensorDataset(val_shard["x"],   val_shard["y"]),   bs, shuffle=False, num_workers=4, pin_memory=use_cuda)

    model = RegressionTripleHidden(config["input_dim"], config["embeddings_dim"], drop_out=dropout)
    # model = EmbeddingRegressor128().to(device)
    # opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    if use_cuda:
        model = model.to(device)

    criterion = torch.nn.MSELoss()
    # cosine alignment loss
    # criterion = nn.CosineEmbeddingLoss(margin=0.0)
    optim     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    for epoch in range(n_epochs):
        # ---- train ----
        model.train(); t0 = time.time(); losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device),  F.normalize(yb.to(device), dim=1)
            optim.zero_grad()
            pred = model(xb)
            # loss = criterion(pred, yb, torch.ones(xb.size(0),device=device))
            loss = criterion(pred, yb)
            
            # Print a sample prediction and its target
            if epoch == n_epochs-2:
                print("Sample model output:", pred[0].detach().cpu().numpy())
                print("Sample target y:", yb[0].detach().cpu().numpy())
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:02d}  train‑MSE {np.mean(losses):.5f}  [{time.time()-t0:.1f}s]")

       # ---- validate ----
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            model.eval()
            vloss = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    # normalise only when yb is already a float embedding
                    if yb.dtype.is_floating_point:
                        yb = F.normalize(yb, dim=1)
                    else:
                        yb = yb.float()            # cast ID tensor to float so loss works

                    loss = criterion(model(xb),
                                    yb)
                    vloss.append(loss.item())

            print(f"  val‑loss {np.mean(vloss):.5f}")


    if save_model:
        torch.save(model.state_dict(), model_path)
        print(f"Saved model → {model_path}")


# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn
# import time
# import pickle
# import random
# from model import RegressionTripleHidden

# from options import config

# def training(dataset_path, master_path, embeddings_version="svd", eval=True, model_save=True, model_filename=None):

#     use_cuda = config['use_cuda']
#     cuda_number = config['device_number']
#     cuda = torch.device(cuda_number)
#     target_dim = config['embeddings_dim']
#     input_dim = config['input_dim']
#     nb_epochs = config['nb_epochs']
#     learning_rate = config['learning_rate']
#     reg_param = config['reg_param']
#     drop_out = config['drop_out']
#     batch_size = config['batch_size']
#     eval_every = config['eval_every']
#     k_val = config['k_val']

#     if not os.path.exists(master_path + "/" + model_filename + ".pt"):

#         print("--- no model pre-existing for "+embeddings_version+" : training regression model running ---")

#         # Load training dataset.
#         training_set_size = int(len(os.listdir("{}/{}/train".format(master_path, embeddings_version))) / 2)
#         train_xs = []
#         train_ys = []
#         for idx in range(training_set_size):
#             train_xs.append(pickle.load(open("{}/{}/train/x_train_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
#             train_ys.append(pickle.load(open("{}/{}/train/y_train_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
#         total_dataset = list(zip(train_xs, train_ys))
#         del(train_xs, train_ys)
#         print("training set size : "+str(training_set_size))
#         if eval:

#             # Load validation dataset.

#             validation_set_size = int(len(os.listdir("{}/{}/validation".format(master_path, embeddings_version))) / 3)
#             validation_xs = []
#             listened_songs_validation_ys = []
#             for idx in range(validation_set_size):
#                 validation_xs.append(pickle.load(open("{}/{}/validation/x_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
#                 listened_songs_validation_ys.append(pickle.load(open("{}/{}/validation/y_listened_songs_{}.pkl".format(master_path, embeddings_version, idx), "rb")))

#             total_validation_dataset = list(zip(validation_xs, listened_songs_validation_ys))
#             del(validation_xs, listened_songs_validation_ys)

#             # Load song embeddings for evaluation

#             song_embeddings_path = dataset_path + "/song_embeddings.parquet"
#             song_embeddings = pd.read_parquet(song_embeddings_path, engine = 'fastparquet')
#             list_features = ["feature_" + str(i) for i in range(len(song_embeddings["features_" + embeddings_version][0]))]
#             song_embeddings[list_features] = pd.DataFrame(song_embeddings["features_" + embeddings_version].tolist(), index= song_embeddings.index)
#             song_embeddings_values = song_embeddings[list_features].values
#             song_embeddings_values_ = torch.FloatTensor(song_embeddings_values.astype(np.float32))
#             print("validation set size : "+str(validation_set_size))
#         if use_cuda:    
#             regression_model = RegressionTripleHidden(input_dim = input_dim, output_dim = target_dim, drop_out = drop_out).cuda(device = cuda)
#         else:
#             regression_model = RegressionTripleHidden(input_dim = input_dim, output_dim = target_dim, drop_out = drop_out)
#         criterion = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(regression_model.parameters(), lr = learning_rate, weight_decay=reg_param )

#         print("training set size : "+str(training_set_size))
#         print("validation set size : "+str(validation_set_size))
#         print("input dimension : " + str(input_dim))
#         print("regression model : "+ str(regression_model))
#         print("training running")

#         loss_train = []

#         for nb in range(nb_epochs):
#             print("nb epoch : "+str(nb))
#             start_time_epoch = time.time()
#             random.Random(nb).shuffle(total_dataset)
#             a,b = zip(*total_dataset)
#             num_batch = int(training_set_size / batch_size)
#             if use_cuda:
#                 regression_model = regression_model.to(device = cuda)                
#             for i in range(num_batch):
#                 optimizer.zero_grad()
#                 if use_cuda:
#                     batch_features_tensor = torch.stack(a[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
#                     batch_target_tensor = torch.stack(b[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
#                 else:
#                     batch_features_tensor = torch.stack(a[batch_size*i:batch_size*(i+1)])
#                     batch_target_tensor = torch.stack(b[batch_size*i:batch_size*(i+1)])
#                 output_tensor = regression_model(batch_features_tensor)
#                 loss = criterion(output_tensor, batch_target_tensor)
#                 # print a single output tensor vs target tensor
#                 print("output tensor (sample 0): ", output_tensor[0])
#                 print("target tensor (sample 0): ", batch_target_tensor[0])
#                 loss.backward()
#                 optimizer.step()
#                 loss_train.append(loss.item())
#             print('epoch ' + str(nb) +  " training loss : "+ str(sum(loss_train)/float(len(loss_train))))
#             print("--- seconds ---" + str(time.time() - start_time_epoch))

#             if nb != 0 and (nb % eval_every == 0 or nb == nb_epochs - 1):
#                 print('testing model')
#                 start_time_eval = time.time()
#                 reg = regression_model.eval()
#                 if use_cuda:
#                     reg = reg.to(device=cuda)
#                 validation_set_size = len(total_validation_dataset)
#                 a,b = zip(*total_validation_dataset)
#                 num_batch_validation = int(validation_set_size / batch_size)
#                 current_precisions = []
#                 with torch.set_grad_enabled(False):
#                     for i in range(num_batch_validation):
#                         if use_cuda:
#                             batch_features_tensor_validation = torch.stack(a[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
#                         else:
#                             batch_features_tensor_validation = torch.stack(a[batch_size*i:batch_size*(i+1)])
#                         predictions_validation = reg(batch_features_tensor_validation)
#                         groundtruth_validation = list(b[batch_size*i:batch_size*(i+1)])
#                         predictions_songs_validation = torch.mm(predictions_validation.cpu(), song_embeddings_values_.transpose(0, 1))
#                         recommendations_validation = (predictions_songs_validation.topk(k= k_val, dim = 1)[1]).tolist()
#                         precisions = list(map(lambda x, y: len(set(x) & set(y))/float(min(len(x), k_val)), groundtruth_validation, recommendations_validation))
#                         current_precisions.extend(precisions)
#                 print('epoch ' + str(nb) +  " precision test : "+ str(sum(current_precisions) / float(len(current_precisions))) )
#                 print("--- %s seconds ---" + str(time.time() - start_time_eval))
#         print("--- training finished ---")

#         if model_save:
#             print("--- saving model ---")
#             torch.save(regression_model.state_dict(), master_path + "/" + model_filename + ".pt")
#             print(regression_model)
#             print("--- model saved ---")

#     else:
#         print("--- there is already a model pre-existing for "+embeddings_version +" : no need to run training again ---")


#  import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn
# import time
# import pickle
# import random
# from model import RegressionTripleHidden
# from options import config


# import pickle
# import os
# import time
# import random
# import glob
# import torch
# import torch.nn
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader, TensorDataset
# from options import config  # unchanged
# from model import RegressionTripleHidden  # unchanged

# ############################################################
# # 1. Helpers to build / load a single shard per split       #
# ############################################################

# def _build_shard(split_dir: str, embeddings_version: str, force: bool = False):
#     """Merge legacy pickle files (<split>/x_*.pkl, y_*.pkl) into one .pt shard.
#     Creates <split>_shard.pt in the same directory.  Run once and re‑use."""

#     shard_path = os.path.join(split_dir, f"{embeddings_version}_{os.path.basename(split_dir)}_shard.pt")
#     if os.path.exists(shard_path) and not force:
#         return shard_path  # already done

#     xs, ys = [], []
#     x_files = sorted(glob.glob(os.path.join(split_dir, "x_*.pkl")))
#     y_files = sorted(glob.glob(os.path.join(split_dir, "y*_*.pkl")))
#     assert len(x_files) == len(y_files), "Mismatch X/Y files"

#     print(f"⋯ Packing {len(x_files)} pickle pairs from {split_dir} → {shard_path}")
#     for xf, yf in zip(x_files, y_files):
#         with open(xf, "rb") as f:
#             xs.append(torch.tensor(pickle.load(f)))
#         with open(yf, "rb") as f:
#             ys.append(torch.tensor(pickle.load(f)))

#     shard = {"x": torch.stack(xs), "y": torch.stack(ys)}
#     torch.save(shard, shard_path)
#     return shard_path


# def _load_shard(shard_path: str):
#     """Memory‑map the shard (no RAM copy) and build TensorDataset."""
#     shard = torch.load(shard_path, map_location="cpu", mmap=True)
#     return TensorDataset(shard["x"], shard["y"])

# ############################################################
# # 2. Main training routine                                 #
# ############################################################

# def training(dataset_path: str, master_path: str, embeddings_version: str = "svd", *,
#              eval_split: bool = True, save_model: bool = True, model_filename: str | None = None):

#     cfg = config  # shorthand
#     device = torch.device(cfg['device_number'] if cfg['use_cuda'] else "cpu")

#     model_filename = model_filename or f"reg_{embeddings_version}"
#     model_path = os.path.join(master_path, model_filename + ".pt")

#     if os.path.exists(model_path):
#         print(f"✔ Model already exists at {model_path}; skip training.")
#         return

#     # ---------- build / load shards ----------
#     train_dir = os.path.join(master_path, embeddings_version, "train")
#     val_dir   = os.path.join(master_path, embeddings_version, "validation")

#     train_shard = _build_shard(train_dir, embeddings_version)
#     train_ds    = _load_shard(train_shard)

#     if eval_split:
#         val_shard = _build_shard(val_dir, embeddings_version)
#         val_ds    = _load_shard(val_shard)

#         # song embeddings for evaluation
#         song_emb_df = pd.read_parquet(os.path.join(dataset_path, "song_embeddings.parquet"), engine="fastparquet")
#         feat_cols = [f"feature_{i}" for i in range(len(song_emb_df[f"features_{embeddings_version}"][0]))]
#         song_emb_df[feat_cols] = pd.DataFrame(song_emb_df[f"features_{embeddings_version}"].tolist(), index=song_emb_df.index)
#         song_matrix = torch.as_tensor(song_emb_df[feat_cols].values, dtype=torch.float32)

#     # ---------- dataloaders ----------
#     train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
#                               num_workers=4, pin_memory=cfg['use_cuda'])

#     if eval_split:
#         val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
#                                 num_workers=4, pin_memory=False)

#     # ---------- model / optimiser ----------
#     model = RegressionTripleHidden(cfg['input_dim'], cfg['embeddings_dim'], cfg['drop_out']).to(device)
#     criterion = torch.nn.MSELoss()
#     optimiser = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['reg_param'])

#     print(model)
#     print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds) if eval_split else 0}")

#     # ---------- training loop ----------
#     for epoch in range(cfg['nb_epochs']):
#         model.train()
#         running_loss = 0.0
#         start_ep = time.time()

#         for xb, yb in train_loader:
#             xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
#             optimiser.zero_grad()
#             loss = criterion(model(xb), yb)
#             loss.backward()
#             optimiser.step()
#             running_loss += loss.item() * xb.size(0)

#         print(f"Epoch {epoch:02d}  train‑loss: {running_loss/len(train_ds):.5f}  [{time.time()-start_ep:.1f}s]")

#         # --------- evaluation ---------
#         if eval_split and (epoch % cfg['eval_every'] == 0 or epoch == cfg['nb_epochs'] - 1):
#             model.eval()
#             correct = 0.0
#             total   = 0
#             with torch.no_grad():
#                 for xb, gt_listened in val_loader:
#                     preds = model(xb.to(device, non_blocking=True)).cpu()
#                     # cosine scores against all song embeddings
#                     scores = torch.mm(preds, song_matrix.T)
#                     topk   = scores.topk(cfg['k_val'], dim=1).indices
#                     batch_precision = [len(set(topk[i].tolist()) & set(gt_listened[i]))/cfg['k_val'] for i in range(len(gt_listened))]
#                     correct += sum(batch_precision)
#                     total   += len(batch_precision)
#             print(f"          precision@{cfg['k_val']}: {correct/total:.4f}")

#     # ---------- save ----------
#     if save_model:
#         torch.save(model.state_dict(), model_path)
#         print(f"Model saved → {model_path}")


