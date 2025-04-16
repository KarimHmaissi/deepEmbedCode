# TEsting  – Card metric‑learning pipeline
# ----------------------------------------------------------------------
import os, random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# pytorch‑metric‑learning ----------------------------------------------
from pytorch_metric_learning.losses   import TripletMarginLoss
from pytorch_metric_learning.miners   import BatchHardMiner
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
# ----------------------------------------------------------------------

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL.Image import Resampling

torch.set_float32_matmul_precision("high")           # 4090 speed hint
# ======================================================================
# ── Utility ────────────────────────────────────────────────────────────
def pad_to_square(img):
    return ImageOps.pad(img, (224, 224),
                        method=Resampling.BICUBIC, color=(0, 0, 0))

def make_contiguous(x):            # avoids non‑contiguous warning in AMP
    return x.contiguous()

class RandomGaussianNoise:
    def __init__(self, mean=0., std=0.02, p=0.3):
        self.mean, self.std, self.p = mean, std, p
    def __call__(self, t):
        if random.random() < self.p:
            t += torch.randn_like(t)*self.std + self.mean
            t.clamp_(0, 1)
        return t

# 2‑crop wrapper used **only** at validation time
class TwoCropTransform:
    """Return two independently augmented views of the same image."""
    def __init__(self, base_transform):
        self.base = base_transform
    def __call__(self, x):
        return self.base(x), self.base(x)
# ----------------------------------------------------------------------
# One set of heavy augs, reused by train & val
base_aug = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomAffine(degrees=8, translate=(0.05,0.05),
                            scale=(0.9,1.1), shear=3),
    transforms.ColorJitter(0.2,0.2,0.1,0.01),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.Lambda(pad_to_square),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02,0.2)),
    transforms.RandomApply([transforms.GaussianBlur(3,(0.3,2.0))], p=0.3),
    RandomGaussianNoise(std=0.05, p=0.3),
    transforms.Lambda(make_contiguous)
])
# ======================================================================
# ── Dataset ────────────────────────────────────────────────────────────
class CardDataset(Dataset):
    def __init__(self, csv_path, root_dir,
                 label2idx=None, validation=False):
        self.data = pd.read_csv(csv_path, header=None,
                                names=["filename", "id_str"])
        self.root_dir = root_dir
        if label2idx is None:
            ids = self.data["id_str"].unique()
            self.label2idx = {i: j for j, i in enumerate(ids)}
        else:
            self.label2idx = label2idx

        self.transform = (TwoCropTransform(base_aug) if validation
                          else base_aug)
        self.validation = validation

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        fname = os.path.join(self.root_dir, row["filename"])
        img   = Image.open(fname).convert("RGB")
        label = self.label2idx[row["id_str"]]

        if self.validation:                # → two augmented views
            img_q, img_k = self.transform(img)
            return img_q, img_k, label
        else:                              # training view
            return self.transform(img), label
# ======================================================================
# ── Model ──────────────────────────────────────────────────────────────
class EmbeddingModel(pl.LightningModule):
    def __init__(self, embedding_dim=512, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.fc      = nn.Linear(2048, embedding_dim)

        self.miner     = BatchHardMiner()
        self.loss_func = TripletMarginLoss(margin=0.2)
        self.lr        = lr

        # validation buffers
        self.val_q, self.val_k, self.val_lab = [], [], []

    def forward(self, x):
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        return F.normalize(self.fc(x), p=2, dim=1)

    # ── train ──────────────────────────────────────────────────────────
    def training_step(self, batch, _):
        imgs, labels = batch
        e = self(imgs)
        loss = self.loss_func(e, labels, self.miner(e, labels))
        self.log("train_loss", loss, prog_bar=True, batch_size=len(imgs))
        return loss

    # ── val ────────────────────────────────────────────────────────────
    def validation_step(self, batch, _):
        img_q, img_k, labels = batch
        e_q = self(img_q)
        e_k = self(img_k)
        # Triplet loss requires one tensor → concat
        e   = torch.cat([e_q, e_k], 0)
        ll  = torch.cat([labels, labels], 0)
        loss = self.loss_func(e, ll, self.miner(e, ll))
        self.log("val_loss", loss, prog_bar=True, batch_size=len(labels))

        self.val_q.append(e_q.cpu())
        self.val_k.append(e_k.cpu())
        self.val_lab.append(labels.cpu())
        return loss

    def on_validation_epoch_end(self):
        q  = torch.cat(self.val_q).numpy()
        k  = torch.cat(self.val_k).numpy()
        y  = torch.cat(self.val_lab).numpy()

        acc = AccuracyCalculator(
            include=["precision_at_1","mean_average_precision",
                     "mean_average_precision_at_r","r_precision"], k=5)
        m = acc.get_accuracy(q, k, y, y)

        self.log("val/precision@1", m["precision_at_1"],       prog_bar=True)
        self.log("val/mAP",         m["mean_average_precision"],prog_bar=True)
        self.log("val/mAP@r",       m["mean_average_precision_at_r"],
                 prog_bar=True)
        self.log("val/r_precision", m["r_precision"],          prog_bar=True)

        self.val_q.clear(); self.val_k.clear(); self.val_lab.clear()

    # ── optim ──────────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.lr/50)
        return [opt], [sch]
# ======================================================================
# ── Inference dataset (unchanged) ─────────────────────────────────────
class CardInferenceDataset(Dataset):
    def __init__(self, csv_path, root_dir):
        self.data = pd.read_csv(csv_path, header=None,
                                names=["filename", "id_str"])
        self.root_dir = root_dir
        self.t = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        r = self.data.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, r["filename"]))\
                  .convert("RGB")
        return self.t(img), r["id_str"], idx
# ======================================================================
# ── MAIN ───────────────────────────────────────────────────────────────
def main():
    base = "/workspace/deepEmbed"
    CSV_TRAIN = f"{base}/dataset/dataset.csv"
    CSV_VAL   = f"{base}/dataset/dataset_backup.csv"
    ROOT_DIR  = f"{base}/dataset"
    CKPT_DIR  = f"{base}/checkpoints"
    EMB_OUT   = f"{base}/all_embeddings.csv"

    BATCH = 440
    EPOCH = 200
    EMB   = 512
    LR    = 1e-4

    # shared label map
    ids = pd.read_csv(CSV_TRAIN, header=None, names=["f","id"])["id"].unique()
    lbl = {id_: i for i, id_ in enumerate(ids)}

    train_ds = CardDataset(CSV_TRAIN, ROOT_DIR, label2idx=lbl, validation=False)
    val_ds   = CardDataset(CSV_VAL,  ROOT_DIR, label2idx=lbl, validation=True)

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=15, persistent_workers=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=True,
                          num_workers=10, persistent_workers=True)

    model = EmbeddingModel(embedding_dim=EMB, lr=LR)

    ckpt = ModelCheckpoint(
        monitor="val/precision@1", mode="max",
        filename="best-checkpoint", save_top_k=1, save_last=True,
        dirpath=CKPT_DIR)
    logger = TensorBoardLogger(CKPT_DIR, "metric_learning_logs")

    tr = pl.Trainer(max_epochs=EPOCH, accelerator="auto", precision="16-mixed",
                    callbacks=[ckpt], logger=logger, default_root_dir=CKPT_DIR)

    resume = os.path.join(CKPT_DIR, "last-v1.ckpt")
    tr.fit(model, train_ld, val_ld,
           ckpt_path=resume if os.path.exists(resume) else None)

    # ── export embeddings on unique set ────────────────────────────────
    best = EmbeddingModel.load_from_checkpoint(ckpt.best_model_path)
    best.eval().freeze()

    inf_ld = DataLoader(CardInferenceDataset(CSV_VAL, ROOT_DIR),
                        batch_size=32, shuffle=False, num_workers=4)
    embs, ids_out, idxs = [], [], []
    with torch.no_grad():
        for x, id_s, i in inf_ld:
            embs.append(best(x.to(best.device)).cpu())
            ids_out.extend(id_s); idxs.extend(i)
    embs = normalize(torch.cat(embs).numpy(), axis=1)
    cols = ["index","id_str"] + [f"emb_{i}" for i in range(EMB)]
    rows = [[idxs[i], ids_out[i], *embs[i]] for i in range(len(ids_out))]
    pd.DataFrame(rows, columns=cols).to_csv(EMB_OUT, index=False)

    print(f"✅ Embeddings saved to {EMB_OUT}")
    print(f"📈 Launch TensorBoard: tensorboard --logdir {CKPT_DIR}/metric_learning_logs")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
