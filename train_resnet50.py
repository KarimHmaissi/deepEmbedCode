# Card metric‑learning pipeline  (ResNet‑50 → 512‑D embeddings)
# =============================================================
import os, random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# pytorch‑metric‑learning
from pytorch_metric_learning.losses   import TripletMarginLoss
from pytorch_metric_learning.miners   import BatchHardMiner
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL.Image import Resampling

torch.set_float32_matmul_precision("high")          # use Tensor‑Cores on 4090
# ----------------------------------------------------------------------
# Utils
def pad_to_square(img):
    return ImageOps.pad(img, (224,224), method=Resampling.BICUBIC, color=(0,0,0))
def make_contiguous(x): return x.contiguous()

class RandomGaussianNoise:
    def __init__(self, std=0.05, p=0.3): self.std, self.p = std, p
    def __call__(self, x):
        if random.random() < self.p:
            x += torch.randn_like(x)*self.std
            x.clamp_(0,1)
        return x
# ----------------------------------------------------------------------
# Dataset
class CardDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, label2idx=None):
        self.data      = pd.read_csv(csv_path, header=None, names=["file","id"])
        self.root_dir  = root_dir
        self.label2idx = label2idx or {id_:i for i,id_ in enumerate(self.data["id"].unique())}

        self.transform = transform or transforms.Compose([
            transforms.RandomPerspective(0.2, p=0.3),
            transforms.RandomAffine(8, translate=(.05,.05), scale=(.9,1.1), shear=3),
            transforms.ColorJitter(.2,.2,.1,.01),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.Lambda(pad_to_square),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(.02,.2)),
            transforms.RandomApply([transforms.GaussianBlur(3,(.3,2.0))], p=0.3),
            RandomGaussianNoise(),
            transforms.Lambda(make_contiguous)
        ])

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row  = self.data.iloc[idx]
        img  = Image.open(os.path.join(self.root_dir, row.file)).convert("RGB")
        img  = self.transform(img)
        lab  = self.label2idx[row.id]
        return img, lab
# ----------------------------------------------------------------------
# Model
class EmbeddingModel(pl.LightningModule):
    def __init__(self, emb_dim=512, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        backbone   = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.fc      = nn.Linear(2048, emb_dim)

        self.miner  = BatchHardMiner()
        self.loss_f = TripletMarginLoss(margin=0.2)
        self.acc    = AccuracyCalculator(include=[
            "precision_at_1","mean_average_precision",
            "mean_average_precision_at_r","r_precision"
        ], k=5)

        self.val_embs, self.val_labels = [], []

    def forward(self,x):
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        return F.normalize(self.fc(x), p=2, dim=1)

    def training_step(self,batch,_):
        imgs, y = batch
        e = self(imgs)
        loss = self.loss_f(e, y, self.miner(e,y))
        self.log("train_loss", loss, prog_bar=True, batch_size=len(imgs))
        return loss

    def validation_step(self,batch,_):
        imgs, y = batch
        e = self(imgs)
        loss = self.loss_f(e, y, self.miner(e,y))
        self.val_embs.append(e.cpu()); self.val_labels.append(y.cpu())
        self.log("val_loss", loss, prog_bar=True, batch_size=len(imgs))

    def on_validation_epoch_end(self):
        e = torch.cat(self.val_embs).numpy()
        y = torch.cat(self.val_labels).numpy()
        m = self.acc.get_accuracy(e, y)          # simplest signature
        self.log_dict({
            "val/precision@1": m["precision_at_1"],
            "val/mAP":         m["mean_average_precision"],
            "val/mAP@r":       m["mean_average_precision_at_r"],
            "val/r_precision": m["r_precision"]
        }, prog_bar=True)
        self.val_embs.clear(); self.val_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs,
                                                         eta_min=self.hparams.lr/50)
        return [opt], [sch]
# ----------------------------------------------------------------------
# Inference DS
class CardInferenceDataset(Dataset):
    def __init__(self, csv, root):
        self.data = pd.read_csv(csv, header=None, names=["file","id"])
        self.root = root
        self.t    = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        row = self.data.iloc[i]
        img = Image.open(os.path.join(self.root,row.file)).convert("RGB")
        return self.t(img), row.id, i
# ----------------------------------------------------------------------
def main():
    BASE  = "/workspace/deepEmbed"
    TRAIN = f"{BASE}/dataset/dataset.csv"          # many rows
    VAL   = f"{BASE}/dataset/dataset_backup.csv"   # 1 per card
    ROOT  = f"{BASE}/dataset"
    CKPT  = f"{BASE}/checkpoints"
    OUT   = f"{BASE}/all_embeddings.csv"

    BS   = 440
    EPOCHS=200
    EMB  = 512
    LR   = 1e-4

    shared = {id_:i for i,id_ in enumerate(
        pd.read_csv(TRAIN, header=None, names=["f","id"])["id"].unique())
    }

    train_ds = CardDataset(TRAIN, ROOT, label2idx=shared)
    val_ds   = CardDataset(VAL,   ROOT, label2idx=shared)

    train_ld = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=15)
    val_ld   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=10)

    model  = EmbeddingModel(emb_dim=EMB, lr=LR)

    ckpt_cb = ModelCheckpoint(dirpath=CKPT, filename="best", monitor="val/precision@1",
                              mode="max", save_top_k=1, save_last=True)
    logger  = TensorBoardLogger(CKPT, name="metric_learning_logs")

    tr = pl.Trainer(max_epochs=EPOCHS, accelerator="auto", precision="16-mixed",
                    callbacks=[ckpt_cb], logger=logger, default_root_dir=CKPT)

    resume = os.path.join(CKPT,"last.ckpt")
    tr.fit(model, train_ld, val_ld, ckpt_path=resume if os.path.exists(resume) else None)

    # ---------- export embeddings on the unique‑card list -------------
    best = EmbeddingModel.load_from_checkpoint(ckpt_cb.best_model_path)
    best.eval().freeze()

    inf_ld = DataLoader(CardInferenceDataset(VAL, ROOT), batch_size=32, shuffle=False)
    embs, ids, idxs = [], [], []
    with torch.no_grad():
        for imgs, id_strs, ixs in inf_ld:
            embs.append(best(imgs.to(best.device)).cpu())
            ids.extend(id_strs); idxs.extend(ixs)
    embs = normalize(torch.cat(embs).numpy(), axis=1)
    cols = ["index","id_str"] + [f"emb_{i}" for i in range(EMB)]
    pd.DataFrame([[idxs[i], ids[i], *embs[i]] for i in range(len(ids))],
                 columns=cols).to_csv(OUT, index=False)
    print(f"✅ Embeddings saved to {OUT}")
    print(f"📈 tensorboard --logdir {CKPT}/metric_learning_logs")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()