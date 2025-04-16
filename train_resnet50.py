# TEsting  – Card metric‑learning pipeline
# ----------------------------------------------------------------------
import os, random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score

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
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
# ----------------------------------------------------------------------

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL.Image import Resampling

# speed hint for 4090
torch.set_float32_matmul_precision("high")

# ======================================================================
# ‑‑‑ Utility -----------------------------------------------------------
def pad_to_square(img):
    return ImageOps.pad(img, (224, 224), method=Resampling.BICUBIC, color=(0, 0, 0))

def make_contiguous(x):        # avoids non‑contiguous warning inside AMP
    return x.contiguous()

class RandomGaussianNoise:
    def __init__(self, mean=0.0, std=0.02, p=0.3):
        self.mean, self.std, self.p = mean, std, p
    def __call__(self, img_tensor):
        if random.random() < self.p:
            img_tensor += torch.randn_like(img_tensor)*self.std + self.mean
            img_tensor.clamp_(0, 1)
        return img_tensor
# ======================================================================
# ‑‑‑ Dataset -----------------------------------------------------------
class CardDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, label2idx=None):
        self.data = pd.read_csv(csv_path, header=None, names=["filename", "id_str"])
        self.root_dir = root_dir

        # keep a shared label mapping so train / val use same indices
        if label2idx is None:
            unique_ids = self.data["id_str"].unique()
            self.label2idx = {id_: i for i, id_ in enumerate(unique_ids)}
        else:
            self.label2idx = label2idx

        self.transform = transform or transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(degrees=8, translate=(0.05,0.05), scale=(0.9,1.1), shear=3),
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

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        img      = Image.open(img_path).convert("RGB")
        img      = self.transform(img)
        label    = self.label2idx[row["id_str"]]
        return img, label
# ======================================================================
# ‑‑‑ Model -------------------------------------------------------------
class EmbeddingModel(pl.LightningModule):
    def __init__(self, embedding_dim=512, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        backbone      = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder  = nn.Sequential(*list(backbone.children())[:-1])
        self.fc       = nn.Linear(2048, embedding_dim)

        self.miner     = BatchHardMiner()
        self.loss_func = TripletMarginLoss(margin=0.2)
        self.lr        = lr

        # buffers for validation metrics
        self.val_embeddings, self.val_labels = [], []

    def forward(self, x):
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        return F.normalize(self.fc(x), p=2, dim=1)

    # ---------- Train / Val steps -------------------------------------
    def training_step(self, batch, _):
        imgs, labels = batch
        embs = self(imgs)
        loss = self.loss_func(embs, labels, self.miner(embs, labels))
        self.log("train_loss", loss, prog_bar=True, batch_size=len(imgs))
        return loss

    def validation_step(self, batch, _):
        imgs, labels = batch
        embs = self(imgs)
        loss = self.loss_func(embs, labels, self.miner(embs, labels))

        # store full batch – queries *and* references are identical
        self.val_embeddings.append(embs.cpu())
        self.val_labels.append(labels.cpu())
        self.log("val_loss", loss, prog_bar=True, batch_size=len(imgs))
        return loss

    def on_validation_epoch_end(self):
        embs   = torch.cat(self.val_embeddings).numpy()
        labels = torch.cat(self.val_labels).numpy()

        acc_calc = AccuracyCalculator(
            include=[
                "precision_at_1",
                "mean_average_precision",
                "mean_average_precision_at_r",
                "r_precision"
            ], k=5
        )
        metrics = acc_calc.get_accuracy(embs, embs, labels, labels)

        self.log("val/precision@1", metrics["precision_at_1"], prog_bar=True)
        self.log("val/mAP",         metrics["mean_average_precision"],      prog_bar=True)
        self.log("val/mAP@r",       metrics["mean_average_precision_at_r"], prog_bar=True)
        self.log("val/r_precision", metrics["r_precision"],                 prog_bar=True)

        self.val_embeddings.clear()
        self.val_labels.clear()

    # ---------- Optim --------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs, eta_min=self.lr/50)
        return [opt], [sch]
# ======================================================================
# ‑‑‑ Inference dataset (unchanged) ------------------------------------
class CardInferenceDataset(Dataset):
    def __init__(self, csv_path, root_dir):
        self.data = pd.read_csv(csv_path, header=None, names=["filename", "id_str"])
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, row["filename"])).convert("RGB")
        return self.transform(img), row["id_str"], idx
# ======================================================================
# ‑‑‑ MAIN --------------------------------------------------------------
def main():
    BASE_DIR   = "/workspace/deepEmbed"
    CSV_FILE   = f"{BASE_DIR}/dataset/dataset.csv"            # many entries
    VAL_CSV    = f"{BASE_DIR}/dataset/dataset_backup.csv"     # 1 per card
    ROOT_DIR   = f"{BASE_DIR}/dataset"
    CKPT_DIR   = f"{BASE_DIR}/checkpoints"
    EMB_OUT    = f"{BASE_DIR}/all_embeddings.csv"

    BATCH_SIZE = 440
    EPOCHS     = 200
    EMB_DIM    = 512
    LR         = 1e-4

    # ---------- shared label mapping ----------------------------------
    shared_labels = {id_: i for i, id_ in enumerate(
        pd.read_csv(CSV_FILE, header=None, names=["f","id"])["id"].unique()
    )}

    train_ds = CardDataset(CSV_FILE, ROOT_DIR, label2idx=shared_labels)
    val_ds   = CardDataset(VAL_CSV,  ROOT_DIR, label2idx=shared_labels)

    # ------ Balanced sampler for validation ---------------------------
    val_labels = [shared_labels[id_] for id_ in val_ds.data["id_str"]]
    M = 2                              # images per class
    val_sampler = MPerClassSampler(
        val_labels, m=M, batch_size=BATCH_SIZE,
        length_before_new_iter=len(val_labels)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=15)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,            num_workers=10)

    # ---------- Lightning plumbing ------------------------------------
    model = EmbeddingModel(embedding_dim=EMB_DIM, lr=LR)

    ckpt_cb = ModelCheckpoint(
        monitor="val/precision@1", mode="max",
        save_top_k=1, filename="best-checkpoint", save_last=True,
        dirpath=CKPT_DIR
    )
    logger = TensorBoardLogger(CKPT_DIR, name="metric_learning_logs")

    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator="auto", precision="16-mixed",
        callbacks=[ckpt_cb], logger=logger, default_root_dir=CKPT_DIR
    )

    resume_path = os.path.join(CKPT_DIR, "last-v1.ckpt")
    trainer.fit(model, train_loader, val_loader,
                ckpt_path=resume_path if os.path.exists(resume_path) else None)

    # ---------- Export full‑set embeddings -----------------------------
    best = EmbeddingModel.load_from_checkpoint(ckpt_cb.best_model_path)
    best.eval().freeze()

    infer_loader = DataLoader(
        CardInferenceDataset(VAL_CSV, ROOT_DIR),
        batch_size=32, shuffle=False, num_workers=4
    )

    embs, ids, idxs = [], [], []
    with torch.no_grad():
        for imgs, id_strs, batch_idxs in infer_loader:
            embs.append(best(imgs.to(best.device)).cpu())
            ids.extend(id_strs); idxs.extend(batch_idxs)

    embs = normalize(torch.cat(embs).numpy(), axis=1)
    cols = ["index","id_str"] + [f"emb_{i}" for i in range(EMB_DIM)]
    rows = [[idxs[i], ids[i], *embs[i]] for i in range(len(ids))]
    pd.DataFrame(rows, columns=cols).to_csv(EMB_OUT, index=False)

    print(f"✅ Embeddings saved to {EMB_OUT}")
    print(f"📈 TensorBoard: tensorboard --logdir {CKPT_DIR}/metric_learning_logs")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
