# Card metric‑learning pipeline  (ResNet‑50 → 512‑D embeddings)
# =============================================================
import os, random, csv, tempfile, atexit
import numpy as np, pandas as pd
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_metric_learning.losses  import TripletMarginLoss
from pytorch_metric_learning.miners  import BatchHardMiner
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL.Image import Resampling

torch.set_float32_matmul_precision("high")      # 40‑series tensor cores
# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
def pad_to_square(img):
    return ImageOps.pad(img, (224, 224), method=Resampling.BICUBIC, color=(0, 0, 0))
def make_contiguous(x): return x.contiguous()

class RandomGaussianNoise:
    def __init__(self, std=0.05, p=0.3): self.std, self.p = std, p
    def __call__(self, x):
        if random.random() < self.p:
            x += torch.randn_like(x)*self.std
            x.clamp_(0,1)
        return x

train_tf = transforms.Compose([
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

val_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(pad_to_square),
    transforms.ToTensor()
])
# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class CardDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, label2idx=None):
        self.root_dir = root_dir
        self.files, self.id_strs = [], []
        with open(csv_path) as f:
            for fname, cid in csv.reader(f):
                self.files.append(fname)
                self.id_strs.append(cid)

        if label2idx is None:
            label2idx = {cid:i for i,cid in enumerate(sorted(set(self.id_strs)))}
        self.label2idx = label2idx
        self.labels = torch.tensor([self.label2idx[c] for c in self.id_strs])
        self.t = transform or val_tf

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.files[idx])).convert("RGB")
        return self.t(img), self.labels[idx]
# ----------------------------------------------------------------------
# LightningModule
# ----------------------------------------------------------------------
class EmbeddingModel(pl.LightningModule):
    def __init__(self, emb_dim=512, lr=1e-4, ref_loader=None):
        super().__init__()
        self.save_hyperparameters(ignore=["ref_loader"])
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.fc      = nn.Linear(2048, emb_dim)

        self.miner  = BatchHardMiner()
        self.loss_f = TripletMarginLoss(margin=0.2)

        self.acc_calc = AccuracyCalculator(
            include=("precision_at_1","mean_average_precision",
                     "mean_average_precision_at_r","r_precision"),
            k="max_bin_count")

        self.val_embs, self.val_labels = [], []
        self.ref_loader = ref_loader
        self._ref_cache = None   # cached (emb, lab)

    # ------------ embedding helper ------------
    def _embed(self, x):
        x = self.encoder(x).flatten(1)
        return F.normalize(self.fc(x), p=2, dim=1)
    def forward(self, x): return self._embed(x)

    # ------------ training --------------------
    def training_step(self, batch, _):
        imgs, y = batch
        e = self(imgs)                          # embed once
        loss = self.loss_f(e, y, self.miner(e, y))
        self.log("train_loss", loss, prog_bar=True, batch_size=len(imgs))
        return loss

    # ------------ validation ------------------
    def validation_step(self, batch, _):
        imgs, y = batch
        self.val_embs.append(self(imgs).cpu())
        self.val_labels.append(y.cpu())

    def on_validation_epoch_end(self):
        q_emb = torch.cat(self.val_embs); self.val_embs.clear()
        q_lab = torch.cat(self.val_labels); self.val_labels.clear()

        # --- always rebuild reference embeddings --------------------
        r_embs, r_labs = [], []
        with torch.no_grad():
            for imgs, y in self.ref_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                r_embs.append(self(imgs).cpu()); r_labs.append(y.cpu())
        r_emb = torch.cat(r_embs);  r_lab = torch.cat(r_labs)

        if hasattr(self.acc_calc, "calculate_in_chunks"):
                m = self.acc_calc.calculate_in_chunks(
                    q_emb, q_lab, r_emb, r_lab,
                    chunk_size=10000, ref_includes_query=False)
        else:                          # <- fallback for stripped builds
            print("⚠️  AccuracyCalculator.calculate_in_chunks unavailable – "
                "falling back to get_accuracy (higher RAM).")
            m = self.acc_calc.get_accuracy(
                q_emb.numpy(), q_lab.numpy(),
                r_emb.numpy(), r_lab.numpy(),
                ref_includes_query=False)
            
        self.log_dict({
            "val/precision@1": m["precision_at_1"],
            "val/mAP":         m["mean_average_precision"],
            "val/mAP@r":       m["mean_average_precision_at_r"],
            "val/r_precision": m["r_precision"]
        }, prog_bar=True, sync_dist=True)

    # ------------ optimiser -------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.hparams.lr/50)
        return [opt], [sch]
# ----------------------------------------------------------------------
# Inference dataset
# ----------------------------------------------------------------------
class CardInferenceDataset(Dataset):
    def __init__(self, csv_path, root_dir):
        self.data = pd.read_csv(csv_path, header=None, names=["file","id"])
        self.root = root_dir; self.t = val_tf
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(os.path.join(self.root, row.file)).convert("RGB")
        return self.t(img), row.id, idx
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    BASE = "/workspace/deepEmbed"
    RAW_TRAIN_CSV = f"{BASE}/dataset/dataset.csv"          # duplicates + aug
    VAL_CSV       = f"{BASE}/dataset/dataset_backup.csv"   # 1 image / card
    ROOT          = f"{BASE}/dataset"
    CKPT_DIR      = f"{BASE}/checkpoints"
    OUT_CSV       = f"{BASE}/all_embeddings.csv"

    BS, EPOCHS, EMB, LR = 440, 200, 512, 1e-4
    tmp_train_csv = None                                   # path if we create one

    # -------- ensure ≥2 rows / card in reference ----------
    df_train = pd.read_csv(RAW_TRAIN_CSV, header=None, names=["file","id"])
    singleton_ids = df_train["id"].value_counts()[lambda s: s==1].index
    if len(singleton_ids):
        print(f"🔄  Duplicating {len(singleton_ids)} singleton card IDs so each has ≥2 rows.")
        df_train = pd.concat([df_train, df_train[df_train["id"].isin(singleton_ids)]],
                             ignore_index=True)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_train.to_csv(tmp.name, header=False, index=False)
        tmp_train_csv = tmp.name
        atexit.register(lambda p=tmp_train_csv: os.remove(p) if p and os.path.exists(p) else None)
    else:
        print("✅  All card IDs already have ≥2 rows; using original dataset.csv.")
    train_csv_path = tmp_train_csv or RAW_TRAIN_CSV

    # -------- label map -----------------------------------
    label_map = {cid:i for i,cid in enumerate(sorted(df_train["id"].unique()))}

    # -------- datasets & loaders --------------------------
    train_ds     = CardDataset(train_csv_path, ROOT, train_tf, label_map)
    val_query_ds = CardDataset(VAL_CSV,        ROOT, val_tf,   label_map)
    val_ref_ds   = CardDataset(train_csv_path, ROOT, val_tf,   label_map)

    train_ld = DataLoader(train_ds, BS, shuffle=True,  num_workers=15, pin_memory=True)
    val_qld  = DataLoader(val_query_ds, BS, shuffle=False, num_workers=10, pin_memory=True)
    val_rld  = DataLoader(val_ref_ds,   BS, shuffle=False, num_workers=10, pin_memory=False)

    # -------- model ---------------------------------------
    model = EmbeddingModel(EMB, LR, ref_loader=val_rld)

    ckpt_cb = ModelCheckpoint(dirpath=CKPT_DIR, filename="best",
                              monitor="val/precision@1", mode="max",
                              save_top_k=1, save_last=True)
    logger = TensorBoardLogger(CKPT_DIR, name="metric_learning_logs")

    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="auto",
                         precision="16-mixed", callbacks=[ckpt_cb],
                         logger=logger, default_root_dir=CKPT_DIR)

    resume = os.path.join(CKPT_DIR, "last.ckpt")
    trainer.fit(model, train_ld, val_qld,
                ckpt_path=resume if os.path.exists(resume) else None)

    # -------- export embeddings ---------------------------
    best_path = ckpt_cb.best_model_path or resume
    best = EmbeddingModel.load_from_checkpoint(best_path, EMB, LR, ref_loader=val_rld)
    best.eval().freeze()

    inf_ld = DataLoader(CardInferenceDataset(VAL_CSV, ROOT),
                        batch_size=32, shuffle=False, num_workers=8)
    embs, ids, idxs = [], [], []
    with torch.no_grad():
        for imgs, id_strs, ixs in inf_ld:
            embs.append(best(imgs.to(best.device)).cpu())
            ids.extend(id_strs); idxs.extend(ixs)
    embs = normalize(torch.cat(embs).numpy(), axis=1)

    cols = ["index","id_str"] + [f"emb_{i}" for i in range(EMB)]
    pd.DataFrame([[idxs[i], ids[i], *embs[i]] for i in range(len(ids))],
                 columns=cols).to_csv(OUT_CSV, index=False)
    print(f"✅ Embeddings saved to {OUT_CSV}")
    print(f"📈 tensorboard --logdir {CKPT_DIR}/metric_learning_logs")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
