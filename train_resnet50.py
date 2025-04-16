#TEsting

import os
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import random
from torchvision.utils import save_image
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics import average_precision_score

# pytorch-metric-learning imports
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner

# For a pretrained ResNet50
from torchvision.models import resnet50, ResNet50_Weights
from PIL.Image import Resampling

# =======================
# Utility Functions
# =======================
def pad_to_square(img):
    return ImageOps.pad(img, (224, 224), method=Resampling.BICUBIC, color=(0, 0, 0))

def make_contiguous(x):
    return x.contiguous()

class RandomGaussianNoise:
    def __init__(self, mean=0.0, std=0.02, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img_tensor):
        if random.random() < self.p:
            noise = torch.randn_like(img_tensor) * self.std + self.mean
            img_tensor = img_tensor + noise
            img_tensor.clamp_(0, 1)
        return img_tensor

def compute_metrics(embeddings, labels):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = labels.cpu().numpy()
    sims = torch.matmul(embeddings, embeddings.T).cpu().numpy()

    top1, recall_at_5, average_precisions = 0, 0, []

    for i in range(len(labels)):
        sims[i, i] = -np.inf  # exclude self
        top_k = np.argsort(sims[i])[::-1]
        top1 += (labels[i] == labels[top_k[0]])
        recall_at_5 += (labels[i] in labels[top_k[:5]])

        y_true = (labels == labels[i]).astype(int)
        y_scores = sims[i]
        average_precisions.append(average_precision_score(y_true, y_scores))

    return top1 / len(labels), recall_at_5 / len(labels), np.mean(average_precisions)

# =======================
# DATASET
# =======================
class CardDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform=None):
        self.data = pd.read_csv(csv_path, header=None, names=["filename", "id_str"])
        self.root_dir = root_dir
        unique_ids = self.data["id_str"].unique()
        self.label2idx = {id_str: i for i, id_str in enumerate(unique_ids)}

        self.transform = transform or transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(224),
            transforms.Lambda(pad_to_square),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 2.0))], p=0.3),
            RandomGaussianNoise(std=0.05, p=0.3),
            transforms.Lambda(make_contiguous)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.label2idx[row["id_str"]]
        return img, label

    @property
    def num_classes(self):
        return len(self.label2idx)

# =======================
# MODEL
# =======================
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class EmbeddingModel(pl.LightningModule):
    def __init__(self, embedding_dim=512, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, embedding_dim)
        self.miner = BatchHardMiner()
        self.loss_func = TripletMarginLoss(margin=0.2)
        self.lr = lr

        # For validation
        self.val_query_embeddings = []
        self.val_query_labels = []
        self.val_reference_embeddings = []
        self.val_reference_labels = []

    def forward(self, x):
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        x = F.normalize(self.fc(x), p=2, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        embeddings = self(imgs)
        hard_triplets = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_triplets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        embeddings = self(imgs)
        hard_triplets = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_triplets)

        # Split into query/reference randomly
        split_point = imgs.size(0) // 2
        self.val_query_embeddings.append(embeddings[:split_point].cpu())
        self.val_query_labels.append(labels[:split_point].cpu())
        self.val_reference_embeddings.append(embeddings[split_point:].cpu())
        self.val_reference_labels.append(labels[split_point:].cpu())

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # concatenate tensors
        query_embs = torch.cat(self.val_query_embeddings).numpy()
        query_labels = torch.cat(self.val_query_labels).numpy()
        ref_embs   = torch.cat(self.val_reference_embeddings).numpy()
        ref_labels = torch.cat(self.val_reference_labels).numpy()

        acc_calc = AccuracyCalculator(
            include=[
                "precision_at_1",
                "mean_average_precision",
                "mean_average_precision_at_r",
                "r_precision"
            ],
            k=5
        )
        metrics = acc_calc.get_accuracy(
            query=query_embs,
            reference=ref_embs,
            query_labels=query_labels,
            reference_labels=ref_labels
        )

        self.log("val/precision@1", metrics["precision_at_1"], prog_bar=True)
        self.log("val/mAP",         metrics["mean_average_precision"],        prog_bar=True)
        self.log("val/mAP@r",       metrics["mean_average_precision_at_r"],     prog_bar=True)
        self.log("val/r_precision", metrics["r_precision"],                  prog_bar=True)

        # clear for next epoch
        self.val_query_embeddings.clear()
        self.val_query_labels.clear()
        self.val_reference_embeddings.clear()
        self.val_reference_labels.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [scheduler]

# =======================
# INFERENCE DATASET
# =======================
class CardInferenceDataset(Dataset):
    def __init__(self, csv_path, root_dir):
        self.data = pd.read_csv(csv_path, header=None, names=["filename", "id_str"])
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, row["id_str"], idx

# =======================
# MAIN
# =======================
def main():
    BASE_DIR = "/workspace/deepEmbed"
    CSV_FILE = os.path.join(BASE_DIR, "dataset/dataset.csv")
    ORIGINAL_CSV_FILE = os.path.join(BASE_DIR, "dataset/dataset_backup.csv")
    ROOT_DIR = os.path.join(BASE_DIR, "dataset")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    EMBEDDING_OUTPUT = os.path.join(BASE_DIR, "all_embeddings.csv")

    BATCH_SIZE = 400
    EPOCHS = 200
    EMBEDDING_DIM = 512
    LR = 1e-4

    trainval_dataset = CardDataset(CSV_FILE, ROOT_DIR)
    val_size = int(0.1 * len(trainval_dataset))
    train_ds, val_ds = random_split(trainval_dataset, [len(trainval_dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    model = EmbeddingModel(embedding_dim=EMBEDDING_DIM, lr=LR)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/precision@1",
        save_top_k=1,
        mode="max",
        filename="best-checkpoint",
        save_last=True,
        dirpath=CHECKPOINT_DIR
    )

    logger = TensorBoardLogger(
        save_dir=CHECKPOINT_DIR,
        name="metric_learning_logs"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=CHECKPOINT_DIR
    )

    last_ckpt_path = os.path.join(CHECKPOINT_DIR, "last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    best_model = EmbeddingModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval().freeze()

    infer_dataset = CardInferenceDataset(ORIGINAL_CSV_FILE, ROOT_DIR)
    infer_loader = DataLoader(infer_dataset, batch_size=32, shuffle=False, num_workers=4)

    embeddings, ids, idxs = [], [], []
    with torch.no_grad():
        for imgs, id_str, idx in infer_loader:
            emb = best_model(imgs.to(best_model.device)).cpu()
            embeddings.append(emb)
            ids.extend(id_str)
            idxs.extend(idx)

    embeddings = normalize(torch.cat(embeddings).numpy(), axis=1)

    df_records = [[idxs[i], ids[i]] + embeddings[i].tolist() for i in range(len(ids))]
    cols = ["index", "id_str"] + [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    pd.DataFrame(df_records, columns=cols).to_csv(EMBEDDING_OUTPUT, index=False)

    print(f"✅ Embeddings written to: {EMBEDDING_OUTPUT}")
    print(f"📈 Run 'tensorboard --logdir {CHECKPOINT_DIR}/metric_learning_logs' to view training logs.")

if __name__ == "__main__":
    main()
