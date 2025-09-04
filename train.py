# train.py

import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.nn import BCEWithLogitsLoss
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import build_dataset
from dataset import ArabicPlagiarismCSVDataset
from model import PlagiarismDetector


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            s_ids  = batch["s_ids"].to(device)
            s_mask = batch["s_mask"].to(device)
            r_ids  = batch["r_ids"].to(device)
            r_mask = batch["r_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type="cuda"):
                logits = model(s_ids, s_mask, r_ids, r_mask).squeeze(-1)
                probs  = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    bin_preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, bin_preds)
    f1  = f1_score(all_labels, bin_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return acc, f1, auc


def train(args):
    # 0) Reproductibilité
    set_seed(args.random_state)

    # 1) Générer les CSV (train/val/test)
    paths, _ = build_dataset(
        xml_dir=args.xml_dir,
        susp_dir=args.susp_dir,
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        augment=args.augment,
        neg_length=args.neg_length,
        neg_ratio=args.neg_ratio,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=None
    )
    train_csv = paths["train"]
    val_csv   = paths["val"]

    # 2) Charger les datasets
    train_ds = ArabicPlagiarismCSVDataset(train_csv, max_len=args.max_len)
    val_ds   = ArabicPlagiarismCSVDataset(val_csv,   max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # 3) Initialisation du modèle, de l’optimiseur, de la loss et du scaler
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = PlagiarismDetector().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = BCEWithLogitsLoss()
    scaler    = GradScaler()

    best_val_loss = float("inf")
    patience_cnt  = 0

    # 4) Boucle d’entraînement
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        loop = tqdm(train_loader, desc=f"[Epoch {epoch}]")
        for batch in loop:
            optimizer.zero_grad()

            s_ids  = batch["s_ids"].to(device)
            s_mask = batch["s_mask"].to(device)
            r_ids  = batch["r_ids"].to(device)
            r_mask = batch["r_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type="cuda"):
                logits = model(s_ids, s_mask, r_ids, r_mask).squeeze(-1)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            loop.set_postfix(train_loss=np.mean(train_losses))

        # 5) Validation
        val_losses, preds, trues = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                s_ids  = batch["s_ids"].to(device)
                s_mask = batch["s_mask"].to(device)
                r_ids  = batch["r_ids"].to(device)
                r_mask = batch["r_mask"].to(device)
                labels = batch["label"].to(device)

                with autocast(device_type="cuda"):
                    logits = model(s_ids, s_mask, r_ids, r_mask).squeeze(-1)
                    loss   = criterion(logits, labels)

                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().tolist()
                preds.extend(probs)
                trues.extend(labels.cpu().tolist())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        bin_preds  = [1 if p >= 0.5 else 0 for p in preds]
        val_acc    = accuracy_score(trues, bin_preds)
        val_f1     = f1_score(trues, bin_preds)
        try:
            val_auc = roc_auc_score(trues, preds)
        except ValueError:
            val_auc = float("nan")

        print(f"[Epoch {epoch}] "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"acc={val_acc:.3f}  "
              f"f1={val_f1:.3f}  "
              f"auc={val_auc:.3f}")

        # 6) Early stopping & checkpoint
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_cnt  = 0

            parent = os.path.dirname(args.model_out)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(model.state_dict(), args.model_out)
            print(f"→ Meilleur modèle sauvegardé: {args.model_out}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print("→ Early stopping déclenché")
                break


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement PlagiarismDetector")
    p.add_argument("--xml_dir",      required=True)
    p.add_argument("--susp_dir",     required=True)
    p.add_argument("--src_dir",      required=True)
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--model_out",    default="best_model.pth")
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--max_len",      type=int,   default=128)
    p.add_argument("--patience",     type=int,   default=2)
    p.add_argument("--augment",      action="store_true")
    p.add_argument("--neg_length",   type=int,   default=50)
    p.add_argument("--neg_ratio",    type=float, default=2.0)
    p.add_argument("--test_size",    type=float, default=0.30)
    p.add_argument("--random_state", type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
