# train.py

import argparse
import os
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import build_dataset
from dataset import ArabicPlagiarismCSVDataset
from model import PlagiarismDetector

def train(args):
    # 1) Préparer les CSV (écrit train.csv, val.csv dans out_dir)
    train_csv, val_csv, _ = build_dataset(
        xml_dir=args.xml_dir,
        susp_dir=args.susp_dir,
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        augment=args.augment,
        neg_length=args.neg_length,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # 2) Charger les datasets et DataLoaders
    train_ds = ArabicPlagiarismCSVDataset(train_csv, max_len=args.max_len)
    val_ds   = ArabicPlagiarismCSVDataset(val_csv,   max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # 3) Initialiser le modèle, l’optimiseur, la loss et le scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlagiarismDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = BCEWithLogitsLoss()
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_cnt = 0

    # 4) Boucle d’entraînement
    for epoch in range(1, args.epochs + 1):
        # --- phase entraînement ---
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            s_ids   = batch['s_ids'].to(device)
            s_mask  = batch['s_mask'].to(device)
            r_ids   = batch['r_ids'].to(device)
            r_mask  = batch['r_mask'].to(device)
            labels  = batch['label'].unsqueeze(1).to(device)

            with autocast():
                logits = model(s_ids, s_mask, r_ids, r_mask)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        # --- phase validation ---
        model.eval()
        val_losses, preds, trues = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                s_ids  = batch['s_ids'].to(device)
                s_mask = batch['s_mask'].to(device)
                r_ids  = batch['r_ids'].to(device)
                r_mask = batch['r_mask'].to(device)
                labels = batch['label'].unsqueeze(1).to(device)

                with autocast():
                    logits = model(s_ids, s_mask, r_ids, r_mask)
                    loss = loss_fn(logits, labels)

                val_losses.append(loss.item())
                prob = torch.sigmoid(logits).cpu().numpy().flatten()
                preds.extend(prob)
                trues.extend(labels.cpu().numpy().flatten())

        # Calcul des métriques
        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        bin_preds  = [1 if p >= 0.5 else 0 for p in preds]
        val_acc    = accuracy_score(trues, bin_preds)
        val_f1     = f1_score(trues, bin_preds)
        val_auc    = roc_auc_score(trues, preds) if len(set(trues)) > 1 else float('nan')

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  acc={val_acc:.3f}  f1={val_f1:.3f}  auc={val_auc:.3f}")

        # Early-stopping & checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            # Créer le dossier parent seulement s’il existe
            model_dir = os.path.dirname(args.model_out)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), args.model_out)
            print(f"→ Meilleur modèle sauvegardé: {args.model_out}")

        #if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #
        #    patience_cnt = 0
         #   os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
          #  torch.save(model.state_dict(), args.model_out)
           # print(f"→ Meilleur modèle sauvegardé: {args.model_out}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print("→ Early stopping triggered")
                break

def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne le détecteur de plagiat arabe")
    parser.add_argument('--xml_dir',     required=True)
    parser.add_argument('--susp_dir',    required=True)
    parser.add_argument('--src_dir',     required=True)
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--model_out',   default="best_model.pth")
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=2e-5)
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--max_len',     type=int,   default=128)
    parser.add_argument('--patience',    type=int,   default=2)
    parser.add_argument('--augment',     action='store_true')
    parser.add_argument('--neg_length',  type=int,   default=50)
    parser.add_argument('--test_size',   type=float, default=0.30)
    parser.add_argument('--random_state',type=int,   default=42)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
