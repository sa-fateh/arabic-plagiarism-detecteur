# train.py

import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import build_dataset
from dataset import ArabicPlagiarismCSVDataset
from model import PlagiarismDetector

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader, 1):
        optimizer.zero_grad()
        logits = model(
            batch["s_ids"].to(device),
            batch["s_mask"].to(device),
            batch["r_ids"].to(device),
            batch["r_mask"].to(device)
        )
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 50 == 0:
            print(f"  [Batch {i}/{len(loader)}] loss={loss.item():.4f}", flush=True)

    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["s_ids"].to(device),
                batch["s_mask"].to(device),
                batch["r_ids"].to(device),
                batch["r_mask"].to(device)
            )
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(batch["label"])
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = (probs > 0.5).astype(int)
    return probs, labels, preds

def main():
    parser = ArgumentParser()
    parser.add_argument("--xml_dir",      required=True)
    parser.add_argument("--susp_dir",     required=True)
    parser.add_argument("--src_dir",      required=True)
    parser.add_argument("--neg_pool_dir", default=None)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--max_len",      type=int,   default=128)
    parser.add_argument("--neg_length",   type=int,   default=50)
    parser.add_argument("--neg_ratio",    type=float, default=0.5)
    parser.add_argument("--slide_per_pos",type=int,   default=1)
    parser.add_argument("--noise_deletion_frac", type=float, default=0.05)
    parser.add_argument("--test_size",    type=float, default=0.3)
    parser.add_argument("--random_state", type=int,   default=42)
    args = parser.parse_args()

    print("→ BUILDING DATASET", flush=True)
    paths = build_dataset(
        xml_dir            = args.xml_dir,
        susp_dir           = args.susp_dir,
        src_dir            = args.src_dir,
        out_dir            = args.out_dir,
        neg_pool_dir       = args.neg_pool_dir,
        neg_length         = args.neg_length,
        neg_ratio          = args.neg_ratio,
        slide_per_pos      = args.slide_per_pos,
        noise_deletion_frac= args.noise_deletion_frac,
        test_size          = args.test_size,
        random_state       = args.random_state
    )
    print("→ DATASET READY", flush=True)

    print("→ INITIALIZING DATA LOADERS", flush=True)
    ds_train = ArabicPlagiarismCSVDataset(paths["train"], max_len=args.max_len)
    labels   = ds_train.df["label"].values.astype(int)
    class_counts  = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler)
    ds_val     = ArabicPlagiarismCSVDataset(paths["val"], max_len=args.max_len)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size)
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True)

    print("→ LOADING MODEL", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlagiarismDetector().to(device)
    print(f"  Using device: {device}", flush=True)

    print("→ SETTING UP OPTIMIZER & CRITERION & SCHEDULER", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = class_counts[0] / class_counts[1]
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)

    best_f1 = 0.0
    print("→ START TRAINING", flush=True)
    try:
        for epoch in range(1, args.epochs+1):
            start_time = time.time()
            print(f"[Epoch {epoch}/{args.epochs}]", flush=True)
            tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            probs, labels, preds = eval_epoch(model, val_loader, device)

            acc = accuracy_score(labels, preds)
            f1  = f1_score(labels, preds)
            auc = roc_auc_score(labels, probs)
            print(f"  train_loss={tr_loss:.4f}  val_acc={acc:.3f}  val_f1={f1:.3f}  val_auc={auc:.3f}", flush=True)
            print(f"  time: {(time.time()-start_time):.1f}s", flush=True)

            scheduler.step(1 - f1)  # on maximise F1 => min(1-F1)
            if f1 > best_f1:
                best_f1 = f1
                best_path = os.path.join(args.out_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"→ New best F1={f1:.3f}, model saved to {best_path}", flush=True)

    except KeyboardInterrupt:
        print("⚠️  Training interrupted by signal. Saving current model...", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "interrupted_model.pth"))
        sys.exit(0)

    print("→ TRAINING COMPLETE", flush=True)

if __name__ == "__main__":
    main()
