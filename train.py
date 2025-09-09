import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import torch
import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = (probs > 0.5).astype(int)
    return probs, labels, preds

def main():
    parser = ArgumentParser()
    parser.add_argument("--xml_dir", required=True)
    parser.add_argument("--susp_dir", required=True)
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--neg_pool_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--neg_length", type=int, default=50)
    parser.add_argument("--neg_ratio", type=float, default=1.0)
    parser.add_argument("--slide_per_pos", type=int, default=2)
    parser.add_argument("--noise_deletion_frac", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    print("→ BUILDING DATASET", flush=True)
    paths = build_dataset(
        xml_dir=args.xml_dir,
        susp_dir=args.susp_dir,
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        neg_pool_dir=args.neg_pool_dir,
        neg_length=args.neg_length,
        neg_ratio=args.neg_ratio,
        slide_per_pos=args.slide_per_pos,
        noise_deletion_frac=args.noise_deletion_frac,
        test_size=args.test_size,
        random_state=args.random_state
    )

    print("→ INITIALIZING DATA LOADERS", flush=True)
    ds_train = ArabicPlagiarismCSVDataset(paths["train"], max_len=args.max_len)
    ds_val   = ArabicPlagiarismCSVDataset(paths["val"],   max_len=args.max_len)

    # Comptage des exemples
    n_pos = (ds_train.df["label"] == 1).sum()
    n_neg = (ds_train.df["label"] == 0).sum()
    print(f"  ✅ Train set: {len(ds_train)} examples → {n_pos} positives / {n_neg} negatives", flush=True)

    labels = ds_train.df["label"].values.astype(int)
    class_counts  = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler)
    val_loader   = DataLoader(ds_val, batch_size=args.batch_size)

    print("→ LOADING MODEL", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlagiarismDetector().to(device)
    print(f"  Using device: {device}", flush=True)

    print("→ SETTING UP OPTIMIZER & SCHEDULER", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    pos_weight = class_counts[0] / class_counts[1]
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"[Epoch {epoch}/{args.epochs}]", flush=True)

        # Freeze/unfreeze BERT
        if epoch == 1:
            for param in model.bert.parameters():
                param.requires_grad = False
            print("  🔒 BERT frozen", flush=True)
        if epoch == 3:
            for param in model.bert.parameters():
                param.requires_grad = True
            print("  🔓 BERT unfrozen", flush=True)

        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        probs, labels, preds = eval_epoch(model, val_loader, device)

        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        print(f"  train_loss={tr_loss:.4f}  val_acc={acc:.3f}  val_f1={f1:.3f}  val_auc={auc:.3f}  time={(time.time()-t0):.1f}s", flush=True)

        scheduler.step()
        if f1 > best_f1:
            best_f1 = f1
            out_fp = os.path.join(args.out_dir, "best_model.pth")
            torch.save(model.state_dict(), out_fp)
            print(f"→ Saved new best model (F1={f1:.3f})", flush=True)

    print("→ TRAINING COMPLETE", flush=True)

if __name__ == "__main__":
    main()
