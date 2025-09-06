import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from data import build_dataset
from dataset import ArabicPlagiarismCSVDataset
from model import PlagiarismDetector

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            batch["s_ids"].to(device),
            batch["s_mask"].to(device),
            batch["r_ids"].to(device),
            batch["r_mask"].to(device)
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, batch["label"].to(device)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["s_ids"].to(device),
                batch["s_mask"].to(device),
                batch["r_ids"].to(device),
                batch["r_mask"].to(device)
            )
            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch["label"])
    preds = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    # compute metrics: acc, f1, auc...
    return preds, labels

def main():
    parser = ArgumentParser()
    parser.add_argument("--xml_dir",        required=True)
    parser.add_argument("--susp_dir",       required=True)
    parser.add_argument("--src_dir",        required=True)
    parser.add_argument("--neg_pool_dir",   default=None)
    parser.add_argument("--out_dir",        required=True)
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--max_len",        type=int, default=128)
    parser.add_argument("--epochs",         type=int, default=10)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--no_augment",     action="store_true")
    args = parser.parse_args()

    # 1) build dataset
    paths = build_dataset(
        xml_dir=args.xml_dir,
        susp_dir=args.susp_dir,
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        neg_pool_dir=args.neg_pool_dir,
        augment=not args.no_augment,
        test_size=0.3,
        random_state=42
    )

    # 2) create DataLoaders
    ds_train = ArabicPlagiarismCSVDataset(paths["train"], max_len=args.max_len)
    ds_val   = ArabicPlagiarismCSVDataset(paths["val"],   max_len=args.max_len)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size)

    # 3) model, optimizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = PlagiarismDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        preds, labels = eval_epoch(model, val_loader, device)
        val_loss = F.binary_cross_entropy(preds, labels).item()

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
            print(f"→ Meilleur modèle sauvegardé")

if __name__ == "__main__":
    main()
