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
    return torch.cat(all_probs), torch.cat(all_labels)

def main():
    parser = ArgumentParser()
    parser.add_argument("--xml_dir",      required=True)
    parser.add_argument("--susp_dir",     required=True)
    parser.add_argument("--src_dir",      required=True)
    parser.add_argument("--neg_pool_dir", default=None,
                        help="Directory with extra .txt files for negatives")
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--max_len",      type=int,   default=128)
    parser.add_argument("--neg_length",   type=int,   default=50)
    parser.add_argument("--neg_ratio",    type=float, default=1.0)
    parser.add_argument("--slide_per_pos",type=int,   default=2)
    parser.add_argument("--noise_deletion_frac", type=float, default=0.1)
    parser.add_argument("--test_size",    type=float, default=0.30)
    parser.add_argument("--random_state", type=int,   default=42)
    args = parser.parse_args()

    # 1) BUILD DATASET
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

    # 2) DATALOADERS
    ds_train = ArabicPlagiarismCSVDataset(paths["train"], max_len=args.max_len)
    ds_val   = ArabicPlagiarismCSVDataset(paths["val"],   max_len=args.max_len)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size)

    # 3) MODEL & OPTIMIZER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlagiarismDetector().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        probs, labels = eval_epoch(model, val_loader, device)
        val_loss = F.binary_cross_entropy(probs, labels).item()

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            out_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save(model.state_dict(), out_path)
            print(f"→ Best model saved: {out_path}")

if __name__ == "__main__":
    main()
