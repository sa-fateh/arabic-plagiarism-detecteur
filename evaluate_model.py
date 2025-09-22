import argparse
import numpy as np
import torch
import torch.serialization as serialization
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from dataset import ArabicPlagiarismCSVDataset
from model import PlagiarismDetector

# 1) Parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--test_csv",  required=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len",    type=int, default=128)
args = parser.parse_args()

# 2) Autoriser le global numpy.scalar (Option 2)
serialization.add_safe_globals([np._core.multiarray.scalar])

# 3) Charger checkpoint (Option 1 ou 2)
#ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
ckpt = torch.load(
    args.model_path,
    map_location="cpu",
    weights_only=False  # force le dé-pickle complet
)

# 4) Construire modèle et DataLoader
model = PlagiarismDetector().eval()
model.load_state_dict(ckpt["model_state"])
threshold = ckpt["threshold"]

test_ds    = ArabicPlagiarismCSVDataset(args.test_csv, max_len=args.max_len)
test_loader = DataLoader(test_ds, batch_size=args.batch_size)

# 5) Inférence
probs, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(
            batch["s_ids"], batch["s_mask"],
            batch["r_ids"], batch["r_mask"],
            batch["lex"], batch["tfidf"],
        )
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(batch["label"].cpu().numpy().tolist())

preds = [int(p > threshold) for p in probs]

# 6) Rapports
print("Accuracy :", accuracy_score(labels, preds))
print("F1       :", f1_score(labels, preds))
print("AUC      :", roc_auc_score(labels, probs))
print("\nClassification Report:\n", classification_report(labels, preds))
