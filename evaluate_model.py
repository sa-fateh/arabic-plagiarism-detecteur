# evaluate_model.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from dataset import ArabicPlagiarismCSVDataset
from model import PlagiarismDetector

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, help="best_model.pth entraîné sur ExAraPlagDet")
parser.add_argument("--test_csv",  required=True, help="test.csv d’AraPlagDet")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len",    type=int, default=128)
args = parser.parse_args()

# 1) Charger modèle + seuil
ckpt      = torch.load(args.model_path, map_location="cpu")
model     = PlagiarismDetector().eval()
model.load_state_dict(ckpt["model_state"])
threshold = ckpt["threshold"]

# 2) DataLoader test
test_ds    = ArabicPlagiarismCSVDataset(args.test_csv, max_len=args.max_len)
test_loader = DataLoader(test_ds, batch_size=args.batch_size)

# 3) Inférence
all_probs, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(
            batch["s_ids"],
            batch["s_mask"],
            batch["r_ids"],
            batch["r_mask"],
            batch["lex"],
            batch["tfidf"],
        )
        probs = torch.sigmoid(logits).numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())

preds = [int(p > threshold) for p in all_probs]

# 4) Reporting
print("Accuracy :", accuracy_score(all_labels, preds))
print("F1       :", f1_score(all_labels, preds))
print("AUC      :", roc_auc_score(all_labels, all_probs))
print("\nClassification Report:\n", classification_report(all_labels, preds))
