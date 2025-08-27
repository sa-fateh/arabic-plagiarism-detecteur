# evaluate.py

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset import ArabicPlagiarismCSVDataset
from model import PlagiarismDetector

def evaluate(args):
    # 1) Charger le dataset de test
    test_ds = ArabicPlagiarismCSVDataset(args.test_csv, max_len=args.max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 2) Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlagiarismDetector().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 3) Boucle d'inférence
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            s_ids  = batch['s_ids'].to(device)
            s_mask = batch['s_mask'].to(device)
            r_ids  = batch['r_ids'].to(device)
            r_mask = batch['r_mask'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            logits = model(s_ids, s_mask, r_ids, r_mask)
            prob   = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(prob)
            trues.extend(labels.cpu().numpy().flatten())

    # 4) Calcul des métriques
    bin_preds = [1 if p >= args.threshold else 0 for p in preds]
    acc = accuracy_score(trues, bin_preds)
    f1  = f1_score(trues, bin_preds)
    auc = roc_auc_score(trues, preds) if len(set(trues)) > 1 else float('nan')

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test F1-score : {f1:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Évalue le détecteur sur test.csv")
    parser.add_argument('--test_csv',   required=True, help="Chemin vers data/test.csv")
    parser.add_argument('--model_path', required=True, help="Chemin vers best_model.pth")
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--max_len',    type=int,   default=128)
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help="Seuil pour transformer prob en label binaire")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
