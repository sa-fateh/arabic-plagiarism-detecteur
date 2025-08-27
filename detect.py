# detect.py

import argparse
import os
import csv
import torch
from tqdm import tqdm
from model import PlagiarismDetector
from preprocess import extract_fragment
from transformers import AutoTokenizer

def detect(args):
    # 1) Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlagiarismDetector().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 2) Initialiser le tokenizer (même que dans dataset)
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")

    # 3) Parcourir les docs suspects
    results = []
    susp_files = sorted(os.listdir(args.susp_dir))
    src_files  = sorted(os.listdir(args.src_dir))

    for susp_name in susp_files:
        susp_path = os.path.join(args.susp_dir, susp_name)
        # On lit tout le texte proprement
        full_susp = extract_fragment(susp_path, 0, 10**6)

        # Balayage glissant sur le doc suspect
        for i in range(0, len(full_susp) - args.window_len + 1, args.stride):
            s_frag = full_susp[i : i + args.window_len]

            # Pour chaque source, on teste les mêmes fenêtres
            for src_name in src_files:
                src_path = os.path.join(args.src_dir, src_name)
                full_src = extract_fragment(src_path, 0, 10**6)
                if len(full_src) < args.window_len:
                    continue

                for j in range(0, len(full_src) - args.window_len + 1, args.stride):
                    r_frag = full_src[j : j + args.window_len]

                    # Tokenisation
                    se = tokenizer(s_frag,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=args.max_len,
                                   return_tensors='pt')
                    re = tokenizer(r_frag,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=args.max_len,
                                   return_tensors='pt')

                    # Inférence
                    with torch.no_grad():
                        logits = model(
                            se['input_ids'].to(device),
                            se['attention_mask'].to(device),
                            re['input_ids'].to(device),
                            re['attention_mask'].to(device)
                        )
                        prob = torch.sigmoid(logits).item()

                    # Si au-dessus du seuil, on enregistre
                    if prob >= args.threshold:
                        results.append({
                            'suspicious_reference': susp_name,
                            'this_offset': i,
                            'this_length': args.window_len,
                            'source_reference': src_name,
                            'source_offset': j,
                            'source_length': args.window_len,
                            'probability': prob
                        })

    # 4) Sauvegarder les résultats
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'suspicious_reference',
            'this_offset','this_length',
            'source_reference',
            'source_offset','source_length',
            'probability'
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Détection terminée ({len(results)} fragments ≥ seuil {args.threshold})")
    print(f"→ Résultats dans : {args.out_csv}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Détecte le plagiat en fenêtres glissantes"
    )
    parser.add_argument('--susp_dir',    required=True,
                        help="Dossier des documents suspects (.txt)")
    parser.add_argument('--src_dir',     required=True,
                        help="Dossier des documents sources (.txt)")
    parser.add_argument('--model_path',  required=True,
                        help="Chemin vers best_model.pth")
    parser.add_argument('--out_csv',     default="detection.csv",
                        help="Fichier de sortie (CSV)")
    parser.add_argument('--window_len',  type=int, default=50,
                        help="Taille de la fenêtre (caractères)")
    parser.add_argument('--stride',      type=int, default=25,
                        help="Pas de la fenêtre (caractères)")
    parser.add_argument('--max_len',     type=int, default=128,
                        help="Longueur max de tokenisation")
    parser.add_argument('--threshold',   type=float, default=0.5,
                        help="Seuil de décision sur la probabilité")
    return parser.parse_args()

if __name__ == '__main__':
    detect(parse_args())
