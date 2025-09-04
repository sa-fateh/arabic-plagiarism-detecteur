import os
import glob
import random
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

from preprocess import extract_fragment  # Assurez-vous que cette fonction est définie

class ArabicPlagiarismCSVDataset(Dataset):
    """
    Charge un CSV de fragments plagiés/non-plagiés et renvoie un dict :
      - input_ids, attention_mask : tenseurs BERT pour suspect+source
      - label : float
      - métadonnées offsets & références
    """
    def __init__(self, csv_path: str, max_len: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Concat des deux textes avec [SEP]
        enc = self.tokenizer(
            row["suspicious_text"],
            row["source_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(row["label"], dtype=torch.float),
            "suspicious_reference": row["suspicious_reference"],
            "source_reference":     row["source_reference"],
            "this_offset":          int(row["this_offset"]),
            "this_length":          int(row["this_length"]),
            "source_offset":        int(row["source_offset"]),
            "source_length":        int(row["source_length"])
        }

def build_dataset(
    xml_dir: str,
    susp_dir: str,
    src_dir: str,
    out_dir: str,
    augment: bool = False,
    neg_length: int = 50,
    neg_ratio: float = 2.0,
    test_size: float = 0.30,
    random_state: int = 42,
    cv_folds: int = None,
    min_len: int = 10,
    max_len: int = 200
):
    """
    Extrait positifs, génère hard/soft négatifs, nettoie, split par document,
    sauvegarde train.csv, val.csv, test.csv (et cv_folds si demandé).
    Retourne dict paths={"train":..., "val":..., "test":...}, cv_splits=list.
    """
    # 1) Collecte positives + mapping susp→sources annotées
    records, ann_sources = [], defaultdict(set)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f">>> {len(xml_files)} XML trouvés dans {xml_dir}")

    for xml_fp in xml_files:
        root = ET.parse(xml_fp).getroot()
        susp = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_path = os.path.join(susp_dir, susp)
        for feat in root.findall("feature"):
            src_ref = feat.get("source_reference")
            ann_sources[susp].add(src_ref)
            to, tl = int(feat.get("this_offset")), int(feat.get("this_length"))
            so, sl = int(feat.get("source_offset")), int(feat.get("source_length"))
            s_frag = extract_fragment(susp_path, to, tl)
            r_frag = extract_fragment(os.path.join(src_dir, src_ref), so, sl)
            if s_frag and r_frag:
                records.append({
                    "suspicious_reference": susp,
                    "this_offset": to, "this_length": tl,
                    "source_reference": src_ref,
                    "source_offset": so, "source_length": sl,
                    "suspicious_text": s_frag,
                    "source_text": r_frag,
                    "label": 1
                })

    # 2) DataFrame & nettoyage
    df = pd.DataFrame(records)
    df = df[df["suspicious_reference"] != df["source_reference"]]
    df = df.drop_duplicates(subset=["suspicious_text", "source_text"])
    df = df[df["this_length"].between(min_len, max_len)]
    print(f"→ Après nettoyage : {len(df)} exemples (pos={df.label.sum()})")

    # 3) Augmentation (shuffle) si demandé
    if augment:
        aug = []
        for _, r in df.iterrows():
            ws, wr = r.suspicious_text.split(), r.source_text.split()
            random.shuffle(ws); random.shuffle(wr)
            aug.append({**r.to_dict(),
                        "suspicious_text":" ".join(ws),
                        "source_text":" ".join(wr)})
        df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
        print(f"→ Après augmentation : {len(df)} exemples")

    # 4) Génération négatifs (exclut sources annotées, ratio nég_ratio)
    all_src = glob.glob(os.path.join(src_dir, "*.txt"))
    negs = []
    pos_counts = df[df.label==1].groupby("suspicious_reference").size().to_dict()
    for susp, n_pos in pos_counts.items():
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**6)
        if len(s_full) < neg_length: continue
        candidates = [p for p in all_src if os.path.basename(p) not in ann_sources[susp]]
        for _ in range(int(n_pos * neg_ratio)):
            src_fp = random.choice(candidates)
            r_full = extract_fragment(src_fp, 0, 10**6)
            if len(r_full) < neg_length: continue
            i = random.randint(0, len(s_full)-neg_length)
            j = random.randint(0, len(r_full)-neg_length)
            negs.append({
                "suspicious_reference": susp,
                "this_offset": i, "this_length": neg_length,
                "source_reference": os.path.basename(src_fp),
                "source_offset": j, "source_length": neg_length,
                "suspicious_text": s_full[i:i+neg_length],
                "source_text": r_full[j:j+neg_length],
                "label": 0
            })
    df = pd.concat([df, pd.DataFrame(negs)], ignore_index=True)
    print(f"→ Total après négatifs : {len(df)} (neg={len(negs)})")

    # 5) Split train/val/test par document
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, tmp_idx = next(gss.split(df, groups=df["suspicious_reference"]))
    tr_df, tmp_df = df.iloc[tr_idx], df.iloc[tmp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    v_idx, te_idx = next(gss2.split(tmp_df, groups=tmp_df["suspicious_reference"]))
    val_df, te_df = tmp_df.iloc[v_idx], tmp_df.iloc[te_idx]

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, d in zip(["train","val","test"], [tr_df,val_df,te_df]):
        path = os.path.join(out_dir, f"{name}.csv")
        d.to_csv(path, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(d)} exemples")
        paths[name] = path

    # 6) Cross-validation (opt.)
    cv_splits = []
    if cv_folds and cv_folds>1:
        gkf = GroupKFold(n_splits=cv_folds)
        for fold,(i_tr,i_va) in enumerate(gkf.split(df, groups=df["suspicious_reference"]), 1):
            dir_fold = os.path.join(out_dir, f"cv_{fold}")
            os.makedirs(dir_fold, exist_ok=True)
            df.iloc[i_tr].to_csv(os.path.join(dir_fold,"train.csv"), index=False)
            df.iloc[i_va].to_csv(os.path.join(dir_fold,"val.csv"),   index=False)
            cv_splits.append(dir_fold)
        print(f"✅ {cv_folds}-folds CV générés")
    return paths, cv_splits
