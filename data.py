# data.py

"""
Construction du dataset CSV pour la détection de plagiat arabe,
avec split par documents, augmentations et option de cross-validation.
"""

import os
import glob
import random
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from preprocess import extract_fragment

def build_dataset(
    xml_dir: str,
    susp_dir: str,
    src_dir: str,
    out_dir: str,
    augment: bool = True,
    neg_length: int = 50,
    test_size: float = 0.30,
    random_state: int = 42,
    cv_folds: int = None
):
    """
    Construit et exporte :
      - train.csv, val.csv, test.csv  (split par document)
      - Optionnellement folds pour cross-validation (train_i.csv / val_i.csv)

    Args:
        xml_dir     : dossier PAN XML (features annotées)
        susp_dir    : dossier des textes suspects (.txt)
        src_dir     : dossier des textes source (.txt)
        out_dir     : dossier de sortie pour les CSV
        augment     : active shuffle de mots pour doublement des positifs
        neg_length  : taille fixe (en caractères) des exemples négatifs
        test_size   : fraction (val+test) pour GroupShuffleSplit
        random_state: graine pour reproductibilité
        cv_folds    : si int>1, génère cross-validation par groupe

    Returns:
        (train_csv, val_csv, test_csv) ou list de tuples si cv_folds.
    """

    # 1) Extraire paires positives depuis chaque XML
    records = []
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f">>> {len(xml_files)} fichiers XML trouvés dans {xml_dir}")

    for xml_fp in xml_files:
        tree = ET.parse(xml_fp)
        root = tree.getroot()
        susp_name = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_path = os.path.join(susp_dir, susp_name)

        for feat in root.findall("feature"):
            to   = int(feat.get("this_offset"))
            tl   = int(feat.get("this_length"))
            src_ref = feat.get("source_reference")
            so   = int(feat.get("source_offset"))
            sl   = int(feat.get("source_length"))
            src_path = os.path.join(src_dir, src_ref)

            s_frag = extract_fragment(susp_path, to, tl)
            r_frag = extract_fragment(src_path, so, sl)
            if not s_frag or not r_frag:
                continue

            records.append({
                "suspicious_reference": susp_name,
                "this_offset": to,
                "this_length": tl,
                "source_reference": src_ref,
                "source_offset": so,
                "source_length": sl,
                "suspicious_text": s_frag,
                "source_text":    r_frag,
                "label": 1
            })

    # 2) Data augmentation : shuffle de mots pour positifs
    if augment:
        augmented = []
        for rec in records:
            ws = rec["suspicious_text"].split()
            wr = rec["source_text"].split()
            random.shuffle(ws)
            random.shuffle(wr)
            aug = rec.copy()
            aug["suspicious_text"] = " ".join(ws)
            aug["source_text"]     = " ".join(wr)
            augmented.append(aug)
        records.extend(augmented)

    # 3) Génération de négatifs
    susp_paths = glob.glob(os.path.join(susp_dir, "*.txt"))
    src_paths  = glob.glob(os.path.join(src_dir,  "*.txt"))
    pos_count  = sum(1 for r in records if r["label"] == 1)

    for _ in range(pos_count):
        sp = random.choice(susp_paths)
        rp = random.choice(src_paths)
        s_full = extract_fragment(sp, 0, 10**6)
        r_full = extract_fragment(rp, 0, 10**6)
        if len(s_full) < neg_length or len(r_full) < neg_length:
            continue

        i = random.randint(0, len(s_full) - neg_length)
        j = random.randint(0, len(r_full) - neg_length)
        records.append({
            "suspicious_reference": os.path.basename(sp),
            "this_offset": i,
            "this_length": neg_length,
            "source_reference": os.path.basename(rp),
            "source_offset": j,
            "source_length": neg_length,
            "suspicious_text": s_full[i:i+neg_length],
            "source_text":    r_full[j:j+neg_length],
            "label": 0
        })

    # 4) Passage en DataFrame
    df = pd.DataFrame(records).reset_index(drop=True)
    print(f"→ total examples avant split: {len(df)} (pos={df.label.sum()}, neg={len(df)-df.label.sum()})")

    # 5) Split train / val / test par document – pas de fuite
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["suspicious_reference"]
    train_idx, temp_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df  = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    groups_temp = temp_df["suspicious_reference"]
    val_idx, test_idx = next(gss2.split(temp_df, groups=groups_temp))
    val_df   = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df  = temp_df.iloc[test_idx].reset_index(drop=True)

    # 6) Sauvegarde des CSV standards
    os.makedirs(out_dir, exist_ok=True)
    train_csv = os.path.join(out_dir, "train.csv")
    val_csv   = os.path.join(out_dir, "val.csv")
    test_csv  = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv,     index=False, encoding="utf-8")
    test_df.to_csv(test_csv,   index=False, encoding="utf-8")
    print(f"✅ Dataset prêt → train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    # 7) Option cross-validation (par document)
    cv_splits = []
    if cv_folds and cv_folds > 1:
        print(f">>> Génération de {cv_folds}-fold CV par documents")
        gkf = GroupKFold(n_splits=cv_folds)
        for fold, (tr_idx, vl_idx) in enumerate(gkf.split(df, groups=df["suspicious_reference"])):
            fold_dir = os.path.join(out_dir, f"cv_fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)

            df.iloc[tr_idx].reset_index(drop=True).to_csv(
                os.path.join(fold_dir, "train.csv"), index=False, encoding="utf-8"
            )
            df.iloc[vl_idx].reset_index(drop=True).to_csv(
                os.path.join(fold_dir, "val.csv"),   index=False, encoding="utf-8"
            )
            cv_splits.append((fold_dir, tr_idx, vl_idx))

        print(f"✅ {cv_folds}-fold CV prêt dans {out_dir}/cv_fold_*")

    # 8) Retourne les chemins CSV
    return (train_csv, val_csv, test_csv) if not cv_splits else {
        "standard": (train_csv, val_csv, test_csv),
        "cv": cv_splits
    }
