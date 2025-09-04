# data.py

import os
import glob
import random
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from preprocess import extract_fragment

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
    1) Extrait les paires positives depuis les XML
    2) Nettoie (doublons, longueur)
    3) Optionnellement shuffle les positifs
    4) Génère négatifs (exclut sources annotées)
    5) Dé-duplicate exact/inversé et split sur pair_group
    6) Sauvegarde train/val/test et CV folds
    """
    # A) Lecture des annotations positives
    records = []
    ann_sources = defaultdict(set)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f">>> {len(xml_files)} XML trouvés")

    for xml_fp in xml_files:
        root = ET.parse(xml_fp).getroot()
        susp_name = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_path = os.path.join(susp_dir, susp_name)
        for feat in root.findall("feature"):
            src_ref = feat.get("source_reference")
            ann_sources[susp_name].add(src_ref)
            to, tl = int(feat.get("this_offset")), int(feat.get("this_length"))
            so, sl = int(feat.get("source_offset")), int(feat.get("source_length"))
            s_frag = extract_fragment(susp_path, to, tl)
            r_frag = extract_fragment(os.path.join(src_dir, src_ref), so, sl)
            if s_frag and r_frag:
                records.append({
                    "suspicious_reference": susp_name,
                    "this_offset": to,
                    "this_length": tl,
                    "source_reference": src_ref,
                    "source_offset": so,
                    "source_length": sl,
                    "suspicious_text": s_frag,
                    "source_text": r_frag,
                    "label": 1
                })

    # B) DataFrame et nettoyage initial
    df = pd.DataFrame(records)
    # B.1) Suppression des doublons exacts (mêmes offsets)
    df = df.drop_duplicates(subset=[
        "suspicious_reference",
        "source_reference",
        "this_offset",
        "source_offset"
    ])
    # B.2) Filtre sur la longueur
    df = df[df["this_length"].between(min_len, max_len)]
    print(f"→ Après nettoyage : {len(df)} exemples (pos={df.label.sum()})")

    # C) Augmentation par shuffle de mots (optionnel)
    if augment:
        aug_recs = []
        for _, row in df.iterrows():
            ws = row.suspicious_text.split()
            wr = row.source_text.split()
            random.shuffle(ws)
            random.shuffle(wr)
            rec = row.to_dict()
            rec["suspicious_text"] = " ".join(ws)
            rec["source_text"] = " ".join(wr)
            aug_recs.append(rec)
        if aug_recs:
            df = pd.concat([df, pd.DataFrame(aug_recs)], ignore_index=True)
            print(f"→ Après augmentation : {len(df)} exemples")

    # D) Génération des exemples négatifs
    all_srcs = glob.glob(os.path.join(src_dir, "*.txt"))
    negs = []
    pos_counts = df[df.label == 1].groupby("suspicious_reference").size().to_dict()
    for susp, n_pos in pos_counts.items():
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**6)
        if len(s_full) < neg_length:
            continue
        candidates = [
            p for p in all_srcs
            if os.path.basename(p) not in ann_sources[susp]
        ]
        for _ in range(int(n_pos * neg_ratio)):
            src_fp = random.choice(candidates)
            r_full = extract_fragment(src_fp, 0, 10**6)
            if len(r_full) < neg_length:
                continue
            i = random.randint(0, len(s_full) - neg_length)
            j = random.randint(0, len(r_full) - neg_length)
            negs.append({
                "suspicious_reference": susp,
                "this_offset": i,
                "this_length": neg_length,
                "source_reference": os.path.basename(src_fp),
                "source_offset": j,
                "source_length": neg_length,
                "suspicious_text": s_full[i:i+neg_length],
                "source_text": r_full[j:j+neg_length],
                "label": 0
            })

    if negs:
        df = pd.concat([df, pd.DataFrame(negs)], ignore_index=True)
        print(f"→ Total exemples après négatifs : {len(df)} (neg={len(negs)})")

    # E) Création du groupe 'pair_group' pour le split
    df["pair_group"] = (
        df["suspicious_reference"] + "__" + df["source_reference"]
    )

    # F) Split train vs (val+test) sur pair_group
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_i, tmp_i = next(gss.split(df, groups=df["pair_group"]))
    tr_df, tmp_df = df.iloc[tr_i], df.iloc[tmp_i]

    # G) Split val vs test (50/50) sur le même pair_group
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    v_i, te_i = next(gss2.split(tmp_df, groups=tmp_df["pair_group"]))
    val_df, te_df = tmp_df.iloc[v_i], tmp_df.iloc[te_i]

    # H) Sauvegarde des CSV
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [tr_df, val_df, te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(fp, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(subset)} exemples")
        paths[name] = fp

    # I) Cross-validation optionnelle sur pair_group
    cv_splits = []
    if cv_folds and cv_folds > 1:
        gkf = GroupKFold(n_splits=cv_folds)
        for fold, (i_tr, i_va) in enumerate(
            gkf.split(df, groups=df["pair_group"]), start=1
        ):
            d = os.path.join(out_dir, f"cv_{fold}")
            os.makedirs(d, exist_ok=True)
            df.iloc[i_tr].to_csv(os.path.join(d, "train.csv"), index=False)
            df.iloc[i_va].to_csv(os.path.join(d, "val.csv"),   index=False)
            cv_splits.append(d)
        print(f"✅ {cv_folds}-fold CV générés")

    return paths, cv_splits
