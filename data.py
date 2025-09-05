# data.py

import os
import glob
import random
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

from preprocess import extract_fragment

def build_dataset(
    xml_dir: str,
    susp_dir: str,
    src_dir: str,
    out_dir: str,
    augment: bool = False,
    neg_length: int = 50,
    neg_ratio: float = 2.0,
    slide_per_pos: int = 2,
    test_size: float = 0.30,
    random_state: int = 42
):
    """
    1) Extraire TOUTES les paires annotées (positives)
    2) Optionnel shuffle des positifs
    3) Générer des négatifs cross-doc
    4) Générer des hard‐negatives Sliding‐Window intra‐doc
    5) Split train/val/test par suspicious_reference (pas de fuite)
    6) Sauvegarder train.csv, val.csv, test.csv
    """

    # 1) Collecte des positifs
    records = []
    ann_sources = defaultdict(set)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f">>> {len(xml_files)} fichiers XML trouvés")

    for xml_fp in xml_files:
        root = ET.parse(xml_fp).getroot()
        susp_name = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_path = os.path.join(susp_dir, susp_name)

        for feat in root.findall("feature"):
            src_ref = feat.get("source_reference")
            ann_sources[susp_name].add(src_ref)

            to = int(feat.get("this_offset"))
            tl = int(feat.get("this_length"))
            so = int(feat.get("source_offset"))
            sl = int(feat.get("source_length"))

            s_frag = extract_fragment(susp_path, to, tl)
            r_frag = extract_fragment(os.path.join(src_dir, src_ref), so, sl)
            if s_frag is None or r_frag is None:
                continue

            records.append({
                "suspicious_reference": susp_name,
                "source_reference":     src_ref,
                "this_offset":          to,
                "this_length":          tl,
                "source_offset":        so,
                "source_length":        sl,
                "suspicious_text":      s_frag,
                "source_text":          r_frag,
                "label":                1
            })

    df = pd.DataFrame(records)
    print(f"→ Positifs extraits : {len(df)}")

    # 2) Augmentation faculative (shuffle mots dans chaque fragment)
    if augment and not df.empty:
        aug = []
        for _, row in df.iterrows():
            ws, wr = row.suspicious_text.split(), row.source_text.split()
            random.shuffle(ws); random.shuffle(wr)
            rec = row.to_dict()
            rec["suspicious_text"] = " ".join(ws)
            rec["source_text"]     = " ".join(wr)
            aug.append(rec)
        df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
        print(f"→ Après augmentation : {len(df)}")

    # 3) Négatifs cross-doc
    all_srcs = glob.glob(os.path.join(src_dir, "*.txt"))
    cross_negs = []
    for susp, grp in df[df.label == 1].groupby("suspicious_reference"):
        count = len(grp)
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**6)
        if s_full is None or len(s_full) < neg_length:
            continue

        candidates = [
            p for p in all_srcs
            if os.path.basename(p) not in ann_sources[susp]
        ]
        for _ in range(int(count * neg_ratio)):
            src_fp = random.choice(candidates)
            r_full = extract_fragment(src_fp, 0, 10**6)
            if r_full is None or len(r_full) < neg_length:
                continue
            i = random.randint(0, len(s_full) - neg_length)
            j = random.randint(0, len(r_full) - neg_length)
            cross_negs.append({
                "suspicious_reference": susp,
                "source_reference":     os.path.basename(src_fp),
                "this_offset":          i,
                "this_length":          neg_length,
                "source_offset":        j,
                "source_length":        neg_length,
                "suspicious_text":      s_full[i:i+neg_length],
                "source_text":          r_full[j:j+neg_length],
                "label":                0
            })

    df = pd.concat([df, pd.DataFrame(cross_negs)], ignore_index=True)
    print(f"→ Total après cross-doc négatifs : {len(df)} (ajoutés {len(cross_negs)})")

    # 4) Hard-negatives : sliding-window dans chaque suspicious document
    slide_negs = []
    for _, row in df[df.label == 1].iterrows():
        susp = row["suspicious_reference"]
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**6)
        if s_full is None or len(s_full) < neg_length:
            continue
        for _ in range(slide_per_pos):
            i = random.randint(0, len(s_full) - neg_length)
            seg = s_full[i : i + neg_length]
            slide_negs.append({
                "suspicious_reference": susp,
                "source_reference":     susp,
                "this_offset":          i,
                "this_length":          neg_length,
                "source_offset":        i,
                "source_length":        neg_length,
                "suspicious_text":      seg,
                "source_text":          seg,
                "label":                0
            })

    df = pd.concat([df, pd.DataFrame(slide_negs)], ignore_index=True)
    print(f"→ Hard-negatives sliding-window ajoutés : {len(slide_negs)} → total {len(df)}")

    # 5) Split TRAIN / VAL / TEST par suspicious_reference pour éviter toute fuite
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    tr_idx, tmp_idx = next(splitter.split(df, groups=df["suspicious_reference"]))
    tr_df, tmp_df   = df.iloc[tr_idx], df.iloc[tmp_idx]

    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=random_state
    )
    v_idx, te_idx   = next(splitter2.split(tmp_df, groups=tmp_df["suspicious_reference"]))
    val_df, te_df   = tmp_df.iloc[v_idx], tmp_df.iloc[te_idx]

    # Sanity-check : aucun document suspect n’apparaît dans deux splits
    assert not set(tr_df.suspicious_reference) & set(val_df.suspicious_reference), "Fuite train→val !"
    assert not set(tr_df.suspicious_reference) & set(te_df.suspicious_reference),  "Fuite train→test !"
    assert not set(val_df.suspicious_reference) & set(te_df.suspicious_reference), "Fuite val→test !"

    # 6) Sauvegarde des CSV
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [tr_df, val_df, te_df]):
        path = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(path, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(subset)} exemples")
        paths[name] = path

    return paths
