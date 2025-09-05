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
    neg_ratio: float = 1.0,
    slide_per_pos: int = 2,
    test_size: float = 0.30,
    random_state: int = 42
):
    # 1) Extraction de toutes les paires positives
    records = []
    ann_sources = defaultdict(set)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f">>> {len(xml_files)} XML analysés")
    for xml_fp in xml_files:
        root = ET.parse(xml_fp).getroot()
        susp = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_path = os.path.join(susp_dir, susp)

        # charger le document suspect complet une seule fois
        s_full = extract_fragment(susp_path, 0, 10**7) or ""
        for feat in root.findall("feature"):
            sr = feat.get("source_reference")
            ann_sources[susp].add(sr)
            to, tl = int(feat.get("this_offset")), int(feat.get("this_length"))
            so, sl = int(feat.get("source_offset")), int(feat.get("source_length"))
            s_frag = s_full[to: to+tl]
            r_full = extract_fragment(os.path.join(src_dir, sr), 0, 10**7) or ""
            r_frag = r_full[so: so+sl]
            if not s_frag or not r_frag:
                continue
            records.append({
                "suspicious_reference": susp,
                "source_reference":     sr,
                "this_offset":          to,
                "this_length":          tl,
                "source_offset":        so,
                "source_length":        sl,
                "suspicious_text":      s_frag,
                "source_text":          r_frag,
                "label":                1
            })

    df = pd.DataFrame(records)
    print(f"→ {len(df)} positifs extraits")

    # 2) (Optionnel) augmentation shuffle des positifs
    if augment and not df.empty:
        aug = []
        for _, r in df.iterrows():
            ws, wr = r.suspicious_text.split(), r.source_text.split()
            random.shuffle(ws); random.shuffle(wr)
            rec = r.to_dict()
            rec["suspicious_text"] = " ".join(ws)
            rec["source_text"]     = " ".join(wr)
            aug.append(rec)
        df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
        print(f"→ Après augmentation : {len(df)}")

    # 3) Négatifs cross-doc : on échantillonne un segment ALÉATOIRE du doc suspect
    all_srcs = glob.glob(os.path.join(src_dir, "*.txt"))
    negs = []
    for susp, grp in df[df.label == 1].groupby("suspicious_reference"):
        # recupérer le doc suspect complet
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**7) or ""
        # candidats source (hors ceux annotés)
        candidates = [p for p in all_srcs if os.path.basename(p) not in ann_sources[susp]]
        for _ in range(int(len(grp) * neg_ratio)):
            if len(s_full) < neg_length or not candidates:
                break
            src_fp = random.choice(candidates)
            r_full = extract_fragment(src_fp, 0, 10**7) or ""
            if len(r_full) < neg_length:
                continue

            # choisir un segment aléatoire de s_full et de r_full
            si = random.randint(0, len(s_full) - neg_length)
            ri = random.randint(0, len(r_full) - neg_length)
            negs.append({
                "suspicious_reference": susp,
                "source_reference":     os.path.basename(src_fp),
                "this_offset":          si,
                "this_length":          neg_length,
                "source_offset":        ri,
                "source_length":        neg_length,
                "suspicious_text":      s_full[si: si+neg_length],
                "source_text":          r_full[ri: ri+neg_length],
                "label":                0
            })

    df = pd.concat([df, pd.DataFrame(negs)], ignore_index=True)
    print(f"→ Total après cross-doc negs : {len(df)} (ajoutés {len(negs)})")

    # 4) Hard-negatives sliding-window : deux fenêtres DÉCALÉES
    slide_negs = []
    for _, pos in df[df.label == 1].iterrows():
        susp = pos.suspicious_reference
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**7) or ""
        if len(s_full) < 2 * neg_length:
            continue
        for _ in range(slide_per_pos):
            i = random.randint(0, len(s_full) - neg_length)
            j = random.randint(0, len(s_full) - neg_length)
            if abs(i - j) < neg_length:
                j = (i + neg_length) % (len(s_full) - neg_length)
            slide_negs.append({
                "suspicious_reference": susp,
                "source_reference":     susp,
                "this_offset":          i,
                "this_length":          neg_length,
                "source_offset":        j,
                "source_length":        neg_length,
                "suspicious_text":      s_full[i: i+neg_length],
                "source_text":          s_full[j: j+neg_length],
                "label":                0
            })

    df = pd.concat([df, pd.DataFrame(slide_negs)], ignore_index=True)
    print(f"→ Hard-negs sliding-window ajoutés : {len(slide_negs)} → total {len(df)}")

    # 5) SPLIT TRAIN / VAL / TEST par document suspect (ZÉRO FUITE)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    ti, tmp = next(splitter.split(df, groups=df["suspicious_reference"]))
    tr_df, tmp_df = df.iloc[ti], df.iloc[tmp]

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=random_state
    )
    vi, tsi = next(splitter2.split(tmp_df, groups=tmp_df["suspicious_reference"]))
    val_df, te_df = tmp_df.iloc[vi], tmp_df.iloc[tsi]

    # sanity-check anti-fuite
    assert not set(tr_df.suspicious_reference) & set(val_df.suspicious_reference),  "Fuite train→val !"
    assert not set(tr_df.suspicious_reference) & set(te_df.suspicious_reference),   "Fuite train→test !"
    assert not set(val_df.suspicious_reference) & set(te_df.suspicious_reference),  "Fuite val→test !"

    # 6) SAUVEGARDE DES CSV
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [tr_df, val_df, te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(fp, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(subset)} exemples")
        paths[name] = fp

    return paths
