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
    # 1) Lire positifs
    records = []
    ann = defaultdict(set)
    for xml_fp in glob.glob(os.path.join(xml_dir, "*.xml")):
        tree = ET.parse(xml_fp)
        root = tree.getroot()
        susp = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_path = os.path.join(susp_dir, susp)
        for feat in root.findall("feature"):
            src_ref = feat.get("source_reference")
            ann[susp].add(src_ref)
            to, tl = int(feat.get("this_offset")), int(feat.get("this_length"))
            so, sl = int(feat.get("source_offset")), int(feat.get("source_length"))
            s_frag = extract_fragment(susp_path, to, tl)
            r_frag = extract_fragment(os.path.join(src_dir, src_ref), so, sl)
            if s_frag and r_frag:
                records.append({
                    "suspicious_reference": susp,
                    "source_reference":     src_ref,
                    "this_offset":          to,
                    "this_length":          tl,
                    "source_offset":        so,
                    "source_length":        sl,
                    "suspicious_text":      s_frag,
                    "source_text":          r_frag,
                    "label":                1
                })

    # 2) DataFrame + nettoyage initial
    df = pd.DataFrame(records)
    # 2.a) enlever doublons EXACTS
    df = df.drop_duplicates(subset=[
        "suspicious_reference",
        "source_reference",
        "this_offset",
        "source_offset"
    ])
    # 2.b) filtrer les lengths
    df = df[df["this_length"].between(min_len, max_len)]
    print(f"→ Positifs nettoyés: {len(df)}")

    # 3) (opt) augmentation shuffle
    if augment:
        aug = []
        for _, r in df.iterrows():
            ws, wr = r.suspicious_text.split(), r.source_text.split()
            random.shuffle(ws); random.shuffle(wr)
            rec = r.to_dict()
            rec["suspicious_text"] = " ".join(ws)
            rec["source_text"]     = " ".join(wr)
            aug.append(rec)
        if aug:
            df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
            print(f"→ Après aug. : {len(df)}")

    # 4) construire NÉGATIFS
    all_srcs = glob.glob(os.path.join(src_dir, "*.txt"))
    negs = []
    pos_grp = df.groupby("suspicious_reference")
    for susp, grp in pos_grp:
        count = len(grp)
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**6)
        if len(s_full) < neg_length:
            continue
        cands = [p for p in all_srcs if os.path.basename(p) not in ann[susp]]
        for _ in range(int(count * neg_ratio)):
            src_fp = random.choice(cands)
            r_full = extract_fragment(src_fp, 0, 10**6)
            if len(r_full) < neg_length:
                continue
            i = random.randint(0, len(s_full) - neg_length)
            j = random.randint(0, len(r_full) - neg_length)
            negs.append({
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

    if negs:
        df = pd.concat([df, pd.DataFrame(negs)], ignore_index=True)
        print(f"→ Total après négatifs : {len(df)}")

    # 5) Groupe ULTRA-FIN = référence+offset
    df["example_id"] = (
        df["suspicious_reference"] + "__" +
        df["source_reference"]    + "__" +
        df["this_offset"].astype(str) + "__" +
        df["source_offset"].astype(str)
    )

    # 6) Split train vs (val+test) sur example_id
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    tr_i, tmp_i = next(splitter.split(df, groups=df["example_id"]))
    tr_df, tmp_df = df.iloc[tr_i], df.iloc[tmp_i]

    # 7) Split val vs test (50/50) sur example_id
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=random_state
    )
    v_i, te_i = next(splitter2.split(tmp_df, groups=tmp_df["example_id"]))
    val_df, te_df = tmp_df.iloc[v_i], tmp_df.iloc[te_i]

    # 8) Sanity‐check anti‐fuite
    train_ids = set(tr_df["example_id"])
    val_ids   = set(val_df["example_id"])
    test_ids  = set(te_df["example_id"])
    assert not (train_ids & val_ids),  "Fuite train→val !"
    assert not (train_ids & test_ids), "Fuite train→test !"
    assert not (val_ids   & test_ids), "Fuite val→test !"

    # 9) Sauvegarde CSV
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [tr_df,val_df,te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(fp, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(subset)}")
        paths[name] = fp

    # 10) Option CV
    cv_splits = []
    if cv_folds and cv_folds > 1:
        gkf = GroupKFold(n_splits=cv_folds)
        for fold, (i_tr, i_va) in enumerate(
            gkf.split(df, groups=df["example_id"]), start=1
        ):
            d = os.path.join(out_dir, f"cv_{fold}")
            os.makedirs(d, exist_ok=True)
            df.iloc[i_tr].to_csv(os.path.join(d, "train.csv"), index=False)
            df.iloc[i_va].to_csv(os.path.join(d, "val.csv"),   index=False)
            cv_splits.append(d)
        print(f"✅ {cv_folds}-fold CV générés")
    return paths, cv_splits
