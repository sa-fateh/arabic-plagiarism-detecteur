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
    cv_folds: int = None
):
    """
    1) Extrait TOUTES les annotations positives depuis les XML (sans filtrage de longueur).
    2) (opt) Augmente par shuffle.
    3) Génère négatifs cross-doc au ratio `neg_ratio`.
    4) Split train/val/test sur un identifiant unique (doc+offset) → zéro fuite.
    5) (opt) Génère CV folds si demandé.
    """

    # 1) Collecte positives
    records = []
    ann_sources = defaultdict(set)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f">>> {len(xml_files)} XML trouvés dans {xml_dir}")

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

    # 2) Dé-duplication offset-level (évite deux fois la même feature)
    df = df.drop_duplicates(subset=[
        "suspicious_reference",
        "source_reference",
        "this_offset",
        "source_offset"
    ])

    # 3) Augmentation (shuffle) si activée
    if augment:
        aug_recs = []
        for _, row in df.iterrows():
            ws, wr = row.suspicious_text.split(), row.source_text.split()
            random.shuffle(ws)
            random.shuffle(wr)
            rec = row.to_dict()
            rec["suspicious_text"] = " ".join(ws)
            rec["source_text"]     = " ".join(wr)
            aug_recs.append(rec)
        if aug_recs:
            df = pd.concat([df, pd.DataFrame(aug_recs)], ignore_index=True)
            print(f"→ Après augmentation : {len(df)}")

    # 4) Génération de négatifs cross-doc
    all_srcs = glob.glob(os.path.join(src_dir, "*.txt"))
    negs = []
    for susp, grp in df.groupby("suspicious_reference"):
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
        print(f"→ Total après négatifs : {len(df)} (neg={len(negs)})")

    # 5) Construction d'un example_id unique par fragment
    df["example_id"] = (
        df["suspicious_reference"] + "__" +
        df["source_reference"]     + "__" +
        df["this_offset"].astype(str) + "__" +
        df["source_offset"].astype(str)
    )

    # 6) Split train vs (val+test) sur example_id
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    tr_idx, tmp_idx = next(splitter.split(df, groups=df["example_id"]))
    tr_df, tmp_df  = df.iloc[tr_idx], df.iloc[tmp_idx]

    # 7) Split val vs test (50/50) sur example_id
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=random_state
    )
    v_idx, te_idx = next(splitter2.split(tmp_df, groups=tmp_df["example_id"]))
    val_df, te_df = tmp_df.iloc[v_idx], tmp_df.iloc[te_idx]

    # 8) Assertions anti-fuite
    assert not set(tr_df.example_id) & set(val_df.example_id),  "Fuite train→val !"
    assert not set(tr_df.example_id) & set(te_df.example_id),  "Fuite train→test !"
    assert not set(val_df.example_id) & set(te_df.example_id), "Fuite val→test !"

    # 9) Sauvegarde des CSV
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [tr_df, val_df, te_df]):
        path = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(path, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(subset)} exemples")
        paths[name] = path

    # 10) Génération optionnelle de CV folds
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
