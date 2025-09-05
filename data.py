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
    test_size: float = 0.30,
    random_state: int = 42
):
    """
    1) Extrait TOUTES les paires annotées (positives) sans filtre de longueur.
    2) (opt) Augmente les positifs par shuffle.
    3) Génère neg_ratio×positifs en négatifs cross-doc.
    4) Split train/val/test sur un example_id unique (doc+offset) -> zéro fuite.
    """

    # 1) Collecte des positifs
    records = []
    ann_sources = defaultdict(set)
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
            if s_frag is None or r_frag is None:
                continue

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

    df = pd.DataFrame(records)
    print(f"→ Positifs extraits : {len(df)}")  # tu devrais voir ~601 si chaque XML a au moins 1 feature

    # 2) Optionnel : augmentation shuffle des positifs
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
            print(f"→ Après augmentation : {len(df)}")

    # 3) Négatifs cross-doc
    all_srcs = glob.glob(os.path.join(src_dir, "*.txt"))
    negs = []
    for susp, grp in df.groupby("suspicious_reference"):
        cnt    = len(grp)
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**6)
        if s_full is None or len(s_full) < neg_length:
            continue
        cands = [p for p in all_srcs if os.path.basename(p) not in ann_sources[susp]]
        for _ in range(int(cnt * neg_ratio)):
            src_fp = random.choice(cands)
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

    # 4) Zero-fuite split sur example_id (doc+offset)
    df["example_id"] = (
        df["suspicious_reference"] + "__" +
        df["source_reference"]     + "__" +
        df["this_offset"].astype(str) + "__" +
        df["source_offset"].astype(str)
    )

    gss  = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_i, tmp_i    = next(gss.split(df, groups=df["example_id"]))
    tr_df, tmp_df  = df.iloc[tr_i], df.iloc[tmp_i]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    v_i, te_i      = next(gss2.split(tmp_df, groups=tmp_df["example_id"]))
    val_df, te_df  = tmp_df.iloc[v_i], tmp_df.iloc[te_i]

    # Sanity checks
    assert not set(tr_df.example_id) & set(val_df.example_id), "Fuite train→val !"
    assert not set(tr_df.example_id) & set(te_df.example_id), "Fuite train→test !"
    assert not set(val_df.example_id) & set(te_df.example_id), "Fuite val→test !"

    # 5) Sauvegarde CSV
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [tr_df, val_df, te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(fp, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(subset)} exemples")
        paths[name] = fp

    return paths, []
