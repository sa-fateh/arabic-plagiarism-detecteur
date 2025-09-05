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
    augment: bool = False,            # on garde le paramètre attendu par train.py
    neg_length: int = 50,
    neg_ratio: float = 1.0,
    slide_per_pos: int = 2,
    noise_deletion_frac: float = 0.1,
    test_size: float = 0.30,
    random_state: int = 42,
    cv_folds: int = None             # si train.py passait aussi ce paramètre
):
    """
    1) EXTRACTION POSITIFS  (tous les <feature>)
    2) NÉGATIFS CROSS-DOC
    3) HARD-NEG SLIDING-WINDOW
    4) NEAR-POSITIVE NOISE
    5) SPLIT PAR PAIRE docSuspect__docSource
    6) EXPORT CSV
    """

    # 1) paires positives
    records, ann = [], defaultdict(set)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f">>> {len(xml_files)} XML analysés")
    for xml_fp in xml_files:
        root      = ET.parse(xml_fp).getroot()
        susp_name = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_full = extract_fragment(os.path.join(susp_dir, susp_name), 0, 10**7) or ""
        for feat in root.findall("feature"):
            src_ref = feat.get("source_reference")
            ann[susp_name].add(src_ref)
            to, tl = int(feat.get("this_offset")), int(feat.get("this_length"))
            so, sl = int(feat.get("source_offset")), int(feat.get("source_length"))
            s_frag   = susp_full[to: to+tl]
            src_full = extract_fragment(os.path.join(src_dir, src_ref), 0, 10**7) or ""
            r_frag   = src_full[so: so+sl]
            if not s_frag or not r_frag:
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

    # 2) Négatifs cross-doc
    all_srcs = glob.glob(os.path.join(src_dir, "*.txt"))
    cross_negs = []
    for susp, grp in df.groupby("suspicious_reference"):
        susp_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**7) or ""
        candidates = [p for p in all_srcs if os.path.basename(p) not in ann[susp]]
        for _ in range(int(len(grp) * neg_ratio)):
            if len(susp_full) < neg_length or not candidates:
                break
            src_fp   = random.choice(candidates)
            src_full = extract_fragment(src_fp, 0, 10**7) or ""
            if len(src_full) < neg_length:
                continue
            si = random.randint(0, len(susp_full) - neg_length)
            ri = random.randint(0, len(src_full) - neg_length)
            cross_negs.append({
                "suspicious_reference": susp,
                "source_reference":     os.path.basename(src_fp),
                "this_offset":          si,
                "this_length":          neg_length,
                "source_offset":        ri,
                "source_length":        neg_length,
                "suspicious_text":      susp_full[si: si+neg_length],
                "source_text":          src_full[ri: ri+neg_length],
                "label":                0
            })

    df = pd.concat([df, pd.DataFrame(cross_negs)], ignore_index=True)
    print(f"→ +{len(cross_negs)} cross-doc negs → {len(df)} tot.")

    # 3) Hard-negatives sliding-window (deux segments distincts)
    slide_negs = []
    for _, pos in df[df.label == 1].iterrows():
        susp_full = extract_fragment(os.path.join(susp_dir, pos.suspicious_reference), 0, 10**7) or ""
        if len(susp_full) < 2 * neg_length:
            continue
        for _ in range(slide_per_pos):
            i = random.randint(0, len(susp_full) - neg_length)
            j = random.randint(0, len(susp_full) - neg_length)
            if abs(i - j) < neg_length:
                j = (i + neg_length) % (len(susp_full) - neg_length)
            slide_negs.append({
                "suspicious_reference": pos.suspicious_reference,
                "source_reference":     pos.suspicious_reference,
                "this_offset":          i,
                "this_length":          neg_length,
                "source_offset":        j,
                "source_length":        neg_length,
                "suspicious_text":      susp_full[i: i+neg_length],
                "source_text":          susp_full[j: j+neg_length],
                "label":                0
            })

    df = pd.concat([df, pd.DataFrame(slide_negs)], ignore_index=True)
    print(f"→ +{len(slide_negs)} sliding negs → {len(df)} tot.")

    # 4) Near-positives “noise deletion”
    noise_negs = []
    for _, pos in df[df.label == 1].iterrows():
        words    = pos.suspicious_text.split()
        k        = max(1, int(len(words) * noise_deletion_frac))
        drop_idx = set(random.sample(range(len(words)), k))
        noisy    = " ".join(w for i,w in enumerate(words) if i not in drop_idx)
        rec      = pos.to_dict()
        rec["suspicious_text"] = noisy
        rec["label"]           = 0
        noise_negs.append(rec)

    df = pd.concat([df, pd.DataFrame(noise_negs)], ignore_index=True)
    print(f"→ +{len(noise_negs)} noise-negs → {len(df)} tot.")

    # 5) Split **par paire** docSuspect__docSource (zéro fuite)
    df["pair_doc"] = df["suspicious_reference"] + "__" + df["source_reference"]
    gss  = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_i, tmp_i   = next(gss.split(df, groups=df["pair_doc"]))
    tr_df, tmp_df = df.iloc[tr_i], df.iloc[tmp_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    v_i, te_i     = next(gss2.split(tmp_df, groups=tmp_df["pair_doc"]))
    val_df, te_df = tmp_df.iloc[v_i], tmp_df.iloc[te_i]

    assert not set(tr_df.pair_doc) & set(val_df.pair_doc),  "Fuite train→val !"
    assert not set(tr_df.pair_doc) & set(te_df.pair_doc),   "Fuite train→test !"
    assert not set(val_df.pair_doc) & set(te_df.pair_doc),  "Fuite val→test !"

    # 6) Sauvegarde CSV
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, sub in zip(["train","val","test"], [tr_df,val_df,te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        sub.to_csv(fp, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(sub)} exemples")
        paths[name] = fp

    return paths
