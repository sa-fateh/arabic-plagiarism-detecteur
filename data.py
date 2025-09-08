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
    neg_pool_dir: str = None,
    neg_length: int = 50,
    neg_ratio: float = 1.0,
    slide_per_pos: int = 2,
    noise_deletion_frac: float = 0.1,
    test_size: float = 0.30,
    random_state: int = 42
):
    # 1) POSITIVES
    records, ann = [], defaultdict(set)
    for xml_fp in glob.glob(os.path.join(xml_dir, "*.xml")):
        root      = ET.parse(xml_fp).getroot()
        susp_name = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_full = extract_fragment(os.path.join(susp_dir, susp_name), 0, 10**7) or ""
        for feat in root.findall("feature"):
            src_ref = feat.get("source_reference")
            ann[susp_name].add(src_ref)
            to, tl = int(feat.get("this_offset")), int(feat.get("this_length"))
            so, sl = int(feat.get("source_offset")), int(feat.get("source_length"))
            s_frag   = susp_full[to:to+tl]
            src_full = extract_fragment(os.path.join(src_dir, src_ref), 0, 10**7) or ""
            r_frag   = src_full[so:so+sl]
            if s_frag and r_frag:
                records.append({
                    "suspicious_reference": susp_name,
                    "source_reference":     src_ref,
                    "suspicious_text":      s_frag,
                    "source_text":          r_frag,
                    "label":                1
                })
    df = pd.DataFrame(records)

    # 2) NEGATIVE POOL = src_dir + neg_pool_dir
    pool = set(glob.glob(os.path.join(src_dir, "*.txt")))
    if neg_pool_dir:
        extra = glob.glob(os.path.join(neg_pool_dir, "**", "*.txt"), recursive=True)
        pool |= set(extra)
    pool = list(pool)

    # 3) CROSS-DOC NEGATIVES
    negs = []
    for susp, grp in df.groupby("suspicious_reference"):
        susp_full  = extract_fragment(os.path.join(susp_dir, susp), 0, 10**7) or ""
        candidates = [f for f in pool if os.path.basename(f) not in ann[susp]]
        for _ in range(int(len(grp) * neg_ratio)):
            if len(susp_full) < neg_length or not candidates:
                break
            src_fp   = random.choice(candidates)
            src_full = extract_fragment(src_fp, 0, 10**7) or ""
            if len(src_full) < neg_length:
                continue
            si = random.randint(0, len(susp_full) - neg_length)
            ri = random.randint(0, len(src_full)   - neg_length)
            negs.append({
                "suspicious_reference": susp,
                "source_reference":     os.path.basename(src_fp),
                "suspicious_text":      susp_full[si:si+neg_length],
                "source_text":          src_full[ri:ri+neg_length],
                "label":                0
            })
    df = pd.concat([df, pd.DataFrame(negs)], ignore_index=True)

    # 4) SLIDING-WINDOW HARD NEGATIVES
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
                "suspicious_text":      susp_full[i:i+neg_length],
                "source_text":          susp_full[j:j+neg_length],
                "label":                0
            })
    df = pd.concat([df, pd.DataFrame(slide_negs)], ignore_index=True)

    # 5) NOISE-DELETION NEAR-POSITIVES
    noise_negs = []
    for _, pos in df[df.label == 1].iterrows():
        words = pos.suspicious_text.split()
        k     = max(1, int(len(words) * noise_deletion_frac))
        drop  = set(random.sample(range(len(words)), k))
        noisy = " ".join(w for idx,w in enumerate(words) if idx not in drop)
        rec   = pos.to_dict()
        rec["suspicious_text"] = noisy
        rec["label"]           = 0
        noise_negs.append(rec)
    df = pd.concat([df, pd.DataFrame(noise_negs)], ignore_index=True)

    # 6) SPLIT train/val/test PAR DOCUMENT SUSPECT
    gss   = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_i, tmp_i   = next(gss.split(df, groups=df["suspicious_reference"]))
    tr_df, tmp_df = df.iloc[tr_i], df.iloc[tmp_i]
    gss2  = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    v_i, te_i     = next(gss2.split(tmp_df, groups=tmp_df["suspicious_reference"]))
    val_df, te_df = tmp_df.iloc[v_i], tmp_df.iloc[te_i]

    # 7) SAVE CSVs
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train","val","test"], [tr_df, val_df, te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(fp, index=False, encoding="utf-8")
        paths[name] = fp

    return paths
