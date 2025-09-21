# data.py

import os
import glob
import random
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import extract_fragment


def build_dataset(
    xml_dirs,
    susp_dirs,
    src_dirs,
    out_dir,
    neg_pool_dirs=None,
    neg_length=50,
    neg_ratio=2.0,
    slide_per_pos=3,
    noise_deletion_frac=0.1,
    test_size=0.3,
    random_state=42,
):
    records, ann = [], defaultdict(set)

    # 1) POSITIVE FRAGMENTS
    for cid, (xml_dir, susp_dir, src_dir) in enumerate(zip(xml_dirs, susp_dirs, src_dirs)):
        prefix = f"corpus{cid}"
        for xml_fp in glob.glob(os.path.join(xml_dir, "*.xml")):
            root = ET.parse(xml_fp).getroot()
            base = os.path.basename(xml_fp).replace(".xml", ".txt")
            susp_key = f"{prefix}_{base}"
            susp_full = extract_fragment(os.path.join(susp_dir, base), 0, 10**7) or ""
            for feat in root.findall("feature"):
                src_ref = feat.get("source_reference")
                src_key = f"{prefix}_{src_ref}"
                ann[susp_key].add(src_key)

                to, tl = int(feat.get("this_offset")), int(feat.get("this_length"))
                so, sl = int(feat.get("source_offset")), int(feat.get("source_length"))

                s_frag = susp_full[to : to + tl]
                src_full = extract_fragment(os.path.join(src_dir, src_ref), 0, 10**7) or ""
                r_frag = src_full[so : so + sl]

                if s_frag and r_frag:
                    records.append({
                        "suspicious_reference": susp_key,
                        "source_reference":     src_key,
                        "suspicious_text":      s_frag,
                        "source_text":          r_frag,
                        "label":                1
                    })

    df = pd.DataFrame(records)

    # 2) NEGATIVE POOL + TF-IDF
    pool = set()
    for d in src_dirs:
        pool |= set(glob.glob(os.path.join(d, "*.txt")))
    if neg_pool_dirs:
        for d in neg_pool_dirs:
            pool |= set(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
    pool = list(pool)
    pool_texts = [extract_fragment(p, 0, 10**7) or "" for p in pool]
    tfidf_pool = TfidfVectorizer(max_features=50000).fit(pool_texts)
    pool_vecs = tfidf_pool.transform(pool_texts)

    # 3) HARD NEGATIVES (TF-IDF)
    negs = []
    for susp_key, grp in df[df.label == 1].groupby("suspicious_reference"):
        cid = int(susp_key.split("_")[0].replace("corpus", ""))
        base = susp_key.split("_", 1)[1]
        susp_full = extract_fragment(
            os.path.join(susp_dirs[cid], base), 0, 10**7
        ) or ""
        q_vec = tfidf_pool.transform([susp_full])
        sims = (pool_vecs @ q_vec.T).toarray().ravel()
        for i, p in enumerate(pool):
            if f"corpus{cid}_{os.path.basename(p)}" in ann[susp_key]:
                sims[i] = -1
        top_idxs = sims.argsort()[-10:]
        for _ in range(int(len(grp) * neg_ratio)):
            src_fp = pool[random.choice(top_idxs)]
            src_full = extract_fragment(src_fp, 0, 10**7) or ""
            if len(src_full) < neg_length or len(susp_full) < neg_length:
                continue
            si = random.randint(0, len(susp_full) - neg_length)
            ri = random.randint(0, len(src_full) - neg_length)
            negs.append({
                "suspicious_reference": susp_key,
                "source_reference":     f"pool_{os.path.basename(src_fp)}",
                "suspicious_text":      susp_full[si:si+neg_length],
                "source_text":          src_full[ri:ri+neg_length],
                "label":                0,
            })

    df = pd.concat([df, pd.DataFrame(negs)], ignore_index=True)

    # 4) SLIDING-WINDOW NEGATIVES
    slide_negs = []
    for _, pos in df[df.label == 1].iterrows():
        susp_key = pos.suspicious_reference
        cid = int(susp_key.split("_")[0].replace("corpus", ""))
        base = susp_key.split("_", 1)[1]
        susp_full = extract_fragment(
            os.path.join(susp_dirs[cid], base), 0, 10**7
        ) or ""
        if len(susp_full) < 2 * neg_length:
            continue
        for _ in range(slide_per_pos):
            i = random.randint(0, len(susp_full) - neg_length)
            j = random.randint(0, len(susp_full) - neg_length)
            if abs(i - j) < neg_length:
                j = (i + neg_length) % (len(susp_full) - neg_length)
            slide_negs.append({
                "suspicious_reference": susp_key,
                "source_reference":     susp_key,
                "suspicious_text":      susp_full[i:i+neg_length],
                "source_text":          susp_full[j:j+neg_length],
                "label":                0,
            })
    df = pd.concat([df, pd.DataFrame(slide_negs)], ignore_index=True)

    # 5) NOISE-DELETION NEGATIVES
    noise_negs = []
    for _, pos in df[df.label == 1].iterrows():
        words = pos.suspicious_text.split()
        k = max(1, int(len(words) * noise_deletion_frac))
        drop = set(random.sample(range(len(words)), k))
        noisy = " ".join(w for idx, w in enumerate(words) if idx not in drop)
        rec = pos.to_dict()
        rec["suspicious_text"] = noisy
        rec["label"] = 0
        noise_negs.append(rec)
    df = pd.concat([df, pd.DataFrame(noise_negs)], ignore_index=True)

    # 6) Compute lexical & TF-IDF similarity features
    df["lex_overlap"] = df.apply(
        lambda row: SequenceMatcher(
            None, row["suspicious_text"], row["source_text"]
        ).ratio(),
        axis=1,
    )
    tfidf_all = TfidfVectorizer(max_features=20000).fit(
        df["suspicious_text"].tolist() + df["source_text"].tolist()
    )
    s_vecs = tfidf_all.transform(df["suspicious_text"])
    r_vecs = tfidf_all.transform(df["source_text"])
    s_norm = (s_vecs.multiply(s_vecs).sum(axis=1).A1) ** 0.5
    r_norm = (r_vecs.multiply(r_vecs).sum(axis=1).A1) ** 0.5
    dot = s_vecs.multiply(r_vecs).sum(axis=1).A1
    df["tfidf_sim"] = dot / (s_norm * r_norm + 1e-8)

    # 7) Split train/val/test by suspicious_reference groups
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_i, tmp_i = next(gss1.split(df, groups=df["suspicious_reference"]))
    tr_df, tmp = df.iloc[tr_i], df.iloc[tmp_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    v_i, te_i = next(gss2.split(tmp, groups=tmp["suspicious_reference"]))
    val_df, te_df = tmp.iloc[v_i], tmp.iloc[te_i]

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [tr_df, val_df, te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(fp, index=False, encoding="utf-8")
        paths[name] = fp

    return paths
