# data.py

import os
import glob
import random
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import GroupShuffleSplit

from preprocess import extract_fragment

def build_dataset(
    xml_dir: str,
    susp_dir: str,
    src_dir: str,
    out_dir: str,
    augment: bool = False,
    neg_length: int = 512,
    neg_ratio: float = 1.0,
    hard_neg_k: int = 2,
    slide_step: int = 256,
    test_size: float = 0.30,
    random_state: int = 42
):
    # 1) EXTRACTION DES POSITIFS
    records, ann = [], defaultdict(set)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f">>> {len(xml_files)} XML analysés")
    for xml_fp in xml_files:
        root = ET.parse(xml_fp).getroot()
        susp = os.path.basename(xml_fp).replace(".xml", ".txt")
        susp_path = os.path.join(susp_dir, susp)
        for feat in root.findall("feature"):
            sr = feat.get("source_reference")
            ann[susp].add(sr)
            to, tl = int(feat.get("this_offset")), int(feat.get("this_length"))
            so, sl = int(feat.get("source_offset")), int(feat.get("source_length"))
            s_frag = extract_fragment(susp_path, to, tl)
            r_frag = extract_fragment(os.path.join(src_dir, sr), so, sl)
            if s_frag and r_frag:
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

    # 2) OPTIONNEL : shuffle-augmentation
    if augment:
        aug = []
        for _, r in df.iterrows():
            ws, wr = r.suspicious_text.split(), r.source_text.split()
            random.shuffle(ws); random.shuffle(wr)
            rec = r.to_dict()
            rec["suspicious_text"] = " ".join(ws)
            rec["source_text"]     = " ".join(wr)
            aug.append(rec)
        df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
        print(f"→ après augmentation : {len(df)}")

    # 3) CRÉATION DU POOL SLIDING-WINDOW SUR TOUS LES SOURCES
    windows = []
    src_files = glob.glob(os.path.join(src_dir, "*.txt"))
    for src_fp in src_files:
        text = extract_fragment(src_fp, 0, 10**7) or ""
        for start in range(0, len(text) - neg_length + 1, slide_step):
            seg = text[start:start+neg_length]
            windows.append({
                "source_reference": os.path.basename(src_fp),
                "source_text": seg
            })
    win_df = pd.DataFrame(windows)
    print(f"→ {len(win_df)} segments sliding‐window prêts (hard negatives)")

    # 4) TF-IDF + hard negative sampling
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=50000)
    corpus = list(df["source_text"]) + list(win_df["source_text"])
    tfidf_mat = tfidf.fit_transform(corpus)
    pos_mat = tfidf_mat[: len(df)]            # positifs
    win_mat = tfidf_mat[len(df):]             # windows

    hard_negs = []
    for idx in range(len(df)):
        sims = linear_kernel(pos_mat[idx:idx+1], win_mat).flatten()
        topk = sims.argsort()[-hard_neg_k:][::-1]
        susp = df.at[idx, "suspicious_reference"]
        for widx in topk:
            row = win_df.iloc[widx]
            # n'inclut pas si c'est une vraie annotation
            if row.source_reference in ann[susp]:
                continue
            hard_negs.append({
                "suspicious_reference": susp,
                "source_reference":     row.source_reference,
                "this_offset":          0,
                "this_length":          neg_length,
                "source_offset":        0,
                "source_length":        neg_length,
                "suspicious_text":      df.at[idx, "suspicious_text"],
                "source_text":          row.source_text,
                "label":                0
            })
    df = pd.concat([df, pd.DataFrame(hard_negs)], ignore_index=True)
    print(f"→ ajout de {len(hard_negs)} hard negatives → total {len(df)}")

    # 5) GENERATION DE NEGATIFS “CROSS-DOC” SIMPLE
    # (facultatif, selon neg_ratio)
    if neg_ratio > 0:
        cross = []
        all_srcs = set(win_df["source_reference"])
        for susp, grp in df[df.label==1].groupby("suspicious_reference"):
            count = len(grp)
            candidates = list(all_srcs - ann[susp])
            for _ in range(int(count * neg_ratio)):
                sr = random.choice(candidates)
                seg = win_df[win_df.source_reference == sr]["source_text"].sample(1).iloc[0]
                cross.append({
                    "suspicious_reference": susp,
                    "source_reference":     sr,
                    "this_offset":          0,
                    "this_length":          neg_length,
                    "source_offset":        0,
                    "source_length":        neg_length,
                    "suspicious_text":      extract_fragment(os.path.join(susp_dir, susp), 0, 10**6),
                    "source_text":          seg,
                    "label":                0
                })
        df = pd.concat([df, pd.DataFrame(cross)], ignore_index=True)
        print(f"→ ajout de {len(cross)} cross-doc negs → total {len(df)}")

    # 6) HOLD-OUT SPLIT PAR DOCUMENT SUSPECT
    df["pair_doc"] = df["suspicious_reference"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_i, tmp_i = next(gss.split(df, groups=df["pair_doc"]))
    tr_df, tmp_df = df.iloc[tr_i], df.iloc[tmp_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    v_i, te_i = next(gss2.split(tmp_df, groups=tmp_df["pair_doc"]))
    val_df, te_df = tmp_df.iloc[v_i], tmp_df.iloc[te_i]

    assert not set(tr_df.pair_doc) & set(val_df.pair_doc), "Fuite train→val !"
    assert not set(tr_df.pair_doc) & set(te_df.pair_doc),  "Fuite train→test !"
    assert not set(val_df.pair_doc) & set(te_df.pair_doc), "Fuite val→test !"

    # 7) SAUVEGARDE
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, sub in zip(["train", "val", "test"], [tr_df, val_df, te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        sub.to_csv(fp, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(sub)} exemples")
        paths[name] = fp

    return paths
