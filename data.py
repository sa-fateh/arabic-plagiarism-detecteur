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
    # 1) Lire les annotations positives
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
                    "source_text":    r_frag,
                    "label": 1
                })

    # 2) Nettoyage initial et DataFrame
    df = pd.DataFrame(records)
    # 2.a) supprimer doublons exacts
    df = df.drop_duplicates(subset=[
        "suspicious_reference", "source_reference",
        "this_offset", "source_offset"
    ])
    # 2.b) garder les fragments dans la plage de longueur
    df = df[df["this_length"].between(min_len, max_len)]
    print(f"→ Après nettoyage : {len(df)} exemples (pos={df.label.sum()})")

    # 3) Augmentation (shuffle mots) si demandé
    if augment:
        aug = []
        for _, row in df.iterrows():
            ws, wr = row.suspicious_text.split(), row.source_text.split()
            random.shuffle(ws); random.shuffle(wr)
            rec = row.to_dict()
            rec["suspicious_text"] = " ".join(ws)
            rec["source_text"]     = " ".join(wr)
            aug.append(rec)
        if aug:
            df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
            print(f"→ Après augmentation : {len(df)} exemples")

    # 4) Génération négatifs (exclut sources annotées)
    all_srcs = glob.glob(os.path.join(src_dir, "*.txt"))
    negs = []
    pos_counts = df[df.label == 1].groupby("suspicious_reference").size().to_dict()
    for susp, cnt in pos_counts.items():
        s_full = extract_fragment(os.path.join(susp_dir, susp), 0, 10**6)
        if len(s_full) < neg_length:
            continue
        candidates = [p for p in all_srcs if os.path.basename(p) not in ann_sources[susp]]
        for _ in range(int(cnt * neg_ratio)):
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
                "source_text":    r_full[j:j+neg_length],
                "label": 0
            })

    if negs:
        df = pd.concat([df, pd.DataFrame(negs)], ignore_index=True)
        print(f"→ Total après négatifs : {len(df)} (neg={len(negs)})")

    # 5) Créer pair_group pour le GroupShuffleSplit
    df["pair_group"] = df["suspicious_reference"] + "__" + df["source_reference"]

    # 6) Split train vs (val+test) sur pair_group
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, tmp_idx = next(gss.split(df, groups=df["pair_group"]))
    tr_df, tmp_df = df.iloc[tr_idx], df.iloc[tmp_idx]

    # 7) Split val vs test (50/50) sur pair_group
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    v_idx, te_idx = next(gss2.split(tmp_df, groups=tmp_df["pair_group"]))
    val_df, te_df = tmp_df.iloc[v_idx], tmp_df.iloc[te_idx]

    # 8) Assertions anti-fuite (optionnel)
    train_groups = set(tr_df["pair_group"])
    val_groups   = set(val_df["pair_group"])
    test_groups  = set(te_df["pair_group"])
    assert not (train_groups & val_groups), "Fuite T→V détectée"
    assert not (train_groups & test_groups), "Fuite T→Test détectée"
    assert not (val_groups   & test_groups), "Fuite V→Test détectée"

    # 9) Sauvegarde CSV
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [tr_df, val_df, te_df]):
        fp = os.path.join(out_dir, f"{name}.csv")
        subset.to_csv(fp, index=False, encoding="utf-8")
        print(f"✅ {name}.csv → {len(subset)} exemples")
        paths[name] = fp

    # 10) Option CV par pair_group
    cv_splits = []
    if cv_folds and cv_folds > 1:
        gkf = GroupKFold(n_splits=cv_folds)
        for fold, (i_tr, i_va) in enumerate(gkf.split(df, groups=df["pair_group"]), start=1):
            d = os.path.join(out_dir, f"cv_{fold}")
            os.makedirs(d, exist_ok=True)
            df.iloc[i_tr].to_csv(os.path.join(d, "train.csv"), index=False)
            df.iloc[i_va].to_csv(os.path.join(d, "val.csv"),   index=False)
            cv_splits.append(d)
        print(f"✅ {cv_folds}-fold CV générés")

    return paths, cv_splits
