# data.py

import os, glob, random, xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from preprocess import extract_fragment

def build_dataset(
    xml_dir:str, susp_dir:str, src_dir:str, out_dir:str,
    augment:bool=False, neg_length:int=50,
    neg_ratio:float=5.0,  # ← plus de négatifs
    slide_per_pos:int=3,  # ← sliding windows négatifs
    test_size:float=0.30,
    random_state:int=42,
    cv_folds:int=5,       # ← pour cross‐validation
    min_len:int=10, max_len:int=200
):
    # 1) Charger positifs
    recs, ann = [], defaultdict(set)
    for xml_fp in glob.glob(os.path.join(xml_dir,"*.xml")):
        root = ET.parse(xml_fp).getroot()
        susp = os.path.basename(xml_fp).replace(".xml",".txt")
        s_path = os.path.join(susp_dir, susp)
        for f in root.findall("feature"):
            sr = f.get("source_reference")
            ann[susp].add(sr)
            to, tl = int(f.get("this_offset")), int(f.get("this_length"))
            so, sl = int(f.get("source_offset")), int(f.get("source_length"))
            s_frag = extract_fragment(s_path, to, tl)
            r_frag = extract_fragment(os.path.join(src_dir, sr), so, sl)
            if s_frag and r_frag:
                recs.append({
                    "suspicious_reference": susp,
                    "this_offset": to, "this_length": tl,
                    "source_reference": sr,
                    "source_offset": so, "source_length": sl,
                    "suspicious_text": s_frag,
                    "source_text":    r_frag,
                    "label": 1
                })

    # 2) DataFrame & nettoyage
    df = pd.DataFrame(recs)
    df = df.drop_duplicates(subset=[
        "suspicious_reference","source_reference",
        "this_offset","source_offset"
    ])
    df = df[df["this_length"].between(min_len,max_len)]
    print(f"Positifs nettoyés : {len(df)}")

    # 3) Augmentation shuffle (optionnel)
    if augment:
        aug = []
        for _,row in df.iterrows():
            ws,wr=row.suspicious_text.split(),row.source_text.split()
            random.shuffle(ws); random.shuffle(wr)
            rec = row.to_dict()
            rec["suspicious_text"]=" ".join(ws)
            rec["source_text"]=" ".join(wr)
            aug.append(rec)
        df = pd.concat([df,pd.DataFrame(aug)],ignore_index=True)
        print(f"Après augmentation : {len(df)}")

    # Séparer positifs pour sliding-window
    pos_df = df[df.label==1]

    # 4) Génération négatifs “cross-doc” (ratio neg_ratio)
    all_src = glob.glob(os.path.join(src_dir,"*.txt"))
    negs = []
    for susp,group in pos_df.groupby("suspicious_reference"):
        count = len(group)
        s_full = extract_fragment(os.path.join(susp_dir,susp),0,10**6)
        # cross‐doc négatifs
        cand = [p for p in all_src if os.path.basename(p) not in ann[susp]]
        for _ in range(int(count*neg_ratio)):
            src_fp=random.choice(cand)
            r_full=extract_fragment(src_fp,0,10**6)
            if len(r_full)<neg_length: continue
            i=random.randint(0,len(s_full)-neg_length)
            j=random.randint(0,len(r_full)-neg_length)
            negs.append({
                "suspicious_reference":susp,
                "this_offset":i,"this_length":neg_length,
                "source_reference":os.path.basename(src_fp),
                "source_offset":j,"source_length":neg_length,
                "suspicious_text":s_full[i:i+neg_length],
                "source_text":r_full[j:j+neg_length],
                "label":0
            })
    print(f"Cross-doc négatifs : {len(negs)}")

    # 5) Sliding-window INTRA-doc négatifs
    slide_negs = []
    for _,row in group.iterrows():
        susp=row.suspicious_reference
        s_full = extract_fragment(os.path.join(susp_dir,susp),0,10**6)
        L = len(s_full)
        for _ in range(slide_per_pos):
            if L<=neg_length: break
            i = random.randint(0, L-neg_length)
            slide_negs.append({
                "suspicious_reference":susp,
                "this_offset":i,"this_length":neg_length,
                "source_reference":susp,
                "source_offset":i,"source_length":neg_length,
                "suspicious_text":s_full[i:i+neg_length],
                "source_text":s_full[i:i+neg_length],
                "label":0
            })
    print(f"Slide-window négatifs : {len(slide_negs)}")

    # Concat tous négatifs
    df = pd.concat([df,pd.DataFrame(negs),pd.DataFrame(slide_negs)],ignore_index=True)
    print(f"Total exemples avant split : {len(df)}")

    # 6) Créer pair_group & splitter sans fuite
    df["pair_group"] = df.suspicious_reference + "__" + df.source_reference
    gss = GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
    tr_i,tmp_i = next(gss.split(df,groups=df["pair_group"]))
    tr_df,tmp_df = df.iloc[tr_i],df.iloc[tmp_i]
    gss2=GroupShuffleSplit(n_splits=1,test_size=0.5,random_state=random_state)
    v_i,te_i = next(gss2.split(tmp_df,groups=tmp_df["pair_group"]))
    val_df,te_df = tmp_df.iloc[v_i],tmp_df.iloc[te_i]

    # 7) Assertions anti-fuite
    tg,vg,tstg = set(tr_df.pair_group),set(val_df.pair_group),set(te_df.pair_group)
    assert not(tg&vg), f"Train-Val fuite {len(tg&vg)}"
    assert not(tg&tstg), f"Train-Test fuite {len(tg&tstg)}"
    assert not(vg&tstg), f"Val-Test fuite {len(vg&tstg)}"

    # 8) Sauvegarde CSV
    os.makedirs(out_dir,exist_ok=True)
    paths={}
    for name,sub in zip(["train","val","test"],[tr_df,val_df,te_df]):
        fp=os.path.join(out_dir,f"{name}.csv")
        sub.to_csv(fp,index=False,encoding="utf-8")
        print(f"{name}: {len(sub)}")
        paths[name]=fp

    # 9) 5-fold CV
    cv=[]
    if cv_folds and cv_folds>1:
        gkf=GroupKFold(n_splits=cv_folds)
        for f,(i_tr,i_va) in enumerate(gkf.split(df,groups=df["pair_group"]),1):
            d=os.path.join(out_dir,f"cv_{f}")
            os.makedirs(d,exist_ok=True)
            df.iloc[i_tr].to_csv(os.path.join(d,"train.csv"),index=False)
            df.iloc[i_va].to_csv(os.path.join(d,"val.csv"),index=False)
            cv.append(d)
    return paths,cv
