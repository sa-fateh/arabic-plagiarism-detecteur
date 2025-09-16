# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ArabicPlagiarismCSVDataset(Dataset):
    def __init__(
        self,
        csv_path,
        max_len=128,
        bert_model="aubmindlab/bert-base-arabertv2"
    ):
        # Lecture du CSV et nettoyage des valeurs manquantes
        self.df = pd.read_csv(csv_path, dtype=str)
        self.df = self.df.fillna("")  # remplace tout NaN par ""
        
        # S’assure que les colonnes de texte sont bien des str
        self.df["suspicious_text"] = self.df["suspicious_text"].astype(str)
        self.df["source_text"]     = self.df["source_text"].astype(str)
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.max_len   = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # On sait que ce sont désormais toujours des str
        s_text = row["suspicious_text"]
        r_text = row["source_text"]

        s_enc = self.tokenizer(
            s_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        r_enc = self.tokenizer(
            r_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "s_ids":  s_enc["input_ids"].squeeze(0),
            "s_mask": s_enc["attention_mask"].squeeze(0),
            "r_ids":  r_enc["input_ids"].squeeze(0),
            "r_mask": r_enc["attention_mask"].squeeze(0),
            "label":  torch.tensor(float(row["label"]), dtype=torch.float),
        }
