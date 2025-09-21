# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ArabicPlagiarismCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        max_len: int = 128,
        bert_model: str = "aubmindlab/bert-base-arabertv2",
    ):
        # 1) Load and clean CSV
        self.df = pd.read_csv(csv_path, dtype=str).fillna("")

        # 2) Convert columns to correct dtypes
        self.df["label"]        = (
            pd.to_numeric(self.df["label"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        self.df["lex_overlap"]  = (
            pd.to_numeric(self.df["lex_overlap"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        self.df["tfidf_sim"]    = (
            pd.to_numeric(self.df["tfidf_sim"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

        # 3) Ensure text columns are str
        self.df["suspicious_text"] = self.df["suspicious_text"].astype(str)
        self.df["source_text"]     = self.df["source_text"].astype(str)

        # 4) Tokenizer and length
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        s_enc = self.tokenizer(
            row["suspicious_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        r_enc = self.tokenizer(
            row["source_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "s_ids":  s_enc["input_ids"].squeeze(0),
            "s_mask": s_enc["attention_mask"].squeeze(0),
            "r_ids":  r_enc["input_ids"].squeeze(0),
            "r_mask": r_enc["attention_mask"].squeeze(0),
            "lex":    torch.tensor(row["lex_overlap"], dtype=torch.float),
            "tfidf":  torch.tensor(row["tfidf_sim"],   dtype=torch.float),
            "label":  torch.tensor(row["label"],       dtype=torch.float),
        }
