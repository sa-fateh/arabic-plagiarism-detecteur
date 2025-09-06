# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ArabicPlagiarismCSVDataset(Dataset):
    def __init__(self, csv_path: str, max_len: int = 128):
        df = pd.read_csv(csv_path, dtype=str)
        df["suspicious_text"] = df["suspicious_text"].fillna("")
        df["source_text"]     = df["source_text"].fillna("")
        df["label"]           = df["label"].astype(int)
        self.df = df.reset_index(drop=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "aubmindlab/bert-base-arabertv2"
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        s_text, r_text = row.suspicious_text, row.source_text

        se = self.tokenizer(
            s_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        re = self.tokenizer(
            r_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "s_ids":  se.input_ids.squeeze(0),
            "s_mask": se.attention_mask.squeeze(0),
            "r_ids":  re.input_ids.squeeze(0),
            "r_mask": re.attention_mask.squeeze(0),
            "label":  torch.tensor(row.label, dtype=torch.float),
        }
