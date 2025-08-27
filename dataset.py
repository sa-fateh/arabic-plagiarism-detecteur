# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ArabicPlagiarismCSVDataset(Dataset):
    """
    Charge un CSV de fragments plagiés / non-plagiés et renvoie un dict :
      - s_ids, s_mask : tenseurs du texte suspect
      - r_ids, r_mask : tenseurs du texte source
      - label          : 0 ou 1 (float)
      - suspicious_reference, this_offset, this_length
      - source_reference, source_offset, source_length
    """

    def __init__(self, csv_path: str, max_len: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "aubmindlab/bert-base-arabertv2"
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]

        # Tokenisation du fragment suspect
        se = self.tokenizer(
            row["suspicious_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        # Tokenisation du fragment source
        re = self.tokenizer(
            row["source_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "s_ids":                se["input_ids"].squeeze(0),
            "s_mask":               se["attention_mask"].squeeze(0),
            "r_ids":                re["input_ids"].squeeze(0),
            "r_mask":               re["attention_mask"].squeeze(0),
            "label":                torch.tensor(row["label"], dtype=torch.float),
            "suspicious_reference": row["suspicious_reference"],
            "this_offset":          int(row["this_offset"]),
            "this_length":          int(row["this_length"]),
            "source_reference":     row["source_reference"],
            "source_offset":        int(row["source_offset"]),
            "source_length":        int(row["source_length"])
        }
