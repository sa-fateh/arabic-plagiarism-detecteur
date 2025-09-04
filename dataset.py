# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ArabicPlagiarismCSVDataset(Dataset):
    """
    Charge un CSV de paires (suspicious_text, source_text, label) 
    et renvoie pour chaque item :
      - s_ids, s_mask : tenseurs BERT du texte suspect
      - r_ids, r_mask : tenseurs BERT du texte source
      - label         : tenseur float (0.0 ou 1.0)
    """
    def __init__(self, csv_path: str, max_len: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

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
            "s_ids":  se["input_ids"].squeeze(0),
            "s_mask": se["attention_mask"].squeeze(0),
            "r_ids":  re["input_ids"].squeeze(0),
            "r_mask": re["attention_mask"].squeeze(0),
            "label":  torch.tensor(row["label"], dtype=torch.float),
        }
