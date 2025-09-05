import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ArabicPlagiarismCSVDataset(Dataset):
    """
    Lit train/val/test.csv et renvoie pour chaque item :
      - s_ids, s_mask : BERT pour suspicious_text
      - r_ids, r_mask : BERT pour source_text
      - label         : float 0.0 ou 1.0
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
        row = self.df.iloc[idx]
        se = self.tokenizer(
            row["suspicious_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
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
