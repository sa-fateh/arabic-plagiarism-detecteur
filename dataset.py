# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ArabicPlagiarismCSVDataset(Dataset):
    """
    Dataset for Arabic plagiarism detection.
    Lit un CSV avec colonnes :
      - suspicious_text
      - source_text
      - label
    et renvoie des batchs tokenisés pour BERT.
    """

    def __init__(
        self,
        csv_path: str,
        max_len: int = 128,
        bert_model: str = "aubmindlab/bert-base-arabertv2",
    ):
        # 1) Lecture du CSV en str et remplissage des valeurs manquantes
        self.df = pd.read_csv(csv_path, dtype=str).fillna("")

        # 2) Conversion de la colonne 'label' en int (0 ou 1)
        self.df["label"] = (
            pd.to_numeric(self.df["label"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # 3) Assurer que les textes sont bien des str
        self.df["suspicious_text"] = self.df["suspicious_text"].astype(str)
        self.df["source_text"]     = self.df["source_text"].astype(str)

        # 4) Initialisation du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Textes à encoder
        s_text = row["suspicious_text"]
        r_text = row["source_text"]

        # Tokenization avec padding et truncation
        s_enc = self.tokenizer(
            s_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        r_enc = self.tokenizer(
            r_text,
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
            "label":  torch.tensor(row["label"], dtype=torch.float),
        }
