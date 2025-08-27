# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class PlagiarismDetector(nn.Module):
    """
    Modèle de détection de plagiat :
      1) Encodage AraBERT
      2) BiLSTM sur les embeddings
      3) Soft-attention alignment
      4) Construction de 4 représentations : sl, aligned, |sl−aligned|, sl*aligned
      5) Concaténation + max-pooling → vecteur de similitude
      6) MLP pour projection à 256 dimensions
      7) Couche finale linéaire → logit (plagiat ou non)
    """

    def __init__(self,
                 bert_model: str = "aubmindlab/bert-base-arabertv2",
                 lstm_hidden: int = 128,
                 dropout: float = 0.5):
        super().__init__()
        # 1) Pré‐chargement d’AraBERT
        self.bert = AutoModel.from_pretrained(bert_model)
        # 2) BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # 3) MLP sur la représentation de similitude
        #    taille d’entrée = 4*(2*lstm_hidden)
        self.sim_mlp = nn.Sequential(
            nn.Linear(4 * lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 4) Couche finale
        self.classifier = nn.Linear(256, 1)

    def forward(self,
                s_ids: torch.Tensor,
                s_mask: torch.Tensor,
                r_ids: torch.Tensor,
                r_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_ids   (N×L)   : input_ids du fragment suspect
            s_mask  (N×L)   : attention_mask du suspect
            r_ids   (N×L)   : input_ids du fragment source
            r_mask  (N×L)   : attention_mask du source

        Returns:
            logits (N×1) : score brut (avant sigmoid)
        """
        # 1) Encodage BERT
        s_emb = self.bert(input_ids=s_ids, attention_mask=s_mask).last_hidden_state
        r_emb = self.bert(input_ids=r_ids, attention_mask=r_mask).last_hidden_state

        # 2) BiLSTM + masquage des paddings
        s_lstm, _ = self.lstm(s_emb)  # (N, L, 2*H)
        r_lstm, _ = self.lstm(r_emb)  # (N, L, 2*H)
        s_lstm = s_lstm * s_mask.unsqueeze(-1)
        r_lstm = r_lstm * r_mask.unsqueeze(-1)

        # 3) Soft‐attention alignment
        #    A = softmax(s_lstm @ r_lstm^T) → (N, L, L)
        A = F.softmax(torch.bmm(s_lstm, r_lstm.transpose(1, 2)), dim=-1)
        #    aligned = A @ r_lstm → (N, L, 2*H)
        aligned = torch.bmm(A, r_lstm)

        # 4) Similarité : sl, aligned, |sl−aligned|, sl*aligned
        diff = torch.abs(s_lstm - aligned)
        prod = s_lstm * aligned
        sims = torch.cat([s_lstm, aligned, diff, prod], dim=-1)  # (N, L, 4*2H)

        # 5) Max‐pooling sur la dimension séquence
        sim_vec = sims.max(dim=1).values  # (N, 4*2H)

        # 6) MLP de réduction
        features = self.sim_mlp(sim_vec)   # (N, 256)

        # 7) Classifieur final
        logits = self.classifier(features) # (N, 1)
        return logits
