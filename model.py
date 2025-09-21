# model.py

import torch
import torch.nn as nn
from transformers import AutoModel


class PlagiarismDetector(nn.Module):
    def __init__(
        self,
        bert_model: str = "aubmindlab/bert-base-arabertv2",
        lstm_hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        hid = 2 * lstm_hidden
        feat_size = 4 * hid + 2  # +2 for lex_overlap & tfidf_sim

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, s_ids, s_mask, r_ids, r_mask, lex, tfidf):
        S = self.bert(input_ids=s_ids, attention_mask=s_mask).last_hidden_state
        R = self.bert(input_ids=r_ids, attention_mask=r_mask).last_hidden_state

        S, _ = self.lstm(S)
        R, _ = self.lstm(R)

        d_k = S.size(-1)
        A = torch.softmax(torch.bmm(S, R.transpose(1, 2)) / (d_k**0.5), dim=-1)
        R_aligned = torch.bmm(A, R)

        mean_s = S.mean(dim=1)
        mean_r = R_aligned.mean(dim=1)
        diff   = (mean_s - mean_r).abs()
        prod   = mean_s * mean_r

        # Add lexical & TF-IDF features
        lex_feat   = lex.unsqueeze(1)
        tfidf_feat = tfidf.unsqueeze(1)

        V = torch.cat([mean_s, mean_r, diff, prod, lex_feat, tfidf_feat], dim=-1)
        logits = self.classifier(V).squeeze(-1)
        return logits
