import torch
import torch.nn as nn
from transformers import AutoModel

class PlagiarismDetector(nn.Module):
    def __init__(self, bert_model: str = "aubmindlab/bert-base-arabertv2",
                 lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        hid = 2 * lstm_hidden  # bidirectional
        # classifier: [h_s; h_r; |h_s-h_r|; h_s*h_r]
        feat_size = 4 * hid
        self.fc = nn.Sequential(
            nn.Linear(feat_size, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1)
        )

    def forward(self, s_ids, s_mask, r_ids, r_mask):
        Hs = self.bert(input_ids=s_ids, attention_mask=s_mask).last_hidden_state
        Hr = self.bert(input_ids=r_ids, attention_mask=r_mask).last_hidden_state

        Hs, _ = self.lstm(Hs)  # (B, S, 2*lstm_hidden)
        Hr, _ = self.lstm(Hr)

        # dot-product attention alignment
        attn = torch.softmax(torch.bmm(Hs, Hr.transpose(1,2)), dim=-1)
        Hr_aligned = torch.bmm(attn, Hr)

        # feature vector
        diff = torch.abs(Hs - Hr_aligned)
        prod = Hs * Hr_aligned
        V = torch.cat([Hs, Hr_aligned, diff, prod], dim=-1)

        # max-pool over sequence
        Vp, _ = torch.max(V, dim=1)  # (B, feat_size)
        logits = self.fc(Vp).squeeze(-1)
        return logits
