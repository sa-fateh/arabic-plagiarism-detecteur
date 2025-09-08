import torch
import torch.nn as nn
from transformers import AutoModel

class PlagiarismDetector(nn.Module):
    """
    Pipeline : AraBERT → BiLSTM → dot-product attention → pooling → classifier
    """
    def __init__(
        self,
        bert_model: str = "aubmindlab/bert-base-arabertv2",
        lstm_hidden: int = 128,
        dropout: float = 0.7
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        hid       = 2 * lstm_hidden
        feat_size = 4 * hid
        self.classifier = nn.Sequential(
            nn.Linear(feat_size, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1)
        )

    def forward(self, s_ids, s_mask, r_ids, r_mask):
        Hs = self.bert(input_ids=s_ids, attention_mask=s_mask).last_hidden_state
        Hr = self.bert(input_ids=r_ids, attention_mask=r_mask).last_hidden_state

        Hs, _ = self.lstm(Hs)
        Hr, _ = self.lstm(Hr)

        attn       = torch.softmax(torch.bmm(Hs, Hr.transpose(1,2)), dim=-1)
        Hr_aligned = torch.bmm(attn, Hr)

        diff   = (Hs - Hr_aligned).abs()
        prod   = Hs * Hr_aligned
        V      = torch.cat([Hs, Hr_aligned, diff, prod], dim=-1)
        Vp, _  = V.max(dim=1)
        logits = self.classifier(Vp).squeeze(-1)
        return logits
