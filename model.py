import torch
import torch.nn as nn
from transformers import AutoModel

class PlagiarismDetector(nn.Module):
    def __init__(self, bert_model="aubmindlab/bert-base-arabertv2", lstm_hidden=128, dropout=0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        hid