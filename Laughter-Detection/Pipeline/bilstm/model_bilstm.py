# model_bilstm.py

import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Binary classification
        )

    def forward(self, x, lengths):
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed)

        # Concatenate last forward and backward hidden states
        hn_fwd = hn[-2, :, :]  # Last layer forward
        hn_bwd = hn[-1, :, :]  # Last layer backward
        hn_combined = torch.cat((hn_fwd, hn_bwd), dim=1)  # (batch_size, 2*hidden_dim)

        logits = self.classifier(hn_combined)
        return logits.squeeze(1)
