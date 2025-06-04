# prepare_dataset_lstm.py
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
from collections import defaultdict

class LaughterLSTMPreprocessor:
    def __init__(self, feature_csv_path, label_col="label"):
        self.df = pd.read_csv(feature_csv_path)
        self.label_col = label_col

    def group_by_participant_segment(self):
        grouped = defaultdict(list)
        for _, row in self.df.iterrows():
            key = (row["participant"], row["segment"])
            feats = row.drop(["video_id", "participant", "segment", "label", "modality"], errors="ignore").values.astype(np.float32)
            grouped[key].append((feats, row[self.label_col]))
        return grouped

    def to_padded_sequences(self):
        grouped = self.group_by_participant_segment()
        sequences, labels = [], []
        for key, values in grouped.items():
            feats_seq = [torch.tensor(v[0]) for v in values]
            label_seq = [torch.tensor(v[1]) for v in values]
            sequences.append(torch.stack(feats_seq))
            labels.append(torch.tensor(label_seq))
        padded_x = pad_sequence(sequences, batch_first=True)
        padded_y = pad_sequence(labels, batch_first=True)
        lengths = torch.tensor([len(seq) for seq in sequences])
        return padded_x, padded_y, lengths
