import os, numpy as np

class LogitCollector:
    def __init__(self):
        self._logits = []
        self._labels = []

    def add_episode(self, logits_2d, labels_1d):
        # logits_2d: [M, C]，labels_1d: [M]  (M = n_way * n_query, C = n_way)
        self._logits.append(np.asarray(logits_2d))
        self._labels.append(np.asarray(labels_1d, dtype=np.int64))

    def save(self, out_path):
        logits = np.concatenate(self._logits, axis=0) if self._logits else np.zeros((0,0))
        labels = np.concatenate(self._labels, axis=0) if self._labels else np.zeros((0,), dtype=np.int64)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, logits=logits, labels=labels)
        print(f"[saved] {out_path} | logits={logits.shape}, labels={labels.shape}")
