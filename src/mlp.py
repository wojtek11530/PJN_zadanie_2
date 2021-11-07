from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F


def avg_embedding_collate(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xx, yy = zip(*batch)
    xx = torch.Tensor([np.mean(embeddings, axis=0) for embeddings in xx])
    yy = torch.Tensor(yy).to(dtype=torch.int64)
    return xx, yy


class MLPClassifier(LightningModule):
    def __init__(self, input_size: int = 100, output_size: int = 4, hidden_layer: int = 24,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5, dropout: float = 0.5):
        super(MLPClassifier, self).__init__()

        self._layer_1 = torch.nn.Linear(input_size, hidden_layer)
        self._layer_2 = torch.nn.Linear(hidden_layer, output_size)
        self._dropout = torch.nn.Dropout(p=dropout)

        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self._layer_1(x)
        x = F.relu(x)
        x = self._dropout(x)
        x = self._layer_2(x)
        return x

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self._learning_rate, weight_decay=self._weight_decay)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) \
            -> Dict[str, Any]:
        loss, correct, total = self._batch_step(batch)
        return {'loss': loss, "correct": correct, "total": total, 'log': {'train_loss': loss}}

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        tensorboard_logs = {'loss': avg_loss, "train_acc": correct / total}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) \
            -> Dict[str, Any]:
        loss, correct, total = self._batch_step(val_batch)
        return {'val_loss': loss, "correct": correct, "total": total}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) \
            -> Dict[str, Any]:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        tensorboard_logs = {'val_loss': avg_loss, "val_acc": correct / total}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def _get_correct_prediction_count(self, logits: torch.Tensor, y_labels: torch.Tensor) -> int:
        probs = torch.softmax(logits, dim=1)
        return int(probs.argmax(dim=1).eq(y_labels).sum().item())

    def _batch_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, int, int]:
        x, y_labels = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y_labels)
        total = len(y_labels)
        correct = self._get_correct_prediction_count(logits, y_labels)
        return loss, correct, total
