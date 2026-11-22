"""Offline, frozen episodic kNN hazard memory.

No test observation changes the memory; this avoids conflating memory
accumulation with the method's effects during held-out evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .core import Snapshot, episode_feature

FEATURE_COUNT = 21


@dataclass
class EpisodicMemory:
    features: np.ndarray
    labels: np.ndarray
    k: int = 7
    max_radius: float = 0.85

    @classmethod
    def from_records(cls, rows: list[dict], max_per_class: int = 600, seed: int = 2026) -> 'EpisodicMemory':
        rng = np.random.default_rng(seed)
        selected = []
        for label in (0, 1):
            indexes = np.asarray([i for i, row in enumerate(rows) if int(row['hazard_label']) == label])
            if indexes.size > max_per_class:
                indexes = rng.choice(indexes, size=max_per_class, replace=False)
            selected.extend(indexes.tolist())
        selected.sort()
        if not selected:
            return cls(np.empty((0, FEATURE_COUNT), np.float32), np.empty((0,), np.int8))
        return cls(
            features=np.asarray([rows[i]['feature'] for i in selected], np.float32),
            labels=np.asarray([rows[i]['hazard_label'] for i in selected], np.int8),
        )

    def lookup(self, state: Snapshot, proposed_action: int) -> tuple[float, float, bool]:
        if len(self.labels) == 0 or not np.any(self.labels):
            return 0.0, float('inf'), False
        query = episode_feature(state, proposed_action)
        distances = np.sqrt(np.square(self.features - query).sum(axis=1))
        k = min(self.k, len(distances))
        indexes = np.argpartition(distances, k - 1)[:k]
        selected_d = distances[indexes]
        if float(np.min(selected_d)) > self.max_radius:
            return 0.0, float(np.min(selected_d)), False
        weights = 1.0 / (selected_d + 0.05)
        score = float(np.sum(weights * self.labels[indexes]) / np.sum(weights))
        return score, float(np.min(selected_d)), True

    def save(self, folder: Path) -> None:
        np.savez_compressed(folder / 'frozen_memory.npz', features=self.features, labels=self.labels,
                            k=self.k, max_radius=self.max_radius)

    @classmethod
    def load(cls, path: Path) -> 'EpisodicMemory':
        with np.load(path) as obj:
            return cls(obj['features'], obj['labels'], int(obj['k']), float(obj['max_radius']))


def save_training_records(rows: list[dict], folder: Path) -> None:
    records = []
    for row in rows:
        entry = {k: v for k, v in row.items() if k != 'feature'}
        entry.update({f'f{i:02d}': float(value) for i, value in enumerate(row['feature'])})
        records.append(entry)
    pd.DataFrame(records).to_csv(folder / 'calibration_transitions.csv', index=False)
