"""
Replay Buffer

学習データを効率的に管理するバッファ

機能:
- 固定サイズのバッファでデータを循環管理
- ランダムサンプリングでミニバッチを生成
- 古いデータを自動的に破棄
"""

import numpy as np
from typing import List, Tuple
from collections import deque
import random


class ReplayBuffer:
    """
    リプレイバッファ

    Self-Play で生成された学習データを保存し、
    ランダムサンプリングでミニバッチを提供する
    """

    def __init__(self, max_size: int = 100000):
        """
        Args:
            max_size (int): バッファの最大サイズ
                古いデータは自動的に破棄される
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, data: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        データをバッファに追加

        Args:
            data: [(state, policy, value), ...]
                - state: (3, 8, 8)
                - policy: (65,)
                - value: float
        """
        for item in data:
            self.buffer.append(item)

    def add_single(self, state: np.ndarray, policy: np.ndarray, value: float):
        """
        単一のデータをバッファに追加

        Args:
            state (np.ndarray): (3, 8, 8)
            policy (np.ndarray): (65,)
            value (float): 勝敗
        """
        self.buffer.append((state, policy, value))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ランダムサンプリングでミニバッチを生成

        Args:
            batch_size (int): バッチサイズ

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - states: (batch_size, 3, 8, 8)
                - policies: (batch_size, 65)
                - values: (batch_size, 1)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer size ({len(self.buffer)}) is smaller than batch size ({batch_size})"
            )

        # ランダムサンプリング
        samples = random.sample(self.buffer, batch_size)

        # バッチに変換
        states = np.array([s[0] for s in samples], dtype=np.float32)
        policies = np.array([s[1] for s in samples], dtype=np.float32)
        values = np.array([[s[2]] for s in samples], dtype=np.float32)  # (batch_size, 1)

        return states, policies, values

    def __len__(self) -> int:
        """バッファのサイズを取得"""
        return len(self.buffer)

    def clear(self):
        """バッファをクリア"""
        self.buffer.clear()

    def is_ready(self, min_size: int) -> bool:
        """
        バッファが学習可能な状態かチェック

        Args:
            min_size (int): 最小サイズ

        Returns:
            bool: バッファサイズが min_size 以上なら True
        """
        return len(self.buffer) >= min_size

    def get_statistics(self) -> dict:
        """
        バッファの統計情報を取得

        Returns:
            dict: 統計情報
                - size: バッファサイズ
                - max_size: 最大サイズ
                - fill_rate: 充填率
                - value_mean: 価値の平均
                - value_std: 価値の標準偏差
        """
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "max_size": self.max_size,
                "fill_rate": 0.0,
                "value_mean": 0.0,
                "value_std": 0.0,
            }

        values = [item[2] for item in self.buffer]

        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "fill_rate": len(self.buffer) / self.max_size,
            "value_mean": np.mean(values),
            "value_std": np.std(values),
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    優先度付きリプレイバッファ（拡張版）

    学習誤差が大きいサンプルを優先的にサンプリング

    NOTE: 現時点では未実装。将来の拡張用にスケルトンのみ定義
    """

    def __init__(self, max_size: int = 100000, alpha: float = 0.6):
        """
        Args:
            max_size (int): バッファの最大サイズ
            alpha (float): 優先度の指数（0: 一様サンプリング, 1: 完全優先）
        """
        super().__init__(max_size)
        self.alpha = alpha
        self.priorities = deque(maxlen=max_size)

    def add_single(self, state: np.ndarray, policy: np.ndarray, value: float, priority: float = 1.0):
        """
        優先度付きでデータを追加

        Args:
            state (np.ndarray): (3, 8, 8)
            policy (np.ndarray): (65,)
            value (float): 勝敗
            priority (float): 優先度
        """
        super().add_single(state, policy, value)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        優先度に基づいてサンプリング

        NOTE: 現時点では一様サンプリング（優先度未実装）
        """
        return super().sample(batch_size)
