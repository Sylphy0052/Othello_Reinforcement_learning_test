"""
MCTSノード定義

AlphaZero方式のMCTSで使用する木構造のノードクラス
PUCT値計算と統計情報を管理
"""

import numpy as np
from typing import Optional, Dict


class MCTSNode:
    """
    MCTSの木構造ノード

    各ノードは以下の情報を保持:
    - 訪問回数 (N)
    - 累積価値 (W)
    - 事前確率 (P) - ニューラルネットワークの出力
    - 子ノードへのリンク

    PUCT式:
        Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Q(s,a) = W(s,a) / N(s,a)  ... 平均価値
    """

    def __init__(self, prior: float, parent: Optional['MCTSNode'] = None):
        """
        Args:
            prior (float): このノードへの事前確率（親ノードのNN出力確率）
            parent (MCTSNode, optional): 親ノード
        """
        self.prior = prior
        self.parent = parent

        # 統計情報
        self.visit_count = 0  # N(s,a)
        self.value_sum = 0.0  # W(s,a)

        # 子ノード: {action: MCTSNode}
        self.children: Dict[int, MCTSNode] = {}

        # 展開済みフラグ
        self.is_expanded = False

    def is_leaf(self) -> bool:
        """リーフノードかどうか"""
        return len(self.children) == 0

    def get_value(self) -> float:
        """
        平均価値 Q(s,a) を取得

        Returns:
            float: 平均価値。訪問回数が0の場合は0を返す
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, policy_probs: np.ndarray, legal_actions: list):
        """
        ノードを展開し、合法手に対応する子ノードを作成

        Args:
            policy_probs (np.ndarray): ニューラルネットワークの方策出力 (65,)
            legal_actions (list): 合法手のリスト [0-64]
        """
        # Action Masking: 合法手のみ確率を残す
        masked_probs = np.zeros_like(policy_probs)
        masked_probs[legal_actions] = policy_probs[legal_actions]

        # 確率を正規化
        prob_sum = masked_probs.sum()
        if prob_sum > 0:
            masked_probs /= prob_sum
        else:
            # ニューラルネットワークがすべて0を出力した場合、均等分布
            masked_probs[legal_actions] = 1.0 / len(legal_actions)

        # 合法手に対して子ノードを作成
        for action in legal_actions:
            self.children[action] = MCTSNode(
                prior=masked_probs[action],
                parent=self
            )

        self.is_expanded = True

    def select_child(self, c_puct: float) -> tuple:
        """
        PUCT値が最大の子ノードを選択

        PUCT式:
            Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct (float): 探索のバランス定数（大きいほど探索重視）

        Returns:
            tuple: (選択されたaction, 選択された子ノード)
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        # 親の訪問回数
        parent_visit_count = self.visit_count

        for action, child in self.children.items():
            # Q値（平均価値）
            q_value = child.get_value()

            # U値（探索ボーナス）
            u_value = c_puct * child.prior * np.sqrt(parent_visit_count) / (1 + child.visit_count)

            # PUCT値
            puct_score = q_value + u_value

            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child

        return best_action, best_child

    def update(self, value: float):
        """
        ノードの統計情報を更新（バックプロパゲーション）

        Args:
            value (float): 更新する価値（手番視点での価値 [-1, 1]）
        """
        self.visit_count += 1
        self.value_sum += value

    def get_visit_counts(self) -> Dict[int, int]:
        """
        子ノードの訪問回数を取得

        Returns:
            Dict[int, int]: {action: visit_count}
        """
        return {action: child.visit_count for action, child in self.children.items()}

    def get_policy_distribution(self, temperature: float = 1.0) -> np.ndarray:
        """
        訪問回数に基づく方策分布を生成

        温度パラメータ:
        - temperature = 1.0: 訪問回数に比例した確率
        - temperature → 0: 最大訪問回数の手に確率を集中（決定的）
        - temperature → ∞: 均等分布

        Args:
            temperature (float): 温度パラメータ

        Returns:
            np.ndarray: 方策分布 (65,)
        """
        policy = np.zeros(65, dtype=np.float32)

        if len(self.children) == 0:
            return policy

        visit_counts = self.get_visit_counts()
        actions = list(visit_counts.keys())
        counts = np.array([visit_counts[a] for a in actions], dtype=np.float32)

        if temperature == 0:
            # 決定的な選択: 最大訪問回数の手に確率1
            best_action = actions[np.argmax(counts)]
            policy[best_action] = 1.0
        else:
            # 温度付きSoftmax
            counts = counts ** (1.0 / temperature)
            counts /= counts.sum()
            for action, prob in zip(actions, counts):
                policy[action] = prob

        return policy

    def __repr__(self) -> str:
        """デバッグ用の文字列表現"""
        return (f"MCTSNode(prior={self.prior:.3f}, "
                f"N={self.visit_count}, "
                f"W={self.value_sum:.3f}, "
                f"Q={self.get_value():.3f}, "
                f"children={len(self.children)})")
