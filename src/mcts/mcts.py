"""
モンテカルロ木探索 (Monte Carlo Tree Search)

AlphaZero方式のMCTS実装:
- ニューラルネットワークによる評価
- PUCT式による選択
- Action Masking（合法手のみ探索）
- ディリクレノイズによる探索促進
"""

import numpy as np
import torch
from typing import Optional
from .node import MCTSNode


class MCTS:
    """
    モンテカルロ木探索

    AlphaZeroの探索アルゴリズム:
    1. Select: PUCT値が最大の子ノードを選択
    2. Expand: ニューラルネットワークで評価し、子ノードを展開
    3. Backpropagate: リーフノードの価値を親ノードに伝播
    """

    def __init__(
        self,
        model,
        device,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        """
        Args:
            model: ニューラルネットワークモデル (OthelloResNet)
            device: torch.device
            c_puct (float): PUCT式の探索定数（大きいほど探索重視）
            dirichlet_alpha (float): ディリクレ分布のパラメータ
            dirichlet_epsilon (float): ノイズの混合比率
        """
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(
        self,
        board,
        num_simulations: int,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
    ) -> tuple:
        """
        MCTS探索を実行し、方策分布を返す

        Args:
            board: OthelloBitboard インスタンス
            num_simulations (int): シミュレーション回数
            temperature (float): 温度パラメータ（0に近いほど決定的）
            add_dirichlet_noise (bool): ルートノードにディリクレノイズを追加するか

        Returns:
            tuple: (policy_distribution, root_value)
                - policy_distribution (np.ndarray): 方策分布 (65,)
                - root_value (float): ルートノードの価値推定
        """
        # ルートノードの作成
        root = MCTSNode(prior=1.0)

        # ルートノードの初期展開
        board_tensor = self._get_board_tensor(board)
        policy_probs, value = self._predict(board_tensor)
        legal_actions = board.get_legal_moves()

        # パスしかできない場合
        if len(legal_actions) == 0:
            legal_actions = [64]  # パスアクション

        root.expand(policy_probs, legal_actions)

        # ディリクレノイズの追加（学習時の探索促進）
        if add_dirichlet_noise:
            self._add_dirichlet_noise(root, legal_actions)

        # シミュレーションを実行
        for _ in range(num_simulations):
            # 盤面のコピー（各シミュレーションで独立）
            board_copy = board.copy()
            self._run_simulation(root, board_copy)

        # 訪問回数に基づく方策分布を生成
        policy_distribution = root.get_policy_distribution(temperature)
        root_value = root.get_value()

        return policy_distribution, root_value

    def _run_simulation(self, node: MCTSNode, board) -> float:
        """
        1回のシミュレーションを実行

        Select -> Expand -> Backpropagate のサイクル

        Args:
            node (MCTSNode): 現在のノード
            board: 現在の盤面状態

        Returns:
            float: リーフノードの価値（現在の手番視点）
        """
        # 1. Select: リーフノードに到達するまで下降
        path = []
        current_node = node

        while not current_node.is_leaf():
            action, child = current_node.select_child(self.c_puct)
            path.append((current_node, action, child))

            # 盤面を進める
            board.make_move(action)
            current_node = child

        # 2. Expand & Evaluate
        # 終局チェック
        if board.is_terminal():
            # 終局の場合、実際の勝敗を価値として使う
            winner = board.get_winner()
            value = float(winner)  # 1: 勝ち, -1: 負け, 0: 引き分け
        else:
            # ニューラルネットワークで評価
            board_tensor = self._get_board_tensor(board)
            policy_probs, value = self._predict(board_tensor)

            # 子ノードを展開
            legal_actions = board.get_legal_moves()
            if len(legal_actions) == 0:
                legal_actions = [64]  # パスアクション

            current_node.expand(policy_probs, legal_actions)

            # NNの価値は現在の手番視点での価値
            value = value.item()

        # 3. Backpropagate: 価値を親ノードに伝播
        # 手番が入れ替わるため、価値の符号を反転させながら伝播
        self._backpropagate(path, value)

        return value

    def _backpropagate(self, path: list, value: float):
        """
        価値をルートノードまで伝播

        Args:
            path (list): [(parent_node, action, child_node), ...]
            value (float): リーフノードの価値
        """
        # リーフノードから順に更新
        for i in range(len(path) - 1, -1, -1):
            parent, action, child = path[i]

            # 子ノードの更新（子の視点での価値）
            child.update(value)

            # 手番が入れ替わるため、価値の符号を反転
            value = -value

        # ルートノードの更新（最後に残った符号反転後の価値）
        # path[0][0] がルートノードだが、ルートノード自体は path に含まれていないため
        # ここでは何もしない（必要に応じて追加可能）

    def _predict(self, board_tensor: torch.Tensor) -> tuple:
        """
        ニューラルネットワークで方策と価値を予測

        Args:
            board_tensor (torch.Tensor): 盤面テンソル (1, 3, 8, 8)

        Returns:
            tuple: (policy_probs, value)
                - policy_probs (np.ndarray): 方策確率 (65,)
                - value (torch.Tensor): 価値推定 (1,)
        """
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)

            # Log確率 -> 確率
            policy_probs = torch.exp(policy_logits).squeeze(0).cpu().numpy()

        return policy_probs, value

    def _get_board_tensor(self, board) -> torch.Tensor:
        """
        盤面をニューラルネットワーク用のテンソルに変換

        Args:
            board: OthelloBitboard インスタンス

        Returns:
            torch.Tensor: (1, 3, 8, 8) - バッチ次元付き
        """
        tensor = board.get_tensor_input()  # (3, 8, 8)
        tensor = torch.from_numpy(tensor).float().unsqueeze(0)  # (1, 3, 8, 8)
        tensor = tensor.to(self.device)
        return tensor

    def _add_dirichlet_noise(self, root: MCTSNode, legal_actions: list):
        """
        ルートノードの事前確率にディリクレノイズを追加

        探索を促進するため、学習時に使用する

        Args:
            root (MCTSNode): ルートノード
            legal_actions (list): 合法手のリスト
        """
        num_actions = len(legal_actions)
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)

        for i, action in enumerate(legal_actions):
            if action in root.children:
                child = root.children[action]
                # 元の事前確率とノイズを混合
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                              self.dirichlet_epsilon * noise[i]

    def get_action_probs(
        self,
        board,
        num_simulations: int,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
    ) -> np.ndarray:
        """
        MCTS探索を実行し、行動確率を返す（便利メソッド）

        Args:
            board: OthelloBitboard インスタンス
            num_simulations (int): シミュレーション回数
            temperature (float): 温度パラメータ
            add_dirichlet_noise (bool): ディリクレノイズを追加するか

        Returns:
            np.ndarray: 行動確率分布 (65,)
        """
        policy, _ = self.search(
            board,
            num_simulations=num_simulations,
            temperature=temperature,
            add_dirichlet_noise=add_dirichlet_noise
        )
        return policy

    def get_best_action(
        self,
        board,
        num_simulations: int,
    ) -> int:
        """
        MCTS探索を実行し、最良の行動を返す（推論用）

        Args:
            board: OthelloBitboard インスタンス
            num_simulations (int): シミュレーション回数

        Returns:
            int: 最良の行動 (0-64)
        """
        policy, _ = self.search(
            board,
            num_simulations=num_simulations,
            temperature=0.0,  # 決定的な選択
            add_dirichlet_noise=False
        )

        return int(np.argmax(policy))
