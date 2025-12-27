"""
並列Self-Playワーカー

複数のゲームを同時に進行し、NN評価をバッチ化することで
GPUスループットを最大化する。

高速化のポイント:
1. 複数ゲームを同時進行
2. 各ゲームのMCTS探索でリーフノード評価をバッチ化
3. GPUの並列処理能力を最大限活用
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class GameState:
    """並列ゲームの状態管理"""
    board: object  # OthelloBitboard
    history: List[Tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)
    move_count: int = 0
    is_finished: bool = False
    winner: int = 0


class BatchMCTS:
    """
    バッチ推論対応のMCTS

    複数のゲーム状態をまとめて評価し、GPUスループットを向上させる
    """

    def __init__(
        self,
        model,
        device,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self._lock = threading.Lock()

    def batch_predict(self, boards: List[object]) -> Tuple[np.ndarray, np.ndarray]:
        """
        複数の盤面をバッチ評価

        Args:
            boards: OthelloBitboardのリスト

        Returns:
            (policies, values): 各盤面の方策と価値
        """
        if len(boards) == 0:
            return np.array([]), np.array([])

        # バッチテンソル作成
        tensors = [board.get_tensor_input() for board in boards]
        batch = np.stack(tensors, axis=0)  # (N, 3, 8, 8)
        batch_tensor = torch.from_numpy(batch).float().to(self.device)

        # 推論
        self.model.eval()
        with torch.no_grad():
            policy_logits, values = self.model(batch_tensor)
            policies = torch.exp(policy_logits).cpu().numpy()
            values = values.cpu().numpy()

        return policies, values

    def search_batch(
        self,
        boards: List[object],
        num_simulations: int,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        複数の盤面に対してMCTS探索を実行

        Args:
            boards: OthelloBitboardのリスト
            num_simulations: シミュレーション回数
            temperature: 温度パラメータ
            add_dirichlet_noise: ディリクレノイズを追加するか

        Returns:
            [(policy, value), ...]: 各盤面の方策と価値
        """
        from src.mcts.node import MCTSNode

        n_boards = len(boards)
        if n_boards == 0:
            return []

        # 各盤面のルートノードを作成
        roots = [MCTSNode(prior=1.0) for _ in range(n_boards)]

        # 初期展開: バッチ評価
        policies, values = self.batch_predict(boards)

        for i, (board, root, policy) in enumerate(zip(boards, roots, policies)):
            legal_actions = board.get_legal_moves()
            if len(legal_actions) == 0:
                legal_actions = [64]
            root.expand(policy, legal_actions)

            if add_dirichlet_noise:
                self._add_dirichlet_noise(root, legal_actions)

        # シミュレーション実行
        for _ in range(num_simulations):
            # 各ゲームのリーフノードを収集
            leaf_boards = []
            leaf_indices = []
            paths = []

            for i, (board, root) in enumerate(zip(boards, roots)):
                board_copy = board.copy()
                path, leaf_board, is_terminal, terminal_value = self._select_leaf(
                    root, board_copy
                )

                if is_terminal:
                    # 終局ノード: 即座にバックプロパゲート
                    self._backpropagate(path, terminal_value)
                else:
                    # 評価が必要なリーフノード
                    leaf_boards.append(leaf_board)
                    leaf_indices.append(i)
                    paths.append(path)

            # リーフノードをバッチ評価
            if len(leaf_boards) > 0:
                policies, values = self.batch_predict(leaf_boards)

                for j, (idx, path) in enumerate(zip(leaf_indices, paths)):
                    board_copy = leaf_boards[j]
                    policy = policies[j]
                    value = values[j].item()

                    # リーフノードを展開
                    leaf_node = path[-1][2] if path else roots[idx]
                    legal_actions = board_copy.get_legal_moves()
                    if len(legal_actions) == 0:
                        legal_actions = [64]

                    if not leaf_node.is_expanded:
                        leaf_node.expand(policy, legal_actions)

                    # バックプロパゲート
                    self._backpropagate(path, value)

        # 方策分布を生成
        results = []
        for root in roots:
            policy = root.get_policy_distribution(temperature)
            value = root.get_value()
            results.append((policy, value))

        return results

    def _select_leaf(
        self,
        root,
        board,
    ) -> Tuple[list, object, bool, float]:
        """
        リーフノードまで選択

        Returns:
            (path, board, is_terminal, terminal_value)
        """
        path = []
        current_node = root

        while not current_node.is_leaf():
            action, child = current_node.select_child(self.c_puct)
            path.append((current_node, action, child))
            board.make_move(action)
            current_node = child

        # 終局チェック
        if board.is_terminal():
            winner = board.get_winner()
            return path, board, True, float(winner)

        return path, board, False, 0.0

    def _backpropagate(self, path: list, value: float):
        """価値をルートまで伝播"""
        for i in range(len(path) - 1, -1, -1):
            parent, action, child = path[i]
            child.update(value)
            value = -value

    def _add_dirichlet_noise(self, root, legal_actions: list):
        """ディリクレノイズを追加"""
        num_actions = len(legal_actions)
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)

        for i, action in enumerate(legal_actions):
            if action in root.children:
                child = root.children[action]
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                              self.dirichlet_epsilon * noise[i]


class ParallelSelfPlayWorker:
    """
    並列Self-Playワーカー

    複数のゲームを同時に進行し、効率的にデータ生成を行う
    """

    def __init__(
        self,
        board_class,
        model,
        device,
        num_simulations: int = 25,
        temperature_threshold: int = 15,
        num_parallel_games: int = 8,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        """
        Args:
            board_class: OthelloBitboard クラス
            model: ニューラルネットワークモデル
            device: torch.device
            num_simulations: MCTSシミュレーション回数
            temperature_threshold: 温度を下げる手数の閾値
            num_parallel_games: 同時に進行するゲーム数
        """
        self.board_class = board_class
        self.num_simulations = num_simulations
        self.temperature_threshold = temperature_threshold
        self.num_parallel_games = num_parallel_games

        # バッチMCTS
        self.batch_mcts = BatchMCTS(
            model=model,
            device=device,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

    def execute_episodes(
        self,
        num_episodes: int,
        add_dirichlet_noise: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        複数エピソードを並列実行

        Args:
            num_episodes: 実行するエピソード数
            add_dirichlet_noise: ディリクレノイズを追加するか

        Returns:
            [(state, policy, value), ...]: すべてのエピソードの学習データ
        """
        all_data = []
        completed = 0

        while completed < num_episodes:
            # 並列ゲーム数を決定
            remaining = num_episodes - completed
            batch_size = min(self.num_parallel_games, remaining)

            # バッチでゲームを実行
            batch_data = self._execute_batch(
                batch_size=batch_size,
                add_dirichlet_noise=add_dirichlet_noise,
            )
            all_data.extend(batch_data)

            completed += batch_size

            # 進捗表示
            if completed % 10 == 0 or completed == num_episodes:
                print(f"Self-Play: {completed}/{num_episodes} episodes completed")

        return all_data

    def _execute_batch(
        self,
        batch_size: int,
        add_dirichlet_noise: bool,
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        バッチ単位でゲームを実行

        Args:
            batch_size: 同時実行するゲーム数
            add_dirichlet_noise: ディリクレノイズを追加するか

        Returns:
            [(state, policy, value), ...]: 学習データ
        """
        # ゲーム状態を初期化
        games = []
        for _ in range(batch_size):
            board = self.board_class()
            board.reset()
            games.append(GameState(board=board))

        # 全ゲームが終了するまでループ
        while True:
            # アクティブなゲームを収集
            active_games = [(i, g) for i, g in enumerate(games) if not g.is_finished]

            if len(active_games) == 0:
                break

            # 温度パラメータを設定
            temperatures = []
            for _, g in active_games:
                temp = 1.0 if g.move_count < self.temperature_threshold else 0.0
                temperatures.append(temp)

            # 盤面を収集
            boards = [g.board for _, g in active_games]

            # 現在の状態を保存
            states = [board.get_tensor_input().copy() for board in boards]

            # バッチMCTS探索
            results = self.batch_mcts.search_batch(
                boards=boards,
                num_simulations=self.num_simulations,
                temperature=1.0,  # バッチでは一旦1.0で探索
                add_dirichlet_noise=add_dirichlet_noise,
            )

            # 各ゲームで着手
            for j, ((idx, game), (policy, _), state, temp) in enumerate(
                zip(active_games, results, states, temperatures)
            ):
                # 温度に応じた方策を生成
                if temp == 0:
                    action = int(np.argmax(policy))
                else:
                    action = np.random.choice(len(policy), p=policy)

                # 現在の手番を記録
                current_player = 1 if (game.move_count % 2 == 0) else -1

                # 履歴に追加
                game.history.append((state, policy.copy(), current_player))

                # 着手
                game.board.make_move(action)
                game.move_count += 1

                # 終局チェック
                if game.board.is_terminal():
                    game.is_finished = True
                    game.winner = game.board.get_winner()

        # 学習データを生成
        all_data = []
        for game in games:
            for state, policy, player in game.history:
                # 各プレイヤーの視点での価値
                value = float(game.winner * player)
                all_data.append((state, policy, value))

        return all_data


def create_parallel_self_play_worker(config: dict, model, device):
    """
    設定からParallelSelfPlayWorkerを作成

    Args:
        config: 設定辞書
        model: ニューラルネットワークモデル
        device: torch.device

    Returns:
        ParallelSelfPlayWorker
    """
    from src.cython.bitboard import OthelloBitboard

    return ParallelSelfPlayWorker(
        board_class=OthelloBitboard,
        model=model,
        device=device,
        num_simulations=config.get('mcts', {}).get('num_simulations', 25),
        temperature_threshold=config.get('self_play', {}).get('temperature_threshold', 15),
        num_parallel_games=config.get('self_play', {}).get('num_parallel_games', 8),
        c_puct=config.get('mcts', {}).get('c_puct', 1.0),
        dirichlet_alpha=config.get('mcts', {}).get('dirichlet_alpha', 0.3),
        dirichlet_epsilon=config.get('mcts', {}).get('dirichlet_epsilon', 0.25),
    )
