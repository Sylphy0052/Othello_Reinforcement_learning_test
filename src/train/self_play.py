"""
Self-Play ワーカー

MCTS を使って自己対戦を行い、学習データを生成する

データ形式:
- state: (3, 8, 8) - NN入力テンソル
- policy: (65,) - MCTS訪問回数分布
- value: 1 or -1 - 最終的な勝敗（手番視点）
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class GameStep:
    """ゲームの1ステップのデータ"""
    state: np.ndarray  # (3, 8, 8)
    policy: np.ndarray  # (65,)
    player: int  # 1 or -1 (手番)


class SelfPlayWorker:
    """
    自己対戦ワーカー

    MCTS を使って1ゲームをプレイし、学習データを生成する
    """

    def __init__(
        self,
        board_class,
        mcts,
        num_simulations: int = 25,
        temperature_threshold: int = 15,
    ):
        """
        Args:
            board_class: OthelloBitboard クラス
            mcts: MCTS インスタンス
            num_simulations (int): MCTSシミュレーション回数
            temperature_threshold (int): 温度を下げる手数の閾値
                （この手数以降は決定的な選択になる）
        """
        self.board_class = board_class
        self.mcts = mcts
        self.num_simulations = num_simulations
        self.temperature_threshold = temperature_threshold

    def execute_episode(
        self,
        add_dirichlet_noise: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        1エピソード（1ゲーム）を実行

        Args:
            add_dirichlet_noise (bool): ディリクレノイズを追加するか
                （学習時はTrue、評価時はFalse）

        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]:
                [(state, policy, value), ...]
                - state: (3, 8, 8)
                - policy: (65,)
                - value: 1.0 (勝ち), -1.0 (負け), 0.0 (引き分け)
        """
        # 盤面の初期化
        board = self.board_class()
        board.reset()

        # ゲームステップを記録
        game_history: List[GameStep] = []

        move_count = 0

        # ゲームが終了するまでループ
        while not board.is_terminal():
            # 現在の手番（1: 黒番, -1: 白番）
            # Bitboard では常に自分の手番視点なので、手番を追跡
            current_player = 1 if (move_count % 2 == 0) else -1

            # 温度パラメータの設定
            # 序盤は確率的、終盤は決定的
            temperature = 1.0 if move_count < self.temperature_threshold else 0.0

            # 現在の盤面状態を保存
            state = board.get_tensor_input()  # (3, 8, 8)

            # MCTS 探索
            policy, _ = self.mcts.search(
                board,
                num_simulations=self.num_simulations,
                temperature=temperature,
                add_dirichlet_noise=add_dirichlet_noise,
            )

            # ゲームステップを記録
            game_history.append(GameStep(
                state=state.copy(),
                policy=policy.copy(),
                player=current_player,
            ))

            # 行動選択
            if temperature == 0:
                # 決定的な選択
                action = int(np.argmax(policy))
            else:
                # 確率的な選択
                action = np.random.choice(len(policy), p=policy)

            # 着手
            board.make_move(action)
            move_count += 1

        # ゲーム終了: 勝者を取得
        winner = board.get_winner()  # 1: 黒勝ち, -1: 白勝ち, 0: 引き分け

        # 学習データを生成（各手番の視点で価値を設定）
        training_data = []
        for step in game_history:
            # 各プレイヤーの視点での価値
            # winner が自分の色なら +1, 相手の色なら -1, 引き分けなら 0
            value = float(winner * step.player)

            training_data.append((
                step.state,
                step.policy,
                value,
            ))

        return training_data

    def execute_episodes(
        self,
        num_episodes: int,
        add_dirichlet_noise: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        複数エピソードを実行

        Args:
            num_episodes (int): 実行するエピソード数
            add_dirichlet_noise (bool): ディリクレノイズを追加するか

        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]:
                すべてのエピソードの学習データ
        """
        all_data = []

        for episode_idx in range(num_episodes):
            episode_data = self.execute_episode(add_dirichlet_noise=add_dirichlet_noise)
            all_data.extend(episode_data)

            # 進捗表示
            if (episode_idx + 1) % 10 == 0:
                print(f"Self-Play: {episode_idx + 1}/{num_episodes} episodes completed")

        return all_data


def augment_data_with_symmetries(
    training_data: List[Tuple[np.ndarray, np.ndarray, float]],
    board_class,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    データ拡張: 盤面の対称性を利用して学習データを8倍に増やす

    Args:
        training_data: 元の学習データ
        board_class: OthelloBitboard クラス

    Returns:
        List[Tuple[np.ndarray, np.ndarray, float]]:
            拡張された学習データ
    """
    augmented_data = []
    board = board_class()

    for state, policy, value in training_data:
        # 元のデータを追加
        augmented_data.append((state, policy, value))

        # 8つの対称性変換を生成
        # get_symmetries は (state, policy) の8パターンを返す
        # ここでは state と policy を個別に変換する必要がある

        # Bitboard の get_symmetries は state と policy をまとめて変換可能
        # しかし、現在の実装では state のみを扱う想定なので、
        # policy も同様に変換する必要がある

        # 簡易実装: 8つの変換を手動で適用
        # （より効率的には Cython 側で実装すべき）

        # 回転と反転のパターン:
        # - 90度回転 x 4
        # - 左右反転
        # - 上下反転
        # - 対角反転

        # 注: 現在の bitboard.get_symmetries はこの機能を持っているため、
        # それを利用する
        # ただし、policy も同じ変換が必要

        # ここでは簡易版として元データのみを使用
        # （TODO: 対称性変換の完全実装）

    return augmented_data
