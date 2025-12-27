"""
オセロプレイヤークラス

評価用の様々なプレイヤーを実装:
- RandomPlayer: ランダムに着手
- GreedyPlayer: 最も多く石を取れる手を選択
- MCTSPlayer: MCTSベースのAI
- EdaxPlayer: Edaxエンジン（外部プログラム）
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
import torch

from src.cython.bitboard import OthelloBitboard


class Player(ABC):
    """
    プレイヤーの基底クラス
    """

    def __init__(self, name: str):
        """
        Args:
            name: プレイヤー名
        """
        self.name = name

    @abstractmethod
    def get_action(self, board: OthelloBitboard) -> int:
        """
        着手を選択

        Args:
            board: 現在の盤面

        Returns:
            int: 着手位置 (0-64)
        """
        pass

    def reset(self):
        """ゲーム開始時の初期化（必要に応じてオーバーライド）"""
        pass


class RandomPlayer(Player):
    """
    ランダムプレイヤー

    合法手の中からランダムに選択
    """

    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def get_action(self, board: OthelloBitboard) -> int:
        """ランダムに着手を選択"""
        legal_moves = board.get_legal_moves()

        if len(legal_moves) == 0:
            return 64  # パス

        return random.choice(legal_moves)


class GreedyPlayer(Player):
    """
    貪欲プレイヤー

    最も多く石を取れる（反転できる）手を選択
    """

    def __init__(self, name: str = "Greedy"):
        super().__init__(name)

    def get_action(self, board: OthelloBitboard) -> int:
        """最も多く石を取れる手を選択"""
        legal_moves = board.get_legal_moves()

        if len(legal_moves) == 0:
            return 64  # パス

        best_action = legal_moves[0]
        best_score = -1

        # 各手を試して、最も石数が増える手を選ぶ
        for action in legal_moves:
            # 盤面をコピーして試す
            test_board = board.copy()
            test_board.make_move(action)

            # 自分の石数を取得（着手後なので相手視点になっている）
            black_count, white_count = test_board.get_stone_counts()

            # 現在の手番が黒か白かを判定（元の盤面の手数から）
            move_count = board.move_count
            if move_count % 2 == 0:
                # 黒番: 着手後は白視点なので white_count が自分の石
                score = white_count
            else:
                # 白番: 着手後は黒視点なので black_count が自分の石
                score = black_count

            if score > best_score:
                best_score = score
                best_action = action

        return best_action


class MCTSPlayer(Player):
    """
    MCTSベースのAIプレイヤー

    学習済みモデルとMCTSを使用
    """

    def __init__(
        self,
        model,
        device: torch.device,
        num_simulations: int = 50,
        name: str = "MCTS-AI",
    ):
        """
        Args:
            model: 学習済みニューラルネットワーク
            device: torch.device
            num_simulations: MCTSシミュレーション回数
            name: プレイヤー名
        """
        super().__init__(name)

        from src.mcts.mcts import MCTS

        self.model = model
        self.device = device
        self.num_simulations = num_simulations

        # MCTS作成
        self.mcts = MCTS(
            model=model,
            device=device,
            c_puct=1.0,
        )

    def get_action(self, board: OthelloBitboard) -> int:
        """MCTSで最良の手を選択"""
        action = self.mcts.get_best_action(
            board,
            num_simulations=self.num_simulations
        )
        return action

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device,
        num_simulations: int = 50,
    ):
        """
        チェックポイントからMCTSPlayerを作成

        Args:
            checkpoint_path: モデルチェックポイントのパス
            device: torch.device
            num_simulations: MCTSシミュレーション回数

        Returns:
            MCTSPlayer: インスタンス
        """
        from src.model.net import OthelloResNet

        # チェックポイント読み込み
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # モデル設定を取得または推測
        config = checkpoint.get("config", {})

        # state_dictからモデル設定を推測
        state_dict = checkpoint["model_state_dict"]

        # フィルタ数を推測（conv_blockの重みから）
        if "conv_block.conv.weight" in state_dict:
            num_filters = state_dict["conv_block.conv.weight"].shape[0]
        else:
            num_filters = config.get("num_filters", 128)

        # ブロック数を推測（res_blocksの数から）
        num_blocks = 0
        for key in state_dict.keys():
            if key.startswith("res_blocks."):
                block_idx = int(key.split(".")[1])
                num_blocks = max(num_blocks, block_idx + 1)

        if num_blocks == 0:
            num_blocks = config.get("num_blocks", 10)

        print(f"Detected model config: blocks={num_blocks}, filters={num_filters}")

        # モデル作成
        model = OthelloResNet(
            num_blocks=num_blocks,
            num_filters=num_filters,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # プレイヤー作成
        player = cls(
            model=model,
            device=device,
            num_simulations=num_simulations,
            name=f"MCTS-AI-{num_simulations}sim"
        )

        return player


class EdaxPlayer(Player):
    """
    Edaxエンジンプレイヤー

    外部プログラムEdaxを使用（要インストール）
    """

    def __init__(
        self,
        level: int = 1,
        edax_path: str = "edax",
        name: Optional[str] = None,
    ):
        """
        Args:
            level: Edaxのレベル (1-21)
            edax_path: Edaxの実行ファイルパス
            name: プレイヤー名
        """
        if name is None:
            name = f"Edax-L{level}"

        super().__init__(name)

        self.level = level
        self.edax_path = edax_path

        # TODO: Edax連携の実装
        # 現在は未実装のため、ランダムプレイヤーとして動作
        print(f"Warning: EdaxPlayer not fully implemented yet. Using random moves.")
        self._fallback_player = RandomPlayer()

    def get_action(self, board: OthelloBitboard) -> int:
        """Edaxに問い合わせて着手を選択"""
        # TODO: Edax実装
        # 現在はフォールバック
        return self._fallback_player.get_action(board)


class HumanPlayer(Player):
    """
    人間プレイヤー（CLI用）

    標準入力から着手を受け付ける
    """

    def __init__(self, name: str = "Human"):
        super().__init__(name)

    def get_action(self, board: OthelloBitboard) -> int:
        """標準入力から着手を受け付ける"""
        legal_moves = board.get_legal_moves()

        if len(legal_moves) == 0:
            print("パスします")
            return 64

        print(f"\n合法手: {legal_moves}")

        while True:
            try:
                move_input = input("着手を入力してください (0-63 or row,col): ").strip()

                # カンマ区切りの場合 (row, col)
                if "," in move_input:
                    row, col = map(int, move_input.split(","))
                    action = row * 8 + col
                else:
                    action = int(move_input)

                if action in legal_moves:
                    return action
                else:
                    print(f"無効な着手です。合法手: {legal_moves}")

            except (ValueError, KeyboardInterrupt):
                print("入力エラー。もう一度入力してください。")
