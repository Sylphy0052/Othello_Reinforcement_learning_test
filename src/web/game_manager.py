"""
ゲーム状態管理クラス

OthelloAppからゲームロジックを抽出し、
TkinterとWebの両方で共有可能にする
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import torch

from src.cython.bitboard import OthelloBitboard
from src.model.net import OthelloResNet
from src.mcts.mcts import MCTS
from .schemas import GameState


class GameManager:
    """
    Webインターフェース向けゲーム状態管理

    OthelloApp(src/gui/app.py)のゲームロジック部分を抽出
    Tkinterに依存しない形で実装
    """

    def __init__(self) -> None:
        # ゲーム状態
        self.board = OthelloBitboard()
        self.board.reset()
        self.game_history: List[OthelloBitboard] = []
        self.player_history: List[int] = []
        self.is_ai_thinking = False
        self.current_player = 1  # 1=黒, -1=白
        self.game_mode = "human_vs_ai"
        self.last_message: Optional[str] = None

        # AI設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[OthelloResNet] = None
        self.mcts: Optional[MCTS] = None
        self.ai_simulations = 50

    def new_game(self, mode: str = "human_vs_ai") -> None:
        """
        新規ゲーム開始

        Args:
            mode: ゲームモード (human_vs_ai, human_vs_human, ai_vs_ai)
        """
        self.board.reset()
        self.game_history = []
        self.player_history = []
        self.is_ai_thinking = False
        self.current_player = 1
        self.game_mode = mode
        self.last_message = "New game started"

    def make_move(self, position: int) -> Tuple[bool, Optional[str]]:
        """
        着手を実行

        Args:
            position: 着手位置 (0-63) またはパス (64)

        Returns:
            (成功フラグ, エラーメッセージ)
        """
        if self.is_ai_thinking:
            return False, "AI is thinking..."

        if self.board.is_terminal():
            return False, "Game has already ended"

        legal_moves = self.board.get_legal_moves()
        if position not in legal_moves:
            return False, f"Invalid move: position {position} is not legal"

        # 履歴保存
        self.game_history.append(self.board.copy())
        self.player_history.append(self.current_player)

        # 着手
        self.board.make_move(position)
        self.current_player *= -1

        # メッセージ更新
        if position == 64:
            self.last_message = "Pass"
        else:
            row, col = position // 8, position % 8
            coord = f"{chr(65 + col)}{row + 1}"
            self.last_message = f"Moved to {coord}"

        return True, None

    def undo(self) -> Tuple[bool, Optional[str]]:
        """
        一手戻す

        Returns:
            (成功フラグ, エラーメッセージ)
        """
        if len(self.game_history) == 0:
            return False, "No moves to undo"

        self.board = self.game_history.pop()
        self.current_player = self.player_history.pop()
        self.last_message = "Move undone"
        return True, None

    def get_ai_move(self) -> Tuple[int, Optional[str]]:
        """
        AIの最善手を取得

        Returns:
            (着手位置, エラーメッセージ)
        """
        if self.mcts is None:
            return -1, "No model loaded"

        if self.board.is_terminal():
            return -1, "Game has ended"

        try:
            action = self.mcts.get_best_action(
                self.board, num_simulations=self.ai_simulations
            )

            # 合法手チェック
            legal_moves = self.board.get_legal_moves()
            if action not in legal_moves:
                # フォールバック: 合法手からランダム選択
                import random

                valid_moves = [m for m in legal_moves if m < 64]
                if valid_moves:
                    action = random.choice(valid_moves)
                else:
                    action = 64  # パス

            return action, None

        except Exception as e:
            return -1, str(e)

    def execute_ai_move(self) -> Tuple[bool, Optional[str]]:
        """
        AIに着手させる

        Returns:
            (成功フラグ, エラーメッセージ)
        """
        action, error = self.get_ai_move()
        if error:
            return False, error

        # 履歴保存
        self.game_history.append(self.board.copy())
        self.player_history.append(self.current_player)

        # 着手
        self.board.make_move(action)
        self.current_player *= -1

        # メッセージ更新
        if action == 64:
            self.last_message = "AI passed"
        else:
            row, col = action // 8, action % 8
            coord = f"{chr(65 + col)}{row + 1}"
            self.last_message = f"AI played at {coord}"

        return True, None

    def get_hint_evaluations(self) -> Tuple[Dict[int, int], Optional[str]]:
        """
        ヒント評価値を取得

        Returns:
            (評価値辞書 {position: value}, エラーメッセージ)
        """
        if self.mcts is None:
            return {}, "No model loaded"

        if self.board.is_terminal():
            return {}, "Game has ended"

        try:
            hint_simulations = max(10, self.ai_simulations // 2)
            evaluations = self.mcts.get_action_evaluations(
                self.board, num_simulations=hint_simulations
            )

            # numpy配列を辞書に変換
            result = {}
            legal_moves = self.board.get_legal_moves()
            for pos in legal_moves:
                if pos < 64:
                    result[pos] = int(evaluations[pos])

            return result, None

        except Exception as e:
            return {}, str(e)

    def load_model(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """
        モデルを読み込む

        OthelloApp.load_model() と同等の処理

        Args:
            model_path: モデルファイルのパス

        Returns:
            (成功フラグ, エラーメッセージ)
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # モデル設定を取得または自動検出
            config = checkpoint.get("config", {})
            state_dict = checkpoint["model_state_dict"]

            # ブロック数を自動検出
            if "num_blocks" not in config:
                block_keys = [
                    k for k in state_dict.keys() if k.startswith("res_blocks.")
                ]
                if block_keys:
                    block_indices = [int(k.split(".")[1]) for k in block_keys]
                    config["num_blocks"] = max(block_indices) + 1

            # フィルタ数を自動検出
            if "num_filters" not in config:
                for key in state_dict.keys():
                    if "res_blocks.0.conv1.weight" in key:
                        config["num_filters"] = state_dict[key].shape[0]
                        break

            self.model = OthelloResNet(
                num_blocks=config.get("num_blocks", 10),
                num_filters=config.get("num_filters", 128),
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.mcts = MCTS(
                model=self.model,
                device=self.device,
                c_puct=1.0,
            )

            self.last_message = f"Model loaded: {model_path}"
            return True, None

        except Exception as e:
            return False, str(e)

    def set_simulations(self, count: int) -> None:
        """
        AIシミュレーション回数を設定

        Args:
            count: シミュレーション回数 (10-500)
        """
        self.ai_simulations = max(10, min(500, count))

    def get_board_array(self) -> np.ndarray:
        """
        現在の盤面を8x8配列に変換

        OthelloApp._get_board_array() と同等の処理

        Returns:
            (8, 8) 配列 (0=空, 1=黒, -1=白)
        """
        tensor = self.board.get_tensor_input()  # (3, 8, 8)

        # move_count % 2 == 0 なら黒番（self=黒）
        is_black_turn = self.board.move_count % 2 == 0

        if is_black_turn:
            board_array = tensor[0] - tensor[1]
        else:
            board_array = tensor[1] - tensor[0]

        return board_array

    def get_state(self) -> GameState:
        """
        現在のゲーム状態を取得

        Returns:
            GameState: Pydanticモデル
        """
        board_array = self.get_board_array()
        legal_moves = self.board.get_legal_moves()

        # パス(64)を除外
        filtered_legal_moves = [m for m in legal_moves if m < 64]

        # 石数を取得
        self_count, opp_count = self.board.get_stone_counts()
        is_black_turn = self.board.move_count % 2 == 0

        if is_black_turn:
            black_count, white_count = self_count, opp_count
            current_player = 1
        else:
            black_count, white_count = opp_count, self_count
            current_player = -1

        # 勝者判定
        winner = None
        if self.board.is_terminal():
            winner = self.board.get_winner()

        return GameState(
            board=board_array.tolist(),
            legal_moves=filtered_legal_moves,
            current_player=current_player,
            black_count=int(black_count),
            white_count=int(white_count),
            is_terminal=self.board.is_terminal(),
            winner=winner,
            is_ai_thinking=self.is_ai_thinking,
            move_count=self.board.move_count,
            message=self.last_message,
            model_loaded=self.model is not None,
        )
