"""
オセロGUIアプリケーション

Tkinterを使ったオセロ対戦アプリ
- 人間 vs AI
- AI vs AI
- 待った機能
- ヒント表示
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import torch
import numpy as np
from typing import Optional

from src.cython.bitboard import OthelloBitboard
from src.model.net import OthelloResNet
from src.mcts.mcts import MCTS
from .board_ui import OthelloBoardUI, InfoPanel


class OthelloApp:
    """
    オセロGUIアプリケーション

    人間 vs AI の対戦を提供
    """

    def __init__(self, root: tk.Tk, model_path: Optional[str] = None):
        """
        Args:
            root: Tkinterルートウィンドウ
            model_path: 学習済みモデルのパス（オプション）
        """
        self.root = root
        self.root.title("Othello AlphaZero")

        # ゲーム状態
        self.board = OthelloBitboard()
        self.board.reset()
        self.game_history = []  # 履歴（待った用）
        self.player_history = []  # 手番履歴（待った用）
        self.is_ai_thinking = False
        self.game_mode = "human_vs_ai"  # "human_vs_ai", "ai_vs_ai", "human_vs_human"
        self.current_player = 1  # 1=黒, -1=白（黒が先手）

        # AI設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mcts = None
        self.ai_simulations = 50  # AI思考のシミュレーション回数

        # モデルロード
        if model_path:
            self.load_model(model_path)

        # GUI構築
        self._build_gui()

        # 初期盤面表示
        self._update_display()

    def _build_gui(self):
        """GUI構築"""
        # メニューバー
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model", command=self._load_model_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # ゲームメニュー
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self.new_game)
        game_menu.add_command(label="Undo", command=self.undo_move)
        game_menu.add_separator()
        game_menu.add_command(label="Show Hint", command=self._toggle_hint)

        # メインフレーム
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        # 左側: 盤面
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)

        self.board_ui = OthelloBoardUI(
            left_frame,
            board_size=8,
            cell_size=60,
            on_click_callback=self._on_board_click,
        )
        self.board_ui.pack()

        # 右側: 情報パネルとコントロール
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH)

        # 情報パネル
        self.info_panel = InfoPanel(right_frame)
        self.info_panel.pack(pady=10)

        # コントロールボタン
        button_frame = tk.Frame(right_frame)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="New Game",
            command=self.new_game,
            width=12,
            height=2,
        ).pack(pady=5)

        tk.Button(
            button_frame,
            text="Undo",
            command=self.undo_move,
            width=12,
            height=2,
        ).pack(pady=5)

        tk.Button(
            button_frame,
            text="AI Move",
            command=self.ai_move,
            width=12,
            height=2,
        ).pack(pady=5)

        # AI設定
        settings_frame = tk.LabelFrame(right_frame, text="AI Settings", padx=10, pady=10)
        settings_frame.pack(pady=10, fill=tk.BOTH)

        tk.Label(settings_frame, text="Simulations:").pack()
        self.sim_scale = tk.Scale(
            settings_frame,
            from_=10,
            to=200,
            orient=tk.HORIZONTAL,
            variable=tk.IntVar(value=50),
            command=self._update_ai_simulations,
        )
        self.sim_scale.pack()

    def load_model(self, model_path: str):
        """
        学習済みモデルを読み込む

        Args:
            model_path: モデルファイルのパス
        """
        try:
            # チェックポイント読み込み
            checkpoint = torch.load(model_path, map_location=self.device)

            # モデル作成（設定から、またはstate_dictから自動検出）
            config = checkpoint.get("config", {})
            state_dict = checkpoint["model_state_dict"]

            # state_dictからブロック数を自動検出
            if "num_blocks" not in config:
                block_keys = [k for k in state_dict.keys() if k.startswith("res_blocks.")]
                if block_keys:
                    block_indices = [int(k.split(".")[1]) for k in block_keys]
                    config["num_blocks"] = max(block_indices) + 1

            # state_dictからフィルタ数を自動検出
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

            # MCTS作成
            self.mcts = MCTS(
                model=self.model,
                device=self.device,
                c_puct=1.0,
            )

            messagebox.showinfo("Success", f"Model loaded:\n{model_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def _load_model_dialog(self):
        """モデル読み込みダイアログ"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialdir="data/models",
        )

        if file_path:
            self.load_model(file_path)

    def new_game(self):
        """新規ゲームを開始"""
        self.board.reset()
        self.game_history = []
        self.player_history = []
        self.is_ai_thinking = False
        self.current_player = 1  # 黒が先手
        self._update_display()
        self.info_panel.set_message("New game started", "green")

    def undo_move(self):
        """待った（1手戻す）"""
        if len(self.game_history) == 0:
            messagebox.showinfo("Undo", "No more moves to undo")
            return

        # 履歴から1つ前の状態を復元
        self.board = self.game_history.pop()
        self.current_player = self.player_history.pop()
        self._update_display()
        self.info_panel.set_message("Move undone", "orange")

    def _on_board_click(self, position: int):
        """
        盤面クリック時のコールバック

        Args:
            position: クリックされた位置 (0-63)
        """
        if self.is_ai_thinking:
            messagebox.showinfo("Wait", "AI is thinking...")
            return

        if self.board.is_terminal():
            messagebox.showinfo("Game Over", "The game has already ended")
            return

        # 合法手チェック
        legal_moves = self.board.get_legal_moves()
        if position not in legal_moves:
            messagebox.showwarning("Invalid Move", "You cannot place here")
            return

        # 履歴保存（待った用）
        self.game_history.append(self.board.copy())
        self.player_history.append(self.current_player)

        # 着手
        self.board.make_move(position)
        self.current_player *= -1  # 手番交代
        self._update_display()

        # 終局チェック
        if self.board.is_terminal():
            self._show_game_result()
            return

        # AIの手番なら自動で着手
        if self.game_mode == "human_vs_ai" and self.mcts is not None:
            self.root.after(500, self.ai_move)  # 0.5秒後にAI着手

    def ai_move(self):
        """AIに着手させる"""
        if self.mcts is None:
            messagebox.showwarning("Warning", "No model loaded")
            return

        if self.board.is_terminal():
            return

        # 別スレッドでAI思考
        self.is_ai_thinking = True
        self.info_panel.set_message("AI thinking...", "blue")

        thread = threading.Thread(target=self._ai_move_thread)
        thread.daemon = True
        thread.start()

    def _ai_move_thread(self):
        """AI思考スレッド"""
        try:
            # MCTS探索
            action = self.mcts.get_best_action(
                self.board,
                num_simulations=self.ai_simulations
            )

            # メインスレッドで着手
            self.root.after(0, lambda: self._execute_ai_move(action))

        except Exception as e:
            err_msg = str(e)
            self.root.after(
                0, lambda msg=err_msg: messagebox.showerror("Error", f"AI error:\n{msg}")
            )
            self.is_ai_thinking = False

    def _execute_ai_move(self, action: int):
        """
        AI着手を実行（メインスレッド）

        Args:
            action: 着手位置
        """
        # AIが選んだ手が合法手かチェック
        legal_moves = self.board.get_legal_moves()
        if action not in legal_moves:
            # 合法手でない場合、合法手からランダムに選択
            import random
            valid_moves = [m for m in legal_moves if m < 64]
            if valid_moves:
                action = random.choice(valid_moves)
            else:
                action = 64  # パス
            self.info_panel.set_message(f"AI fallback to {action}", "orange")

        # 履歴保存
        self.game_history.append(self.board.copy())
        self.player_history.append(self.current_player)

        # 着手
        self.board.make_move(action)
        self.current_player *= -1  # 手番交代
        self._update_display()

        self.is_ai_thinking = False
        row, col = action // 8, action % 8
        coord = f"{chr(65 + col)}{row + 1}"
        self.info_panel.set_message(f"AI played at {coord}", "green")

        # 終局チェック
        if self.board.is_terminal():
            self._show_game_result()

    def _update_display(self):
        """盤面表示を更新"""
        # 盤面状態を取得
        board_array = self._get_board_array()
        legal_moves = self.board.get_legal_moves()

        # パス(64)を除外
        legal_moves = [m for m in legal_moves if m < 64]

        # 盤面更新
        self.board_ui.update_board(board_array, legal_moves)

        # 石数更新（get_stone_countsは自分視点なので変換が必要）
        self_count, opp_count = self.board.get_stone_counts()
        is_black_turn = self.board.move_count % 2 == 0
        if is_black_turn:
            # 黒番: self=黒, opp=白
            black_count, white_count = self_count, opp_count
        else:
            # 白番: self=白, opp=黒
            black_count, white_count = opp_count, self_count
        self.info_panel.update_scores(black_count, white_count)

        # 手番更新（move_countから判定）
        current_player = 1 if is_black_turn else -1
        self.info_panel.update_turn(current_player)

    def _get_board_array(self) -> np.ndarray:
        """
        現在の盤面を8x8配列に変換

        Returns:
            np.ndarray: (8, 8) 配列 (0=空, 1=黒, -1=白)
        """
        tensor = self.board.get_tensor_input()  # (3, 8, 8)

        # ビットボードのself_boardが黒か白かを判定
        # move_count % 2 == 0 なら黒番（self=黒）
        # パスがあるとずれるが、パス時もmove_countを増加させるように修正済み
        is_black_turn = self.board.move_count % 2 == 0

        if is_black_turn:
            # self_board = 黒
            board_array = tensor[0] - tensor[1]
        else:
            # self_board = 白
            board_array = tensor[1] - tensor[0]

        return board_array

    def _show_game_result(self):
        """ゲーム終了時の結果表示"""
        winner = self.board.get_winner()
        black_count, white_count = self.board.get_stone_counts()

        if winner == 1:
            result = f"Black wins!\nBlack: {black_count} - White: {white_count}"
        elif winner == -1:
            result = f"White wins!\nBlack: {black_count} - White: {white_count}"
        else:
            result = f"Draw\nBlack: {black_count} - White: {white_count}"

        messagebox.showinfo("Game Over", result)
        self.info_panel.set_message(result.replace("\n", " "), "red")

    def _toggle_hint(self):
        """ヒント表示の切り替え"""
        self.board_ui.toggle_evaluation()

    def _update_ai_simulations(self, value):
        """AI思考時間（シミュレーション回数）を更新"""
        self.ai_simulations = int(value)

    def run(self):
        """アプリケーションを実行"""
        self.root.mainloop()
