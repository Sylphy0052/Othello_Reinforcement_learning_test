"""
オセロ盤面描画コンポーネント

Tkinterを使って盤面を描画し、マウスクリックで着手できるようにする
"""

import tkinter as tk
from tkinter import messagebox
from typing import Optional, Callable, List
import numpy as np


class OthelloBoardUI(tk.Canvas):
    """
    オセロ盤面描画用Canvas

    機能:
    - 8x8盤面の描画
    - 石の描画（黒・白）
    - 合法手のハイライト表示
    - マウスクリックでの着手
    - 評価値の表示（オプション）
    """

    def __init__(
        self,
        parent,
        board_size: int = 8,
        cell_size: int = 60,
        on_click_callback: Optional[Callable[[int], None]] = None,
    ):
        """
        Args:
            parent: 親ウィジェット
            board_size: 盤面サイズ（デフォルト: 8）
            cell_size: セルのサイズ（ピクセル）
            on_click_callback: クリック時のコールバック関数 (position: int) -> None
        """
        self.board_size = board_size
        self.cell_size = cell_size
        self.on_click_callback = on_click_callback

        # Canvasの作成
        canvas_size = board_size * cell_size
        super().__init__(
            parent,
            width=canvas_size,
            height=canvas_size,
            bg="#006400",  # 深緑
            highlightthickness=0,
        )

        # 現在の盤面状態
        self.board_state = None  # (8, 8) numpy配列: 0=空, 1=黒, -1=白
        self.legal_moves = []    # 合法手のリスト

        # 評価値表示フラグ
        self.show_evaluation = False
        self.evaluation_values = None  # (65,) numpy配列

        # グリッド描画
        self._draw_grid()

        # マウスクリックイベント
        self.bind("<Button-1>", self._on_click)

    def _draw_grid(self):
        """グリッド線を描画"""
        for i in range(self.board_size + 1):
            # 縦線
            x = i * self.cell_size
            self.create_line(
                x, 0, x, self.board_size * self.cell_size,
                fill="#004d00", width=2
            )

            # 横線
            y = i * self.cell_size
            self.create_line(
                0, y, self.board_size * self.cell_size, y,
                fill="#004d00", width=2
            )

    def update_board(
        self,
        board_state: np.ndarray,
        legal_moves: List[int],
        evaluation: Optional[np.ndarray] = None,
    ):
        """
        盤面を更新

        Args:
            board_state: (8, 8) numpy配列 (0=空, 1=黒, -1=白)
            legal_moves: 合法手のリスト [0-63]
            evaluation: 評価値（オプション）
        """
        self.board_state = board_state
        self.legal_moves = legal_moves
        self.evaluation_values = evaluation

        self._redraw()

    def _redraw(self):
        """盤面全体を再描画"""
        # 既存の石をクリア
        self.delete("stone")
        self.delete("legal")
        self.delete("eval")

        if self.board_state is None:
            return

        # 各マスを描画
        for row in range(self.board_size):
            for col in range(self.board_size):
                pos = row * self.board_size + col
                cell_value = self.board_state[row, col]

                # 石を描画
                if cell_value != 0:
                    self._draw_stone(row, col, cell_value)

                # 合法手のハイライト
                elif pos in self.legal_moves:
                    self._draw_legal_hint(row, col)

                # 評価値の表示
                if self.show_evaluation and self.evaluation_values is not None:
                    if pos < len(self.evaluation_values):
                        self._draw_evaluation(row, col, self.evaluation_values[pos])

    def _draw_stone(self, row: int, col: int, player: int):
        """
        石を描画

        Args:
            row: 行
            col: 列
            player: 1=黒, -1=白
        """
        x1 = col * self.cell_size + 5
        y1 = row * self.cell_size + 5
        x2 = (col + 1) * self.cell_size - 5
        y2 = (row + 1) * self.cell_size - 5

        color = "black" if player == 1 else "white"

        self.create_oval(
            x1, y1, x2, y2,
            fill=color,
            outline="gray",
            width=2,
            tags="stone"
        )

    def _draw_legal_hint(self, row: int, col: int):
        """
        合法手のヒント（小さい円）を描画

        Args:
            row: 行
            col: 列
        """
        center_x = col * self.cell_size + self.cell_size // 2
        center_y = row * self.cell_size + self.cell_size // 2
        radius = 8

        self.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            fill="#90EE90",  # ライトグリーン
            outline="",
            tags="legal"
        )

    def _draw_evaluation(self, row: int, col: int, value: float):
        """
        評価値を描画

        Args:
            row: 行
            col: 列
            value: 評価値
        """
        center_x = col * self.cell_size + self.cell_size // 2
        center_y = row * self.cell_size + self.cell_size // 2

        # 評価値を0-100のスケールに変換
        eval_str = f"{value * 100:.1f}"

        self.create_text(
            center_x, center_y,
            text=eval_str,
            fill="yellow",
            font=("helvetica", 8),
            tags="eval"
        )

    def _on_click(self, event):
        """
        マウスクリックイベント

        Args:
            event: Tkinterイベント
        """
        # クリック位置からマス目を計算
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        # 範囲チェック
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return

        pos = row * self.board_size + col

        # 合法手かチェック
        if pos not in self.legal_moves:
            messagebox.showwarning("Invalid Move", "You cannot place here")
            return

        # コールバック実行
        if self.on_click_callback:
            self.on_click_callback(pos)

    def toggle_evaluation(self):
        """評価値表示の切り替え"""
        self.show_evaluation = not self.show_evaluation
        self._redraw()

    def clear(self):
        """盤面をクリア"""
        self.board_state = None
        self.legal_moves = []
        self.evaluation_values = None
        self.delete("stone")
        self.delete("legal")
        self.delete("eval")


class InfoPanel(tk.Frame):
    """
    情報パネル

    ゲーム状態、石数、メッセージなどを表示
    """

    def __init__(self, parent):
        super().__init__(parent, bg="#f0f0f0", padx=10, pady=10)

        # 現在の手番表示
        self.turn_label = tk.Label(
            self, text="Turn: Black", font=("helvetica", 14, "bold"), bg="#f0f0f0"
        )
        self.turn_label.pack(pady=5)

        # 石数表示
        self.score_frame = tk.Frame(self, bg="#f0f0f0")
        self.score_frame.pack(pady=5)

        tk.Label(
            self.score_frame, text="Black: ", font=("helvetica", 12), bg="#f0f0f0"
        ).grid(row=0, column=0, sticky="e")

        self.black_score_label = tk.Label(
            self.score_frame, text="2", font=("helvetica", 12, "bold"), bg="#f0f0f0"
        )
        self.black_score_label.grid(row=0, column=1, sticky="w")

        tk.Label(
            self.score_frame, text="White: ", font=("helvetica", 12), bg="#f0f0f0"
        ).grid(row=1, column=0, sticky="e")

        self.white_score_label = tk.Label(
            self.score_frame, text="2", font=("helvetica", 12, "bold"), bg="#f0f0f0"
        )
        self.white_score_label.grid(row=1, column=1, sticky="w")

        # メッセージ表示
        self.message_label = tk.Label(
            self, text="Game Start", font=("helvetica", 10), bg="#f0f0f0", fg="blue"
        )
        self.message_label.pack(pady=5)

    def update_turn(self, player: int):
        """
        手番を更新

        Args:
            player: 1=黒, -1=白
        """
        turn_text = "Turn: Black" if player == 1 else "Turn: White"
        self.turn_label.config(text=turn_text)

    def update_scores(self, black_count: int, white_count: int):
        """
        石数を更新

        Args:
            black_count: 黒石の数
            white_count: 白石の数
        """
        self.black_score_label.config(text=str(black_count))
        self.white_score_label.config(text=str(white_count))

    def set_message(self, message: str, color: str = "blue"):
        """
        メッセージを設定

        Args:
            message: メッセージ
            color: 文字色
        """
        self.message_label.config(text=message, fg=color)
