"""
GUI基本動作テスト（非表示モード）

GUIコンポーネントのロジックをテスト
"""

import numpy as np
from src.cython.bitboard import OthelloBitboard
from src.gui.board_ui import OthelloBoardUI, InfoPanel

# tkinterがない環境でもテストできるように try-except
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Tkinter not available, skipping GUI tests")


def test_board_ui_basic():
    """OthelloBoardUI基本テスト"""
    if not TKINTER_AVAILABLE:
        print("SKIP: Tkinter not available")
        return

    print("Testing OthelloBoardUI...")

    # ルートウィンドウ作成
    root = tk.Tk()
    root.withdraw()  # 非表示

    # ボード作成
    board_ui = OthelloBoardUI(root, board_size=8, cell_size=60)

    # 盤面データ作成
    board = OthelloBitboard()
    board.reset()

    # 盤面を8x8配列に変換
    tensor = board.get_tensor_input()
    board_array = tensor[0] - tensor[1]  # 黒番視点

    legal_moves = board.get_legal_moves()

    # 更新テスト
    board_ui.update_board(board_array, legal_moves)

    print("✓ OthelloBoardUI basic test passed")

    root.destroy()


def test_info_panel():
    """InfoPanel基本テスト"""
    if not TKINTER_AVAILABLE:
        print("SKIP: Tkinter not available")
        return

    print("Testing InfoPanel...")

    root = tk.Tk()
    root.withdraw()

    info_panel = InfoPanel(root)

    # 手番更新
    info_panel.update_turn(1)
    info_panel.update_turn(-1)

    # 石数更新
    info_panel.update_scores(10, 15)

    # メッセージ設定
    info_panel.set_message("テストメッセージ", "blue")

    print("✓ InfoPanel basic test passed")

    root.destroy()


def test_game_logic():
    """ゲームロジックテスト"""
    print("Testing game logic...")

    board = OthelloBitboard()
    board.reset()

    # 初期状態チェック
    legal_moves = board.get_legal_moves()
    assert len(legal_moves) == 4, f"Expected 4 legal moves, got {len(legal_moves)}"

    # 着手テスト
    board.make_move(legal_moves[0])

    # 手数が増えているか
    assert board.move_count == 1

    # 石数取得
    black_count, white_count = board.get_stone_counts()
    assert black_count + white_count >= 4

    print("✓ Game logic test passed")


def main():
    """テスト実行"""
    print("=" * 60)
    print("GUI Basic Tests")
    print("=" * 60)

    test_game_logic()
    test_board_ui_basic()
    test_info_panel()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
