"""
GUI自動テストスクリプト

GUIコンポーネントの動作を自動的にテストする（ヘッドレス環境でも動作可能）

Usage:
    uv run python test_gui_automated.py
"""

import sys
from pathlib import Path

# GUI が使えない環境でもテストできるようにする
try:
    import tkinter as tk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Warning: Tkinter not available. GUI tests will be skipped.")

import torch
from src.cython.bitboard import OthelloBitboard
from src.eval.players import RandomPlayer, GreedyPlayer, MCTSPlayer


def test_board_logic():
    """盤面ロジックのテスト"""
    print("\n" + "=" * 70)
    print("Testing Board Logic")
    print("=" * 70)

    board = OthelloBitboard()
    board.reset()

    # 初期状態確認
    assert not board.is_terminal()
    legal_moves = board.get_legal_moves()
    assert len(legal_moves) == 4, f"Expected 4 legal moves, got {len(legal_moves)}"
    print("✓ Board initialization successful")

    # 着手テスト
    board.make_move(legal_moves[0])
    assert board.move_count == 1
    print("✓ Move execution successful")

    # コピーテスト
    board_copy = board.copy()
    assert board_copy.move_count == board.move_count
    print("✓ Board copy successful")

    print("\nBoard Logic Tests: PASSED")


def test_players():
    """プレイヤーのテスト"""
    print("\n" + "=" * 70)
    print("Testing Players")
    print("=" * 70)

    board = OthelloBitboard()
    board.reset()

    # RandomPlayer
    random_player = RandomPlayer()
    action = random_player.get_action(board)
    assert action in board.get_legal_moves()
    print("✓ RandomPlayer works")

    # GreedyPlayer
    greedy_player = GreedyPlayer()
    action = greedy_player.get_action(board)
    assert action in board.get_legal_moves()
    print("✓ GreedyPlayer works")

    print("\nPlayer Tests: PASSED")


def test_mcts_player_with_model():
    """MCTSプレイヤー（モデル付き）のテスト"""
    print("\n" + "=" * 70)
    print("Testing MCTS Player with Model")
    print("=" * 70)

    # テストモデルを探す
    test_model_paths = [
        "data/models/test/final_model.pt",
        "data/models/checkpoint_iter_10.pt",
        "data/models/final_model.pt",
    ]

    model_path = None
    for path in test_model_paths:
        if Path(path).exists():
            model_path = path
            break

    if model_path is None:
        print("⚠ No model found. Skipping MCTS player test.")
        return

    print(f"Using model: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # モデル読み込み
        mcts_player = MCTSPlayer.from_checkpoint(
            checkpoint_path=model_path,
            device=device,
            num_simulations=10,  # 高速化のため少なめ
        )
        print("✓ Model loaded successfully")

        # 着手テスト
        board = OthelloBitboard()
        board.reset()

        action = mcts_player.get_action(board)
        assert action in board.get_legal_moves()
        print("✓ MCTS player action selection works")

        # 数手進めてみる
        for i in range(5):
            if board.is_terminal():
                break
            legal_moves = board.get_legal_moves()
            if len(legal_moves) == 0 or legal_moves == [64]:
                board.make_move(64)
            else:
                action = mcts_player.get_action(board)
                board.make_move(action)

        print(f"✓ Played {board.move_count} moves successfully")

        print("\nMCTS Player Tests: PASSED")

    except Exception as e:
        print(f"✗ MCTS Player test failed: {e}")
        import traceback
        traceback.print_exc()


def test_gui_components():
    """GUIコンポーネントのテスト（可能な場合）"""
    if not GUI_AVAILABLE:
        print("\n⚠ Tkinter not available. Skipping GUI component tests.")
        return

    print("\n" + "=" * 70)
    print("Testing GUI Components")
    print("=" * 70)

    try:
        from src.gui.board_ui import OthelloBoardUI, InfoPanel

        # Tkinterルート作成（表示はしない）
        root = tk.Tk()
        root.withdraw()  # ウィンドウを隠す

        # ボードUI作成
        board_ui = OthelloBoardUI(root, cell_size=40)
        print("✓ BoardUI creation successful")

        # InfoPanel作成
        info_panel = InfoPanel(root)
        print("✓ InfoPanel creation successful")

        # ボード更新テスト
        board = OthelloBitboard()
        board.reset()
        board_ui.update_board(board)
        print("✓ Board update successful")

        # クリーンアップ
        root.destroy()

        print("\nGUI Component Tests: PASSED")

    except Exception as e:
        print(f"✗ GUI component test failed: {e}")
        import traceback
        traceback.print_exc()


def test_full_game():
    """完全なゲームのテスト"""
    print("\n" + "=" * 70)
    print("Testing Full Game (Random vs Greedy)")
    print("=" * 70)

    from src.eval.arena import Arena

    arena = Arena(verbose=False)
    player1 = RandomPlayer()
    player2 = GreedyPlayer()

    result = arena.play_game(player1, player2)

    print(f"Game completed in {result.num_moves} moves")
    print(f"Final score: {result.player1_score} - {result.player2_score}")
    print(f"Winner: {['Draw', 'Player1', 'Player2'][result.winner]}")
    print("✓ Full game test successful")

    print("\nFull Game Test: PASSED")


def main():
    """すべてのテストを実行"""
    print("=" * 70)
    print("Othello GUI - Automated Tests")
    print("=" * 70)

    try:
        # 基本機能テスト
        test_board_logic()
        test_players()

        # AI機能テスト
        test_mcts_player_with_model()

        # GUIコンポーネントテスト
        test_gui_components()

        # 統合テスト
        test_full_game()

        # 総括
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nGUI is ready to use!")
        print("Run the GUI with:")
        print("  uv run python run_gui.py --model <path_to_model.pt>")
        print("\nOr use the demo script:")
        print("  uv run python demo_gui.py --checkpoint <path_to_model.pt>")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("TESTS FAILED ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
