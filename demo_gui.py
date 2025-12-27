"""
GUI デモ・テストスクリプト

様々なAI対戦相手とのGUI対戦をテストする

Usage:
    # テストモデルとの対戦
    uv run python demo_gui.py --checkpoint data/models/test/final_model.pt

    # ランダムプレイヤーとの対戦
    uv run python demo_gui.py --opponent random

    # Greedyプレイヤーとの対戦
    uv run python demo_gui.py --opponent greedy

    # AI vs AI 観戦モード
    uv run python demo_gui.py --checkpoint data/models/checkpoint_iter_10.pt --ai-vs-ai
"""

import tkinter as tk
import argparse
from pathlib import Path
import torch

from src.gui.app import OthelloApp


def main():
    """GUIデモアプリケーションを起動"""
    parser = argparse.ArgumentParser(description="Othello GUI Demo/Test")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to AI model checkpoint"
    )

    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "greedy", "mcts"],
        default="mcts",
        help="AI opponent type (default: mcts)"
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="MCTS simulations for AI player (default: 50)"
    )

    parser.add_argument(
        "--ai-vs-ai",
        action="store_true",
        help="AI vs AI mode (watch two AIs play)"
    )

    args = parser.parse_args()

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Othello GUI Demo")
    print("=" * 70)

    # モデル読み込み
    model_path = args.checkpoint

    # チェックポイントがない場合、デフォルトモデルを探す
    if model_path is None:
        default_paths = [
            "data/models/final_model.pt",
            "data/models/test/final_model.pt",
        ]
        for path in default_paths:
            if Path(path).exists():
                model_path = path
                print(f"Using default model: {model_path}")
                break

    if model_path and Path(model_path).exists():
        print(f"Model: {model_path}")
    elif args.opponent == "mcts":
        print("Warning: No model found. Using Random opponent instead.")
        args.opponent = "random"
    else:
        print(f"Opponent: {args.opponent.title()}")

    print(f"MCTS Simulations: {args.simulations}")
    print(f"Device: {device}")

    if args.ai_vs_ai:
        print("\nMode: AI vs AI (観戦モード)")
        print("Note: AI vs AI mode requires manual step-through in current version.")
    else:
        print("\nMode: Human vs AI")
        print("You play as Black (●)")

    print("\n" + "=" * 70)
    print("GUI Controls:")
    print("  - Click on board to place stones")
    print("  - 'Undo' button: Take back last move")
    print("  - 'Hint' button: Show AI's recommended move")
    print("  - 'New Game' button: Start over")
    print("=" * 70)

    # Tkinter ルートウィンドウ作成
    root = tk.Tk()

    # アプリケーション作成
    app = OthelloApp(root, model_path=model_path)

    # GUIを起動
    print("\nStarting GUI...")
    app.run()

    print("\nGUI closed. Demo complete.")


if __name__ == "__main__":
    main()
