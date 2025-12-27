"""
Othello AlphaZero - GUI エントリポイント

Tkinterベースのオセロ対戦アプリを起動
"""

import tkinter as tk
import argparse
from pathlib import Path
from src.gui.app import OthelloApp


def main():
    """GUIアプリケーションを起動"""
    parser = argparse.ArgumentParser(description="Othello AlphaZero - GUI")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint (optional)"
    )

    args = parser.parse_args()

    # Tkinter ルートウィンドウ作成
    root = tk.Tk()

    # アプリケーション作成
    app = OthelloApp(root, model_path=args.model)

    # 初期メッセージ
    if args.model:
        print(f"Model loaded: {args.model}")
    else:
        print("No model loaded. You can load a model from the menu.")
        print("Or start GUI with: uv run python run_gui.py --model <path_to_model.pt>")

    # アプリケーション実行
    app.run()


if __name__ == "__main__":
    main()
