#!/usr/bin/env python
"""
Webサーバー起動スクリプト

使用方法:
    uv run python run_web.py [--host HOST] [--port PORT] [--model MODEL_PATH]

例:
    uv run python run_web.py
    uv run python run_web.py --port 8080
    uv run python run_web.py --model data/models/best.pt
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Othello AlphaZero Web Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file to load on startup",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # モデル事前読み込み
    if args.model:
        from src.web.api import get_game_manager

        manager = get_game_manager()
        success, error = manager.load_model(args.model)
        if success:
            print(f"Model loaded: {args.model}")
        else:
            print(f"Failed to load model: {error}")

    print(f"Starting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        "src.web.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
