"""
Othello AlphaZero - CLIエントリポイント

学習・評価用のコマンドラインインターフェース
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.cython.bitboard import OthelloBitboard
from src.model.net import OthelloResNet
from src.mcts.mcts import MCTS
from src.train.buffer import ReplayBuffer
from src.train.self_play import SelfPlayWorker
from src.train.parallel_self_play import ParallelSelfPlayWorker, create_parallel_self_play_worker
from src.train.trainer import AlphaZeroTrainer


def load_config(config_path: str) -> dict:
    """
    YAML設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        dict: 設定辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: dict) -> torch.device:
    """
    デバイスを設定

    Args:
        config: 設定辞書

    Returns:
        torch.device: デバイス
    """
    device_config = config.get('system', {}).get('device', 'auto')

    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    return device


def set_seed(seed: int):
    """
    乱数シードを設定

    Args:
        seed: シード値
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_command(args):
    """
    学習コマンド

    Args:
        args: argparseの引数
    """
    # 設定ファイル読み込み
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    # シード設定
    seed = config.get('system', {}).get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # デバイス設定
    device = setup_device(config)

    # モデル作成
    print("\nCreating model...")
    model = OthelloResNet(
        num_blocks=config['model']['num_blocks'],
        num_filters=config['model']['num_filters'],
        board_size=config['model'].get('board_size', 8),
    )
    model.to(device)

    param_count = model.get_param_count()
    print(f"Model: OthelloResNet")
    print(f"  Blocks: {config['model']['num_blocks']}")
    print(f"  Filters: {config['model']['num_filters']}")
    print(f"  Parameters: {param_count['total']:,}")

    # Self-Play Worker作成
    # 並列ゲーム数が設定されている場合はParallelSelfPlayWorkerを使用
    num_parallel_games = config.get('self_play', {}).get('num_parallel_games', 1)

    if num_parallel_games > 1:
        print(f"\nCreating Parallel Self-Play Worker (parallel_games={num_parallel_games})...")
        self_play_worker = create_parallel_self_play_worker(config, model, device)
    else:
        print("\nCreating MCTS...")
        mcts = MCTS(
            model=model,
            device=device,
            c_puct=config['mcts']['c_puct'],
            dirichlet_alpha=config['mcts']['dirichlet_alpha'],
            dirichlet_epsilon=config['mcts'].get('dirichlet_epsilon', 0.25),
        )

        print("\nCreating Self-Play Worker...")
        self_play_worker = SelfPlayWorker(
            board_class=OthelloBitboard,
            mcts=mcts,
            num_simulations=config['mcts']['num_simulations'],
            temperature_threshold=config['self_play'].get('temperature_threshold', 15),
        )

    # Replay Buffer作成
    print("\nCreating Replay Buffer...")
    replay_buffer = ReplayBuffer(
        max_size=config['training']['replay_buffer_size']
    )

    # Trainer作成
    print("\nCreating Trainer...")
    trainer = AlphaZeroTrainer(
        model=model,
        device=device,
        replay_buffer=replay_buffer,
        self_play_worker=self_play_worker,
        config=config['training'],
        checkpoint_dir=config['paths']['checkpoint_dir'],
        log_dir=config['paths']['log_dir'],
    )

    # 学習開始
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    trainer.train(
        num_iterations=config['training']['num_iterations'],
        self_play_episodes_per_iter=config['training']['self_play_episodes_per_iter'],
        train_epochs_per_iter=config['training']['train_epochs_per_iter'],
        batch_size=config['training']['batch_size'],
        checkpoint_interval=config['training']['checkpoint_interval'],
    )


def eval_command(args):
    """
    評価コマンド

    Args:
        args: argparseの引数
    """
    import json
    from datetime import datetime
    from src.eval.players import RandomPlayer, GreedyPlayer, MCTSPlayer
    from src.eval.arena import evaluate_player

    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Games per opponent: {args.games}")
    print(f"MCTS simulations: {args.simulations}")

    # デバイス設定
    device = setup_device({'system': {'device': 'auto'}})

    # モデル読み込み
    print("\nLoading model...")
    ai_player = MCTSPlayer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        num_simulations=args.simulations,
    )
    print(f"Model loaded: {ai_player.name}\n")

    # 対戦相手リスト
    opponents = [
        RandomPlayer(name="Random"),
        GreedyPlayer(name="Greedy"),
    ]

    # 評価結果
    results_summary = {}

    # 各対戦相手と対戦
    for opponent in opponents:
        print(f"\n{'=' * 70}")
        print(f"Evaluating against {opponent.name}")
        print('=' * 70)

        eval_result = evaluate_player(
            player=ai_player,
            opponent=opponent,
            num_games=args.games,
            verbose=args.verbose,
        )

        results_summary[opponent.name] = {
            "win_rate": eval_result["win_rate"],
            "avg_score": eval_result["avg_score"],
            "avg_moves": eval_result["avg_moves"],
        }

        print(f"\nResult vs {opponent.name}:")
        print(f"  Win Rate: {eval_result['win_rate'] * 100:.1f}%")
        print(f"  Avg Score: {eval_result['avg_score']:.1f}")
        print(f"  Avg Moves: {eval_result['avg_moves']:.1f}")

    # サマリー表示
    print(f"\n{'=' * 70}")
    print("Evaluation Summary")
    print('=' * 70)
    for opponent_name, result in results_summary.items():
        print(f"{opponent_name:15s}: {result['win_rate']*100:5.1f}% win rate, "
              f"avg score: {result['avg_score']:5.1f}")

    # 結果を保存（オプション）
    if args.save_results:
        from pathlib import Path
        output_dir = Path("data/eval")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"eval_{timestamp}.json"

        eval_data = {
            "checkpoint": args.checkpoint,
            "timestamp": datetime.now().isoformat(),
            "mcts_simulations": args.simulations,
            "games_per_opponent": args.games,
            "results": results_summary,
        }

        with open(result_file, "w") as f:
            json.dump(eval_data, f, indent=2)

        print(f"\nResults saved to: {result_file}")

    print("\n" + "=" * 70)


def play_command(args):
    """
    対戦コマンド（人間 vs AI）

    Args:
        args: argparseの引数
    """
    print("Play mode (not implemented yet)")
    # TODO: 対戦モードの実装


def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(description="Othello AlphaZero - CLI")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train コマンド
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='configs/default_8x8.yaml',
        help='Path to config file (default: configs/default_8x8.yaml)'
    )
    train_parser.set_defaults(func=train_command)

    # Eval コマンド
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    eval_parser.add_argument(
        '--games',
        type=int,
        default=20,
        help='Number of games per opponent (default: 20)'
    )
    eval_parser.add_argument(
        '--simulations',
        type=int,
        default=50,
        help='MCTS simulations per move (default: 50)'
    )
    eval_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed game progress'
    )
    eval_parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save evaluation results to JSON file'
    )
    eval_parser.set_defaults(func=eval_command)

    # Play コマンド
    play_parser = subparsers.add_parser('play', help='Play against AI')
    play_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    play_parser.set_defaults(func=play_command)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
