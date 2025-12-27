"""
AIベンチマークスクリプト

学習済みモデルを様々な対戦相手と戦わせて評価する

Usage:
    uv run python benchmark_ai.py --checkpoint data/models/checkpoint_iter_10.pt
"""

import argparse
import torch
import json
from pathlib import Path
from datetime import datetime

from src.eval.players import RandomPlayer, GreedyPlayer, MCTSPlayer
from src.eval.arena import Arena, evaluate_player


def benchmark_model(
    checkpoint_path: str,
    num_games_per_opponent: int = 20,
    mcts_simulations: int = 50,
    output_dir: str = "data/benchmark",
):
    """
    モデルをベンチマーク

    Args:
        checkpoint_path: モデルチェックポイントのパス
        num_games_per_opponent: 各対戦相手とのゲーム数
        mcts_simulations: MCTSシミュレーション回数
        output_dir: 結果保存ディレクトリ
    """
    print("=" * 70)
    print("AI Benchmark")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Games per opponent: {num_games_per_opponent}")
    print(f"MCTS simulations: {mcts_simulations}")

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # モデル読み込み
    print("Loading model...")
    ai_player = MCTSPlayer.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        num_simulations=mcts_simulations,
    )
    print(f"Model loaded: {ai_player.name}\n")

    # 対戦相手リスト
    opponents = [
        RandomPlayer(name="Random"),
        GreedyPlayer(name="Greedy"),
    ]

    # 評価結果
    benchmark_results = {
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
        "mcts_simulations": mcts_simulations,
        "num_games_per_opponent": num_games_per_opponent,
        "device": str(device),
        "results": {},
        "summary_statistics": {},
    }

    # 各対戦相手と対戦
    for opponent in opponents:
        print(f"\n{'=' * 70}")
        print(f"Evaluating against {opponent.name}")
        print('=' * 70)

        eval_result = evaluate_player(
            player=ai_player,
            opponent=opponent,
            num_games=num_games_per_opponent,
            verbose=True,
        )

        # 勝敗統計を計算
        wins = sum(1 for r in eval_result["results"] if r.winner == 1)
        losses = sum(1 for r in eval_result["results"] if r.winner == -1)
        draws = sum(1 for r in eval_result["results"] if r.winner == 0)

        # スコア統計
        scores = [r.player1_score for r in eval_result["results"]]
        min_score = min(scores)
        max_score = max(scores)

        # 結果を記録
        benchmark_results["results"][opponent.name] = {
            "win_rate": eval_result["win_rate"],
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_score": eval_result["avg_score"],
            "min_score": min_score,
            "max_score": max_score,
            "avg_moves": eval_result["avg_moves"],
            "num_games": num_games_per_opponent,
        }

        print(f"\nResult vs {opponent.name}:")
        print(f"  Win Rate: {eval_result['win_rate'] * 100:.1f}% ({wins}W-{losses}L-{draws}D)")
        print(f"  Avg Score: {eval_result['avg_score']:.1f} (min: {min_score}, max: {max_score})")
        print(f"  Avg Moves: {eval_result['avg_moves']:.1f}")

    # サマリー統計を計算
    total_games = sum(r["num_games"] for r in benchmark_results["results"].values())
    total_wins = sum(r["wins"] for r in benchmark_results["results"].values())
    total_losses = sum(r["losses"] for r in benchmark_results["results"].values())
    total_draws = sum(r["draws"] for r in benchmark_results["results"].values())
    overall_win_rate = total_wins / total_games if total_games > 0 else 0

    benchmark_results["summary_statistics"] = {
        "total_games": total_games,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_draws": total_draws,
        "overall_win_rate": overall_win_rate,
    }

    # 結果を保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"benchmark_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Benchmark Complete")
    print('=' * 70)
    print(f"Results saved to: {result_file}")

    # サマリー表示
    print("\n=== Summary ===")
    for opponent_name, result in benchmark_results["results"].items():
        print(f"{opponent_name:15s}: {result['win_rate']*100:5.1f}% win rate "
              f"({result['wins']}W-{result['losses']}L-{result['draws']}D), "
              f"avg score: {result['avg_score']:5.1f}")

    print(f"\n{'Overall':15s}: {overall_win_rate*100:5.1f}% win rate "
          f"({total_wins}W-{total_losses}L-{total_draws}D) in {total_games} games")

    return benchmark_results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="AI Benchmark")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games per opponent (default: 20)"
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="MCTS simulations (default: 50)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmark",
        help="Output directory (default: data/benchmark)"
    )

    args = parser.parse_args()

    # チェックポイント存在確認
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    # ベンチマーク実行
    benchmark_model(
        checkpoint_path=args.checkpoint,
        num_games_per_opponent=args.games,
        mcts_simulations=args.simulations,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
