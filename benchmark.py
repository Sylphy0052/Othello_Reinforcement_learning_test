"""ビットボードのベンチマーク

ランダムプレイヤー同士の対戦で1秒あたりの対局数（Games/sec）を計測する。

使用方法:
    uv run python benchmark.py
"""

import random
import time
from typing import Tuple

from src.cython.bitboard import OthelloBitboard


def play_random_game() -> Tuple[int, int]:
    """ランダムプレイヤー同士で1局対戦

    Returns:
        (勝者, 手数): 勝者は1=先手勝ち, -1=後手勝ち, 0=引き分け
    """
    board = OthelloBitboard()
    consecutive_passes = 0

    while not board.is_terminal():
        legal_moves = board.get_legal_moves()

        if legal_moves == [64]:
            # パス
            board.make_move(64)
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
        else:
            move = random.choice(legal_moves)
            board.make_move(move)
            consecutive_passes = 0

    return board.get_winner(), board.move_count


def benchmark_games(num_games: int = 10000) -> None:
    """ベンチマーク実行

    Args:
        num_games: 対局数
    """
    print(f"=== オセロ ビットボード ベンチマーク ===")
    print(f"対局数: {num_games:,}")
    print()

    # ウォームアップ（JITコンパイル等の影響を排除）
    print("ウォームアップ中...")
    for _ in range(100):
        play_random_game()

    # 本計測
    print("計測中...")
    wins = {1: 0, -1: 0, 0: 0}
    total_moves = 0

    start_time = time.perf_counter()

    for _ in range(num_games):
        winner, moves = play_random_game()
        wins[winner] += 1
        total_moves += moves

    elapsed_time = time.perf_counter() - start_time

    # 結果表示
    games_per_sec = num_games / elapsed_time
    moves_per_sec = total_moves / elapsed_time
    avg_moves = total_moves / num_games

    print()
    print("=== 結果 ===")
    print(f"総対局数:     {num_games:,}")
    print(f"経過時間:     {elapsed_time:.2f} 秒")
    print(f"対局速度:     {games_per_sec:,.0f} games/sec")
    print(f"着手速度:     {moves_per_sec:,.0f} moves/sec")
    print(f"平均手数:     {avg_moves:.1f} 手/局")
    print()
    print("=== 勝敗統計 ===")
    print(f"先手勝ち:     {wins[1]:,} ({100*wins[1]/num_games:.1f}%)")
    print(f"後手勝ち:     {wins[-1]:,} ({100*wins[-1]/num_games:.1f}%)")
    print(f"引き分け:     {wins[0]:,} ({100*wins[0]/num_games:.1f}%)")
    print()

    # 目標達成チェック
    target_speed = 5000  # 目標: 5,000 games/sec
    if games_per_sec >= target_speed:
        print(f"✓ 目標速度 {target_speed:,} games/sec を達成!")
    else:
        print(f"✗ 目標速度 {target_speed:,} games/sec 未達成 (現在: {games_per_sec:,.0f})")


def benchmark_operations() -> None:
    """個別操作のベンチマーク"""
    print()
    print("=== 個別操作ベンチマーク ===")

    board = OthelloBitboard()
    iterations = 100000

    # get_legal_moves のベンチマーク
    start = time.perf_counter()
    for _ in range(iterations):
        board.get_legal_moves()
    elapsed = time.perf_counter() - start
    print(f"get_legal_moves: {iterations/elapsed:,.0f} calls/sec")

    # get_tensor_input のベンチマーク
    start = time.perf_counter()
    for _ in range(iterations):
        board.get_tensor_input()
    elapsed = time.perf_counter() - start
    print(f"get_tensor_input: {iterations/elapsed:,.0f} calls/sec")

    # copy のベンチマーク
    start = time.perf_counter()
    for _ in range(iterations):
        board.copy()
    elapsed = time.perf_counter() - start
    print(f"copy: {iterations/elapsed:,.0f} calls/sec")


if __name__ == "__main__":
    benchmark_games(10000)
    benchmark_operations()
