"""
対戦管理システム (Arena)

2つのプレイヤーを対戦させ、結果を記録する
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
from src.cython.bitboard import OthelloBitboard
from .players import Player


@dataclass
class MatchResult:
    """
    対戦結果

    Attributes:
        player1_name: プレイヤー1の名前
        player2_name: プレイヤー2の名前
        winner: 勝者 (1: player1, -1: player2, 0: 引き分け)
        player1_score: プレイヤー1の最終石数
        player2_score: プレイヤー2の最終石数
        num_moves: 総手数
        duration: 対戦時間（秒）
    """
    player1_name: str
    player2_name: str
    winner: int
    player1_score: int
    player2_score: int
    num_moves: int
    duration: float

    def __str__(self) -> str:
        """結果の文字列表現"""
        if self.winner == 1:
            result = f"{self.player1_name} wins"
        elif self.winner == -1:
            result = f"{self.player2_name} wins"
        else:
            result = "Draw"

        return (
            f"{result} | "
            f"{self.player1_name}: {self.player1_score} - "
            f"{self.player2_name}: {self.player2_score} | "
            f"Moves: {self.num_moves} | "
            f"Time: {self.duration:.2f}s"
        )


class Arena:
    """
    対戦管理システム

    2つのプレイヤーを対戦させ、結果を記録する
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 詳細な出力を行うか
        """
        self.verbose = verbose

    def play_game(
        self,
        player1: Player,
        player2: Player,
        starting_player: int = 1,
    ) -> MatchResult:
        """
        1ゲームを実行

        Args:
            player1: プレイヤー1（黒番から開始の場合）
            player2: プレイヤー2（白番から開始の場合）
            starting_player: 先手 (1: player1, -1: player2)

        Returns:
            MatchResult: 対戦結果
        """
        # 盤面初期化
        board = OthelloBitboard()
        board.reset()

        # プレイヤーリセット
        player1.reset()
        player2.reset()

        # 先手・後手の割り当て
        if starting_player == 1:
            current_player = player1
            next_player = player2
            current_color = 1  # 黒
        else:
            current_player = player2
            next_player = player1
            current_color = -1  # 白

        start_time = time.time()

        # ゲームループ
        while not board.is_terminal():
            # 着手取得
            action = current_player.get_action(board)

            if self.verbose:
                legal_moves = board.get_legal_moves()
                print(f"{current_player.name} plays: {action} (legal: {legal_moves})")

            # 着手
            board.make_move(action)

            # 手番交代
            current_player, next_player = next_player, current_player
            current_color = -current_color

        duration = time.time() - start_time

        # 結果取得
        winner_color = board.get_winner()  # 1: 黒勝ち, -1: 白勝ち, 0: 引き分け
        black_count, white_count = board.get_stone_counts()

        # プレイヤー視点での勝者判定
        if starting_player == 1:
            # player1が黒、player2が白
            if winner_color == 1:
                winner = 1
            elif winner_color == -1:
                winner = -1
            else:
                winner = 0
            player1_score = black_count
            player2_score = white_count
        else:
            # player1が白、player2が黒
            if winner_color == 1:
                winner = -1
            elif winner_color == -1:
                winner = 1
            else:
                winner = 0
            player1_score = white_count
            player2_score = black_count

        result = MatchResult(
            player1_name=player1.name,
            player2_name=player2.name,
            winner=winner,
            player1_score=player1_score,
            player2_score=player2_score,
            num_moves=board.move_count,
            duration=duration,
        )

        if self.verbose:
            print(f"\n{result}\n")

        return result

    def play_matches(
        self,
        player1: Player,
        player2: Player,
        num_games: int = 10,
        alternate_colors: bool = True,
    ) -> List[MatchResult]:
        """
        複数ゲームを実行

        Args:
            player1: プレイヤー1
            player2: プレイヤー2
            num_games: ゲーム数
            alternate_colors: 先後を交代するか

        Returns:
            List[MatchResult]: 対戦結果のリスト
        """
        results = []

        for game_idx in range(num_games):
            if self.verbose:
                print(f"=== Game {game_idx + 1}/{num_games} ===")

            # 先後を交代
            if alternate_colors:
                starting_player = 1 if (game_idx % 2 == 0) else -1
            else:
                starting_player = 1

            result = self.play_game(player1, player2, starting_player)
            results.append(result)

        # サマリー表示
        if self.verbose:
            self._print_summary(results, player1.name, player2.name)

        return results

    def _print_summary(
        self,
        results: List[MatchResult],
        player1_name: str,
        player2_name: str,
    ):
        """対戦結果のサマリーを表示"""
        print("\n" + "=" * 70)
        print("Match Summary")
        print("=" * 70)

        player1_wins = sum(1 for r in results if r.winner == 1)
        player2_wins = sum(1 for r in results if r.winner == -1)
        draws = sum(1 for r in results if r.winner == 0)

        total_games = len(results)
        player1_win_rate = player1_wins / total_games * 100 if total_games > 0 else 0
        player2_win_rate = player2_wins / total_games * 100 if total_games > 0 else 0

        avg_moves = sum(r.num_moves for r in results) / total_games if total_games > 0 else 0
        avg_duration = sum(r.duration for r in results) / total_games if total_games > 0 else 0

        print(f"\nTotal Games: {total_games}")
        print(f"{player1_name}: {player1_wins} wins ({player1_win_rate:.1f}%)")
        print(f"{player2_name}: {player2_wins} wins ({player2_win_rate:.1f}%)")
        print(f"Draws: {draws}")
        print(f"\nAverage Moves: {avg_moves:.1f}")
        print(f"Average Duration: {avg_duration:.2f}s")
        print("=" * 70 + "\n")


def evaluate_player(
    player: Player,
    opponent: Player,
    num_games: int = 10,
    verbose: bool = True,
) -> dict:
    """
    プレイヤーを評価

    Args:
        player: 評価対象のプレイヤー
        opponent: 対戦相手
        num_games: ゲーム数
        verbose: 詳細な出力

    Returns:
        dict: 評価結果
            - win_rate: 勝率
            - avg_score: 平均スコア
            - avg_moves: 平均手数
            - results: 対戦結果リスト
    """
    arena = Arena(verbose=verbose)
    results = arena.play_matches(player, opponent, num_games=num_games)

    player_wins = sum(1 for r in results if r.winner == 1)
    win_rate = player_wins / num_games if num_games > 0 else 0

    avg_score = sum(r.player1_score for r in results) / num_games if num_games > 0 else 0
    avg_moves = sum(r.num_moves for r in results) / num_games if num_games > 0 else 0

    return {
        "win_rate": win_rate,
        "avg_score": avg_score,
        "avg_moves": avg_moves,
        "results": results,
    }
