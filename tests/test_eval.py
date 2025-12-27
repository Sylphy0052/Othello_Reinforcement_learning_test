"""
評価システムのテストケース

- プレイヤーの動作テスト
- Arena（対戦管理）のテスト
- 統合テスト
"""

import pytest
from src.cython.bitboard import OthelloBitboard
from src.eval.players import RandomPlayer, GreedyPlayer
from src.eval.arena import Arena, evaluate_player


class TestPlayers:
    """プレイヤークラスのテスト"""

    def test_random_player(self):
        """RandomPlayerの動作テスト"""
        player = RandomPlayer()
        board = OthelloBitboard()
        board.reset()

        # 着手取得
        action = player.get_action(board)

        # 合法手が選ばれているか
        legal_moves = board.get_legal_moves()
        assert action in legal_moves

    def test_greedy_player(self):
        """GreedyPlayerの動作テスト"""
        player = GreedyPlayer()
        board = OthelloBitboard()
        board.reset()

        # 着手取得
        action = player.get_action(board)

        # 合法手が選ばれているか
        legal_moves = board.get_legal_moves()
        assert action in legal_moves

    def test_player_multiple_moves(self):
        """複数手での動作テスト"""
        player = RandomPlayer()
        board = OthelloBitboard()
        board.reset()

        # 10手進める
        for _ in range(10):
            if board.is_terminal():
                break

            action = player.get_action(board)
            board.make_move(action)

        # エラーなく進んだことを確認
        assert board.move_count <= 10


class TestArena:
    """Arenaクラスのテスト"""

    def test_play_single_game(self):
        """1ゲームの実行テスト"""
        arena = Arena(verbose=False)

        player1 = RandomPlayer("Player1")
        player2 = RandomPlayer("Player2")

        result = arena.play_game(player1, player2)

        # 結果が正しく返されるか
        assert result.player1_name == "Player1"
        assert result.player2_name == "Player2"
        assert result.winner in [-1, 0, 1]
        assert result.player1_score + result.player2_score <= 64
        assert result.player1_score > 0
        assert result.player2_score > 0
        assert result.num_moves > 0
        assert result.duration > 0

    def test_play_multiple_games(self):
        """複数ゲームの実行テスト"""
        arena = Arena(verbose=False)

        player1 = RandomPlayer("Player1")
        player2 = RandomPlayer("Player2")

        results = arena.play_matches(player1, player2, num_games=5)

        # 5ゲーム実行されたか
        assert len(results) == 5

        # 各結果が正しいか
        for result in results:
            assert result.winner in [-1, 0, 1]
            # 石数の合計は64以下（空マスが残る場合がある）
            assert result.player1_score + result.player2_score <= 64
            assert result.player1_score > 0
            assert result.player2_score > 0

    def test_alternating_colors(self):
        """先後交代のテスト"""
        arena = Arena(verbose=False)

        player1 = RandomPlayer("Player1")
        player2 = RandomPlayer("Player2")

        results = arena.play_matches(
            player1, player2,
            num_games=4,
            alternate_colors=True
        )

        assert len(results) == 4


class TestEvaluation:
    """評価機能のテスト"""

    def test_evaluate_player(self):
        """プレイヤー評価のテスト"""
        player = GreedyPlayer("Greedy")
        opponent = RandomPlayer("Random")

        eval_result = evaluate_player(
            player=player,
            opponent=opponent,
            num_games=5,
            verbose=False
        )

        # 結果が正しく返されるか
        assert "win_rate" in eval_result
        assert "avg_score" in eval_result
        assert "avg_moves" in eval_result
        assert "results" in eval_result

        assert 0.0 <= eval_result["win_rate"] <= 1.0
        assert eval_result["avg_score"] > 0
        assert len(eval_result["results"]) == 5

    def test_greedy_vs_random(self):
        """Greedy vs Random対戦テスト"""
        greedy = GreedyPlayer()
        random_player = RandomPlayer()

        eval_result = evaluate_player(
            player=greedy,
            opponent=random_player,
            num_games=10,
            verbose=False
        )

        # Greedyは通常Randomより強いはず（勝率50%以上を期待）
        # ただし確率的要素があるので、必ずしもそうとは限らない
        print(f"\nGreedy vs Random: {eval_result['win_rate'] * 100:.1f}% win rate")
        assert 0.0 <= eval_result["win_rate"] <= 1.0


class TestGameCompletion:
    """ゲーム完了のテスト"""

    def test_game_always_terminates(self):
        """ゲームが必ず終了することを確認"""
        arena = Arena(verbose=False)

        player1 = RandomPlayer()
        player2 = RandomPlayer()

        # 10ゲーム実行
        for _ in range(10):
            result = arena.play_game(player1, player2)

            # ゲームが終了しているか
            assert result.num_moves > 0
            assert result.num_moves <= 60  # 最大60手

            # 石数の合計が64以下
            assert result.player1_score + result.player2_score <= 64
            assert result.player1_score > 0
            assert result.player2_score > 0
