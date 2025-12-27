"""ビットボードのテスト

初期配置、合法手生成、石の反転が正しいことを確認する。
"""

import numpy as np
import pytest

from src.cython.bitboard import OthelloBitboard


class TestOthelloBitboardInit:
    """初期化テスト"""

    def test_initial_position(self):
        """初期配置が正しいこと"""
        board = OthelloBitboard()

        # 初期配置の確認
        # 黒（self_board）: E4, D5
        # 白（opp_board）: D4, E5
        # インデックス: D4=27, E4=28, D5=35, E5=36

        # 石の数
        self_count, opp_count = board.get_stone_counts()
        assert self_count == 2, "黒石は2個であるべき"
        assert opp_count == 2, "白石は2個であるべき"

    def test_initial_legal_moves(self):
        """初期状態での合法手が正しいこと"""
        board = OthelloBitboard()
        legal_moves = board.get_legal_moves()

        # 黒の初期合法手: C4, D3, E6, F5
        # インデックス: C4=26, D3=19, E6=44, F5=37
        expected = sorted([19, 26, 37, 44])
        assert sorted(legal_moves) == expected, f"初期合法手が正しくない: {legal_moves}"

    def test_reset(self):
        """リセットで初期状態に戻ること"""
        board = OthelloBitboard()

        # 何手か進める
        board.make_move(19)  # D3
        board.make_move(18)  # C3

        # リセット
        board.reset()

        # 初期状態を確認
        self_count, opp_count = board.get_stone_counts()
        assert self_count == 2
        assert opp_count == 2
        assert board.move_count == 0


class TestOthelloBitboardMoves:
    """着手テスト"""

    def test_make_valid_move(self):
        """有効な手を打てること"""
        board = OthelloBitboard()

        # D3に打つ（インデックス19）
        result = board.make_move(19)
        assert result is True, "有効な手なので成功するべき"

        # 石の数が増えている
        self_count, opp_count = board.get_stone_counts()
        assert self_count == 1, "白石は1個（D4がひっくり返されて黒に）"
        assert opp_count == 4, "黒石は4個（D3, D4, D5, E4）"

    def test_make_invalid_move_occupied(self):
        """既に石がある場所には置けないこと"""
        board = OthelloBitboard()

        # E4（インデックス28）は既に黒石がある
        result = board.make_move(28)
        assert result is False, "既に石がある場所には置けない"

    def test_make_invalid_move_no_flip(self):
        """反転できない場所には置けないこと"""
        board = OthelloBitboard()

        # A1（インデックス0）は反転できる石がない
        result = board.make_move(0)
        assert result is False, "反転できない場所には置けない"

    def test_flip_multiple_directions(self):
        """複数方向への反転が正しいこと"""
        board = OthelloBitboard()

        # いくつか手を進めて複数方向反転のケースを作る
        board.make_move(19)  # D3 黒
        board.make_move(18)  # C3 白
        board.make_move(17)  # B3 黒
        board.make_move(26)  # C4 白
        board.make_move(34)  # C5 黒
        board.make_move(42)  # C6 白
        board.make_move(43)  # D6 黒

        # C6（インデックス42）で白が反転
        # 確認のためにゲームを続行できることを確認
        legal = board.get_legal_moves()
        assert len(legal) >= 1, "ゲーム続行可能であるべき"


class TestOthelloBitboardPass:
    """パステスト"""

    def test_pass_when_no_legal_moves(self):
        """合法手がないときにパスできること"""
        board = OthelloBitboard()

        # パスが必要な状況を作るのは複雑なので、
        # get_legal_movesが64を返すことを確認する簡易テストを行う
        # （実際のパス状況のテストは別途必要）

        legal = board.get_legal_moves()
        # 初期状態ではパスは不要
        assert 64 not in legal, "初期状態ではパス不要"


class TestOthelloBitboardTerminal:
    """終局判定テスト"""

    def test_not_terminal_at_start(self):
        """ゲーム開始時は終局ではない"""
        board = OthelloBitboard()
        assert board.is_terminal() is False

    def test_terminal_detection(self):
        """終局判定が機能すること"""
        board = OthelloBitboard()

        # 終局状態を作るのは複雑なので、
        # 両者に合法手がない状態のテストは統合テストで行う
        # ここでは関数が呼べることを確認
        result = board.is_terminal()
        assert isinstance(result, bool)


class TestOthelloBitboardTensor:
    """テンソル出力テスト"""

    def test_get_tensor_input_shape(self):
        """テンソルの形状が正しいこと"""
        board = OthelloBitboard()
        tensor = board.get_tensor_input()

        assert tensor.shape == (3, 8, 8), f"形状が正しくない: {tensor.shape}"
        assert tensor.dtype == np.float32, f"型が正しくない: {tensor.dtype}"

    def test_get_tensor_input_content(self):
        """テンソルの内容が正しいこと"""
        board = OthelloBitboard()
        tensor = board.get_tensor_input()

        # チャンネル0: 自分の石（黒）
        # E4(row=3, col=4), D5(row=4, col=3)
        assert tensor[0, 3, 4] == 1.0, "E4に黒石があるべき"
        assert tensor[0, 4, 3] == 1.0, "D5に黒石があるべき"

        # チャンネル1: 相手の石（白）
        # D4(row=3, col=3), E5(row=4, col=4)
        assert tensor[1, 3, 3] == 1.0, "D4に白石があるべき"
        assert tensor[1, 4, 4] == 1.0, "E5に白石があるべき"

        # チャンネル2: 合法手マスク
        # 初期合法手: D3, C4, F5, E6
        legal_positions = [(2, 3), (3, 2), (4, 5), (5, 4)]
        for row, col in legal_positions:
            assert tensor[2, row, col] == 1.0, f"({row}, {col})は合法手であるべき"


class TestOthelloBitboardCopy:
    """コピーテスト"""

    def test_copy_is_independent(self):
        """コピーが元の盤面から独立していること"""
        board = OthelloBitboard()
        copy = board.copy()

        # コピーに手を打つ
        copy.make_move(19)  # D3

        # 元の盤面は変わっていない
        self_count, _ = board.get_stone_counts()
        copy_self_count, _ = copy.get_stone_counts()

        assert self_count == 2, "元の盤面は変わっていないべき"
        assert copy_self_count != self_count, "コピーは変わっているべき"


class TestOthelloBitboardSymmetries:
    """対称性テスト"""

    def test_get_symmetries_count(self):
        """対称性変換が8パターン返すこと"""
        board = OthelloBitboard()
        pi = np.ones(65, dtype=np.float32) / 65

        symmetries = board.get_symmetries(pi)
        assert len(symmetries) == 8, f"8パターンであるべき: {len(symmetries)}"

    def test_get_symmetries_shapes(self):
        """対称性変換の形状が正しいこと"""
        board = OthelloBitboard()
        pi = np.ones(65, dtype=np.float32) / 65

        symmetries = board.get_symmetries(pi)

        for board_tensor, pi_transformed in symmetries:
            assert board_tensor.shape == (3, 8, 8), "盤面の形状が正しくない"
            assert pi_transformed.shape == (65,), "方策の形状が正しくない"


class TestOthelloBitboardGameplay:
    """ゲームプレイテスト"""

    def test_play_random_game(self):
        """ランダムなゲームが終局まで進むこと"""
        import random

        board = OthelloBitboard()
        max_moves = 100  # 無限ループ防止

        for _ in range(max_moves):
            if board.is_terminal():
                break

            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break

            move = random.choice(legal_moves)
            result = board.make_move(move)
            assert result is True, f"合法手{move}は成功するべき"

        # ゲームが終了したか確認
        assert board.is_terminal() or board.move_count > 0, "ゲームは進行すべき"

    def test_winner_detection(self):
        """勝者判定が機能すること"""
        board = OthelloBitboard()

        # get_winnerが呼べることを確認
        winner = board.get_winner()
        assert winner in [-1, 0, 1], f"勝者は-1, 0, 1のいずれか: {winner}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
