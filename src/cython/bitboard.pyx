# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""オセロのビットボード実装

Cythonを使用した高速なオセロ盤面管理。
uint64×2で盤面を表現し、ビット演算で合法手生成・石反転を行う。
"""

cimport numpy as np
import numpy as np
from libc.stdint cimport uint64_t

# NumPy配列の型定義
np.import_array()

# 8方向のシフト量と対応するマスク
# 方向: 上, 下, 左, 右, 左上, 右上, 左下, 右下
cdef int[8] DIRECTIONS = [-8, 8, -1, 1, -9, -7, 7, 9]

# 左右端のマスク（端を超えないように）
# 左側の列（A列）を除外するマスク
cdef uint64_t NOT_A_FILE = 0xFEFEFEFEFEFEFEFE
# 右側の列（H列）を除外するマスク
cdef uint64_t NOT_H_FILE = 0x7F7F7F7F7F7F7F7F

# 各方向に対応するマスク
cdef uint64_t[8] DIRECTION_MASKS = [
    0xFFFFFFFFFFFFFFFF,  # 上: 制限なし
    0xFFFFFFFFFFFFFFFF,  # 下: 制限なし
    NOT_A_FILE,           # 左: A列を除外
    NOT_H_FILE,           # 右: H列を除外
    NOT_A_FILE,           # 左上: A列を除外
    NOT_H_FILE,           # 右上: H列を除外
    NOT_A_FILE,           # 左下: A列を除外
    NOT_H_FILE,           # 右下: H列を除外
]


cdef class OthelloBitboard:
    """オセロのビットボード実装クラス

    盤面を2つの64ビット整数で表現することで、
    ビット演算による高速な合法手計算・石反転を実現する。
    """

    def __init__(self):
        """コンストラクタ: 初期盤面をセットアップ"""
        self.reset()

    cpdef void reset(self):
        """盤面を初期状態にリセット

        初期配置:
            - D4, E5: 白石
            - D5, E4: 黒石
        黒（先手）から開始。self_board = 黒, opp_board = 白
        """
        # 初期配置のビット位置
        # D4=27, E4=28, D5=35, E5=36
        cdef uint64_t white_init = (1ULL << 27) | (1ULL << 36)  # D4, E5
        cdef uint64_t black_init = (1ULL << 28) | (1ULL << 35)  # E4, D5

        # 黒が先手なので、self_board = 黒
        self.self_board = black_init
        self.opp_board = white_init
        self.move_count = 0
        self.passed = False

    cdef uint64_t _get_flip_direction(self, int pos, int direction, uint64_t self_b, uint64_t opp_b, uint64_t mask) noexcept:
        """指定方向に反転できる石のビットマスクを取得

        Args:
            pos: 着手位置（0-63）
            direction: シフト量（-8, 8, -1, 1, -9, -7, 7, 9）
            self_b: 自分の石のビットボード
            opp_b: 相手の石のビットボード
            mask: 端を超えないためのマスク

        Returns:
            反転する石のビットマスク
        """
        cdef uint64_t flip = 0
        cdef uint64_t cursor
        cdef int shift

        # 正の方向と負の方向で処理を分ける
        if direction > 0:
            shift = direction
            cursor = (1ULL << pos) << shift
            cursor = cursor & mask

            # 相手の石が続く限り探索
            while cursor & opp_b:
                flip |= cursor
                cursor = (cursor << shift) & mask

            # 自分の石で挟んでいなければ反転なし
            if not (cursor & self_b):
                flip = 0
        else:
            shift = -direction
            cursor = (1ULL << pos) >> shift
            cursor = cursor & mask

            while cursor & opp_b:
                flip |= cursor
                cursor = (cursor >> shift) & mask

            if not (cursor & self_b):
                flip = 0

        return flip

    cdef uint64_t _get_flip_bits(self, int pos, uint64_t self_b, uint64_t opp_b) noexcept:
        """指定位置に着手した場合の全方向の反転ビットを取得

        Args:
            pos: 着手位置（0-63）
            self_b: 自分の石のビットボード
            opp_b: 相手の石のビットボード

        Returns:
            全方向の反転する石のビットマスク
        """
        cdef uint64_t flip = 0
        cdef int i

        for i in range(8):
            flip |= self._get_flip_direction(pos, DIRECTIONS[i], self_b, opp_b, DIRECTION_MASKS[i])

        return flip

    cdef uint64_t _compute_legal_moves(self, uint64_t self_b, uint64_t opp_b) noexcept:
        """合法手のビットマスクを計算

        _get_flip_bitsと一貫した結果を保証するため、
        全空きマスについて反転可能かチェックする方式を採用。

        Args:
            self_b: 自分の石のビットボード
            opp_b: 相手の石のビットボード

        Returns:
            合法手位置のビットマスク（1が立っている位置が合法手）
        """
        cdef uint64_t empty = ~(self_b | opp_b)
        cdef uint64_t legal = 0
        cdef int pos

        # 各空きマスについて、反転可能な石があるかチェック
        for pos in range(64):
            if empty & (1ULL << pos):
                if self._get_flip_bits(pos, self_b, opp_b) != 0:
                    legal |= (1ULL << pos)

        return legal

    cdef void _swap_players(self) noexcept:
        """手番を入れ替える（自分と相手のビットボードを交換）"""
        cdef uint64_t temp = self.self_board
        self.self_board = self.opp_board
        self.opp_board = temp

    cpdef list get_legal_moves(self):
        """合法手のリストを取得

        Returns:
            合法手のインデックスリスト（0-63）
            置ける場所がない場合はパス用に64を含むリストを返す
        """
        cdef uint64_t legal = self._compute_legal_moves(self.self_board, self.opp_board)
        cdef list moves = []
        cdef int i

        if legal == 0:
            # 置ける場所がない場合はパス（64）
            return [64]

        for i in range(64):
            if legal & (1ULL << i):
                moves.append(i)

        return moves

    cpdef uint64_t get_legal_moves_bits(self):
        """合法手のビットマスクを取得

        Returns:
            合法手位置のビットマスク
        """
        return self._compute_legal_moves(self.self_board, self.opp_board)

    cpdef bint make_move(self, int pos):
        """指定位置に着手

        Args:
            pos: 着手位置（0-63）、パスの場合は64

        Returns:
            True: 着手成功, False: 無効な手
        """
        cdef uint64_t flip
        cdef uint64_t legal
        cdef uint64_t pos_bit

        # パス処理
        if pos == 64:
            legal = self._compute_legal_moves(self.self_board, self.opp_board)
            if legal == 0:
                # パスが有効（合法手がない）
                self._swap_players()
                self.move_count += 1  # パスも1手としてカウント
                self.passed = True
                return True
            else:
                # パスは無効（合法手がある）
                return False

        # 位置が範囲外
        if pos < 0 or pos > 63:
            return False

        pos_bit = 1ULL << pos

        # 既に石がある
        if (self.self_board | self.opp_board) & pos_bit:
            return False

        # 反転する石を計算
        flip = self._get_flip_bits(pos, self.self_board, self.opp_board)

        # 反転できる石がない（無効な手）
        if flip == 0:
            return False

        # 石を配置して反転
        self.self_board |= pos_bit | flip
        self.opp_board &= ~flip

        # 手番交代
        self._swap_players()
        self.move_count += 1
        self.passed = False

        return True

    cpdef bint is_terminal(self):
        """ゲーム終了判定

        Returns:
            True: ゲーム終了, False: 継続中
        """
        cdef uint64_t self_legal = self._compute_legal_moves(self.self_board, self.opp_board)
        cdef uint64_t opp_legal

        # 自分に合法手がある場合は継続
        if self_legal != 0:
            return False

        # 相手にも合法手がない場合は終了
        opp_legal = self._compute_legal_moves(self.opp_board, self.self_board)
        return opp_legal == 0

    cpdef int get_winner(self):
        """勝者を判定（終局時のみ有効）

        Returns:
            1: 現在手番プレイヤーの勝ち
            -1: 相手プレイヤーの勝ち
            0: 引き分け
        """
        cdef int self_count = self._popcount(self.self_board)
        cdef int opp_count = self._popcount(self.opp_board)

        if self_count > opp_count:
            return 1
        elif self_count < opp_count:
            return -1
        else:
            return 0

    cdef int _popcount(self, uint64_t x) noexcept:
        """ビットカウント（立っているビットの数を数える）"""
        cdef int count = 0
        while x:
            count += 1
            x &= x - 1
        return count

    cpdef tuple get_stone_counts(self):
        """石の数を取得

        Returns:
            (自分の石の数, 相手の石の数)
        """
        return (self._popcount(self.self_board), self._popcount(self.opp_board))

    cpdef np.ndarray get_tensor_input(self):
        """ニューラルネットワーク入力用のテンソルを生成

        Returns:
            shape=(3, 8, 8) のfloat32配列
            - チャンネル0: 自分の石
            - チャンネル1: 相手の石
            - チャンネル2: 合法手マスク
        """
        cdef np.ndarray[np.float32_t, ndim=3] tensor = np.zeros((3, 8, 8), dtype=np.float32)
        cdef uint64_t legal = self._compute_legal_moves(self.self_board, self.opp_board)
        cdef int i, row, col

        for i in range(64):
            row = i // 8
            col = i % 8
            if self.self_board & (1ULL << i):
                tensor[0, row, col] = 1.0
            if self.opp_board & (1ULL << i):
                tensor[1, row, col] = 1.0
            if legal & (1ULL << i):
                tensor[2, row, col] = 1.0

        return tensor

    cpdef OthelloBitboard copy(self):
        """盤面のコピーを作成

        Returns:
            現在の盤面のディープコピー
        """
        cdef OthelloBitboard new_board = OthelloBitboard.__new__(OthelloBitboard)
        new_board.self_board = self.self_board
        new_board.opp_board = self.opp_board
        new_board.move_count = self.move_count
        new_board.passed = self.passed
        return new_board

    cpdef list get_symmetries(self, np.ndarray pi):
        """盤面と方策の対称性を利用した拡張データを生成

        Args:
            pi: 方策ベクトル（65次元: 64マス + パス）

        Returns:
            (board_tensor, pi)のタプルのリスト（8パターン）
        """
        cdef list symmetries = []
        cdef np.ndarray board = self.get_tensor_input()
        cdef np.ndarray pi_board = pi[:64].reshape(8, 8)
        cdef int k
        cdef np.ndarray rotated_board, rotated_pi, new_pi

        for k in range(4):
            # 回転
            rotated_board = np.rot90(board, k, axes=(1, 2))
            rotated_pi = np.rot90(pi_board, k)
            new_pi = np.zeros(65, dtype=np.float32)
            new_pi[:64] = rotated_pi.flatten()
            new_pi[64] = pi[64]  # パスの確率
            symmetries.append((rotated_board.copy(), new_pi))

            # 反転 + 回転
            flipped_board = np.flip(rotated_board, axis=2)
            flipped_pi = np.flip(rotated_pi, axis=1)
            new_pi = np.zeros(65, dtype=np.float32)
            new_pi[:64] = flipped_pi.flatten()
            new_pi[64] = pi[64]
            symmetries.append((flipped_board.copy(), new_pi))

        return symmetries

    def __repr__(self):
        """盤面の文字列表現"""
        cdef list lines = ["  A B C D E F G H"]
        cdef int row, col, i
        cdef str line

        for row in range(8):
            line = f"{row + 1} "
            for col in range(8):
                i = row * 8 + col
                if self.self_board & (1ULL << i):
                    line += "● "
                elif self.opp_board & (1ULL << i):
                    line += "○ "
                else:
                    line += ". "
            lines.append(line)

        return "\n".join(lines)

    cpdef str to_string(self):
        """盤面の文字列表現（__repr__のcpdef版）"""
        return self.__repr__()
