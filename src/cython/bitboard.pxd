# cython: language_level=3
"""ビットボードクラスのヘッダ定義

uint64を使用して64マスの盤面を効率的に表現する。
"""

cimport numpy as np
from libc.stdint cimport uint64_t


cdef class OthelloBitboard:
    """オセロのビットボード実装

    盤面を2つの64ビット整数で表現:
    - self_board: 現在手番プレイヤーの石
    - opp_board: 相手プレイヤーの石

    盤面インデックス:
        0  1  2  3  4  5  6  7
        8  9  10 11 12 13 14 15
        ...
        56 57 58 59 60 61 62 63
    """

    cdef public uint64_t self_board
    cdef public uint64_t opp_board
    cdef public int move_count
    cdef public bint passed

    # 内部メソッド（cdefで高速化）
    cdef uint64_t _get_flip_bits(self, int pos, uint64_t self_b, uint64_t opp_b) noexcept
    cdef uint64_t _get_flip_direction(self, int pos, int direction, uint64_t self_b, uint64_t opp_b, uint64_t mask) noexcept
    cdef uint64_t _compute_legal_moves(self, uint64_t self_b, uint64_t opp_b) noexcept
    cdef void _swap_players(self) noexcept
    cdef int _popcount(self, uint64_t x) noexcept

    # 公開メソッド（cpdef）
    cpdef void reset(self)
    cpdef list get_legal_moves(self)
    cpdef uint64_t get_legal_moves_bits(self)
    cpdef bint make_move(self, int pos)
    cpdef bint is_terminal(self)
    cpdef int get_winner(self)
    cpdef tuple get_stone_counts(self)
    cpdef np.ndarray get_tensor_input(self)
    cpdef OthelloBitboard copy(self)
    cpdef list get_symmetries(self, np.ndarray pi)
    cpdef str to_string(self)
