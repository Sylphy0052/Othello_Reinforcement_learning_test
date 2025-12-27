"""Cython拡張モジュールのビルドスクリプト

使用方法:
    uv run python setup.py build_ext --inplace
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "src.cython.bitboard",
        sources=["src/cython/bitboard.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],  # 最適化オプション
    )
]

setup(
    name="othello-cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,  # 境界チェック無効（高速化）
            "wraparound": False,  # 負のインデックス無効（高速化）
            "cdivision": True,  # C言語形式の除算（高速化）
        },
    ),
)
