"""
GUI アプリケーションモジュール

Tkinterを使ったオセロ対戦アプリ
"""

from .app import OthelloApp
from .board_ui import OthelloBoardUI, InfoPanel

__all__ = [
    "OthelloApp",
    "OthelloBoardUI",
    "InfoPanel",
]
