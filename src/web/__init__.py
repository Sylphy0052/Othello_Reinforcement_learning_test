"""
Webインターフェースモジュール

FastAPIを使ったWeb版オセロアプリ
"""

from .game_manager import GameManager
from .schemas import (
    GameState,
    MoveRequest,
    MoveResponse,
    NewGameRequest,
    HintResponse,
    LoadModelRequest,
    SimulationsRequest,
)

__all__ = [
    "GameManager",
    "GameState",
    "MoveRequest",
    "MoveResponse",
    "NewGameRequest",
    "HintResponse",
    "LoadModelRequest",
    "SimulationsRequest",
]
