"""
Pydanticスキーマ定義

Web APIのリクエスト/レスポンスモデル
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


# === リクエストモデル ===


class NewGameRequest(BaseModel):
    """新規ゲームリクエスト"""

    mode: str = Field(
        default="human_vs_ai",
        description="ゲームモード: human_vs_ai, human_vs_human, ai_vs_ai",
    )


class MoveRequest(BaseModel):
    """着手リクエスト"""

    position: int = Field(..., ge=0, le=64, description="着手位置 (0-63) またはパス (64)")


class LoadModelRequest(BaseModel):
    """モデル読み込みリクエスト"""

    model_path: str = Field(..., description="モデルファイルのパス")


class SimulationsRequest(BaseModel):
    """シミュレーション数設定リクエスト"""

    count: int = Field(..., ge=10, le=500, description="シミュレーション回数 (10-500)")


# === レスポンスモデル ===


class GameState(BaseModel):
    """ゲーム状態レスポンス"""

    board: List[List[int]] = Field(..., description="8x8盤面 (0=空, 1=黒, -1=白)")
    legal_moves: List[int] = Field(..., description="合法手リスト (0-63)")
    current_player: int = Field(..., description="現在の手番 (1=黒, -1=白)")
    black_count: int = Field(..., description="黒石の数")
    white_count: int = Field(..., description="白石の数")
    is_terminal: bool = Field(..., description="ゲーム終了フラグ")
    winner: Optional[int] = Field(None, description="勝者 (1=黒, -1=白, 0=引き分け, None=未終了)")
    is_ai_thinking: bool = Field(..., description="AI思考中フラグ")
    move_count: int = Field(..., description="手数")
    message: Optional[str] = Field(None, description="ステータスメッセージ")
    model_loaded: bool = Field(..., description="モデル読み込み済みフラグ")


class MoveResponse(BaseModel):
    """着手レスポンス"""

    success: bool = Field(..., description="成功フラグ")
    game_state: GameState = Field(..., description="着手後のゲーム状態")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class HintResponse(BaseModel):
    """ヒントレスポンス"""

    evaluations: Dict[int, int] = Field(
        ..., description="各位置の評価値 (position -> value 0-100)"
    )
    success: bool = Field(..., description="成功フラグ")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class AIStatusResponse(BaseModel):
    """AI状態レスポンス"""

    is_thinking: bool = Field(..., description="AI思考中フラグ")
    game_state: GameState = Field(..., description="現在のゲーム状態")


class ModelListResponse(BaseModel):
    """利用可能モデル一覧レスポンス"""

    models: List[str] = Field(..., description="モデルファイルパスのリスト")


class SuccessResponse(BaseModel):
    """汎用成功レスポンス"""

    success: bool = Field(..., description="成功フラグ")
    message: Optional[str] = Field(None, description="メッセージ")
    error: Optional[str] = Field(None, description="エラーメッセージ")
