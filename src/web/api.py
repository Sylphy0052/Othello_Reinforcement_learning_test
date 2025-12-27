"""
FastAPI APIエンドポイント

オセロゲームのWeb API
"""

from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .game_manager import GameManager
from .schemas import (
    GameState,
    MoveRequest,
    MoveResponse,
    NewGameRequest,
    HintResponse,
    LoadModelRequest,
    SimulationsRequest,
    AIStatusResponse,
    ModelListResponse,
    SuccessResponse,
)


# FastAPIアプリ
app = FastAPI(
    title="Othello AlphaZero Web",
    description="オセロAIのWebインターフェース",
    version="0.1.0",
)

# ゲームマネージャ（シングルトン）
_game_manager: Optional[GameManager] = None

# AI処理用スレッドプール
_executor = ThreadPoolExecutor(max_workers=1)


def get_game_manager() -> GameManager:
    """ゲームマネージャを取得"""
    global _game_manager
    if _game_manager is None:
        _game_manager = GameManager()
    return _game_manager


# === 静的ファイル配信 ===

# 静的ファイルのパス
STATIC_DIR = Path(__file__).parent / "static"

# 静的ファイルをマウント
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index():
    """メインページを配信"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


# === ゲームAPI ===


@app.post("/api/game/new", response_model=GameState)
async def new_game(request: NewGameRequest):
    """新規ゲーム開始"""
    manager = get_game_manager()
    manager.new_game(mode=request.mode)
    return manager.get_state()


@app.get("/api/game/state", response_model=GameState)
async def get_game_state():
    """現在のゲーム状態を取得"""
    manager = get_game_manager()
    return manager.get_state()


@app.post("/api/game/move", response_model=MoveResponse)
async def make_move(request: MoveRequest):
    """着手を実行"""
    manager = get_game_manager()

    success, error = manager.make_move(request.position)

    return MoveResponse(
        success=success,
        game_state=manager.get_state(),
        error=error,
    )


@app.post("/api/game/undo", response_model=MoveResponse)
async def undo_move():
    """一手戻す"""
    manager = get_game_manager()

    success, error = manager.undo()

    return MoveResponse(
        success=success,
        game_state=manager.get_state(),
        error=error,
    )


@app.post("/api/game/ai-move", response_model=AIStatusResponse)
async def request_ai_move(background_tasks: BackgroundTasks):
    """
    AIに着手させる（非同期）

    バックグラウンドでMCTS探索を実行し、
    フロントエンドは /api/game/ai-status でポーリング
    """
    manager = get_game_manager()

    if manager.is_ai_thinking:
        return AIStatusResponse(
            is_thinking=True,
            game_state=manager.get_state(),
        )

    if manager.mcts is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    if manager.board.is_terminal():
        raise HTTPException(status_code=400, detail="Game has ended")

    # AI思考開始
    manager.is_ai_thinking = True
    manager.last_message = "AI thinking..."

    def run_ai():
        """バックグラウンドでAI着手"""
        try:
            manager.execute_ai_move()
        finally:
            manager.is_ai_thinking = False

    # バックグラウンドタスクとして実行
    background_tasks.add_task(run_ai)

    return AIStatusResponse(
        is_thinking=True,
        game_state=manager.get_state(),
    )


@app.get("/api/game/ai-status", response_model=AIStatusResponse)
async def get_ai_status():
    """AI思考状態を取得"""
    manager = get_game_manager()

    return AIStatusResponse(
        is_thinking=manager.is_ai_thinking,
        game_state=manager.get_state(),
    )


@app.get("/api/game/hint", response_model=HintResponse)
async def get_hint():
    """ヒント評価値を取得"""
    manager = get_game_manager()

    if manager.mcts is None:
        return HintResponse(
            evaluations={},
            success=False,
            error="No model loaded",
        )

    # ヒント計算はブロッキング処理なので、スレッドプールで実行
    loop = asyncio.get_event_loop()
    evaluations, error = await loop.run_in_executor(
        _executor, manager.get_hint_evaluations
    )

    return HintResponse(
        evaluations=evaluations,
        success=error is None,
        error=error,
    )


# === AI設定API ===


@app.post("/api/ai/load-model", response_model=SuccessResponse)
async def load_model(request: LoadModelRequest):
    """モデルを読み込む"""
    manager = get_game_manager()

    # モデル読み込みはブロッキング処理なので、スレッドプールで実行
    loop = asyncio.get_event_loop()
    success, error = await loop.run_in_executor(
        _executor, lambda: manager.load_model(request.model_path)
    )

    return SuccessResponse(
        success=success,
        message="Model loaded successfully" if success else None,
        error=error,
    )


@app.put("/api/ai/simulations", response_model=SuccessResponse)
async def set_simulations(request: SimulationsRequest):
    """シミュレーション回数を設定"""
    manager = get_game_manager()
    manager.set_simulations(request.count)

    return SuccessResponse(
        success=True,
        message=f"Simulations set to {manager.ai_simulations}",
    )


@app.get("/api/ai/simulations")
async def get_simulations():
    """現在のシミュレーション回数を取得"""
    manager = get_game_manager()
    return {"count": manager.ai_simulations}


@app.get("/api/ai/models", response_model=ModelListResponse)
async def list_models():
    """利用可能なモデル一覧を取得"""
    models_dir = Path("data/models")
    models = []

    if models_dir.exists():
        for model_file in models_dir.glob("**/*.pt"):
            models.append(str(model_file))

    return ModelListResponse(models=sorted(models))
