"""
Web UI E2Eテスト (Playwright)

オセロWebアプリのエンドツーエンドテスト
"""

import pytest
import subprocess
import time
import socket
from contextlib import closing

from playwright.sync_api import Page, expect


def find_free_port() -> int:
    """空いているポートを探す"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    """テスト用Webサーバーを起動"""
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    # サーバー起動
    proc = subprocess.Popen(
        ["uv", "run", "python", "run_web.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # サーバー起動を待機
    max_wait = 10
    for _ in range(max_wait * 10):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.connect(("127.0.0.1", port))
                break
        except ConnectionRefusedError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("Server failed to start")

    yield url

    # サーバー停止
    proc.terminate()
    proc.wait(timeout=5)


class TestWebUI:
    """Web UI テスト"""

    def test_page_loads(self, page: Page, server_url: str):
        """ページが正常に読み込まれる"""
        page.goto(server_url)

        # タイトル確認
        expect(page).to_have_title("Othello AlphaZero")

        # 主要要素が存在する
        expect(page.locator("#board-canvas")).to_be_visible()
        expect(page.locator("#new-game-btn")).to_be_visible()
        expect(page.locator("#undo-btn")).to_be_visible()

    def test_initial_board_state(self, page: Page, server_url: str):
        """初期盤面が正しく表示される"""
        page.goto(server_url)

        # 石数が2-2
        expect(page.locator("#black-count")).to_have_text("2")
        expect(page.locator("#white-count")).to_have_text("2")

        # 黒の手番
        expect(page.locator("#turn-value")).to_have_text("Black")

    def test_new_game_button(self, page: Page, server_url: str):
        """New Gameボタンが動作する"""
        page.goto(server_url)

        # まず着手する
        canvas = page.locator("#board-canvas")
        # D3 (position 19) をクリック - 座標計算: col=3, row=2 -> x=210, y=150
        canvas.click(position={"x": 210, "y": 150})

        # 少し待つ
        page.wait_for_timeout(500)

        # New Gameをクリック
        page.locator("#new-game-btn").click()

        # 石数がリセットされる
        expect(page.locator("#black-count")).to_have_text("2")
        expect(page.locator("#white-count")).to_have_text("2")

    def test_click_on_board(self, page: Page, server_url: str):
        """盤面クリックで着手できる"""
        page.goto(server_url)

        # 初期状態確認
        expect(page.locator("#black-count")).to_have_text("2")

        # D3 (position 19) をクリック - 合法手
        canvas = page.locator("#board-canvas")
        # position 19 = row 2, col 3 -> center: (3*60+30, 2*60+30) = (210, 150)
        canvas.click(position={"x": 210, "y": 150})

        # 着手後、石数が変わる
        page.wait_for_timeout(500)
        # 黒が増えるはず (2 -> 4: 1個置く + 1個返す)
        expect(page.locator("#black-count")).to_have_text("4")

    def test_undo_button(self, page: Page, server_url: str):
        """Undoボタンが動作する"""
        page.goto(server_url)

        # 着手
        canvas = page.locator("#board-canvas")
        canvas.click(position={"x": 210, "y": 150})  # D3

        page.wait_for_timeout(500)
        expect(page.locator("#black-count")).to_have_text("4")

        # Undo
        page.locator("#undo-btn").click()

        page.wait_for_timeout(300)
        expect(page.locator("#black-count")).to_have_text("2")

    def test_simulations_slider(self, page: Page, server_url: str):
        """シミュレーション数スライダーが動作する"""
        page.goto(server_url)

        slider = page.locator("#sim-slider")
        sim_value = page.locator("#sim-value")

        # 初期値確認
        expect(sim_value).to_have_text("50")

        # スライダーを変更
        slider.fill("100")
        slider.dispatch_event("change")

        page.wait_for_timeout(300)
        expect(sim_value).to_have_text("100")

    def test_model_select_exists(self, page: Page, server_url: str):
        """モデル選択ドロップダウンが存在する"""
        page.goto(server_url)

        model_select = page.locator("#model-select")
        expect(model_select).to_be_visible()

        load_btn = page.locator("#load-model-btn")
        expect(load_btn).to_be_visible()

    def test_hint_button_without_model(self, page: Page, server_url: str):
        """モデルなしでHintボタンを押すとメッセージが表示される"""
        page.goto(server_url)

        page.locator("#hint-btn").click()

        page.wait_for_timeout(300)
        message = page.locator("#message-area")
        expect(message).to_contain_text("model")

    def test_ai_move_button_without_model(self, page: Page, server_url: str):
        """モデルなしでAI Moveボタンを押すとメッセージが表示される"""
        page.goto(server_url)

        page.locator("#ai-move-btn").click()

        page.wait_for_timeout(300)
        message = page.locator("#message-area")
        expect(message).to_contain_text("model")

    def test_invalid_move_shows_message(self, page: Page, server_url: str):
        """不正な位置をクリックするとメッセージが表示される"""
        page.goto(server_url)

        # 不正な位置（A1 = position 0）をクリック
        canvas = page.locator("#board-canvas")
        canvas.click(position={"x": 30, "y": 30})  # A1

        page.wait_for_timeout(300)
        message = page.locator("#message-area")
        expect(message).to_contain_text("Invalid")


class TestAPIDirectly:
    """API直接テスト"""

    def test_api_game_state(self, page: Page, server_url: str):
        """API: /api/game/state が正しいレスポンスを返す"""
        response = page.request.get(f"{server_url}/api/game/state")

        assert response.ok
        data = response.json()

        assert "board" in data
        assert "legal_moves" in data
        assert "current_player" in data
        assert len(data["board"]) == 8
        assert len(data["board"][0]) == 8

    def test_api_new_game(self, page: Page, server_url: str):
        """API: /api/game/new が動作する"""
        response = page.request.post(
            f"{server_url}/api/game/new",
            data={"mode": "human_vs_ai"},
        )

        assert response.ok
        data = response.json()
        assert data["move_count"] == 0

    def test_api_make_move(self, page: Page, server_url: str):
        """API: /api/game/move が動作する"""
        # まず新規ゲーム
        page.request.post(f"{server_url}/api/game/new", data={"mode": "human_vs_ai"})

        # 着手
        response = page.request.post(
            f"{server_url}/api/game/move",
            data={"position": 19},
        )

        assert response.ok
        data = response.json()
        assert data["success"] is True
        assert data["game_state"]["move_count"] == 1

    def test_api_undo(self, page: Page, server_url: str):
        """API: /api/game/undo が動作する"""
        # 新規ゲーム
        page.request.post(f"{server_url}/api/game/new", data={"mode": "human_vs_ai"})

        # 着手
        page.request.post(f"{server_url}/api/game/move", data={"position": 19})

        # Undo
        response = page.request.post(f"{server_url}/api/game/undo")

        assert response.ok
        data = response.json()
        assert data["success"] is True
        assert data["game_state"]["move_count"] == 0

    def test_api_invalid_move(self, page: Page, server_url: str):
        """API: 不正な着手はエラーを返す"""
        # 新規ゲーム
        page.request.post(f"{server_url}/api/game/new", data={"mode": "human_vs_ai"})

        # 不正な位置に着手
        response = page.request.post(
            f"{server_url}/api/game/move",
            data={"position": 0},  # A1は合法手ではない
        )

        assert response.ok
        data = response.json()
        assert data["success"] is False
        assert data["error"] is not None
