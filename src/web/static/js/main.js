/**
 * メインアプリケーション
 *
 * 初期化とイベントハンドリング
 */

class OthelloApp {
    constructor() {
        // コンポーネント初期化
        this.board = new OthelloBoard('board-canvas', {
            boardSize: 8,
            cellSize: 60,
        });
        this.ui = new UI();

        // ゲーム状態
        this.gameState = null;
        this.showingHints = false;

        // イベントハンドラ設定
        this._setupEventHandlers();

        // 初期化
        this._initialize();
    }

    /**
     * 初期化
     */
    async _initialize() {
        try {
            // ゲーム状態を取得
            this.gameState = await API.getState();
            this._updateDisplay();

            // モデル一覧を取得
            const modelList = await API.listModels();
            this.ui.updateModelList(modelList.models);

            // シミュレーション回数を取得
            const simData = await API.getSimulations();
            this.ui.elements.simSlider.value = simData.count;
            this.ui.updateSimulationsDisplay(simData.count);

            this.ui.showMessage('Ready to play!');

        } catch (error) {
            this.ui.showError(error.message);
        }
    }

    /**
     * イベントハンドラ設定
     */
    _setupEventHandlers() {
        // 盤面クリック
        this.board.onClick((position, row, col) => {
            this._handleBoardClick(position);
        });

        // New Gameボタン
        this.ui.elements.newGameBtn.addEventListener('click', () => {
            this._handleNewGame();
        });

        // Undoボタン
        this.ui.elements.undoBtn.addEventListener('click', () => {
            this._handleUndo();
        });

        // AI Moveボタン
        this.ui.elements.aiMoveBtn.addEventListener('click', () => {
            this._handleAiMove();
        });

        // Hintボタン
        this.ui.elements.hintBtn.addEventListener('click', () => {
            this._handleHint();
        });

        // シミュレーション数スライダー
        this.ui.elements.simSlider.addEventListener('input', (e) => {
            this.ui.updateSimulationsDisplay(e.target.value);
        });

        this.ui.elements.simSlider.addEventListener('change', (e) => {
            this._handleSimulationsChange(parseInt(e.target.value));
        });

        // モデル読み込みボタン
        this.ui.elements.loadModelBtn.addEventListener('click', () => {
            this._handleLoadModel();
        });
    }

    /**
     * 表示を更新
     */
    _updateDisplay() {
        if (!this.gameState) return;

        // 盤面更新
        this.board.update(
            this.gameState.board,
            this.gameState.legal_moves,
            this.showingHints ? this.board.evaluations : null
        );

        // UI更新
        this.ui.updateFromState(this.gameState);
    }

    /**
     * 盤面クリック処理
     */
    async _handleBoardClick(position) {
        if (!this.ui.controlsEnabled) return;
        if (!this.gameState) return;
        if (this.gameState.is_terminal) return;

        // 合法手チェック
        if (!this.board.isLegalMove(position)) {
            this.ui.showMessage('Invalid move', 'warning');
            return;
        }

        try {
            // 着手
            const response = await API.makeMove(position);

            if (response.success) {
                this.gameState = response.game_state;
                this.board.setLastMove(position);
                this.showingHints = false;
                this._updateDisplay();

                // 終局でなければAIの手番
                if (!this.gameState.is_terminal && this.gameState.model_loaded) {
                    // 少し待ってからAIに着手させる
                    setTimeout(() => this._handleAiMove(), 500);
                }
            } else {
                this.ui.showError(response.error);
            }

        } catch (error) {
            this.ui.showError(error.message);
        }
    }

    /**
     * 新規ゲーム処理
     */
    async _handleNewGame() {
        try {
            this.gameState = await API.newGame('human_vs_ai');
            this.board.setLastMove(null);
            this.showingHints = false;
            this._updateDisplay();
            this.ui.showMessage('New game started!', 'success');

        } catch (error) {
            this.ui.showError(error.message);
        }
    }

    /**
     * Undo処理
     */
    async _handleUndo() {
        try {
            const response = await API.undoMove();

            if (response.success) {
                this.gameState = response.game_state;
                this.board.setLastMove(null);
                this.showingHints = false;
                this._updateDisplay();
            } else {
                this.ui.showMessage(response.error, 'warning');
            }

        } catch (error) {
            this.ui.showError(error.message);
        }
    }

    /**
     * AI着手処理
     */
    async _handleAiMove() {
        if (!this.gameState || this.gameState.is_terminal) return;
        if (!this.gameState.model_loaded) {
            this.ui.showMessage('Load a model first', 'warning');
            return;
        }

        try {
            this.ui.showThinking();
            this.showingHints = false;

            // AI着手リクエスト
            await API.requestAiMove();

            // 完了を待機
            this.gameState = await API.waitForAiMove((status) => {
                // 進捗更新
            });

            this._updateDisplay();
            this.ui.hideThinking();

        } catch (error) {
            this.ui.hideThinking();
            this.ui.showError(error.message);
        }
    }

    /**
     * ヒント表示処理
     */
    async _handleHint() {
        if (this.showingHints) {
            // ヒントを非表示
            this.showingHints = false;
            this.board.setShowEvaluations(false);
            this.ui.showMessage('Hints hidden');
            return;
        }

        if (!this.gameState || this.gameState.is_terminal) return;
        if (!this.gameState.model_loaded) {
            this.ui.showMessage('Load a model first', 'warning');
            return;
        }

        try {
            this.ui.showMessage('Calculating hints...', 'thinking');

            const response = await API.getHint();

            if (response.success) {
                this.showingHints = true;
                this.board.setShowEvaluations(true, response.evaluations);
                this.ui.showMessage('Hint: Higher = Better move', 'success');
            } else {
                this.ui.showError(response.error);
            }

        } catch (error) {
            this.ui.showError(error.message);
        }
    }

    /**
     * シミュレーション数変更処理
     */
    async _handleSimulationsChange(count) {
        try {
            await API.setSimulations(count);
        } catch (error) {
            this.ui.showError(error.message);
        }
    }

    /**
     * モデル読み込み処理
     */
    async _handleLoadModel() {
        const modelPath = this.ui.elements.modelSelect.value;

        if (!modelPath) {
            this.ui.showMessage('Select a model first', 'warning');
            return;
        }

        try {
            this.ui.showMessage('Loading model...', 'thinking');
            this.ui.setControlsEnabled(false);

            const response = await API.loadModel(modelPath);

            if (response.success) {
                this.ui.setModelLoaded(true, modelPath);
                this.ui.showMessage('Model loaded!', 'success');

                // ゲーム状態を更新
                this.gameState = await API.getState();
                this._updateDisplay();
            } else {
                this.ui.showError(response.error);
            }

        } catch (error) {
            this.ui.showError(error.message);
        } finally {
            this.ui.setControlsEnabled(true);
        }
    }
}

// アプリケーション起動
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OthelloApp();
});
