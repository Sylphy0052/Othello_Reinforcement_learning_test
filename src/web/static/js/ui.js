/**
 * UI管理モジュール
 *
 * UIコンポーネントの更新と制御
 */

class UI {
    constructor() {
        // DOM要素の取得
        this.elements = {
            turnValue: document.getElementById('turn-value'),
            turnStone: document.getElementById('turn-stone'),
            blackCount: document.getElementById('black-count'),
            whiteCount: document.getElementById('white-count'),
            messageArea: document.getElementById('message-area'),
            newGameBtn: document.getElementById('new-game-btn'),
            undoBtn: document.getElementById('undo-btn'),
            aiMoveBtn: document.getElementById('ai-move-btn'),
            hintBtn: document.getElementById('hint-btn'),
            simSlider: document.getElementById('sim-slider'),
            simValue: document.getElementById('sim-value'),
            modelSelect: document.getElementById('model-select'),
            loadModelBtn: document.getElementById('load-model-btn'),
            modelStatus: document.getElementById('model-status'),
        };

        // コントロール状態
        this.controlsEnabled = true;
    }

    /**
     * ゲーム状態からUIを更新
     */
    updateFromState(state) {
        // 手番表示
        if (state.current_player === 1) {
            this.elements.turnValue.textContent = 'Black';
            this.elements.turnStone.className = 'black';
        } else {
            this.elements.turnValue.textContent = 'White';
            this.elements.turnStone.className = 'white';
        }

        // 石数表示
        this.elements.blackCount.textContent = state.black_count;
        this.elements.whiteCount.textContent = state.white_count;

        // メッセージ表示
        if (state.message) {
            this.showMessage(state.message);
        }

        // 終局時
        if (state.is_terminal) {
            this._showGameResult(state);
        }

        // モデル状態
        if (state.model_loaded) {
            this.elements.modelStatus.textContent = 'Model loaded';
            this.elements.modelStatus.className = 'loaded';
        }
    }

    /**
     * ゲーム結果を表示
     */
    _showGameResult(state) {
        let message;
        let type;

        if (state.winner === 1) {
            message = `Black wins! (${state.black_count}-${state.white_count})`;
            type = 'success';
        } else if (state.winner === -1) {
            message = `White wins! (${state.black_count}-${state.white_count})`;
            type = 'error';
        } else {
            message = `Draw! (${state.black_count}-${state.white_count})`;
            type = 'warning';
        }

        this.showMessage(message, type);
    }

    /**
     * メッセージを表示
     */
    showMessage(message, type = '') {
        this.elements.messageArea.textContent = message;
        this.elements.messageArea.className = type;
    }

    /**
     * 思考中表示
     */
    showThinking() {
        this.showMessage('AI thinking...', 'thinking');
        this.setControlsEnabled(false);
    }

    /**
     * 思考完了
     */
    hideThinking() {
        this.setControlsEnabled(true);
    }

    /**
     * コントロールの有効/無効
     */
    setControlsEnabled(enabled) {
        this.controlsEnabled = enabled;

        this.elements.newGameBtn.disabled = !enabled;
        this.elements.undoBtn.disabled = !enabled;
        this.elements.aiMoveBtn.disabled = !enabled;
        this.elements.hintBtn.disabled = !enabled;
        this.elements.loadModelBtn.disabled = !enabled;
    }

    /**
     * スライダー値を更新
     */
    updateSimulationsDisplay(value) {
        this.elements.simValue.textContent = value;
    }

    /**
     * モデル一覧を更新
     */
    updateModelList(models) {
        const select = this.elements.modelSelect;

        // 既存のオプションをクリア（最初のプレースホルダーは残す）
        while (select.options.length > 1) {
            select.remove(1);
        }

        // モデルを追加
        for (const model of models) {
            const option = document.createElement('option');
            option.value = model;
            // パスからファイル名を抽出
            option.textContent = model.split('/').pop();
            select.appendChild(option);
        }
    }

    /**
     * モデル読み込み状態を更新
     */
    setModelLoaded(loaded, modelPath = '') {
        if (loaded) {
            const modelName = modelPath.split('/').pop();
            this.elements.modelStatus.textContent = `Loaded: ${modelName}`;
            this.elements.modelStatus.className = 'loaded';
        } else {
            this.elements.modelStatus.textContent = 'No model loaded';
            this.elements.modelStatus.className = '';
        }
    }

    /**
     * エラーを表示
     */
    showError(error) {
        this.showMessage(`Error: ${error}`, 'error');
    }
}
