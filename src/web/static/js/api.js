/**
 * API通信レイヤー
 *
 * バックエンドとの通信を担当
 */

const API = {
    baseUrl: '/api',

    /**
     * GETリクエスト
     */
    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    },

    /**
     * POSTリクエスト
     */
    async post(endpoint, data = {}) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    },

    /**
     * PUTリクエスト
     */
    async put(endpoint, data = {}) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    },

    // === ゲームAPI ===

    /**
     * 新規ゲーム開始
     */
    async newGame(mode = 'human_vs_ai') {
        return this.post('/game/new', { mode });
    },

    /**
     * ゲーム状態取得
     */
    async getState() {
        return this.get('/game/state');
    },

    /**
     * 着手
     */
    async makeMove(position) {
        return this.post('/game/move', { position });
    },

    /**
     * 一手戻す
     */
    async undoMove() {
        return this.post('/game/undo');
    },

    /**
     * AI着手リクエスト
     */
    async requestAiMove() {
        return this.post('/game/ai-move');
    },

    /**
     * AI状態確認
     */
    async getAiStatus() {
        return this.get('/game/ai-status');
    },

    /**
     * ヒント取得
     */
    async getHint() {
        return this.get('/game/hint');
    },

    // === AI設定API ===

    /**
     * モデル読み込み
     */
    async loadModel(modelPath) {
        return this.post('/ai/load-model', { model_path: modelPath });
    },

    /**
     * シミュレーション回数設定
     */
    async setSimulations(count) {
        return this.put('/ai/simulations', { count });
    },

    /**
     * シミュレーション回数取得
     */
    async getSimulations() {
        return this.get('/ai/simulations');
    },

    /**
     * 利用可能モデル一覧
     */
    async listModels() {
        return this.get('/ai/models');
    },

    // === AI着手ポーリング ===

    /**
     * AI着手を待機（ポーリング）
     */
    async waitForAiMove(onUpdate, pollInterval = 200, timeout = 60000) {
        const startTime = Date.now();

        return new Promise((resolve, reject) => {
            const poll = async () => {
                try {
                    const status = await this.getAiStatus();

                    // 進捗コールバック
                    if (onUpdate) {
                        onUpdate(status);
                    }

                    if (!status.is_thinking) {
                        // AI思考完了
                        resolve(status.game_state);
                        return;
                    }

                    // タイムアウトチェック
                    if (Date.now() - startTime > timeout) {
                        reject(new Error('AI move timeout'));
                        return;
                    }

                    // 次のポーリング
                    setTimeout(poll, pollInterval);

                } catch (error) {
                    reject(error);
                }
            };

            poll();
        });
    },
};
