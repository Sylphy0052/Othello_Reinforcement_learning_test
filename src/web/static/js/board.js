/**
 * 盤面描画モジュール
 *
 * HTML5 Canvasを使った8x8オセロ盤面の描画
 */

class OthelloBoard {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        // 設定
        this.boardSize = options.boardSize || 8;
        this.cellSize = options.cellSize || 60;
        this.margin = options.margin || 0;

        // 色設定
        this.colors = {
            board: '#0a6522',
            line: '#064d17',
            black: '#111',
            white: '#eee',
            legalHint: 'rgba(144, 238, 144, 0.6)',
            evalGood: 'rgba(76, 175, 80, 0.8)',
            evalNeutral: 'rgba(255, 193, 7, 0.8)',
            evalBad: 'rgba(255, 87, 34, 0.8)',
        };

        // 状態
        this.boardState = null;
        this.legalMoves = [];
        this.evaluations = null;
        this.showEvaluations = false;
        this.lastMove = null;

        // クリックハンドラ
        this.onClickCallback = null;

        // イベントリスナー設定
        this.canvas.addEventListener('click', (e) => this._handleClick(e));

        // 初期描画
        this.clear();
    }

    /**
     * クリックコールバック設定
     */
    onClick(callback) {
        this.onClickCallback = callback;
    }

    /**
     * クリックイベント処理
     */
    _handleClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const col = Math.floor(x / this.cellSize);
        const row = Math.floor(y / this.cellSize);

        if (row >= 0 && row < this.boardSize && col >= 0 && col < this.boardSize) {
            const position = row * this.boardSize + col;

            if (this.onClickCallback) {
                this.onClickCallback(position, row, col);
            }
        }
    }

    /**
     * 盤面をクリア
     */
    clear() {
        const size = this.boardSize * this.cellSize;

        // 盤面背景
        this.ctx.fillStyle = this.colors.board;
        this.ctx.fillRect(0, 0, size, size);

        // グリッド線
        this.ctx.strokeStyle = this.colors.line;
        this.ctx.lineWidth = 1;

        for (let i = 0; i <= this.boardSize; i++) {
            const pos = i * this.cellSize;

            // 縦線
            this.ctx.beginPath();
            this.ctx.moveTo(pos, 0);
            this.ctx.lineTo(pos, size);
            this.ctx.stroke();

            // 横線
            this.ctx.beginPath();
            this.ctx.moveTo(0, pos);
            this.ctx.lineTo(size, pos);
            this.ctx.stroke();
        }

        // 星（中央4点）
        const starPositions = [
            [2, 2], [2, 6], [6, 2], [6, 6]
        ];
        this.ctx.fillStyle = this.colors.line;
        for (const [row, col] of starPositions) {
            const x = col * this.cellSize;
            const y = row * this.cellSize;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 4, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    /**
     * 盤面を更新
     */
    update(boardState, legalMoves = [], evaluations = null) {
        this.boardState = boardState;
        this.legalMoves = legalMoves;
        this.evaluations = evaluations;

        this.render();
    }

    /**
     * 盤面を描画
     */
    render() {
        this.clear();

        if (!this.boardState) return;

        // 合法手ヒント表示
        for (const pos of this.legalMoves) {
            const row = Math.floor(pos / this.boardSize);
            const col = pos % this.boardSize;
            this._drawLegalHint(row, col);
        }

        // 評価値表示
        if (this.showEvaluations && this.evaluations) {
            for (const pos of this.legalMoves) {
                if (pos in this.evaluations) {
                    const row = Math.floor(pos / this.boardSize);
                    const col = pos % this.boardSize;
                    this._drawEvaluation(row, col, this.evaluations[pos]);
                }
            }
        }

        // 石を描画
        for (let row = 0; row < this.boardSize; row++) {
            for (let col = 0; col < this.boardSize; col++) {
                const value = this.boardState[row][col];
                if (value !== 0) {
                    this._drawStone(row, col, value);
                }
            }
        }

        // 最後の着手位置をハイライト
        if (this.lastMove !== null) {
            const row = Math.floor(this.lastMove / this.boardSize);
            const col = this.lastMove % this.boardSize;
            this._drawLastMoveIndicator(row, col);
        }
    }

    /**
     * 石を描画
     */
    _drawStone(row, col, player) {
        const x = col * this.cellSize + this.cellSize / 2;
        const y = row * this.cellSize + this.cellSize / 2;
        const radius = this.cellSize / 2 - 5;

        // グラデーション
        const gradient = this.ctx.createRadialGradient(
            x - radius / 3, y - radius / 3, 0,
            x, y, radius
        );

        if (player === 1) {
            // 黒石
            gradient.addColorStop(0, '#444');
            gradient.addColorStop(1, '#000');
        } else {
            // 白石
            gradient.addColorStop(0, '#fff');
            gradient.addColorStop(1, '#ccc');
        }

        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();

        // 縁
        this.ctx.strokeStyle = player === 1 ? '#000' : '#999';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }

    /**
     * 合法手ヒントを描画
     */
    _drawLegalHint(row, col) {
        const x = col * this.cellSize + this.cellSize / 2;
        const y = row * this.cellSize + this.cellSize / 2;
        const radius = 8;

        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = this.colors.legalHint;
        this.ctx.fill();
    }

    /**
     * 評価値を描画
     */
    _drawEvaluation(row, col, value) {
        const x = col * this.cellSize + this.cellSize / 2;
        const y = row * this.cellSize + this.cellSize / 2;

        // 背景色（評価値に応じて変更）
        let bgColor;
        if (value >= 60) {
            bgColor = this.colors.evalGood;
        } else if (value >= 40) {
            bgColor = this.colors.evalNeutral;
        } else {
            bgColor = this.colors.evalBad;
        }

        // 背景円
        this.ctx.beginPath();
        this.ctx.arc(x, y, 18, 0, Math.PI * 2);
        this.ctx.fillStyle = bgColor;
        this.ctx.fill();

        // 数値
        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 14px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(value.toString(), x, y);
    }

    /**
     * 最後の着手位置インジケータ
     */
    _drawLastMoveIndicator(row, col) {
        const x = col * this.cellSize + this.cellSize / 2;
        const y = row * this.cellSize + this.cellSize / 2;

        this.ctx.strokeStyle = '#ff0';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.arc(x, y, this.cellSize / 2 - 3, 0, Math.PI * 2);
        this.ctx.stroke();
    }

    /**
     * 評価値表示切り替え
     */
    setShowEvaluations(show, evaluations = null) {
        this.showEvaluations = show;
        if (evaluations !== null) {
            this.evaluations = evaluations;
        }
        this.render();
    }

    /**
     * 最後の着手位置を設定
     */
    setLastMove(position) {
        this.lastMove = position;
        this.render();
    }

    /**
     * 合法手かどうかチェック
     */
    isLegalMove(position) {
        return this.legalMoves.includes(position);
    }
}
