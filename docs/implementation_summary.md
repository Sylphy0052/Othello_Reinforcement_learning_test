# AlphaZero Othello - 実装サマリー

## 概要

AlphaZero方式のオセロAI。自己対戦のみで学習し、人間の専門知識に依存しない強力なAIを構築する。

- **技術スタック**: Python 3.x + Cython + PyTorch
- **対象ハードウェア**: RTX 4050 (6GB VRAM)
- **盤面サイズ**: 8x8

---

## プロジェクト構造

```
├── main.py                 # CLIエントリポイント
├── run_gui.py              # GUIエントリポイント
├── configs/
│   └── default_8x8.yaml    # 学習設定ファイル
├── src/
│   ├── cython/
│   │   └── bitboard.pyx    # 高速ビットボード実装
│   ├── model/
│   │   └── net.py          # ResNetモデル
│   ├── mcts/
│   │   ├── mcts.py         # MCTS探索
│   │   └── node.py         # MCTSノード
│   ├── train/
│   │   ├── trainer.py      # 学習ループ
│   │   ├── self_play.py    # 自己対戦
│   │   └── buffer.py       # リプレイバッファ
│   ├── eval/
│   │   ├── players.py      # プレイヤークラス
│   │   └── arena.py        # 対戦評価
│   └── gui/
│       ├── app.py          # GUIアプリ
│       └── board_ui.py     # 盤面描画
├── tests/                   # テストコード
└── data/
    ├── models/             # チェックポイント保存先
    └── logs/               # TensorBoardログ
```

---

## コンポーネント詳細

### 1. Bitboard (`src/cython/bitboard.pyx`)

**概要**: Cythonによる高速オセロ盤面管理

**データ構造**:
- `self_board`: 自分の石 (uint64)
- `opp_board`: 相手の石 (uint64)
- 64ビット整数で8x8盤面を表現

**主要メソッド**:

| メソッド | 説明 |
|----------|------|
| `reset()` | 盤面を初期状態にリセット |
| `get_legal_moves()` | 合法手リストを取得 (0-63, パス=64) |
| `make_move(pos)` | 指定位置に着手 |
| `is_terminal()` | ゲーム終了判定 |
| `get_winner()` | 勝者判定 (1:自分, -1:相手, 0:引分) |
| `get_tensor_input()` | NN入力用テンソル (3, 8, 8) を生成 |
| `copy()` | 盤面のディープコピー |
| `get_symmetries(pi)` | 対称性拡張データ生成 |

**性能目標**: 5,000-10,000 games/sec

**ビット演算の仕組み**:
```
8方向: 上(-8), 下(+8), 左(-1), 右(+1), 斜め4方向
マスク: A列除外 (0xFEFE...), H列除外 (0x7F7F...)
```

---

### 2. ResNet モデル (`src/model/net.py`)

**概要**: AlphaZero方式のDual-Head ResNet

**アーキテクチャ**:

```
入力: (Batch, 3, 8, 8)
  ├── Channel 0: 自分の石
  ├── Channel 1: 相手の石
  └── Channel 2: 合法手マスク

      ↓
ConvBlock (3 → 128ch, 3x3, BN, ReLU)
      ↓
ResBlock × 10 (128ch, 3x3, BN, Skip Connection)
      ↓
      ├── PolicyHead → (Batch, 65) Log確率分布
      └── ValueHead  → (Batch, 1) 価値 [-1, 1]
```

**設定パラメータ** (default_8x8.yaml):
- `num_blocks`: 10
- `num_filters`: 128
- **総パラメータ数**: 約300万

**クラス構成**:

| クラス | 説明 |
|--------|------|
| `ConvBlock` | 初期畳み込みブロック |
| `ResBlock` | 残差ブロック (Skip Connection) |
| `PolicyHead` | 方策出力ヘッド (65次元) |
| `ValueHead` | 価値出力ヘッド (スカラー) |
| `OthelloResNet` | メインモデルクラス |

---

### 3. MCTS (`src/mcts/mcts.py`, `src/mcts/node.py`)

**概要**: AlphaZero方式のモンテカルロ木探索

**PUCT式** (子ノード選択):
```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))

Q(s,a) = W(s,a) / N(s,a)  ... 平均価値
P(s,a) = NNの事前確率
N(s,a) = 訪問回数
```

**探索フロー**:
```
1. Select:    PUCT最大の子ノードを選択して下降
2. Expand:    リーフノードでNNを評価、子ノードを展開
3. Backprop:  価値を親ノードに逆伝播 (符号反転)
4. 繰り返し: num_simulations回
```

**MCTSノード** (`MCTSNode`):
- `prior`: 事前確率 (NNの出力)
- `visit_count`: 訪問回数 N
- `value_sum`: 累積価値 W
- `children`: 子ノード辞書 {action: MCTSNode}

**主要パラメータ**:

| パラメータ | デフォルト値 | 説明 |
|------------|--------------|------|
| `num_simulations` | 25 (学習) / 50 (評価) | シミュレーション回数 |
| `c_puct` | 1.0 | 探索定数 |
| `dirichlet_alpha` | 0.3 | ディリクレノイズ α |
| `dirichlet_epsilon` | 0.25 | ノイズ混合比率 |

---

### 4. 学習システム (`src/train/`)

#### 4.1 Trainer (`trainer.py`)

**概要**: 学習ループ全体のオーケストレーション

**学習フロー**:
```
for iteration in range(num_iterations):
    1. Self-Play: episodes数のゲームを実行、データ生成
    2. Replay Buffer: 生成データを格納
    3. Train: epochs回ミニバッチ学習 (AMP使用)
    4. Checkpoint: 定期的に保存
    5. TensorBoard: ログ記録
```

**損失関数**:
- **方策損失**: クロスエントロピー `-Σ target × log(pred)`
- **価値損失**: MSE `(pred - target)²`
- **合計損失**: policy_loss + value_loss

**学習設定**:

| パラメータ | デフォルト値 |
|------------|--------------|
| `batch_size` | 256 |
| `lr` | 0.001 |
| `momentum` | 0.9 |
| `weight_decay` | 0.0001 |
| `num_iterations` | 1000 |
| `self_play_episodes_per_iter` | 100 |
| `train_epochs_per_iter` | 10 |
| `use_mixed_precision` | true |

#### 4.2 Self-Play Worker (`self_play.py`)

**概要**: MCTSを使った自己対戦でデータ生成

**データ形式**:
```python
(state, policy, value)
  - state: (3, 8, 8) NN入力テンソル
  - policy: (65,) MCTS訪問回数分布
  - value: 1.0 (勝ち) / -1.0 (負け) / 0.0 (引分)
```

**温度パラメータ**:
- 序盤 (< `temperature_threshold`手): `temperature=1.0` (確率的選択)
- 終盤: `temperature=0.0` (決定的選択)

#### 4.3 Replay Buffer (`buffer.py`)

**概要**: 学習データの循環管理

**機能**:
- 固定サイズバッファ (deque)
- ランダムサンプリング
- 古いデータを自動破棄

**設定**:
- `replay_buffer_size`: 100,000 サンプル

---

### 5. 評価システム (`src/eval/`)

#### 5.1 Players (`players.py`)

| クラス | 説明 |
|--------|------|
| `RandomPlayer` | ランダムに合法手を選択 |
| `GreedyPlayer` | 最も多く石を取れる手を選択 |
| `MCTSPlayer` | 学習済みモデル + MCTS |
| `EdaxPlayer` | 外部エンジン (スタブ実装) |
| `HumanPlayer` | CLI対戦用 |

#### 5.2 Arena (`arena.py`)

**概要**: プレイヤー間の対戦管理・評価

**機能**:
- 指定回数の対戦を実行
- 先手・後手を交互に
- 勝率・平均スコア・平均手数を計算

---

### 6. GUI (`src/gui/`)

**概要**: Tkinterベースのオセロ対戦アプリ

**機能**:
- 人間 vs AI 対戦
- 待った機能 (履歴から復元)
- ヒント表示
- AIシミュレーション回数調整
- モデル読込ダイアログ

**主要クラス**:

| クラス | ファイル | 説明 |
|--------|----------|------|
| `OthelloApp` | `app.py` | メインアプリケーション |
| `OthelloBoardUI` | `board_ui.py` | 盤面描画 Canvas |
| `InfoPanel` | `board_ui.py` | 石数・手番表示 |

---

### 7. CLI (`main.py`)

**コマンド**:

```bash
# 学習
uv run python main.py train --config configs/default_8x8.yaml

# 評価
uv run python main.py eval --checkpoint data/models/final_model.pt \
    --games 20 --simulations 50 --verbose --save-results

# 対戦 (未実装)
uv run python main.py play --checkpoint data/models/final_model.pt
```

---

## 設定ファイル (`configs/default_8x8.yaml`)

```yaml
game:
  size: 8

model:
  num_blocks: 10
  num_filters: 128
  board_size: 8

training:
  batch_size: 256
  lr: 0.001
  num_iterations: 1000
  self_play_episodes_per_iter: 100
  train_epochs_per_iter: 10
  replay_buffer_size: 100000

mcts:
  num_simulations: 25
  num_simulations_eval: 50
  c_puct: 1.0
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25

self_play:
  temperature_threshold: 15

paths:
  checkpoint_dir: "data/models"
  log_dir: "data/logs"

system:
  device: "auto"
  seed: 42
  use_mixed_precision: true
```

---

## データフロー

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────────┐     ┌───────────────┐     ┌──────────────┐ │
│   │   Self-Play   │────▶│ Replay Buffer │────▶│   Trainer    │ │
│   │    Worker     │     │   (100,000)   │     │   (SGD+AMP)  │ │
│   └───────┬───────┘     └───────────────┘     └──────┬───────┘ │
│           │                                          │          │
│           │                                          │          │
│           ▼                                          ▼          │
│   ┌───────────────┐                         ┌──────────────┐   │
│   │     MCTS      │◀────────────────────────│   ResNet     │   │
│   │  (探索木構築)  │         予測            │ (Policy+Value)│   │
│   └───────────────┘                         └──────────────┘   │
│           │                                          ▲          │
│           ▼                                          │          │
│   ┌───────────────┐                                  │          │
│   │   Bitboard    │                                  │          │
│   │  (ゲーム状態)  │──────────────────────────────────┘          │
│   └───────────────┘       get_tensor_input()                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ファイル一覧と行数

| ファイル | 行数 | 説明 |
|----------|------|------|
| `src/cython/bitboard.pyx` | 394 | ビットボード実装 |
| `src/model/net.py` | 266 | ResNetモデル |
| `src/mcts/mcts.py` | 280 | MCTS探索 |
| `src/mcts/node.py` | 191 | MCTSノード |
| `src/train/trainer.py` | 334 | 学習ループ |
| `src/train/self_play.py` | 213 | 自己対戦 |
| `src/train/buffer.py` | 178 | リプレイバッファ |
| `src/eval/players.py` | 303 | プレイヤークラス |
| `src/eval/arena.py` | ~150 | 対戦評価 |
| `src/gui/app.py` | 367 | GUIアプリ |
| `src/gui/board_ui.py` | ~200 | 盤面描画 |
| `main.py` | 334 | CLIエントリポイント |

---

## テスト

```bash
# 全テスト実行
uv run python -m pytest

# カバレッジ付き
uv run python -m pytest --cov=src
```

**テスト数**: 64個 (全パス)

---

## パフォーマンス最適化

### RTX 4050 (6GB VRAM) 向け

1. **混合精度学習 (AMP)**: `torch.amp.autocast('cuda')` + `GradScaler`
2. **バッチサイズ**: 256-512 (VRAM監視で調整)
3. **モデルサイズ**: 10ブロック × 128フィルタ (約300万パラメータ)

### Cython最適化

- `boundscheck=False`
- `wraparound=False`
- `cdivision=True`
- uint64ビット演算で高速な盤面操作

---

## 実装ステータス

| コンポーネント | 状態 |
|----------------|------|
| Bitboard | ✅ 完了 |
| ResNet | ✅ 完了 |
| MCTS | ✅ 完了 |
| Training Loop | ✅ 完了 |
| Replay Buffer | ✅ 完了 |
| Self-Play | ✅ 完了 |
| CLI (train) | ✅ 完了 |
| CLI (eval) | ✅ 完了 |
| CLI (play) | ⏳ 未実装 |
| GUI | ✅ 完了 |
| Edax統合 | ⏳ スタブのみ |
| ONNX変換 | ⏳ 未実装 |

---

## 使用方法

### ビルド

```bash
uv run python setup.py build_ext --inplace
```

### 学習

```bash
uv run python main.py train --config configs/default_8x8.yaml
```

### 評価

```bash
uv run python main.py eval --checkpoint data/models/final_model.pt --games 100
```

### GUI

```bash
uv run python run_gui.py --model data/models/final_model.pt
```

### TensorBoard

```bash
tensorboard --logdir data/logs
```

---

## 参考文献

- [Silver et al., 2017] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
- [Silver et al., 2016] Mastering the game of Go with deep neural networks and tree search
