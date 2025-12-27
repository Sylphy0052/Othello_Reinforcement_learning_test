# Othello AI - AlphaZero

AlphaZero方式による強化学習オセロAI。自己対戦のみで学習し、人間の専門知識に依存しない強力なAIを構築する。

## 🎯 概要

このプロジェクトは、DeepMindのAlphaZeroアルゴリズムを採用し、以下の特徴を持つオセロAIを実装しています：

- **自己学習**: 人間の棋譜データを一切使用せず、自己対戦のみで強化
- **高速シミュレーション**: Cythonビットボード実装で **10,000+ games/sec** を実現
- **デュアルヘッドNN**: Policy（着手確率）とValue（勝率予測）を同時出力するResNet
- **MCTS探索**: ニューラルネットワークの評価値を用いたモンテカルロ木探索
- **完全なパイプライン**: 学習・評価・GUI対戦まで完全実装

## ✨ 実装状況

- ✅ Cython Bitboard実装（10,000+ games/sec達成）
- ✅ ResNet実装（10ブロック、128フィルタ、300万パラメータ）
- ✅ MCTS実装（PUCT、Action Masking、ディリクレノイズ）
- ✅ 学習ループ実装（Self-Play、Replay Buffer、AMP対応）
- ✅ 評価システム実装（Arena、複数プレイヤー対応）
- ✅ Tkinter GUI実装（対戦UI、待った機能、ヒント機能）
- ✅ CLI実装（train、eval コマンド）
- ✅ **全64テストパス**

## 🔧 技術スタック

| レイヤー | 技術 | 用途 |
|---------|------|------|
| 言語 | Python 3.13, Cython | 本体 / 高速化 |
| 学習 | PyTorch (AMP) | GPU活用、ResNet学習 |
| ゲームロジック | Cython Bitboard | uint64×2による高速盤面処理 |
| GUI | Tkinter | 対戦インターフェース |
| テスト | pytest | 64テストケース |
| パッケージ管理 | uv | 高速な依存関係管理 |

## 📁 プロジェクト構成

```
othello_alphazero/
├── configs/              # 設定ファイル (YAML)
│   ├── default_8x8.yaml  # 本番学習設定
│   └── test.yaml         # テスト用設定
├── data/                 # 学習データ・ログ保存先
│   ├── logs/             # TensorBoardログ
│   ├── models/           # 学習済みモデル (.pt)
│   ├── benchmark/        # ベンチマーク結果 (JSON)
│   └── eval/             # 評価結果 (JSON)
├── src/
│   ├── cython/           # 高速化モジュール
│   │   └── bitboard.pyx  # Cythonビットボード
│   ├── model/            # PyTorch ResNet
│   │   └── net.py        # OthelloResNet
│   ├── mcts/             # モンテカルロ木探索
│   │   ├── node.py       # MCTSNode
│   │   └── mcts.py       # MCTS探索
│   ├── train/            # 学習システム
│   │   ├── self_play.py  # Self-Play Worker
│   │   ├── buffer.py     # Replay Buffer
│   │   └── trainer.py    # AlphaZero Trainer
│   ├── eval/             # 評価システム
│   │   ├── players.py    # 各種プレイヤー
│   │   └── arena.py      # 対戦管理
│   └── gui/              # GUIアプリケーション
│       ├── board_ui.py   # ボードUI
│       └── app.py        # メインアプリ
├── tests/                # 単体テスト (64テスト)
│   ├── test_bitboard.py
│   ├── test_model.py
│   ├── test_mcts.py
│   ├── test_train.py
│   └── test_eval.py
├── main.py               # CLIエントリポイント
├── run_gui.py            # GUIエントリポイント
├── demo_gui.py           # GUIデモスクリプト
├── benchmark.py          # Bitboardベンチマーク
├── benchmark_ai.py       # AIベンチマーク
└── test_gui_automated.py # GUI自動テスト
```

## ⚙️ セットアップ

### 必要要件

- Python 3.10以上（推奨: 3.13）
- NVIDIA GPU (CUDA対応、学習時推奨)
  - テスト済み: RTX 4050 (6GB VRAM)
  - CPU版でも動作可能（学習は低速）
- uv (パッケージマネージャ)

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/your-username/Othello_Reinforcement_learning_test.git
cd Othello_Reinforcement_learning_test

# 依存関係のインストール
uv sync

# Cython拡張のビルド
uv run python setup.py build_ext --inplace

# テスト実行（動作確認）
uv run python -m pytest
# 全64テストがパスすることを確認
```

## 🚀 使用方法

### 1. 学習の実行

```bash
# デフォルト設定で学習開始（1000イテレーション）
uv run python main.py train --config configs/default_8x8.yaml

# テスト学習（10イテレーション、高速）
uv run python main.py train --config configs/test.yaml
```

**学習設定**:
- イテレーション: 1000回
- Self-Play: 100エピソード/イテレーション
- バッチサイズ: 256
- MCTS: 50シミュレーション/手
- 混合精度学習（AMP）有効

### 2. モデル評価

```bash
# モデルをRandom/Greedyと対戦させて評価
uv run python main.py eval \
  --checkpoint data/models/checkpoint_iter_100.pt \
  --games 100 \
  --simulations 100 \
  --save-results

# 簡易評価
uv run python main.py eval --checkpoint data/models/test/final_model.pt
```

**出力例**:
```
Random  : 75.0% win rate, avg score: 42.3
Greedy  : 65.0% win rate, avg score: 38.1

Results saved to: data/eval/eval_20251227_100418.json
```

### 3. ベンチマーク

```bash
# AIモデルの詳細ベンチマーク
uv run python benchmark_ai.py \
  --checkpoint data/models/checkpoint_iter_100.pt \
  --games 100 \
  --simulations 100

# ビットボードの性能測定
uv run python benchmark.py
# 期待結果: 10,000+ games/sec
```

### 4. GUI対戦

```bash
# GUIで人間 vs AI対戦
uv run python run_gui.py --model data/models/checkpoint_iter_100.pt

# GUIデモ（詳細説明付き）
uv run python demo_gui.py --checkpoint data/models/test/final_model.pt

# GUI自動テスト（ヘッドレス環境対応）
uv run python test_gui_automated.py
```

**GUI操作**:
- クリックで石を配置
- `Undo` ボタン: 1手戻る
- `Hint` ボタン: AIの推奨手を表示
- `New Game` ボタン: 新規ゲーム開始

### 5. TensorBoard（学習監視）

```bash
# 学習曲線の可視化
tensorboard --logdir data/logs
# ブラウザで http://localhost:6006 を開く
```

## 🛠️ 開発

### テスト実行

```bash
# 全テスト実行
uv run python -m pytest

# 詳細出力
uv run python -m pytest -v

# カバレッジ測定
uv run python -m pytest --cov=src
```

**テスト状況**: ✅ 64/64 パス

### コード品質チェック

```bash
# リント
uv run flake8 .

# 型チェック
uv run mypy .
```

## 🏗️ アーキテクチャ

### コアコンポーネント

#### 1. ビットボード (Cython)
- `uint64`×2で盤面を表現（黒石・白石）
- ビット演算による高速な合法手生成・石反転
- **実績**: 10,000+ games/sec

**主要メソッド**:
```python
board.reset()                    # 初期化
board.get_legal_moves()          # 合法手取得
board.make_move(action)          # 着手
board.get_tensor_input()         # NN入力テンソル取得
board.copy()                     # 盤面コピー
```

#### 2. ResNet (PyTorch)
- **構成**: 10 ResBlocks × 128フィルタ
- **パラメータ数**: 約300万
- **入力**: `(3, 8, 8)` - 自軍石、敵軍石、合法手マスク
- **出力**:
  - Policy Head: 65次元（64マス + パス）
  - Value Head: スカラー値（勝率予測）

#### 3. MCTS (モンテカルロ木探索)
- **アルゴリズム**: PUCT（Polynomial Upper Confidence Trees）
- **特徴**:
  - ロールアウトなし（NN評価値を使用）
  - Action Maskingで合法手のみ選択
  - ディリクレノイズによる探索多様化
- **性能**: 50-100シミュレーション/手

#### 4. 学習ループ
```
Self-Play (100 games)
  ↓
Replay Buffer (max 50,000)
  ↓
Training (5 epochs, batch 256, AMP)
  ↓
Checkpoint保存
  ↓
繰り返し (1000イテレーション)
```

#### 5. 評価システム
- **プレイヤー**: Random, Greedy, MCTS, Human
- **Arena**: 自動対戦管理、統計計算
- **ベンチマーク**: JSON形式での結果保存

### ハードウェア最適化

**RTX 4050 (6GB VRAM) 対応**:
- ✅ 混合精度学習 (AMP)
- ✅ バッチサイズ 256（VRAM 使用量: ~4GB）
- ✅ Gradient Accumulation対応
- ✅ CUDA最適化（cuDNN）

## 📈 開発フェーズ

| フェーズ | 内容 | 状態 |
|---------|------|------|
| **Phase 1** | Cython環境構築、ビットボード実装 | ✅ **完了** (10,000+ games/sec達成) |
| **Phase 2** | NN実装、MCTS、学習ループ構築 | ✅ **完了** (全64テストパス) |
| **Phase 3** | 評価システム、GUI実装 | ✅ **完了** (CLI/GUI完全実装) |
| **Phase 4** | 本格学習、最適化 | 🔄 **学習中** (1000イテレーション実行中) |
| **Phase 5** | ONNX化、デプロイ最適化 | ⏳ **計画中** (オプション) |

## 📊 性能指標

| 項目 | 実績値 |
|-----|--------|
| Bitboard速度 | **10,000+ games/sec** |
| モデルサイズ | 約300万パラメータ |
| 学習速度 | 100 games/イテレーション、~5分/イテレーション (RTX 4050) |
| VRAM使用量 | 約4GB (混合精度学習時) |
| テストカバレッジ | **64/64 テストパス** |

## 📚 ドキュメント

### 設計ドキュメント
詳細な設計情報は `docs/` ディレクトリを参照：

- [要件定義書](docs/要件定義書.md) - システム要件・技術スタック
- [外部設計書](docs/外部設計書.md) - CLI/GUI設計・データフロー
- [内部設計書](docs/内部設計書.md) - クラス設計・モジュール詳細
- [実装計画書](docs/実装計画書.md) - フェーズ別タスク・リスク管理

### その他のドキュメント
- [REMAINING_TASKS.md](REMAINING_TASKS.md) - 残りタスク一覧
- [CLAUDE.md](CLAUDE.md) - AI開発アシスタント向けガイド

## 🎓 参考文献

- [AlphaGo Zero論文](https://www.nature.com/articles/nature24270) - DeepMind, 2017
- [AlphaZero論文](https://arxiv.org/abs/1712.01815) - DeepMind, 2017
- [Othelloビットボード技術](https://www.chessprogramming.org/Othello#Bitboards)

## 📝 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照

## 👥 貢献

Issue報告・Pull Requestを歓迎します。

---

**Last Updated**: 2025-12-27
**Status**: ✅ 全主要機能実装完了、学習実行中
