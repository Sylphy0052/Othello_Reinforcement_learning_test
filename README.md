# Othello AI - AlphaZero

AlphaZero方式による強化学習オセロAI。自己対戦のみで学習し、人間の専門知識に依存しない強力なAIを構築する。

## 概要

このプロジェクトは、DeepMindのAlphaZeroアルゴリズムを採用し、以下の特徴を持つオセロAIを開発します：

- **自己学習**: 人間の棋譜データを一切使用せず、自己対戦のみで強化
- **高速シミュレーション**: Cythonによるビットボード実装で数千局面/秒を実現
- **デュアルヘッドNN**: Policy（着手確率）とValue（勝率予測）を同時出力するResNet
- **MCTS探索**: ニューラルネットワークの評価値を用いたモンテカルロ木探索

## 技術スタック

| レイヤー | 技術 | 用途 |
|---------|------|------|
| 言語 | Python 3.10+, Cython | 本体 / 高速化 |
| 学習 | PyTorch | GPU活用、ResNet学習 |
| 推論 | ONNX Runtime | GUIアプリ用軽量推論 |
| GUI | Tkinter / PySide6 | 対戦インターフェース |

## プロジェクト構成

```
othello_alphazero/
├── configs/              # 設定ファイル (YAML)
├── data/                 # 学習データ・ログ保存先
│   ├── logs/             # TensorBoardログ
│   └── models/           # 学習済みモデル (.pt, .onnx)
├── src/
│   ├── cython/           # 高速化モジュール (ビットボード)
│   ├── model/            # PyTorch ResNet
│   ├── mcts/             # モンテカルロ木探索
│   ├── train/            # 自己対戦・学習ループ
│   └── gui/              # GUIアプリケーション
├── tests/                # 単体テスト
├── docs/                 # 設計ドキュメント
├── main.py               # CLIエントリポイント
└── run_gui.py            # GUIエントリポイント
```

## セットアップ

### 必要要件

- Python 3.10以上
- NVIDIA GPU (CUDA対応、学習時)
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
```

## 使用方法

### 学習の実行

```bash
# デフォルト設定で学習開始
uv run python main.py train --config configs/default_8x8.yaml

# 学習の再開
uv run python main.py train --resume checkpoints/best_model.pt
```

### Edax対戦（ベンチマーク）

```bash
uv run python main.py pit --agent_model checkpoints/best_model.pt --edax_level 10
```

### GUIアプリ起動

```bash
uv run python run_gui.py
```

## 開発

### テスト実行

```bash
uv run python -m pytest
```

### コード品質チェック

```bash
# リント
uv run flake8 .

# 型チェック
uv run mypy .
```

## アーキテクチャ

### コアコンポーネント

1. **ビットボード (Cython)**
   - `uint64`×2で盤面を表現
   - ビット演算による高速な合法手生成・石反転
   - 目標: 5,000-10,000 games/sec

2. **ResNet**
   - 10ブロック × 128フィルタ
   - 入力: `(3, 8, 8)` - 自軍石、敵軍石、合法手マスク
   - 出力: Policy(65次元) + Value(スカラー)

3. **MCTS**
   - ロールアウトなし（NN評価値を使用）
   - Action Maskingで合法手のみ選択
   - PUCT アルゴリズム採用

4. **学習ループ**
   - Self-Play → Replay Buffer → Train (AMP) → Evaluate
   - 混合精度学習で VRAM 節約

### ハードウェア最適化

RTX 4050 (6GB VRAM) 向けに以下の最適化を実施：

- 混合精度学習 (AMP) 必須
- バッチサイズ 256-512 を調整可能
- VRAM 監視によるモデル規模調整

## 開発フェーズ

| フェーズ | 内容 | 状態 |
|---------|------|------|
| Phase 1 | Cython環境構築、ビットボード実装 | 計画中 |
| Phase 2 | NN実装、MCTS、学習ループ構築 | 計画中 |
| Phase 3 | 本格学習、Edaxベンチマーク | 計画中 |
| Phase 4 | ONNX化、GUIアプリ実装 | 計画中 |

## ドキュメント

詳細な設計情報は `docs/` ディレクトリを参照：

- [要件定義書](docs/要件定義書.md) - システム要件・技術スタック
- [外部設計書](docs/外部設計書.md) - CLI/GUI設計・データフロー
- [内部設計書](docs/内部設計書.md) - クラス設計・モジュール詳細
- [実装計画書](docs/実装計画書.md) - フェーズ別タスク・リスク管理

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照
