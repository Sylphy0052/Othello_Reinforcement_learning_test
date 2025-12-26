# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero方式のオセロAI。自己対戦のみで学習し、人間の専門知識に依存しない強力なAIを構築する。

## Commands

```bash
# Build Cython extensions
uv run python setup.py build_ext --inplace

# Run tests
uv run python -m pytest

# Code quality
uv run flake8 .
uv run mypy .

# Training (once implemented)
uv run python main.py train --config configs/default_8x8.yaml

# GUI (once implemented)
uv run python run_gui.py
```

## Architecture

### Directory Structure (Planned)

```
src/
├── cython/          # 高速化モジュール (Cython bitboard)
│   ├── bitboard.pyx # uint64×2 ビットボード実装
│   └── setup.py     # Cython build script
├── model/           # PyTorch ResNet (Policy + Value dual head)
├── mcts/            # Monte Carlo Tree Search with PUCT
├── train/           # Self-play, replay buffer, trainer
└── gui/             # Tkinter/PySide6 interface
```

### Core Components

1. **Bitboard (Cython)**: `uint64`×2 で盤面表現。ビット演算で合法手生成・石反転。目標: 5,000-10,000 games/sec
2. **ResNet**: 10ブロック×128フィルタ。入力 `(3,8,8)`、出力 Policy(65) + Value(scalar)
3. **MCTS**: NNの価値出力で探索。Action Maskingで合法手のみ選択
4. **Training Loop**: Self-Play → Replay Buffer → Train (AMP) → Evaluate

### Hardware Constraints

RTX 4050 (6GB VRAM) 向け最適化:
- 混合精度学習 (AMP) 必須
- バッチサイズ 256-512 を調整可能に
- モデル規模は VRAM 監視しながら調整

## Development Guidelines

### Core Principles

1. **TDD**: 実装前にテストを書く。既存テストを壊さない
2. **Clean Architecture**: UI/ゲームロジック/MLパイプラインを分離
3. **Performance First**: 強化学習の試行回数を稼ぐため計算量に注意

### Code Style

- コメントは日本語
- 複雑なアルゴリズム（MCTS、評価関数）には数式・意図をコメント
- 型ヒント必須 (mypy strict)

## Workflow

- Issue番号が分かっているときは、必ずIssueの内容を確認してから実装
- PRは変更内容を簡潔にまとめたタイトルをつける

## Key Documentation

詳細仕様は `docs/` を参照:
- `要件定義書.md`: システム要件・技術スタック
- `外部設計書.md`: CLI/GUI設計・データフロー
- `内部設計書.md`: クラス設計・モジュール詳細
- `実装計画書.md`: フェーズ別タスク・リスク管理
