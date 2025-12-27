# 残りタスク

## ✅ 完了済み

### Phase 1: コア・ロジックと高速化
- [x] Cython Bitboard実装
- [x] 動作テストとベンチマーク（5,000-10,000 games/sec達成）

### Phase 2: 学習システムの構築
- [x] ResNet実装（10ブロック、128フィルタ、300万パラメータ）
- [x] MCTS実装（PUCT、Action Masking、ディリクレノイズ）
- [x] 学習ループ実装（Self-Play、Replay Buffer、AMP対応）
- [x] CLI実装（trainコマンド）
- [x] 全テスト（64個）パス

### Phase 3: 評価システム
- [x] プレイヤークラス実装（Random、Greedy、MCTS、Human）
- [x] Arena（対戦管理）実装
- [x] ベンチマークスクリプト実装

### Phase 4: GUI
- [x] Tkinter盤面描画実装
- [x] 対戦UI実装
- [x] 待った、ヒント機能実装
- [x] run_gui.pyエントリポイント実装

---

## 🔄 実行中

### 本格学習（バックグラウンド）
- [実行中] default_8x8.yaml設定で学習実行中
  - プロセスID: 571761
  - イテレーション: 1000回
  - チェックポイント: `data/models/checkpoint_iter_*.pt`
  - 推定完了時間: 数時間〜数日

---

## 📋 残りタスク

### 1. 未実装のCLIコマンド

#### 1.1 evalコマンド（評価モード） ✅ 完了
**ファイル**: `main.py` の `eval_command()`

**実装内容**:
```bash
# 使用例
uv run python main.py eval --checkpoint data/models/final_model.pt --games 20 --simulations 50 --save-results
```

**機能**:
- チェックポイント読み込み
- ベンチマーク自動実行
- 結果サマリー表示
- JSON結果保存（オプション）

**状態**: ✅ 実装完了 (2025-12-27)

#### 1.2 playコマンド（対戦モード）
**ファイル**: `main.py` の `play_command()`

**実装内容**:
```bash
# 使用例
uv run python main.py play --checkpoint data/models/final_model.pt
```

**機能**:
- CLI対戦インターフェース
- 人間 vs AI
- 盤面ASCII表示

**優先度**: 中

---

### 2. Edax統合（オプション）

#### 2.1 Edaxインストール
```bash
# Edaxのインストール（必要に応じて）
git clone https://github.com/abulmo/edax-reversi.git
cd edax-reversi
make
```

#### 2.2 EdaxPlayerの完全実装
**ファイル**: `src/eval/players.py` の `EdaxPlayer`

**現状**: スタブ実装のみ（フォールバックでRandomPlayer使用）

**実装内容**:
- サブプロセスでEdax起動
- GGS/GTP プロトコル通信
- レベル別対戦機能

**優先度**: 低（Edaxなしでも評価可能）

---

### 3. モデル軽量化（ONNX変換）

#### 3.1 ONNX エクスポート
**ファイル**: `src/model/utils.py`（新規作成）

**実装内容**:
```python
def export_to_onnx(model_path, output_path):
    """PyTorchモデルをONNX形式にエクスポート"""
    # モデル読み込み
    # torch.onnx.export()で変換
    # 動作確認
```

**用途**:
- デプロイ用の軽量化
- ONNXRuntime での高速推論
- 他プラットフォームへの移植

**優先度**: 低

---

### 4. 配布用パッケージング

#### 4.1 PyInstaller での実行ファイル化
```bash
# 実行ファイル化
pyinstaller --onefile run_gui.py
```

**課題**:
- Cython拡張モジュールの同梱
- PyTorchの大きいサイズ
- WSL環境では実行ファイル化が難しい

**優先度**: 低（開発者向けには不要）

---

### 5. ドキュメント・品質向上

#### 5.1 ユーザーマニュアル
**ファイル**: `docs/user_manual.md`（新規作成）

**内容**:
- インストール手順
- 学習の開始方法
- GUIの使い方
- ベンチマーク方法

**優先度**: 中

#### 5.2 API ドキュメント
- Sphinx等でAPIドキュメント生成
- docstringの充実

**優先度**: 低

#### 5.3 コード品質チェック
```bash
# 現在実装済み
uv run flake8 .
uv run mypy .
```

**残作業**:
- 型ヒントの完全化
- flake8/mypy エラーの修正

**優先度**: 低

---

### 6. 高度な評価・分析

#### 6.1 学習曲線の可視化
**ファイル**: `visualize_training.py`（新規作成）

**機能**:
- TensorBoardログの読み込み
- 損失グラフ
- 勝率推移グラフ

**優先度**: 中

#### 6.2 棋譜解析
**機能**:
- 対戦ログのSGF形式保存
- 棋譜の再生・分析
- 評価値の可視化

**優先度**: 低

---

### 7. 拡張機能（オプション）

#### 7.1 6x6盤面サポート
- 設定ファイルで盤面サイズ変更可能に
- 小規模実験用

**優先度**: 低

#### 7.2 Web UI
- Flaskベースのブラウザ対戦
- オンライン対戦機能

**優先度**: 低

#### 7.3 AI vs AI観戦モード
- 2つのモデルを戦わせる
- リアルタイム表示

**優先度**: 低

---

## 🎯 推奨される次のアクション

### 短期（今すぐできること）

1. ✅ **evalコマンド実装** - 完了 (2025-12-27)
   - ベンチマーク機能を統合
   - 結果サマリー表示
   - JSON保存機能

2. **playコマンド実装** (1時間)
   - CLI対戦インターフェース
   - 盤面表示

3. **学習完了を待つ** (数時間〜数日)
   - チェックポイント確認
   - 中間評価実施

### 中期（学習完了後）

4. ✅ **本格ベンチマーク機能強化** - 完了 (2025-12-27)
   ```bash
   # 強化された統計情報（W-L-D、スコア範囲など）を表示
   uv run python benchmark_ai.py \
     --checkpoint data/models/final_model.pt \
     --games 100 \
     --simulations 100
   ```

5. ✅ **GUIでの対戦テスト用スクリプト** - 完了 (2025-12-27)
   ```bash
   # GUIデモ起動
   uv run python demo_gui.py --checkpoint data/models/final_model.pt

   # 自動テスト（ヘッドレス環境でも動作）
   uv run python test_gui_automated.py

   # 通常のGUI起動
   uv run python run_gui.py --model data/models/final_model.pt
   ```

6. **学習曲線の可視化**
   ```bash
   tensorboard --logdir data/logs
   ```

### 長期（オプション）

7. **Edax統合** - 外部エンジンとの対戦
8. **ONNX変換** - デプロイ用軽量化
9. **ユーザーマニュアル作成** - 使いやすさ向上

---

## 📊 現在の状態

| 項目 | 状態 |
|------|------|
| **コア実装** | ✅ 100% 完了 |
| **テスト** | ✅ 64/64 パス |
| **学習** | 🔄 実行中 |
| **GUI** | ✅ 実装完了（表示確認待ち） |
| **評価システム** | ✅ 実装完了 |
| **CLI (train)** | ✅ 完了 |
| **CLI (eval)** | ✅ 完了 (2025-12-27) |
| **CLI (play)** | ⏳ 未実装 |
| **ベンチマーク強化** | ✅ 完了 (2025-12-27) |
| **GUIテストスクリプト** | ✅ 完了 (2025-12-27) |
| **Edax統合** | ⏳ 準備のみ |
| **ONNX** | ⏳ 未実装 |

---

## 💡 優先順位まとめ

### 高優先度
- なし（主要機能はすべて完成）

### 中優先度
1. ✅ evalコマンド実装 - 完了
2. playコマンド実装
3. 学習完了待ち & 本格評価

### 低優先度
4. Edax統合
5. ONNX変換
6. PyInstaller パッケージング
7. ドキュメント充実

**結論**: 主要な実装はすべて完了しており、学習の完了を待つのが次のステップです！
