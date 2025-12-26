# Test Generation

品質保証のためのテストコードを作成するコマンド。

## 🎯 Objective

<goal>
QA Engineer の視点で、エッジケースを網羅した `pytest` コードを作成すること。
</goal>

## 📋 Instructions

1. **Case Extraction:** 正常系、異常系、境界値（Boundary Value）を洗い出す。
2. **Fixture Setup:** テストに必要な前提データ（特定の盤面状態など）を準備する。
3. **Coding:** `pytest` を用いて実装する。

## 🚫 Constraints

<rules>
- **Parametrization:** `@pytest.mark.parametrize` を使用してテストデータを分離すること。
- **Mocking:** 外部ライブラリやランダム要素には `unittest.mock` を使用すること。
</rules>

## 📤 Output Format

- テストファイル名 (`tests/test_xxx.py`)
- 即時実行可能なテストコードブロック。
