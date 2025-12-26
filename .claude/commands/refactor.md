# Code Refactoring

挙動を変えずにコードの内部構造を改善するコマンド。

## 🎯 Objective

<goal>
Tech Lead の視点で、可読性・保守性を向上させるリファクタリングを行うこと。
</goal>

## 📋 Instructions

1. **Analyze:** 重複コード（DRY違反）や、長すぎる関数（God Function）を特定する。
2. **Plan:** 安全な分割・統合案を作成する。
3. **Apply:** ロジックを変更せずに構造のみを変更する。

## 🚫 Constraints

<rules>
- **No Logic Change:** 入力に対する出力結果を変えてはならない（バグ修正はここで行わない）。
- **Type Hints:** リファクタリングのついでに型ヒントを完備させること。
</rules>

## 📤 Output Format

- リファクタリング前後の比較説明。
- 修正後のコードブロック。
