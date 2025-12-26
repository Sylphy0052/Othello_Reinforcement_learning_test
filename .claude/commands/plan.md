# Implementation Planning

ユーザーの要望を分析し、堅実な実装計画を立てるコマンド。

## 🎯 Objective

<goal>
Product Manager (PdM) の視点で要望を分析し、実装に必要なタスクを洗い出し、`docs/todo.md` を更新すること。
</goal>

## 📋 Instructions

以下のステップで実行してください：

1. **Context Analysis:**
   - ユーザーの要望と、現在のプロジェクト構造 (`tree` や `ls`) を照らし合わせる。
   - 既存機能との競合や、影響範囲を特定する。

2. **Requirement Definition:**
   - 必要な機能を「Must (必須)」と「Nice-to-have (歓迎)」に分類する。

3. **Plan Creation:**
   - 実装手順をステップバイステップで書き出す。
   - 各ステップで変更すべきファイル名を特定する。

## 📤 Output Format

<format>
Markdown形式で以下を出力してください：

### 1. 概要

(機能の要約)

### 2. 影響範囲

- 変更: `src/othello.py`
- 新規: `src/strategies.py`

### 3. 実装ステップ (ToDo)

- [ ] ステップ1: ...
- [ ] ステップ2: ...

</format>
