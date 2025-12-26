# System Architecture Design

機能実装のための構造設計を行うコマンド。

## 🎯 Objective

<goal>
Software Architect の視点で、拡張性と保守性の高いファイル構成とクラス設計を行うこと。
</goal>

## 📋 Instructions

1. **Dependencies Check:** 既存のモジュール間の依存関係を確認し、循環参照を回避する。
2. **Structure Design:** 機能の責務（Responsibility）に基づいて配置場所を決める。
3. **Interface Design:** クラスのメソッドシグネチャ（引数と戻り値）を定義する。

## 🚫 Constraints

<rules>
- **SOLID原則:** 特に「単一責任の原則」を遵守すること。
- **Othello Specifics:** 盤面データ(`Board`)とAIロジック(`Strategy`)を明確に分離すること。
</rules>

## 📤 Output Format

1. **ディレクトリ構造:** `tree` コマンド形式。
2. **クラス図:** Mermaid `classDiagram` 形式。
3. **作成ファイル一覧:** ファイルパスのリスト。
