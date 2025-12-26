# Full Feature Development Flow

要件定義から実装、検証までを一気通貫で行う統合フロー。

## 🎯 Objective

<goal>
ユーザーの抽象的なアイデアを、動作するコードとして納品可能な状態にすること。
</goal>

## 📋 Workflow

ユーザーに対し、以下のステップを順次案内し、承認を得ながら進めてください。

<step1>
**Planning** (`/plan` 相当)
- 要望を聞き出し、TODOリストを作成する。
- ユーザーに「この計画で良いですか？」と確認する。
</step1>

<step2>
**Architecture** (`/architect` 相当)
- ファイル構成やクラス設計を提示する。
</step2>

<step3>
**Coding** (`/code` 相当)
- 計画に基づき実装を行う。
</step3>

<step4>
**Review & Test** (`/review`, `/test` 相当)
- 実装コードを自己レビューし、テストケースを作成する。
</step4>

## 🚫 Constraints

- 各ステップの区切りで必ずユーザーの承認（Confirmation）を待つこと。
- 勝手に次のステップへ進まないこと。
