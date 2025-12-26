# Ship / Pull Request

作業内容をコミットし、PRを作成する納品コマンド。

## 🎯 Objective

<goal>
実装済みのコードをリポジトリにプッシュし、GitHub Pull Request を作成する。
</goal>

## ⚙️ Prerequisites (重要)

<condition>
自動実行されるHook (`hooks/pre_ship.sh`) により、全てのテストが通過していること。
テストが失敗した場合、このコマンドは中断される。
</condition>

## 📋 Instructions

1. **Branching:** 内容を表す適切なブランチ名 (`feat/name` or `fix/name`) を作成し移動する。
2. **Commit:** `[Prefix]: Summary` 形式でコミットする。
3. **Push:** リモートリポジトリへプッシュする。
4. **PR Creation:** GitHub MCP を使用してPRを作成する。

## 📤 Output Format

- 作成されたPRのURL。
