# Code Inspector Skill

Pythonコードの構造品質（複雑度）を静的解析するスキル。

## 🎯 Objective

<goal>
指定されたファイルの「循環的複雑度 (Cyclomatic Complexity)」と「行数」を測定する。
AIがコードを修正する前に、リファクタリングが必要な危険な関数（God Function）を特定するために使用する。
</goal>

## 📋 Usage Rules

<rules>
1. **Threshold:**
   - 複雑度が **10** を超える関数には警告を出すこと。
   - 複雑度が **20** を超える場合は、必ずリファクタリング（分割）を提案すること。
2. **Timing:**
   - `/refactor` コマンドが呼ばれた時。
   - 大規模なロジック修正を行う前の現状分析として。
</rules>

## 💻 Command

uv run python ${SKILL_DIR}/analyze.py --file "{FILE_PATH}"
