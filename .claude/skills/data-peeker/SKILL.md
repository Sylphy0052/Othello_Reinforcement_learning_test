# Data Peeker Skill

巨大なデータファイルの中身を安全に確認するスキル。

## 🎯 Objective

<goal>
JSON, CSV, Pickle などのデータファイルを、中身をすべて展開せずに「構造」や「先頭データ」だけ確認する。
学習データ(`replay_buffer.pkl`)の中身確認や、設定ファイル(`config.json`)の構造把握に使用する。
</goal>

## 📋 Usage Rules

<rules>
1. **Safety First:**
   - ファイルサイズが不明な場合は、決して `cat` や `read()` を使わず、このスキルを使うこと。
2. **Supported Formats:**
   - `.json`, `.csv`, `.pkl` (pickle), `.txt`
</rules>

## 💻 Command

uv run python ${SKILL_DIR}/peek.py --file "{FILE_PATH}" --lines 5
