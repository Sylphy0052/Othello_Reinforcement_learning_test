# Board Visualizer Skill

生の盤面データ（配列）を、人間とAIが理解できるASCIIアートに変換するスキル。

## 🎯 Objective

<goal>
ログに出力された数値データ（`[0, 0, 1, -1...]`など）を視覚化し、バグの原因や盤面の状況を把握する。
「なぜそこに置いたのか？」を分析する際に使用する。
</goal>

## 📋 Usage Rules

<rules>
1. **Input:** 1次元リスト(64要素)、2次元リスト(8x8)、またはそれらの文字列表現を受け付ける。
2. **Timing:** エラーログに盤面配列が含まれている時は、ユーザーに聞く前に自律的に実行して確認すること。
</rules>

## 💻 Command

uv run python ${SKILL_DIR}/render.py --data "{DATA}"
