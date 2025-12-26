# Benchmark Skill

現在のAIモデルの「強さ（勝率）」を測定するスキル。

## 🎯 Objective

<goal>
現在の実装コードと対戦相手（ランダムAIや過去バージョン）を戦わせ、定量的な勝率データを取得する。
ユーザーから「強くなった？」「弱くなった？」と問われた際に、客観的な数値で回答するために使用する。
</goal>

## 📋 Usage Rules

<rules>
1. **Interpretation:**
   - 勝率が 55% を超えていれば「改善 (Improved)」、45% 未満なら「劣化 (Degraded)」と判断する。
   - 試合数が少ないと誤差が出るため、最低でも 100試合 (`--games 100`) は実行すること。
</rules>

## 💻 Command

uv run python ${SKILL_DIR}/run.py --games 100
