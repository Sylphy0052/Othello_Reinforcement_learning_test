# Training Log Analyst Skill

学習ログを解析し、モデルの成長推移や異常を診断するスキル。

## 🎯 Objective

<goal>
学習プロセスが正常に進んでいるか（収束しているか）を統計的に判断する。
Loss（損失）の推移や Epsilon（探索率）の減衰を確認し、ハイパーパラメータの調整提案につなげる。
</goal>

## 📋 Usage Rules

<rules>
1. **Diagnosis:**
   - Lossが減っていない場合 → 「学習率が高すぎる」または「バグ」の可能性を示唆する。
   - Epsilonが高いままの場合 → 「探索過多」の可能性を示唆する。
</rules>

## 💻 Command

uv run python ${SKILL_DIR}/parser.py --log-dir ./logs
