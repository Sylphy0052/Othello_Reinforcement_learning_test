# Code Review

変更内容を監査し、品質を担保するコマンド。

## 🎯 Objective

<goal>
Tech Lead として、現在の `git diff` または指定ファイルをレビューし、改善点を指摘すること。
</goal>

## 🔍 Checklist

<items>
1. **Complexity:** 計算量が `O(N^3)` を超える非効率なループはないか？
2. **Safety:** 配列外参照や型エラーの可能性はないか？
3. **Style:** 変数名は具体的か？ Docstringはあるか？
4. **Logic:** オセロのルール（打てる場所判定など）に違反していないか？
</items>

## 📤 Output Format

1. **判定:** `Pass` / `Request Changes`
2. **指摘事項:** ファイル名と行番号付きのリスト。
3. **修正案:** 改善コードの提示。
