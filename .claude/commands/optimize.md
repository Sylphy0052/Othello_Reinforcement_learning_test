# Performance Optimization

コードのボトルネックを特定し、高速化するコマンド。

## 🎯 Objective

<goal>
アルゴリズムの計算量を削減し、実行速度を向上させること。
</goal>

## 🧠 Thinking Process

<process>
1. **Identify Hotspots:** ループのネストが深い箇所 (`O(N^3)`以上) を探す。
2. **Select Strategy:**
   - Vectorization (NumPy)
   - Caching (Memoization)
   - Pruning (枝刈り)
3. **Apply:** 挙動を変えずに実装を置換する。
</process>

## 🚫 Constraints

<rules>
- ロジックの結果（出力）が変わってはならない。
- 可読性を著しく損なう「黒魔術的な最適化」は避けること。
</rules>

## 📤 Output Format

1. **最適化戦略:** 何をどう変えるかの説明。
2. **コード差分:** 修正後のコード。
3. **期待効果:** 「計算量が O(N^3) から O(N^2) に削減」などの見積もり。
