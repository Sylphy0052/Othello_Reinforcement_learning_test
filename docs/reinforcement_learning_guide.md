# AlphaZero強化学習ガイド

本プロジェクトの実装を元に、AlphaZero方式の強化学習アルゴリズムを解説します。

## 目次

1. [AlphaZeroとは](#1-alphazeroとは)
2. [強化学習の基本要素](#2-強化学習の基本要素)
3. [ニューラルネットワーク](#3-ニューラルネットワーク)
4. [モンテカルロ木探索 (MCTS)](#4-モンテカルロ木探索-mcts)
5. [学習ループ](#5-学習ループ)
6. [実装上の工夫](#6-実装上の工夫)
7. [数式まとめ](#7-数式まとめ)

---

## 1. AlphaZeroとは

AlphaZeroは、DeepMind社が2017年に発表した強化学習アルゴリズムです。
**人間の知識を一切使わず、自己対戦のみで超人的な強さに到達**することが特徴です。

### 従来手法との違い

| 手法 | 特徴 |
|------|------|
| **従来のゲームAI** | 人間が設計した評価関数を使用 |
| **AlphaGo** | 人間の棋譜で事前学習 + 強化学習 |
| **AlphaZero** | **完全にゼロから自己対戦のみで学習** |

### 3つの核心技術

```
┌─────────────────────────────────────────────────────────────┐
│                      AlphaZero                              │
├─────────────────────────────────────────────────────────────┤
│  1. 深層ニューラルネットワーク（方策 + 価値の同時予測）      │
│  2. モンテカルロ木探索（MCTS）による先読み                  │
│  3. 自己対戦による強化学習                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 強化学習の基本要素

### 2.1 状態 (State)

盤面の状態を3チャネルの8x8テンソルで表現します。

```python
# 状態表現: (3, 8, 8) のテンソル
# 実装: src/cython/bitboard.pyx の get_tensor_input()

Channel 0: 自分の石の配置（1: 石あり, 0: 石なし）
Channel 1: 相手の石の配置（1: 石あり, 0: 石なし）
Channel 2: 手番情報（全マス同じ値、黒番=1, 白番=0）
```

**例: 初期盤面**
```
Channel 0 (自分=黒):     Channel 1 (相手=白):     Channel 2 (手番):
0 0 0 0 0 0 0 0         0 0 0 0 0 0 0 0         1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0         0 0 0 0 0 0 0 0         1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0         0 0 0 0 0 0 0 0         1 1 1 1 1 1 1 1
0 0 0 1 0 0 0 0         0 0 0 0 1 0 0 0         1 1 1 1 1 1 1 1
0 0 0 0 1 0 0 0         0 0 0 1 0 0 0 0         1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0         0 0 0 0 0 0 0 0         1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0         0 0 0 0 0 0 0 0         1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0         0 0 0 0 0 0 0 0         1 1 1 1 1 1 1 1
```

### 2.2 アクション (Action)

65種類の離散アクションで表現します。

```python
# アクション空間: 65次元
# 実装: src/model/net.py の PolicyHead

Action 0-63: 盤面のマス (row * 8 + col)
Action 64:   パス（合法手がない場合）

例:
  0  1  2  3  4  5  6  7
  8  9 10 11 12 13 14 15
 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 38 39
 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55
 56 57 58 59 60 61 62 63
```

### 2.3 報酬 (Reward)

**AlphaZeroの特徴: ゲーム終了時のみ報酬を与える**

```python
# 報酬設定
# 実装: src/train/self_play.py

勝利:     +1
敗北:     -1
引き分け:  0

# 重要: 途中の手には報酬を与えない（スパース報酬）
# ゲーム終了後、その結果を全ての手に遡って割り当てる
```

**報酬の割り当て方法:**
```python
# 実装: src/train/self_play.py の execute_episode()

# ゲーム終了後
winner = board.get_winner()  # 1: 黒勝ち, -1: 白勝ち, 0: 引き分け

# 各手番の視点で報酬を計算
for step in game_history:
    # 自分の色 × 勝者の色 = 自分視点の報酬
    value = float(winner * step.player)
    # 黒番で黒勝ち: 1 * 1 = +1
    # 黒番で白勝ち: 1 * -1 = -1
    # 白番で黒勝ち: -1 * 1 = -1
```

---

## 3. ニューラルネットワーク

### 3.1 全体構造（Dual-Head ResNet）

```
┌─────────────────────────────────────────────────────────────┐
│                    入力: (Batch, 3, 8, 8)                   │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ConvBlock (3ch → 128ch)                        │
│              Conv(3x3) → BatchNorm → ReLU                   │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ResBlock × 10 (残差ブロック)                   │
│     ┌──────────────────────────────────────────┐            │
│     │  Conv(3x3) → BN → ReLU → Conv(3x3) → BN  │            │
│     │              ↓                           │            │
│     │         Add Skip Connection              │            │
│     │              ↓                           │            │
│     │            ReLU                          │            │
│     └──────────────────────────────────────────┘            │
└────────────────────────┬────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│    Policy Head      │     │     Value Head      │
│                     │     │                     │
│ Conv(1x1, 2ch)      │     │ Conv(1x1, 1ch)      │
│ → BN → ReLU         │     │ → BN → ReLU         │
│ → Flatten           │     │ → Flatten           │
│ → FC(128 → 65)      │     │ → FC(64 → 256)      │
│ → LogSoftmax        │     │ → ReLU              │
│                     │     │ → FC(256 → 1)       │
│                     │     │ → Tanh              │
├─────────────────────┤     ├─────────────────────┤
│ 出力: (Batch, 65)   │     │ 出力: (Batch, 1)    │
│ 各手の確率分布      │     │ 勝率予測 [-1, 1]    │
└─────────────────────┘     └─────────────────────┘
```

### 3.2 Policy Head（方策ヘッド）

**役割:** 各アクションの確率分布を出力

```python
# 実装: src/model/net.py の PolicyHead

class PolicyHead(nn.Module):
    def forward(self, x):
        # x: (Batch, 128, 8, 8)
        x = self.conv(x)      # → (Batch, 2, 8, 8)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # → (Batch, 128)
        x = self.fc(x)        # → (Batch, 65)
        x = F.log_softmax(x, dim=1)  # 確率分布
        return x
```

### 3.3 Value Head（価値ヘッド）

**役割:** 現在の局面の勝率を [-1, 1] で予測

```python
# 実装: src/model/net.py の ValueHead

class ValueHead(nn.Module):
    def forward(self, x):
        # x: (Batch, 128, 8, 8)
        x = self.conv(x)      # → (Batch, 1, 8, 8)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # → (Batch, 64)
        x = self.fc1(x)       # → (Batch, 256)
        x = F.relu(x)
        x = self.fc2(x)       # → (Batch, 1)
        x = torch.tanh(x)     # → [-1, 1]
        return x
```

### 3.4 なぜDual-Headなのか？

```
従来: 方策ネットワークと価値ネットワークを別々に学習
      → 計算コストが2倍、特徴量の共有ができない

AlphaZero: 1つのネットワークで両方を同時に出力
      → 効率的な学習、特徴表現の共有
      → 「どの手が良いか」と「勝てそうか」を同時に学習
```

---

## 4. モンテカルロ木探索 (MCTS)

### 4.1 MCTSの概要

MCTSは、ゲーム木を効率的に探索するアルゴリズムです。

```
┌─────────────────────────────────────────────────────────────┐
│                    MCTSの4ステップ                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Select      ルートから葉ノードまでPUCT値で選択          │
│       ↓                                                     │
│  2. Expand      葉ノードでNNを評価し、子ノードを展開        │
│       ↓                                                     │
│  3. Evaluate    NNのValue出力を取得（従来のrolloutの代わり）│
│       ↓                                                     │
│  4. Backup      評価値を親ノードに伝播                      │
│                                                             │
│  これを num_simulations 回繰り返す                          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 PUCT式（選択アルゴリズム）

**UCB1の改良版で、NNの事前確率を活用**

```
PUCT(s,a) = Q(s,a) + U(s,a)

Q(s,a) = W(s,a) / N(s,a)    ... 平均価値（exploitation）

U(s,a) = c_puct × P(s,a) × √N(s) / (1 + N(s,a))   ... 探索ボーナス（exploration）

記号:
- s: 現在の状態
- a: アクション
- N(s,a): ノードの訪問回数
- W(s,a): 累積価値
- P(s,a): NNの出力した事前確率
- c_puct: 探索と活用のバランス定数（本実装では1.0〜1.5）
```

```python
# 実装: src/mcts/node.py の select_child()

def select_child(self, c_puct: float) -> tuple:
    best_score = -float('inf')

    for action, child in self.children.items():
        # Q値（平均価値）= 活用項
        q_value = child.get_value()  # W / N

        # U値（探索ボーナス）= 探索項
        u_value = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)

        # PUCT値 = Q + U
        puct_score = q_value + u_value

        if puct_score > best_score:
            best_score = puct_score
            best_action = action
            best_child = child

    return best_action, best_child
```

### 4.3 PUCT式の直感的理解

```
┌─────────────────────────────────────────────────────────────┐
│  Q(s,a): 「この手を選んだ過去の結果、平均してどうだったか」 │
│          → よく勝てた手は Q が高い（活用）                  │
│                                                             │
│  U(s,a): 「この手はまだあまり試してないから試すべき」       │
│          → 訪問回数 N が少ないと U が高い（探索）           │
│          → NNが「良さそう」と言った手（P が高い）も優先     │
│                                                             │
│  c_puct: 探索と活用のバランスを調整                         │
│          → 大きい: 探索重視（新しい手を試す）               │
│          → 小さい: 活用重視（良い手を深掘り）               │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Action Masking（合法手マスク）

**非合法手を確率0にする重要な処理**

```python
# 実装: src/mcts/node.py の expand()

def expand(self, policy_probs: np.ndarray, legal_actions: list):
    # NNの出力から合法手のみ残す
    masked_probs = np.zeros_like(policy_probs)
    masked_probs[legal_actions] = policy_probs[legal_actions]

    # 正規化（確率の合計を1にする）
    prob_sum = masked_probs.sum()
    if prob_sum > 0:
        masked_probs /= prob_sum
    else:
        # NNが全て0を出力した場合、均等分布
        masked_probs[legal_actions] = 1.0 / len(legal_actions)

    # 合法手のみ子ノードを作成
    for action in legal_actions:
        self.children[action] = MCTSNode(prior=masked_probs[action])
```

### 4.5 ディリクレノイズ

**学習時の探索を促進するノイズ**

```python
# 実装: src/mcts/mcts.py の _add_dirichlet_noise()

# ルートノードの事前確率にノイズを追加
noise = np.random.dirichlet([alpha] * num_actions)

# 元の確率とノイズを混合
child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

# 本実装のパラメータ:
# alpha = 0.3 (オセロ用、チェスは0.3、囲碁は0.03)
# epsilon = 0.25 (25%のノイズを混合)
```

**なぜディリクレノイズが必要か？**
```
NNの出力だけだと、同じ局面では常に同じ手を選んでしまう
→ 多様な棋譜が生成されない
→ 学習が局所解に陥る

ディリクレノイズを加えることで:
- NNが「良くない」と判断した手も時々選ばれる
- 多様な局面を経験できる
- 新しい戦術を発見できる可能性
```

### 4.6 温度パラメータ

**行動選択の確率的/決定的を制御**

```python
# 実装: src/mcts/node.py の get_policy_distribution()

def get_policy_distribution(self, temperature: float) -> np.ndarray:
    visit_counts = self.get_visit_counts()  # 各手の訪問回数

    if temperature == 0:
        # 決定的: 最も訪問回数が多い手を選択
        best_action = argmax(counts)
        policy[best_action] = 1.0
    else:
        # 確率的: 訪問回数^(1/T) に比例した確率
        counts = counts ** (1.0 / temperature)
        counts /= counts.sum()
        for action, prob in zip(actions, counts):
            policy[action] = prob

    return policy
```

```
温度パラメータの効果:
- T = 1.0: 訪問回数に比例した確率（探索的）
- T → 0:   最大訪問回数の手に集中（決定的）
- T → ∞:   均等分布

本実装の戦略:
- 序盤（〜15手目）: T = 1.0（多様な局面を学習）
- 終盤:            T = 0（最善手を選択）
```

---

## 5. 学習ループ

### 5.1 全体フロー

```
┌─────────────────────────────────────────────────────────────┐
│                    AlphaZero 学習ループ                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  for iteration in range(num_iterations):                    │
│                                                             │
│    ┌─────────────────────────────────────────────────────┐  │
│    │  1. Self-Play (自己対戦)                            │  │
│    │     - 現在のNNを使ってMCTSで対戦                    │  │
│    │     - (state, policy, value) のデータを生成         │  │
│    └─────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│    ┌─────────────────────────────────────────────────────┐  │
│    │  2. Replay Buffer に格納                            │  │
│    │     - 過去のデータも保持（経験再生）                │  │
│    └─────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│    ┌─────────────────────────────────────────────────────┐  │
│    │  3. Training (学習)                                 │  │
│    │     - Replay Bufferからランダムサンプリング         │  │
│    │     - Policy Loss + Value Loss を最小化             │  │
│    └─────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│    ┌─────────────────────────────────────────────────────┐  │
│    │  4. Checkpoint 保存                                 │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Self-Play（自己対戦）

```python
# 実装: src/train/self_play.py の execute_episode()

def execute_episode(self):
    board = OthelloBitboard()
    board.reset()
    game_history = []

    while not board.is_terminal():
        # 1. 現在の状態を保存
        state = board.get_tensor_input()

        # 2. MCTSで方策を計算
        policy, _ = self.mcts.search(
            board,
            num_simulations=100,  # 100回シミュレーション
            temperature=1.0 if move_count < 15 else 0.0,
            add_dirichlet_noise=True,
        )

        # 3. 記録
        game_history.append((state, policy, current_player))

        # 4. 行動選択・実行
        action = np.random.choice(65, p=policy)
        board.make_move(action)

    # 5. 勝敗に基づいてvalueを割り当て
    winner = board.get_winner()
    training_data = []
    for state, policy, player in game_history:
        value = float(winner * player)  # 勝ったら+1, 負けたら-1
        training_data.append((state, policy, value))

    return training_data
```

### 5.3 損失関数

```python
# 実装: src/train/trainer.py

# 総損失 = 方策損失 + 価値損失
total_loss = policy_loss + value_loss
```

**方策損失（クロスエントロピー）:**
```python
# MCTSの訪問回数分布を正解として、NNの出力を近づける
def _policy_loss(self, policy_logits, target_policies):
    # KLダイバージェンス: -Σ target * log(pred)
    return -torch.mean(torch.sum(target_policies * policy_logits, dim=1))
```

**価値損失（MSE）:**
```python
# ゲームの勝敗を正解として、NNの出力を近づける
def _value_loss(self, value_pred, target_values):
    return F.mse_loss(value_pred, target_values)
```

### 5.4 Replay Buffer（経験再生）

```python
# 実装: src/train/buffer.py

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, training_data):
        # 新しいデータを追加
        self.buffer.extend(training_data)

    def sample(self, batch_size):
        # ランダムにサンプリング
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        return batch
```

**なぜReplay Bufferが必要か？**
```
1. サンプル効率: 同じデータを複数回学習に使える
2. 相関の除去: 連続した局面は類似 → ランダムサンプリングで相関を切る
3. 安定性: 古い経験も学習に使うことで、急激な方策変化を防ぐ
```

---

## 6. 実装上の工夫

### 6.1 混合精度学習 (AMP)

```python
# 実装: src/train/trainer.py

# VRAM使用量を削減し、学習を高速化
self.scaler = torch.amp.GradScaler('cuda')

with torch.amp.autocast('cuda'):
    policy_logits, value_pred = self.model(states)
    loss = policy_loss + value_loss

self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

### 6.2 ビットボード

```
盤面を64ビット整数×2で表現（Cython実装）
- 高速な合法手生成（ビット演算）
- メモリ効率が良い
- 目標: 5,000-10,000 games/sec
```

### 6.3 残差接続（Skip Connection）

```python
# ResBlockで入力を出力に足し合わせる
out = conv1(x) → bn1 → relu → conv2 → bn2
out = out + x  # Skip Connection（勾配消失を防ぐ）
out = relu(out)
```

### 6.4 BatchNormalization

```
- 学習の安定化
- 内部共変量シフトの軽減
- より高い学習率が使える
```

---

## 7. 数式まとめ

### PUCT式
```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))

Q(s,a) = W(s,a) / N(s,a)
```

### 損失関数
```
L = L_policy + L_value

L_policy = -Σ π_MCTS(a|s) × log π_θ(a|s)    (クロスエントロピー)

L_value = (z - v_θ(s))²                      (MSE)

π_MCTS: MCTSの訪問回数に基づく方策
π_θ:    NNの方策出力
z:      ゲームの最終結果 (+1, -1, 0)
v_θ:    NNの価値出力
```

### 温度付きSoftmax
```
π(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)

τ → 0: argmax（決定的）
τ = 1: 訪問回数に比例
τ → ∞: 均等分布
```

### ディリクレノイズ
```
P'(a) = (1 - ε) × P(a) + ε × η_a

η ~ Dir(α, α, ..., α)

オセロ: α = 0.3, ε = 0.25
```

---

## 参考文献

1. Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
2. Silver, D., et al. (2017). "Mastering the game of Go without human knowledge"
3. AlphaZero論文解説: https://arxiv.org/abs/1712.01815

---

## 本プロジェクトのファイル構成

```
src/
├── cython/
│   └── bitboard.pyx      # 状態表現、合法手生成
├── model/
│   └── net.py            # Dual-Head ResNet
├── mcts/
│   ├── node.py           # MCTSノード、PUCT計算
│   └── mcts.py           # MCTS探索アルゴリズム
└── train/
    ├── self_play.py      # 自己対戦データ生成
    ├── buffer.py         # Replay Buffer
    └── trainer.py        # 学習ループ
```
