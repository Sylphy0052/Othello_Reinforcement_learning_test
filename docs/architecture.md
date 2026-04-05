# アーキテクチャ設計書

## 概要

本システムはAlphaZero方式の強化学習を用いたオセロAIです。主要な設計原則は以下の通りです：

1. **自己対戦学習**: 人間の棋譜データに依存せず、自己対戦のみで学習
2. **GPU効率化**: RTX 4050 (6GB VRAM) 向けに混合精度学習 (AMP) を採用
3. **高速盤面処理**: Cythonによるビットボード実装で高速なゲーム進行
4. **モジュラー設計**: UI/ゲームロジック/MLパイプラインを明確に分離

## システムアーキテクチャ

```mermaid
flowchart LR
    subgraph Presentation["プレゼンテーション層"]
        TK["Tkinter GUI"]
        FA["FastAPI"]
    end

    subgraph Application["アプリケーション層"]
        GM["GameManager"]
        SP["SelfPlayWorker"]
        AR["Arena"]
    end

    subgraph Domain["ドメイン層"]
        BB["OthelloBitboard"]
        MCTS["MCTS"]
        NN["OthelloResNet"]
    end

    subgraph Infrastructure["インフラ層"]
        BUF["ReplayBuffer"]
        CHK["Checkpoint"]
        TB["TensorBoard"]
    end

    TK --> GM
    FA --> GM
    GM --> BB
    GM --> MCTS

    SP --> MCTS
    SP --> BB
    SP --> BUF

    AR --> BB

    MCTS --> NN
    MCTS --> BB
```

## コンポーネント詳細

### 1. ゲームエンジン層

```mermaid
classDiagram
    class OthelloBitboard {
        +uint64 self_board
        +uint64 opp_board
        +int move_count
        +bool passed
        +reset()
        +get_legal_moves() list
        +make_move(pos) bool
        +is_terminal() bool
        +get_winner() int
        +get_tensor_input() ndarray
        +copy() OthelloBitboard
    }

    note for OthelloBitboard "Cython実装\nuint64×2でビットボード表現\n高速な合法手生成・石反転"
```

**設計判断**:
- `uint64`×2 のビットボード表現を採用（メモリ効率、ビット演算による高速化）
- 8方向の反転処理を個別メソッドで実装（デバッグ容易性）

### 2. ニューラルネットワーク層

```mermaid
classDiagram
    class OthelloResNet {
        +int num_blocks
        +int num_filters
        +ConvBlock conv_block
        +ModuleList res_blocks
        +PolicyHead policy_head
        +ValueHead value_head
        +forward(x) tuple
        +predict(tensor) tuple
    }

    class ConvBlock {
        +Conv2d conv
        +BatchNorm2d bn
        +forward(x) Tensor
    }

    class ResBlock {
        +Conv2d conv1, conv2
        +BatchNorm2d bn1, bn2
        +forward(x) Tensor
    }

    class PolicyHead {
        +Conv2d conv
        +Linear fc
        +forward(x) Tensor
    }

    class ValueHead {
        +Conv2d conv
        +Linear fc1, fc2
        +forward(x) Tensor
    }

    OthelloResNet *-- ConvBlock
    OthelloResNet *-- ResBlock
    OthelloResNet *-- PolicyHead
    OthelloResNet *-- ValueHead
```

**デフォルト構成**:
- ResBlock: 10ブロック
- フィルタ数: 128
- 入力: `(batch, 3, 8, 8)` - 自分の石/相手の石/合法手マスク
- 出力: Policy `(batch, 65)` + Value `(batch, 1)`

### 3. MCTS探索層

```mermaid
classDiagram
    class MCTS {
        +model: OthelloResNet
        +device: torch.device
        +float c_puct
        +float dirichlet_alpha
        +float dirichlet_epsilon
        +search(board, num_sims) tuple
        +get_best_action(board, num_sims) int
        +get_action_evaluations(board, num_sims) ndarray
    }

    class MCTSNode {
        +float prior
        +MCTSNode parent
        +int visit_count
        +float value_sum
        +Dict children
        +bool is_expanded
        +expand(policy, legal_actions)
        +select_child(c_puct) tuple
        +update(value)
        +get_policy_distribution(temp) ndarray
    }

    MCTS --> MCTSNode
```

**PUCT式**:
```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
```

- `Q(s,a)`: 平均価値 = W(s,a) / N(s,a)
- `P(s,a)`: ニューラルネットワークの事前確率
- `N(s)`: 親ノードの訪問回数
- `N(s,a)`: 子ノードの訪問回数

### 4. 学習パイプライン層

```mermaid
sequenceDiagram
    participant Trainer
    participant Worker as SelfPlayWorker
    participant Buffer as ReplayBuffer
    participant Model

    loop 学習イテレーション
        Trainer->>Worker: execute_episodes(N)
        Worker->>Worker: MCTS探索 × N games
        Worker-->>Trainer: training_data
        Trainer->>Buffer: add(training_data)

        loop 学習エポック
            Trainer->>Buffer: sample(batch_size)
            Buffer-->>Trainer: states, policies, values
            Trainer->>Model: forward + backward (AMP)
            Model-->>Trainer: loss
        end

        opt チェックポイント
            Trainer->>Trainer: save_checkpoint()
        end
    end
```

**並列Self-Play**:
- `ParallelSelfPlayWorker`: 複数ゲームを同時進行
- `BatchMCTS`: NN評価をバッチ化してGPUスループット向上

### 5. 評価システム層

```mermaid
classDiagram
    class Player {
        <<abstract>>
        +str name
        +get_action(board) int
        +reset()
    }

    class RandomPlayer {
        +get_action(board) int
    }

    class GreedyPlayer {
        +get_action(board) int
    }

    class MCTSPlayer {
        +MCTS mcts
        +int num_simulations
        +get_action(board) int
    }

    class Arena {
        +bool verbose
        +play_game(p1, p2) MatchResult
        +play_matches(p1, p2, n) List
    }

    Player <|-- RandomPlayer
    Player <|-- GreedyPlayer
    Player <|-- MCTSPlayer
    Arena --> Player
```

### 6. UI層

```mermaid
flowchart TB
    subgraph Desktop["デスクトップ (Tkinter)"]
        OthelloApp --> OthelloBoardUI
        OthelloApp --> InfoPanel
        OthelloApp --> MCTS1[MCTS]
    end

    subgraph Web["Web (FastAPI)"]
        API --> GameManager
        GameManager --> MCTS2[MCTS]
        GameManager --> Schemas
    end

    subgraph Shared["共有"]
        BB[OthelloBitboard]
        NN[OthelloResNet]
    end

    MCTS1 --> BB
    MCTS1 --> NN
    MCTS2 --> BB
    MCTS2 --> NN
```

## データモデル

### 盤面表現

```
入力テンソル: (3, 8, 8)
├── Channel 0: 自分の石 (1=石あり, 0=なし)
├── Channel 1: 相手の石 (1=石あり, 0=なし)
└── Channel 2: 合法手マスク (1=合法, 0=不可)
```

### 学習データ

```
(state, policy, value)
├── state: ndarray (3, 8, 8) - 盤面テンソル
├── policy: ndarray (65,) - MCTS訪問回数分布
└── value: float - 最終勝敗 (+1=勝ち, -1=負け, 0=引き分け)
```

## 設計判断とトレードオフ

| 項目 | 選択 | 理由 |
|------|------|------|
| 盤面表現 | ビットボード | メモリ効率、ビット演算による高速化 |
| NN構造 | ResNet | AlphaZeroオリジナル踏襲、残差接続による安定学習 |
| 探索 | MCTS+PUCT | 探索と活用のバランス、NN価値推定との相性 |
| 学習 | SGD+Momentum | AlphaZeroオリジナル踏襲 |
| バッチサイズ | 256-512 | VRAM 6GBの制約下で調整可能 |
| AMP | 有効 | VRAM使用量削減、学習速度向上 |
