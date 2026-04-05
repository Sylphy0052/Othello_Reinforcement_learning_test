# API & インターフェース定義

すべてのモジュールの入出力定義を集約したリファレンス。

## 目次

1. [Bitboard API](#bitboard-api)
2. [Model API](#model-api)
3. [MCTS API](#mcts-api)
4. [Training API](#training-api)
5. [Evaluation API](#evaluation-api)
6. [Web API](#web-api)
7. [データスキーマ](#データスキーマ)

---

## Bitboard API

`src/cython/bitboard.pyx`

### OthelloBitboard

#### コンストラクタ

```python
OthelloBitboard()
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `OthelloBitboard` インスタンス |
| 副作用 | なし |

#### reset()

```python
def reset() -> None
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `None` |
| 副作用 | 盤面を初期状態（中央4石）にリセット |

#### get_legal_moves()

```python
def get_legal_moves() -> list[int]
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `list[int]` - 合法手の位置リスト (0-63) |
| 副作用 | なし |

#### make_move()

```python
def make_move(pos: int) -> bool
```

| 項目 | 内容 |
|------|------|
| 入力 | `pos: int` - 着手位置 (0-63) または 64 (パス) |
| 出力 | `bool` - 成功なら `True` |
| 副作用 | 盤面更新、手番交代、履歴追加 |
| エラー | 不正な手の場合 `False` |

#### undo_move()

```python
def undo_move() -> bool
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `bool` - 成功なら `True` |
| 副作用 | 盤面を1手前に戻す |
| エラー | 履歴がない場合 `False` |

#### copy()

```python
def copy() -> OthelloBitboard
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `OthelloBitboard` - 盤面のコピー |
| 副作用 | なし |

#### is_terminal()

```python
def is_terminal() -> bool
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `bool` - 終局なら `True` |
| 副作用 | なし |

#### get_winner()

```python
def get_winner() -> int
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `int` - 1 (黒勝), -1 (白勝), 0 (引分) |
| 前提 | 終局状態であること |

#### to_tensor()

```python
def to_tensor() -> np.ndarray
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `np.ndarray` - shape `(3, 8, 8)`, dtype `float32` |
| 副作用 | なし |

**テンソル構成:**
- Channel 0: 自分の石
- Channel 1: 相手の石
- Channel 2: 合法手マスク

#### count_stones()

```python
def count_stones() -> tuple[int, int]
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `tuple[int, int]` - (黒石数, 白石数) |
| 副作用 | なし |

---

## Model API

`src/model/net.py`

### OthelloResNet

#### コンストラクタ

```python
OthelloResNet(
    num_blocks: int = 10,
    num_filters: int = 128,
    board_size: int = 8
)
```

| 項目 | 内容 |
|------|------|
| 入力 | `num_blocks`, `num_filters`, `board_size` |
| 出力 | `OthelloResNet` インスタンス |
| 副作用 | なし |

#### forward()

```python
def forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

| 項目 | 内容 |
|------|------|
| 入力 | `x: Tensor` - shape `(Batch, 3, 8, 8)` |
| 出力 | `tuple[Tensor, Tensor]` - (policy_logits, value) |
| policy | shape `(Batch, 65)` - Log確率 |
| value | shape `(Batch, 1)` - 範囲 [-1, 1] |

#### predict()

```python
def predict(board_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

| 項目 | 内容 |
|------|------|
| 入力 | `board_tensor: Tensor` - shape `(3,8,8)` or `(B,3,8,8)` |
| 出力 | `tuple[Tensor, Tensor]` - (policy_probs, value) |
| モード | eval(), no_grad() |
| policy | 確率に変換済み (softmax適用) |

#### get_param_count()

```python
def get_param_count() -> dict
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `dict` - `{"total": int, "trainable": int}` |

### create_model()

```python
def create_model(config: dict) -> OthelloResNet
```

| 項目 | 内容 |
|------|------|
| 入力 | `config: dict` - `{"num_blocks", "num_filters", "board_size"}` |
| 出力 | `OthelloResNet` インスタンス |

---

## MCTS API

`src/mcts/node.py`, `src/mcts/mcts.py`

### MCTSNode

#### コンストラクタ

```python
MCTSNode(prior: float = 0.0, parent: MCTSNode = None)
```

| 項目 | 内容 |
|------|------|
| 入力 | `prior`, `parent` |
| 出力 | `MCTSNode` インスタンス |

#### get_value()

```python
def get_value() -> float
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `float` - Q(s,a) = value_sum / visit_count |

#### expand()

```python
def expand(policy_probs: np.ndarray, legal_actions: list[int]) -> None
```

| 項目 | 内容 |
|------|------|
| 入力 | `policy_probs: ndarray(65,)`, `legal_actions: list[int]` |
| 出力 | `None` |
| 副作用 | 子ノード作成、is_expanded = True |

#### select_child()

```python
def select_child(c_puct: float) -> tuple[int, MCTSNode]
```

| 項目 | 内容 |
|------|------|
| 入力 | `c_puct: float` - 探索定数 |
| 出力 | `tuple[int, MCTSNode]` - (action, child_node) |
| 選択基準 | PUCT値最大 |

#### update()

```python
def update(value: float) -> None
```

| 項目 | 内容 |
|------|------|
| 入力 | `value: float` - 手番視点の価値 [-1, 1] |
| 出力 | `None` |
| 副作用 | visit_count++, value_sum += value |

#### get_policy_distribution()

```python
def get_policy_distribution(temperature: float) -> np.ndarray
```

| 項目 | 内容 |
|------|------|
| 入力 | `temperature: float` |
| 出力 | `np.ndarray(65,)` - 方策分布 |

### MCTS

#### コンストラクタ

```python
MCTS(
    model,
    device,
    c_puct: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25
)
```

| 項目 | 内容 |
|------|------|
| 入力 | `model`, `device`, `c_puct`, `dirichlet_alpha`, `dirichlet_epsilon` |
| 出力 | `MCTS` インスタンス |

#### search()

```python
def search(
    board: OthelloBitboard,
    num_simulations: int,
    temperature: float,
    add_dirichlet_noise: bool
) -> tuple[np.ndarray, float]
```

| 項目 | 内容 |
|------|------|
| 入力 | `board`, `num_simulations`, `temperature`, `add_dirichlet_noise` |
| 出力 | `tuple[ndarray(65,), float]` - (policy_distribution, root_value) |

#### get_best_action()

```python
def get_best_action(board: OthelloBitboard, num_simulations: int) -> int
```

| 項目 | 内容 |
|------|------|
| 入力 | `board`, `num_simulations` |
| 出力 | `int` - 最良行動 (0-64) |

#### get_action_evaluations()

```python
def get_action_evaluations(board: OthelloBitboard, num_simulations: int) -> np.ndarray
```

| 項目 | 内容 |
|------|------|
| 入力 | `board`, `num_simulations` |
| 出力 | `np.ndarray(65,)` - 評価値 (0-100) |

---

## Training API

`src/train/buffer.py`, `src/train/self_play.py`, `src/train/trainer.py`

### ReplayBuffer

#### コンストラクタ

```python
ReplayBuffer(max_size: int = 100000)
```

#### add()

```python
def add(data: list[tuple]) -> None
```

| 項目 | 内容 |
|------|------|
| 入力 | `data: list[(state, policy, value)]` |
| state | `ndarray(3, 8, 8)` |
| policy | `ndarray(65,)` |
| value | `float` |
| 副作用 | 古いデータは自動削除 |

#### sample()

```python
def sample(batch_size: int) -> tuple
```

| 項目 | 内容 |
|------|------|
| 入力 | `batch_size: int` |
| 出力 | `tuple[ndarray, ndarray, ndarray]` |
| states | shape `(B, 3, 8, 8)` |
| policies | shape `(B, 65)` |
| values | shape `(B, 1)` |

#### is_ready()

```python
def is_ready(min_size: int) -> bool
```

| 項目 | 内容 |
|------|------|
| 入力 | `min_size: int` |
| 出力 | `bool` |

#### get_statistics()

```python
def get_statistics() -> dict
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `dict` - `{size, max_size, fill_rate, value_mean, value_std}` |

### SelfPlayWorker

#### コンストラクタ

```python
SelfPlayWorker(
    board_class,
    mcts,
    num_simulations: int = 25,
    temperature_threshold: int = 15
)
```

#### execute_episode()

```python
def execute_episode(add_dirichlet_noise: bool) -> list[tuple]
```

| 項目 | 内容 |
|------|------|
| 入力 | `add_dirichlet_noise: bool` |
| 出力 | `list[(state, policy, value)]` |

#### execute_episodes()

```python
def execute_episodes(num_episodes: int, add_dirichlet_noise: bool) -> list[tuple]
```

| 項目 | 内容 |
|------|------|
| 入力 | `num_episodes`, `add_dirichlet_noise` |
| 出力 | `list[(state, policy, value)]` |

### AlphaZeroTrainer

#### コンストラクタ

```python
AlphaZeroTrainer(
    model: nn.Module,
    device: torch.device,
    replay_buffer,
    self_play_worker,
    config: dict,
    checkpoint_dir: str = "data/models",
    log_dir: str = "data/logs"
)
```

**config パラメータ:**

| キー | 型 | 説明 |
|------|-----|------|
| `lr` | `float` | 学習率 |
| `momentum` | `float` | モメンタム |
| `weight_decay` | `float` | 重み減衰 |

#### train()

```python
def train(
    num_iterations: int,
    self_play_episodes_per_iter: int,
    train_epochs_per_iter: int,
    batch_size: int,
    checkpoint_interval: int
) -> None
```

| 項目 | 内容 |
|------|------|
| 入力 | 各種学習パラメータ |
| 出力 | `None` |
| 副作用 | モデル更新、チェックポイント保存、ログ出力 |

#### save_checkpoint()

```python
def save_checkpoint(filename: str) -> None
```

| 項目 | 内容 |
|------|------|
| 入力 | `filename: str` |
| 出力 | `None` |
| 保存内容 | model_state_dict, optimizer_state_dict, scheduler_state_dict, global_step, epoch, config |

#### load_checkpoint()

```python
def load_checkpoint(checkpoint_path: str) -> None
```

| 項目 | 内容 |
|------|------|
| 入力 | `checkpoint_path: str` |
| 出力 | `None` |
| 副作用 | モデル・オプティマイザの状態復元 |

---

## Evaluation API

`src/eval/arena.py`, `src/eval/players.py`

### Player (抽象基底クラス)

#### get_action()

```python
def get_action(board: OthelloBitboard) -> int
```

| 項目 | 内容 |
|------|------|
| 入力 | `board: OthelloBitboard` |
| 出力 | `int` - 着手位置 (0-64) |

#### reset()

```python
def reset() -> None
```

| 項目 | 内容 |
|------|------|
| 入力 | なし |
| 出力 | `None` |

### MCTSPlayer

#### from_checkpoint() (クラスメソッド)

```python
@classmethod
def from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_simulations: int = 50
) -> MCTSPlayer
```

| 項目 | 内容 |
|------|------|
| 入力 | `checkpoint_path`, `device`, `num_simulations` |
| 出力 | `MCTSPlayer` インスタンス |

### Arena

#### コンストラクタ

```python
Arena(verbose: bool = True)
```

#### play_game()

```python
def play_game(
    player1: Player,
    player2: Player,
    starting_player: int
) -> MatchResult
```

| 項目 | 内容 |
|------|------|
| 入力 | `player1`, `player2`, `starting_player` (1 or -1) |
| 出力 | `MatchResult` |

#### play_matches()

```python
def play_matches(
    player1: Player,
    player2: Player,
    num_games: int,
    alternate_colors: bool
) -> list[MatchResult]
```

| 項目 | 内容 |
|------|------|
| 入力 | `player1`, `player2`, `num_games`, `alternate_colors` |
| 出力 | `list[MatchResult]` |

### MatchResult

```python
@dataclass
class MatchResult:
    player1_name: str
    player2_name: str
    winner: int           # 1, -1, 0
    player1_score: int
    player2_score: int
    num_moves: int
    duration: float
```

### evaluate_player()

```python
def evaluate_player(
    player: Player,
    opponent: Player,
    num_games: int = 10,
    verbose: bool = True
) -> dict
```

| 項目 | 内容 |
|------|------|
| 入力 | `player`, `opponent`, `num_games`, `verbose` |
| 出力 | `dict` - `{win_rate, avg_score, avg_moves, results}` |

---

## Web API

`src/web/api.py`

### ゲーム操作エンドポイント

#### POST /api/game/new

新規ゲーム開始。

**リクエスト:**
```json
{
  "mode": "human_vs_ai"  // optional, default: "human_vs_ai"
}
```

**レスポンス:**
```json
{
  "success": true,
  "game_state": { ... }
}
```

#### GET /api/game/state

現在の状態取得。

**レスポンス:**
```json
{
  "board": [[0, 0, ...], ...],  // 8x8
  "legal_moves": [19, 26, ...],
  "current_player": 1,
  "black_count": 2,
  "white_count": 2,
  "is_terminal": false,
  "winner": null,
  "is_ai_thinking": false,
  "move_count": 0,
  "message": null,
  "model_loaded": true
}
```

#### POST /api/game/move

着手実行。

**リクエスト:**
```json
{
  "position": 19  // 0-64
}
```

**レスポンス:**
```json
{
  "success": true,
  "game_state": { ... },
  "error": null
}
```

#### POST /api/game/undo

一手戻す。

**レスポンス:**
```json
{
  "success": true,
  "game_state": { ... },
  "error": null
}
```

#### POST /api/game/ai-move

AI着手（非同期）。

**レスポンス:**
```json
{
  "success": true,
  "is_thinking": true,
  "game_state": { ... }
}
```

#### GET /api/game/ai-status

AI思考状態確認。

**レスポンス:**
```json
{
  "is_thinking": false,
  "game_state": { ... }
}
```

#### GET /api/game/hint

ヒント取得。

**レスポンス:**
```json
{
  "evaluations": {
    "19": 65,
    "26": 45,
    ...
  },
  "success": true,
  "error": null
}
```

### AI設定エンドポイント

#### POST /api/ai/load-model

モデル読込。

**リクエスト:**
```json
{
  "model_path": "data/models/final_model.pt"
}
```

**レスポンス:**
```json
{
  "success": true,
  "error": null
}
```

#### PUT /api/ai/simulations

シミュレーション数設定。

**リクエスト:**
```json
{
  "count": 50  // 10-500
}
```

**レスポンス:**
```json
{
  "success": true,
  "count": 50
}
```

#### GET /api/ai/simulations

現在のシミュレーション数取得。

**レスポンス:**
```json
{
  "count": 50
}
```

#### GET /api/ai/models

利用可能モデル一覧。

**レスポンス:**
```json
{
  "models": [
    "data/models/checkpoint_iter_100.pt",
    "data/models/final_model.pt"
  ]
}
```

---

## データスキーマ

`src/web/schemas.py`

### リクエストスキーマ

```python
class NewGameRequest(BaseModel):
    mode: str = "human_vs_ai"

class MoveRequest(BaseModel):
    position: int  # 0-64

class LoadModelRequest(BaseModel):
    model_path: str

class SimulationsRequest(BaseModel):
    count: int  # 10-500
```

### レスポンススキーマ

```python
class GameState(BaseModel):
    board: List[List[int]]      # 8x8盤面 (0=空, 1=黒, -1=白)
    legal_moves: List[int]      # 合法手位置リスト
    current_player: int         # 1=黒, -1=白
    black_count: int
    white_count: int
    is_terminal: bool
    winner: Optional[int]       # 1, -1, 0, None
    is_ai_thinking: bool
    move_count: int
    message: Optional[str]
    model_loaded: bool

class MoveResponse(BaseModel):
    success: bool
    game_state: GameState
    error: Optional[str]

class HintResponse(BaseModel):
    evaluations: Dict[int, int]  # {position: 0-100}
    success: bool
    error: Optional[str]

class AIStatusResponse(BaseModel):
    is_thinking: bool
    game_state: GameState

class ModelsListResponse(BaseModel):
    models: List[str]
```

---

## エラーハンドリング

### 共通エラーレスポンス形式

```json
{
  "success": false,
  "error": "エラーメッセージ"
}
```

### HTTPステータスコード

| コード | 意味 | 使用場面 |
|--------|------|----------|
| 200 | OK | 正常処理 |
| 400 | Bad Request | 不正なパラメータ |
| 404 | Not Found | モデルファイルが存在しない |
| 409 | Conflict | AI思考中に操作しようとした |
| 500 | Internal Server Error | 予期しないエラー |
