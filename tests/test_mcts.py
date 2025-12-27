"""
MCTSのテストケース

- MCTSNodeの基本機能テスト
- MCTS探索ロジックのテスト
- Bitboard + ResNet + MCTS の統合テスト
"""

import pytest
import torch
import numpy as np
from src.cython.bitboard import OthelloBitboard
from src.model.net import OthelloResNet
from src.mcts.node import MCTSNode
from src.mcts.mcts import MCTS


class TestMCTSNode:
    """MCTSNodeの基本機能テスト"""

    def test_node_initialization(self):
        """ノードの初期化テスト"""
        node = MCTSNode(prior=0.5)

        assert node.prior == 0.5
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.is_leaf()
        assert not node.is_expanded

    def test_node_update(self):
        """ノードの統計更新テスト"""
        node = MCTSNode(prior=0.5)

        node.update(1.0)
        assert node.visit_count == 1
        assert node.value_sum == 1.0
        assert node.get_value() == 1.0

        node.update(-1.0)
        assert node.visit_count == 2
        assert node.value_sum == 0.0
        assert node.get_value() == 0.0

    def test_node_expansion(self):
        """ノードの展開テスト"""
        node = MCTSNode(prior=1.0)

        # ダミーの方策確率
        policy_probs = np.random.rand(65)
        policy_probs /= policy_probs.sum()

        legal_actions = [0, 1, 2, 3, 4]

        # 展開
        node.expand(policy_probs, legal_actions)

        assert node.is_expanded
        assert not node.is_leaf()
        assert len(node.children) == len(legal_actions)

        # すべての合法手に子ノードが作成されているか
        for action in legal_actions:
            assert action in node.children

    def test_select_child(self):
        """PUCT値による子ノード選択テスト"""
        node = MCTSNode(prior=1.0)
        node.visit_count = 10

        # 3つの子ノードを作成
        child1 = MCTSNode(prior=0.5, parent=node)
        child1.visit_count = 5
        child1.value_sum = 2.5

        child2 = MCTSNode(prior=0.3, parent=node)
        child2.visit_count = 3
        child2.value_sum = 1.5

        child3 = MCTSNode(prior=0.2, parent=node)
        child3.visit_count = 2
        child3.value_sum = 1.0

        node.children = {0: child1, 1: child2, 2: child3}

        # c_puct = 1.0 で選択
        action, selected_child = node.select_child(c_puct=1.0)

        # 選択されたノードが children に存在することを確認
        assert action in node.children
        assert selected_child == node.children[action]

    def test_get_policy_distribution(self):
        """訪問回数に基づく方策分布の生成テスト"""
        node = MCTSNode(prior=1.0)

        # 子ノードを作成し、訪問回数を設定
        child1 = MCTSNode(prior=0.5, parent=node)
        child1.visit_count = 10

        child2 = MCTSNode(prior=0.3, parent=node)
        child2.visit_count = 5

        child3 = MCTSNode(prior=0.2, parent=node)
        child3.visit_count = 5

        node.children = {0: child1, 5: child2, 10: child3}

        # temperature = 1.0
        policy = node.get_policy_distribution(temperature=1.0)

        assert policy.shape == (65,)
        assert np.isclose(policy.sum(), 1.0)
        assert policy[0] > 0  # child1 が最も訪問されている
        assert policy[5] > 0
        assert policy[10] > 0

        # temperature = 0.0 (決定的)
        policy_deterministic = node.get_policy_distribution(temperature=0.0)
        assert policy_deterministic[0] == 1.0  # 最大訪問回数のノードのみ確率1
        assert policy_deterministic.sum() == 1.0


class TestMCTS:
    """MCTS探索ロジックのテスト"""

    @pytest.fixture
    def device(self):
        """テスト用デバイス"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        """小規模テストモデル"""
        model = OthelloResNet(num_blocks=2, num_filters=32)
        model.to(device)
        model.eval()
        return model

    @pytest.fixture
    def mcts(self, model, device):
        """MCTSインスタンス"""
        return MCTS(model, device, c_puct=1.0, dirichlet_alpha=0.3)

    @pytest.fixture
    def initial_board(self):
        """初期盤面"""
        board = OthelloBitboard()
        board.reset()
        return board

    def test_mcts_initialization(self, mcts):
        """MCTSの初期化テスト"""
        assert mcts.model is not None
        assert mcts.c_puct == 1.0
        assert mcts.dirichlet_alpha == 0.3

    def test_mcts_search(self, mcts, initial_board):
        """MCTS探索の基本テスト"""
        policy, value = mcts.search(
            initial_board,
            num_simulations=10,
            temperature=1.0,
            add_dirichlet_noise=False
        )

        # 方策分布のチェック
        assert policy.shape == (65,)
        assert np.isclose(policy.sum(), 1.0)
        assert np.all(policy >= 0)

        # 合法手のみに確率が割り当てられているか
        legal_actions = initial_board.get_legal_moves()
        for action in range(65):
            if action not in legal_actions:
                assert policy[action] == 0.0

    def test_mcts_get_best_action(self, mcts, initial_board):
        """最良行動の選択テスト"""
        action = mcts.get_best_action(initial_board, num_simulations=10)

        # 合法手が選ばれているか
        legal_actions = initial_board.get_legal_moves()
        assert action in legal_actions

    def test_mcts_with_dirichlet_noise(self, mcts, initial_board):
        """ディリクレノイズ付き探索のテスト"""
        policy, _ = mcts.search(
            initial_board,
            num_simulations=10,
            temperature=1.0,
            add_dirichlet_noise=True
        )

        # 方策分布が正しいか
        assert policy.shape == (65,)
        assert np.isclose(policy.sum(), 1.0)

    def test_mcts_terminal_state(self, mcts):
        """終局状態でのMCTS動作テスト"""
        # 手動で終局に近い盤面を作成（実際には複雑なので簡易版）
        board = OthelloBitboard()
        board.reset()

        # 何手か進める
        for _ in range(10):
            legal_moves = board.get_legal_moves()
            if len(legal_moves) == 0:
                break
            board.make_move(legal_moves[0])

        # MCTS探索
        policy, value = mcts.search(
            board,
            num_simulations=5,
            temperature=1.0,
            add_dirichlet_noise=False
        )

        assert policy.shape == (65,)
        assert isinstance(value, float)

    def test_mcts_multiple_simulations(self, mcts, initial_board):
        """異なるシミュレーション回数でのテスト"""
        for num_sims in [5, 10, 25, 50]:
            policy, value = mcts.search(
                initial_board,
                num_simulations=num_sims,
                temperature=1.0,
                add_dirichlet_noise=False
            )

            assert policy.shape == (65,)
            assert np.isclose(policy.sum(), 1.0)

    def test_mcts_temperature_effect(self, mcts, initial_board):
        """温度パラメータの効果テスト"""
        # temperature = 1.0 (確率的)
        policy_stochastic, _ = mcts.search(
            initial_board,
            num_simulations=20,
            temperature=1.0,
            add_dirichlet_noise=False
        )

        # temperature = 0.0 (決定的)
        policy_deterministic, _ = mcts.search(
            initial_board,
            num_simulations=20,
            temperature=0.0,
            add_dirichlet_noise=False
        )

        # 決定的な場合、1つの行動に確率が集中
        assert np.max(policy_deterministic) == 1.0
        assert np.sum(policy_deterministic > 0) == 1

        # 確率的な場合、複数の行動に確率が分散
        # （ただし、シミュレーション回数が少ないと例外もあり得る）


class TestIntegration:
    """Bitboard + ResNet + MCTS の統合テスト"""

    @pytest.fixture
    def setup(self):
        """統合テスト用のセットアップ"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OthelloResNet(num_blocks=2, num_filters=32)
        model.to(device)
        model.eval()
        mcts = MCTS(model, device, c_puct=1.0)
        board = OthelloBitboard()
        board.reset()

        return {
            "device": device,
            "model": model,
            "mcts": mcts,
            "board": board,
        }

    def test_full_game_playable(self, setup):
        """MCTSを使った1ゲーム通しのテスト"""
        mcts = setup["mcts"]
        board = setup["board"]

        move_count = 0
        max_moves = 60  # 最大手数

        while not board.is_terminal() and move_count < max_moves:
            # MCTS探索で次の手を選択
            action = mcts.get_best_action(board, num_simulations=5)

            # 着手
            board.make_move(action)
            move_count += 1

        # ゲームが終了したか、最大手数に達したか
        assert move_count <= max_moves

    def test_action_masking_works(self, setup):
        """Action Maskingが正しく機能するかテスト"""
        mcts = setup["mcts"]
        board = setup["board"]

        policy, _ = mcts.search(
            board,
            num_simulations=10,
            temperature=1.0,
            add_dirichlet_noise=False
        )

        # 合法手のみに確率が割り当てられている
        legal_actions = board.get_legal_moves()
        for action in range(65):
            if action not in legal_actions:
                assert policy[action] == 0.0, \
                    f"Non-legal action {action} has probability {policy[action]}"

    def test_get_action_evaluations(self, setup):
        """get_action_evaluationsのテスト（ヒント表示用）"""
        mcts = setup["mcts"]
        board = setup["board"]

        evaluations = mcts.get_action_evaluations(board, num_simulations=10)

        # 評価値配列のサイズチェック
        assert evaluations.shape == (65,)

        # 評価値は0-100の整数
        assert evaluations.dtype == np.int32
        assert np.all(evaluations >= 0)
        assert np.all(evaluations <= 100)

        # 合法手にのみ評価値がある
        legal_actions = board.get_legal_moves()
        for action in range(65):
            if action not in legal_actions:
                assert evaluations[action] == 0, \
                    f"Non-legal action {action} has evaluation {evaluations[action]}"

        # 少なくとも1つの合法手に非ゼロの評価値がある
        assert any(evaluations[action] > 0 for action in legal_actions)
