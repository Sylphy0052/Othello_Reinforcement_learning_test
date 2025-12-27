"""
学習ループのテストケース

- ReplayBuffer の動作テスト
- SelfPlayWorker の動作テスト
- Trainer の統合テスト（小規模）
"""

import pytest
import torch
import numpy as np
from src.cython.bitboard import OthelloBitboard
from src.model.net import OthelloResNet
from src.mcts.mcts import MCTS
from src.train.buffer import ReplayBuffer
from src.train.self_play import SelfPlayWorker
from src.train.trainer import AlphaZeroTrainer


class TestReplayBuffer:
    """ReplayBufferのテスト"""

    def test_buffer_initialization(self):
        """バッファの初期化テスト"""
        buffer = ReplayBuffer(max_size=1000)

        assert len(buffer) == 0
        assert buffer.max_size == 1000

    def test_buffer_add_data(self):
        """データ追加テスト"""
        buffer = ReplayBuffer(max_size=100)

        # ダミーデータ
        state = np.random.rand(3, 8, 8).astype(np.float32)
        policy = np.random.rand(65).astype(np.float32)
        value = 1.0

        buffer.add_single(state, policy, value)

        assert len(buffer) == 1

    def test_buffer_sample(self):
        """サンプリングテスト"""
        buffer = ReplayBuffer(max_size=100)

        # 50サンプル追加
        for i in range(50):
            state = np.random.rand(3, 8, 8).astype(np.float32)
            policy = np.random.rand(65).astype(np.float32)
            value = float(i % 3 - 1)  # -1, 0, 1
            buffer.add_single(state, policy, value)

        # バッチサイズ32でサンプリング
        states, policies, values = buffer.sample(batch_size=32)

        assert states.shape == (32, 3, 8, 8)
        assert policies.shape == (32, 65)
        assert values.shape == (32, 1)

    def test_buffer_overflow(self):
        """バッファオーバーフロー時の動作テスト"""
        buffer = ReplayBuffer(max_size=10)

        # 20サンプル追加（max_sizeを超える）
        for i in range(20):
            state = np.random.rand(3, 8, 8).astype(np.float32)
            policy = np.random.rand(65).astype(np.float32)
            value = 1.0
            buffer.add_single(state, policy, value)

        # バッファサイズはmax_sizeに制限される
        assert len(buffer) == 10

    def test_buffer_statistics(self):
        """統計情報取得テスト"""
        buffer = ReplayBuffer(max_size=100)

        # ダミーデータ追加
        for i in range(30):
            state = np.random.rand(3, 8, 8).astype(np.float32)
            policy = np.random.rand(65).astype(np.float32)
            value = float(i % 3 - 1)
            buffer.add_single(state, policy, value)

        stats = buffer.get_statistics()

        assert stats["size"] == 30
        assert stats["max_size"] == 100
        assert 0.0 <= stats["fill_rate"] <= 1.0
        assert "value_mean" in stats
        assert "value_std" in stats


class TestSelfPlayWorker:
    """SelfPlayWorkerのテスト"""

    @pytest.fixture
    def setup(self):
        """テスト用セットアップ"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OthelloResNet(num_blocks=2, num_filters=32)
        model.to(device)
        model.eval()
        mcts = MCTS(model, device, c_puct=1.0, dirichlet_alpha=0.3)

        worker = SelfPlayWorker(
            board_class=OthelloBitboard,
            mcts=mcts,
            num_simulations=5,  # テスト用に少なめ
            temperature_threshold=10,
        )

        return {"worker": worker, "device": device}

    def test_self_play_single_episode(self, setup):
        """1エピソードの実行テスト"""
        worker = setup["worker"]

        # 1エピソード実行
        training_data = worker.execute_episode(add_dirichlet_noise=False)

        # データが生成されていることを確認
        assert len(training_data) > 0

        # データ形式の確認
        state, policy, value = training_data[0]
        assert state.shape == (3, 8, 8)
        assert policy.shape == (65,)
        assert value in [-1.0, 0.0, 1.0]

    def test_self_play_multiple_episodes(self, setup):
        """複数エピソードの実行テスト"""
        worker = setup["worker"]

        # 3エピソード実行
        all_data = worker.execute_episodes(num_episodes=3, add_dirichlet_noise=False)

        # データが生成されていることを確認
        assert len(all_data) > 0

    def test_self_play_with_dirichlet_noise(self, setup):
        """ディリクレノイズ付きエピソードのテスト"""
        worker = setup["worker"]

        training_data = worker.execute_episode(add_dirichlet_noise=True)

        assert len(training_data) > 0


class TestAlphaZeroTrainer:
    """AlphaZeroTrainerのテスト（統合テスト）"""

    @pytest.fixture
    def setup(self):
        """テスト用セットアップ"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 小規模モデル
        model = OthelloResNet(num_blocks=2, num_filters=16)
        model.to(device)

        # MCTS
        mcts = MCTS(model, device, c_puct=1.0)

        # Self-Play Worker
        self_play_worker = SelfPlayWorker(
            board_class=OthelloBitboard,
            mcts=mcts,
            num_simulations=3,  # テスト用に非常に少なめ
            temperature_threshold=5,
        )

        # Replay Buffer
        replay_buffer = ReplayBuffer(max_size=1000)

        # Config
        config = {
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        }

        # Trainer
        trainer = AlphaZeroTrainer(
            model=model,
            device=device,
            replay_buffer=replay_buffer,
            self_play_worker=self_play_worker,
            config=config,
            checkpoint_dir="data/models/test",
            log_dir="data/logs/test",
        )

        return trainer

    def test_trainer_initialization(self, setup):
        """Trainerの初期化テスト"""
        trainer = setup

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.replay_buffer is not None
        assert trainer.self_play_worker is not None

    def test_trainer_train_step(self, setup):
        """1ステップの学習テスト"""
        trainer = setup

        # ダミーデータをバッファに追加
        for _ in range(64):  # batch_size分
            state = np.random.rand(3, 8, 8).astype(np.float32)
            policy = np.random.rand(65).astype(np.float32)
            policy /= policy.sum()
            value = np.random.choice([-1.0, 0.0, 1.0])
            trainer.replay_buffer.add_single(state, policy, value)

        # ミニバッチでサンプリング
        states, target_policies, target_values = trainer.replay_buffer.sample(batch_size=32)

        # Tensorに変換
        states = torch.from_numpy(states).to(trainer.device)
        target_policies = torch.from_numpy(target_policies).to(trainer.device)
        target_values = torch.from_numpy(target_values).to(trainer.device)

        # 1ステップ学習
        loss = trainer._train_step(states, target_policies, target_values)

        # 損失が計算されていることを確認
        assert isinstance(loss, float)
        assert loss > 0

    def test_trainer_short_training_loop(self, setup):
        """短い学習ループのテスト"""
        trainer = setup

        # 非常に短い学習ループ（1イテレーション）
        trainer.train(
            num_iterations=1,
            self_play_episodes_per_iter=2,  # 2エピソードのみ
            train_epochs_per_iter=2,
            batch_size=16,
            checkpoint_interval=1,
        )

        # バッファにデータが追加されていることを確認
        assert len(trainer.replay_buffer) > 0
