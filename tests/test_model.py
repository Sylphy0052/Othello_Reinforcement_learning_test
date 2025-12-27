"""
OthelloResNetのテストケース

内部設計書の仕様に基づき、以下を検証:
- モデル初期化とパラメータ設定
- 入出力テンソルの形状チェック
- Policy/Valueヘッドの出力範囲検証
- GPU動作確認（CUDA利用可能な場合）
"""

import pytest
import torch
import numpy as np


class TestOthelloResNet:
    """OthelloResNetの基本機能テスト"""

    @pytest.fixture
    def model(self):
        """デフォルト設定のモデルを作成"""
        from src.model.net import OthelloResNet
        return OthelloResNet(num_blocks=10, num_filters=128)

    @pytest.fixture
    def small_model(self):
        """小規模モデル（テスト高速化用）"""
        from src.model.net import OthelloResNet
        return OthelloResNet(num_blocks=2, num_filters=32)

    @pytest.fixture
    def dummy_input(self):
        """ダミー入力テンソル (Batch=4, 3ch, 8x8)"""
        return torch.randn(4, 3, 8, 8)

    def test_model_initialization_default(self, model):
        """デフォルトパラメータでの初期化テスト"""
        assert model is not None
        # パラメータが存在することを確認
        params = list(model.parameters())
        assert len(params) > 0

    def test_model_initialization_custom(self):
        """カスタムパラメータでの初期化テスト"""
        from src.model.net import OthelloResNet
        model = OthelloResNet(num_blocks=5, num_filters=64)
        assert model is not None

    def test_forward_output_shape(self, small_model, dummy_input):
        """Forward パスの出力形状を検証"""
        policy_logits, value = small_model(dummy_input)

        batch_size = dummy_input.shape[0]

        # PolicyHead: (Batch, 65)
        assert policy_logits.shape == (batch_size, 65), \
            f"Expected policy shape ({batch_size}, 65), got {policy_logits.shape}"

        # ValueHead: (Batch, 1)
        assert value.shape == (batch_size, 1), \
            f"Expected value shape ({batch_size}, 1), got {value.shape}"

    def test_policy_output_is_log_probabilities(self, small_model, dummy_input):
        """Policy出力がLog確率であることを確認（LogSoftmax）"""
        policy_logits, _ = small_model(dummy_input)

        # Log確率なので、すべて負の値になるはず
        assert torch.all(policy_logits <= 0), \
            "Policy output should be log probabilities (all <= 0)"

        # 各サンプルの合計が1に近いことを確認（指数関数で戻す）
        probs = torch.exp(policy_logits)
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(dummy_input.shape[0]), atol=1e-5), \
            "Policy probabilities should sum to 1"

    def test_value_output_range(self, small_model, dummy_input):
        """Value出力が[-1, 1]の範囲内にあることを確認（Tanh）"""
        _, value = small_model(dummy_input)

        assert torch.all(value >= -1.0) and torch.all(value <= 1.0), \
            "Value output should be in range [-1, 1] (Tanh activation)"

    def test_single_sample_inference(self, small_model):
        """単一サンプル推論（Batch=1）のテスト"""
        single_input = torch.randn(1, 3, 8, 8)
        policy_logits, value = small_model(single_input)

        assert policy_logits.shape == (1, 65)
        assert value.shape == (1, 1)

    def test_batch_inference(self, small_model):
        """異なるバッチサイズでの推論テスト"""
        for batch_size in [1, 4, 16, 32]:
            batch_input = torch.randn(batch_size, 3, 8, 8)
            policy_logits, value = small_model(batch_input)

            assert policy_logits.shape == (batch_size, 65)
            assert value.shape == (batch_size, 1)

    def test_gradient_flow(self, small_model, dummy_input):
        """勾配が正常に流れるかテスト"""
        policy_logits, value = small_model(dummy_input)

        # ダミーのターゲット
        target_policy = torch.randn(4, 65)
        target_value = torch.randn(4, 1)

        # ダミーの損失
        loss = torch.nn.functional.mse_loss(policy_logits, target_policy) + \
               torch.nn.functional.mse_loss(value, target_value)

        loss.backward()

        # すべてのパラメータに勾配が計算されていることを確認
        for param in small_model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradients should be computed"
                assert not torch.all(param.grad == 0), "Gradients should not be all zeros"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self, small_model):
        """GPU推論のテスト（CUDA利用可能な場合のみ）"""
        device = torch.device("cuda")
        model_gpu = small_model.to(device)
        input_gpu = torch.randn(4, 3, 8, 8).to(device)

        policy_logits, value = model_gpu(input_gpu)

        # 出力がGPU上にあることを確認
        assert policy_logits.device.type == "cuda"
        assert value.device.type == "cuda"

        # 形状チェック
        assert policy_logits.shape == (4, 65)
        assert value.shape == (4, 1)

    def test_model_eval_mode(self, small_model, dummy_input):
        """評価モード（eval）での動作確認"""
        small_model.eval()

        with torch.no_grad():
            policy_logits, value = small_model(dummy_input)

        assert policy_logits.shape == (4, 65)
        assert value.shape == (4, 1)

    def test_model_train_mode(self, small_model, dummy_input):
        """訓練モード（train）での動作確認"""
        small_model.train()

        policy_logits, value = small_model(dummy_input)

        assert policy_logits.shape == (4, 65)
        assert value.shape == (4, 1)

    def test_deterministic_output_in_eval_mode(self, small_model):
        """評価モードで同じ入力に対して同じ出力が得られることを確認"""
        small_model.eval()
        input_tensor = torch.randn(1, 3, 8, 8)

        with torch.no_grad():
            policy1, value1 = small_model(input_tensor)
            policy2, value2 = small_model(input_tensor)

        assert torch.allclose(policy1, policy2), "Output should be deterministic in eval mode"
        assert torch.allclose(value1, value2), "Output should be deterministic in eval mode"

    def test_model_parameters_count(self, model):
        """パラメータ数の妥当性チェック"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # デフォルト設定（10ブロック、128フィルタ）では数百万〜千万パラメータのはず
        assert total_params > 100000, f"Model seems too small: {total_params} params"
        assert trainable_params == total_params, "All parameters should be trainable"
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
