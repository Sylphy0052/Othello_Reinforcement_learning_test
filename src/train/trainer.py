"""
AlphaZero Trainer

学習ループ全体のオーケストレーション:
1. Self-Play でデータ生成
2. Replay Buffer に格納
3. ミニバッチで学習（AMP使用）
4. チェックポイント保存
5. TensorBoard ロギング
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional
import time


class AlphaZeroTrainer:
    """
    AlphaZero学習ループ

    Self-Play → Replay Buffer → Train のサイクルを管理
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        replay_buffer,
        self_play_worker,
        config: dict,
        checkpoint_dir: str = "data/models",
        log_dir: str = "data/logs",
    ):
        """
        Args:
            model: OthelloResNet
            device: torch.device
            replay_buffer: ReplayBuffer
            self_play_worker: SelfPlayWorker
            config: 設定辞書
            checkpoint_dir: チェックポイント保存先
            log_dir: TensorBoard ログ保存先
        """
        self.model = model
        self.device = device
        self.replay_buffer = replay_buffer
        self.self_play_worker = self_play_worker
        self.config = config

        # ディレクトリ作成
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # オプティマイザー
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.get("lr", 0.001),
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0001),
        )

        # 学習率スケジューラ（オプション）
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get("lr_step_size", 100),
            gamma=config.get("lr_gamma", 0.1),
        )

        # AMP Scaler（混合精度学習）
        self.scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # 学習統計
        self.global_step = 0
        self.epoch = 0

    def train(
        self,
        num_iterations: int,
        self_play_episodes_per_iter: int = 100,
        train_epochs_per_iter: int = 10,
        batch_size: int = 256,
        checkpoint_interval: int = 10,
    ):
        """
        学習ループを実行

        Args:
            num_iterations (int): 学習イテレーション数
            self_play_episodes_per_iter (int): イテレーションあたりのSelf-Playエピソード数
            train_epochs_per_iter (int): イテレーションあたりの学習エポック数
            batch_size (int): バッチサイズ
            checkpoint_interval (int): チェックポイント保存間隔（イテレーション）
        """
        print("=" * 70)
        print("AlphaZero Training Started")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Batch Size: {batch_size}")
        print(f"Iterations: {num_iterations}")
        print("=" * 70)

        for iteration in range(1, num_iterations + 1):
            iter_start_time = time.time()

            print(f"\n[Iteration {iteration}/{num_iterations}]")

            # 1. Self-Play でデータ生成
            print(f"  Self-Play: Generating {self_play_episodes_per_iter} episodes...")
            self_play_start = time.time()

            training_data = self.self_play_worker.execute_episodes(
                num_episodes=self_play_episodes_per_iter,
                add_dirichlet_noise=True,
            )

            self.replay_buffer.add(training_data)
            self_play_time = time.time() - self_play_start

            print(f"  Self-Play: Generated {len(training_data)} samples in {self_play_time:.2f}s")
            print(f"  Replay Buffer: {len(self.replay_buffer)} total samples")

            # 2. 学習
            if self.replay_buffer.is_ready(batch_size):
                print(f"  Training: {train_epochs_per_iter} epochs...")
                train_start = time.time()

                avg_loss = self._train_epochs(
                    num_epochs=train_epochs_per_iter,
                    batch_size=batch_size,
                )

                train_time = time.time() - train_start
                print(f"  Training: Avg Loss = {avg_loss:.4f} ({train_time:.2f}s)")

                # TensorBoard ロギング
                self.writer.add_scalar("Loss/train", avg_loss, iteration)
                self.writer.add_scalar("Time/self_play", self_play_time, iteration)
                self.writer.add_scalar("Time/train", train_time, iteration)
                self.writer.add_scalar("Buffer/size", len(self.replay_buffer), iteration)

                # バッファ統計
                buffer_stats = self.replay_buffer.get_statistics()
                self.writer.add_scalar("Buffer/value_mean", buffer_stats["value_mean"], iteration)
                self.writer.add_scalar("Buffer/value_std", buffer_stats["value_std"], iteration)

            # 3. チェックポイント保存
            if iteration % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_iter_{iteration}.pt")

            iter_time = time.time() - iter_start_time
            print(f"  Iteration Time: {iter_time:.2f}s")

        print("\n" + "=" * 70)
        print("Training Completed")
        print("=" * 70)

        # 最終モデルを保存
        self.save_checkpoint("final_model.pt")
        self.writer.close()

    def _train_epochs(
        self,
        num_epochs: int,
        batch_size: int,
    ) -> float:
        """
        指定エポック数だけ学習

        Args:
            num_epochs (int): エポック数
            batch_size (int): バッチサイズ

        Returns:
            float: 平均損失
        """
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(num_epochs):
            # ミニバッチをサンプリング
            states, target_policies, target_values = self.replay_buffer.sample(batch_size)

            # Tensor に変換
            states = torch.from_numpy(states).to(self.device)
            target_policies = torch.from_numpy(target_policies).to(self.device)
            target_values = torch.from_numpy(target_values).to(self.device)

            # 学習ステップ
            loss = self._train_step(states, target_policies, target_values)

            total_loss += loss
            total_batches += 1

            self.global_step += 1
            self.epoch += 1

        avg_loss = total_loss / total_batches
        return avg_loss

    def _train_step(
        self,
        states: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> float:
        """
        1回の学習ステップ

        Args:
            states: (batch_size, 3, 8, 8)
            target_policies: (batch_size, 65)
            target_values: (batch_size, 1)

        Returns:
            float: 損失
        """
        self.optimizer.zero_grad()

        # AMP を使う場合
        if self.scaler is not None and self.device.type == "cuda":
            with torch.amp.autocast('cuda'):
                policy_logits, value_pred = self.model(states)

                # 損失計算
                policy_loss = self._policy_loss(policy_logits, target_policies)
                value_loss = self._value_loss(value_pred, target_values)
                total_loss = policy_loss + value_loss

            # Backward pass with scaler
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # AMP なし
            policy_logits, value_pred = self.model(states)

            policy_loss = self._policy_loss(policy_logits, target_policies)
            value_loss = self._value_loss(value_pred, target_values)
            total_loss = policy_loss + value_loss

            total_loss.backward()
            self.optimizer.step()

        return total_loss.item()

    def _policy_loss(
        self,
        policy_logits: torch.Tensor,
        target_policies: torch.Tensor,
    ) -> torch.Tensor:
        """
        方策損失（クロスエントロピー）

        Args:
            policy_logits: (batch_size, 65) - LogSoftmax出力
            target_policies: (batch_size, 65) - ターゲット確率分布

        Returns:
            torch.Tensor: 損失
        """
        # KL Divergence: -sum(target * log(pred))
        # policy_logits は既に LogSoftmax なので、直接使える
        return -torch.mean(torch.sum(target_policies * policy_logits, dim=1))

    def _value_loss(
        self,
        value_pred: torch.Tensor,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        価値損失（MSE）

        Args:
            value_pred: (batch_size, 1)
            target_values: (batch_size, 1)

        Returns:
            torch.Tensor: 損失
        """
        return nn.functional.mse_loss(value_pred, target_values)

    def save_checkpoint(self, filename: str):
        """
        チェックポイントを保存

        Args:
            filename: ファイル名
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        チェックポイントを読み込み

        Args:
            checkpoint_path: チェックポイントファイルパス
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Global Step: {self.global_step}")
        print(f"  Epoch: {self.epoch}")
