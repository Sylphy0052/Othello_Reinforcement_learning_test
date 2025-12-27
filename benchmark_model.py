"""
OthelloResNetのベンチマーク

RTX 4050での動作確認:
- GPUメモリ使用量
- 推論速度（バッチサイズ別）
- 訓練時のメモリ使用量（AMP有無の比較）
"""

import torch
import time
from src.model.net import OthelloResNet


def benchmark_inference(model, device, batch_sizes=[1, 4, 16, 32, 64, 128, 256]):
    """推論速度ベンチマーク"""
    print("\n" + "=" * 70)
    print("推論速度ベンチマーク")
    print("=" * 70)

    model.eval()
    model.to(device)

    for batch_size in batch_sizes:
        # ダミー入力
        dummy_input = torch.randn(batch_size, 3, 8, 8).to(device)

        # ウォームアップ
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # ベンチマーク
        num_iterations = 100
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        fps = (batch_size * num_iterations) / elapsed_time

        print(f"Batch Size: {batch_size:4d} | "
              f"Time: {elapsed_time:.3f}s | "
              f"Throughput: {fps:8.1f} samples/sec")


def benchmark_training_memory(model, device, batch_sizes=[64, 128, 256, 512]):
    """訓練時のメモリ使用量ベンチマーク（AMP有無）"""
    print("\n" + "=" * 70)
    print("訓練時メモリ使用量ベンチマーク")
    print("=" * 70)

    if device.type != "cuda":
        print("CUDA not available. Skipping memory benchmark.")
        return

    # FP32訓練
    print("\n--- FP32 (通常精度) ---")
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            dummy_input = torch.randn(batch_size, 3, 8, 8).to(device)
            target_policy = torch.randn(batch_size, 65).to(device)
            target_value = torch.randn(batch_size, 1).to(device)

            # Forward pass
            policy_logits, value = model(dummy_input)

            # Loss computation
            loss = torch.nn.functional.mse_loss(policy_logits, target_policy) + \
                   torch.nn.functional.mse_loss(value, target_value)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            print(f"Batch Size: {batch_size:4d} | Peak Memory: {peak_memory:7.1f} MB")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch Size: {batch_size:4d} | OOM (Out of Memory)")
                torch.cuda.empty_cache()
            else:
                raise

    # AMP訓練
    print("\n--- AMP (混合精度) ---")
    torch.cuda.empty_cache()
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')

    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            dummy_input = torch.randn(batch_size, 3, 8, 8).to(device)
            target_policy = torch.randn(batch_size, 65).to(device)
            target_value = torch.randn(batch_size, 1).to(device)

            optimizer.zero_grad()

            # Forward pass with autocast
            with torch.amp.autocast('cuda'):
                policy_logits, value = model(dummy_input)
                loss = torch.nn.functional.mse_loss(policy_logits, target_policy) + \
                       torch.nn.functional.mse_loss(value, target_value)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()

            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            print(f"Batch Size: {batch_size:4d} | Peak Memory: {peak_memory:7.1f} MB")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch Size: {batch_size:4d} | OOM (Out of Memory)")
                torch.cuda.empty_cache()
            else:
                raise


def main():
    print("=" * 70)
    print("OthelloResNet ベンチマーク")
    print("=" * 70)

    # デバイス確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # モデル作成（デフォルト設定: 10 blocks, 128 filters）
    print("\n--- モデル設定 ---")
    print("Blocks: 10")
    print("Filters: 128")

    model = OthelloResNet(num_blocks=10, num_filters=128)
    param_count = model.get_param_count()
    print(f"Total Parameters: {param_count['total']:,}")
    print(f"Trainable Parameters: {param_count['trainable']:,}")

    # 推論ベンチマーク
    benchmark_inference(model, device)

    # 訓練メモリベンチマーク
    if device.type == "cuda":
        benchmark_training_memory(model, device)

    print("\n" + "=" * 70)
    print("ベンチマーク完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
