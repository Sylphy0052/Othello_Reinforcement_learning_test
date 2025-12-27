"""
OthelloResNet - AlphaZero方式のDual-Head ResNet

内部設計書に基づく実装:
- ConvBlock + ResBlocks + PolicyHead + ValueHead
- RTX 4050 (6GB VRAM) 向けの最適化
- 混合精度学習 (AMP) 対応
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    初期畳み込みブロック
    入力: (Batch, 3, 8, 8)
    出力: (Batch, num_filters, 8, 8)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResBlock(nn.Module):
    """
    残差ブロック (Residual Block)
    構造: Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> Add Skip -> ReLU
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        out += residual
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """
    方策ヘッド (Policy Head)
    出力: 65次元のLog確率（64マス + パスアクション）

    構造: Conv(1x1, 2ch) -> BN -> ReLU -> Flatten -> FC(65) -> LogSoftmax
    """

    def __init__(self, num_filters: int, board_size: int = 8):
        super().__init__()
        self.board_size = board_size

        # 1x1畳み込みで2チャネルに削減
        self.conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)

        # 全結合層: (2 * 8 * 8) -> 65
        self.fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

    def forward(self, x):
        # x: (Batch, num_filters, 8, 8)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        # Flatten: (Batch, 2*8*8)
        x = x.view(x.size(0), -1)

        # 全結合 -> LogSoftmax
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x


class ValueHead(nn.Module):
    """
    価値ヘッド (Value Head)
    出力: スカラ値 [-1, 1]（勝率推定）

    構造: Conv(1x1, 1ch) -> BN -> ReLU -> Flatten -> FC(256) -> ReLU -> FC(1) -> Tanh
    """

    def __init__(self, num_filters: int, board_size: int = 8, hidden_size: int = 256):
        super().__init__()
        self.board_size = board_size

        # 1x1畳み込みで1チャネルに削減
        self.conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

        # 全結合層: (1 * 8 * 8) -> 256 -> 1
        self.fc1 = nn.Linear(board_size * board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (Batch, num_filters, 8, 8)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        # Flatten: (Batch, 1*8*8)
        x = x.view(x.size(0), -1)

        # 全結合層
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Tanh: [-1, 1]
        x = torch.tanh(x)

        return x


class OthelloResNet(nn.Module):
    """
    AlphaZero方式のDual-Head ResNet

    入力: (Batch, 3, 8, 8)
        - Channel 0: 自分の石
        - Channel 1: 相手の石
        - Channel 2: 手番情報（全マス同じ値）

    出力:
        - policy_logits: (Batch, 65) - Log確率分布
        - value: (Batch, 1) - 勝率推定 [-1, 1]

    Args:
        num_blocks (int): ResBlockの数（デフォルト: 10）
        num_filters (int): フィルタ数（デフォルト: 128）
        board_size (int): 盤面サイズ（デフォルト: 8）
    """

    def __init__(
        self,
        num_blocks: int = 10,
        num_filters: int = 128,
        board_size: int = 8,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.board_size = board_size

        # 初期畳み込み: 3ch -> num_filters
        self.conv_block = ConvBlock(3, num_filters)

        # 残差ブロック × num_blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_blocks)
        ])

        # Dual Head
        self.policy_head = PolicyHead(num_filters, board_size)
        self.value_head = ValueHead(num_filters, board_size)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): 入力テンソル (Batch, 3, 8, 8)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - policy_logits: (Batch, 65)
                - value: (Batch, 1)
        """
        # 初期畳み込み
        x = self.conv_block(x)

        # 残差ブロックを順次適用
        for res_block in self.res_blocks:
            x = res_block(x)

        # Dual Head
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    def predict(self, board_tensor):
        """
        盤面テンソルから方策と価値を予測（推論用ヘルパー）

        Args:
            board_tensor (torch.Tensor): (3, 8, 8) or (Batch, 3, 8, 8)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - policy_probs: (65,) or (Batch, 65) - 確率分布
                - value: (1,) or (Batch, 1) - 勝率推定
        """
        # 入力が単一サンプルの場合、バッチ次元を追加
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(board_tensor)
            policy_probs = torch.exp(policy_logits)  # Log確率 -> 確率

        if squeeze_output:
            policy_probs = policy_probs.squeeze(0)
            value = value.squeeze(0)

        return policy_probs, value

    def get_param_count(self):
        """モデルのパラメータ数を取得"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def create_model(config: dict) -> OthelloResNet:
    """
    設定ファイルからモデルを作成

    Args:
        config (dict): 設定辞書
            例: {"num_blocks": 10, "num_filters": 128}

    Returns:
        OthelloResNet: インスタンス化されたモデル
    """
    num_blocks = config.get("num_blocks", 10)
    num_filters = config.get("num_filters", 128)
    board_size = config.get("board_size", 8)

    model = OthelloResNet(
        num_blocks=num_blocks,
        num_filters=num_filters,
        board_size=board_size,
    )

    return model
