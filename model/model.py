#!/usr/bin/env python3
"""
MicroCaptchaNet - Ultra-lightweight CNN for 4-character captcha recognition.

Target: <1MiB model size, >2000 img/s inference speed
Architecture: Depthwise Separable Convolutions + Multi-head Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - reduces parameters by ~90%"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class MicroCaptchaNet(nn.Module):
    """
    Ultra-lightweight captcha recognition network.
    
    Architecture:
    - Input: 3x64x176
    - 3 depthwise separable conv blocks with progressive channel expansion
    - Global Average Pooling to eliminate FC layer parameters
    - 4 independent classification heads (one per character position)
    
    Parameters: ~50K (approximately 200KB in float32)
    """
    
    def __init__(
        self, 
        num_classes: int = 22, 
        captcha_length: int = 4,
        channels: list[int] = [16, 32, 64],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.captcha_length = captcha_length
        
        # Initial convolution to expand channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # Depthwise separable conv blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(nn.Sequential(
                DepthwiseSeparableConv(channels[i], channels[i + 1], stride=2),
                DepthwiseSeparableConv(channels[i + 1], channels[i + 1]),
            ))
        
  # After previous blocks: (N, C, 8, 22)
        # After this downsample: (N, C, 4, 11)
        self.downsample = DepthwiseSeparableConv(channels[-1], channels[-1], stride=2)
        
        # Split feature map horizontally into 4 parts for 4 characters
        # Each part will be globally pooled
        # Input: (N, C, 4, 11), we'll use avg pool with kernel (4, 3) and proper padding
        # to get (N, C, 1, 4) output
        self.final_pool = nn.AvgPool2d(kernel_size=(4, 3), stride=(1, 3), padding=(0, 1))
        
        # Classification heads - one per character position
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(channels[-1], num_classes) for _ in range(captcha_length)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 3, 64, 176)
            
        Returns:
            Output tensor of shape (N, 4, 22) - logits for each character position
        """
        # Feature extraction
        x = self.stem(x)  # (N, C0, 32, 88)
        
        for block in self.blocks:
            x = block(x)
        # After blocks: (N, C, 8, 22)
        
        x = self.downsample(x)  # (N, C, 4, 11)
        x = self.final_pool(x)  # (N, C, 1, 4)
        x = x.squeeze(2)  # (N, C, 4)
        x = x.permute(0, 2, 1)  # (N, 4, C)
        
        x = self.dropout(x)
        
        # Classification for each position
        outputs = []
        for i, head in enumerate(self.heads):
            outputs.append(head(x[:, i, :]))  # (N, 22)
        
        return torch.stack(outputs, dim=1)  # (N, 4, 22)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted character indices."""
        logits = self.forward(x)
        return logits.argmax(dim=2)  # (N, 4)


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)  # float32
    return {
        "total": total,
        "trainable": trainable,
        "size_mb": size_mb,
    }


if __name__ == "__main__":
    # Test the model
    model = MicroCaptchaNet(num_classes=22, captcha_length=4)
    
    # Count parameters
    stats = count_parameters(model)
    print(f"Total parameters: {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Model size: {stats['size_mb']:.3f} MB")
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 176)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test prediction
    pred = model.predict(x)
    print(f"Prediction shape: {pred.shape}")
    
    # Benchmark inference speed
    import time
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Benchmark
        n_iterations = 100
        batch_size = 32
        x_batch = torch.randn(batch_size, 3, 64, 176)
        
        start = time.time()
        for _ in range(n_iterations):
            _ = model(x_batch)
        elapsed = time.time() - start
        
        throughput = (n_iterations * batch_size) / elapsed
        print(f"Throughput (CPU): {throughput:.2f} images/sec")
