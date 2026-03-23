"""
model.py — CNN pupil detection
- ~200 training images
- smaller model
- stronger dropout
- rgb instead of grayscale (keeping info)

Architecture
  3×128×128
  → ConvBlock(3→32)   → 64×64
  → ConvBlock(32→64)  → 32×32
  → ConvBlock(64→128) → 16×16
  → GlobalAvgPool     → 128        ← less parameters than Flatten
  → FC(128→64) → FC(64→3) → Sigmoid
  Output: [x, y, r] normalized [0,1]

  Flatten(128×16×16) = 32,768 → FC(512) = 16M parameters → overfitting
  GlobalAvgPool(128) → FC(64) =  8K parameters → regularization
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)


class GazeCNN(nn.Module):
    def __init__(self, img_size=128, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32),    # 128→64
            ConvBlock(32,  64),    # 64→32
            ConvBlock(64,  128),   # 32→16
        )

        # GlobalAveragePooling: 128 feature-maps → 1 number
        # Output is always 128-dim, no matter the img_size
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)   # (B, 128, 16, 16)
        x = self.gap(x)        # (B, 128, 1, 1)
        x = self.regressor(x)  # (B, 3)
        return x

    def predict_pixels(self, img_tensor, orig_size=256):
        self.eval()
        with torch.no_grad():
            out = self.forward(img_tensor)[0].tolist()
        return {'x': out[0]*orig_size, 'y': out[1]*orig_size,
                'r': out[2]*orig_size}


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameteres: {total:,}")
    print(f"Trainable:       {trainable:,}")
    return trainable


if __name__ == '__main__':
    print("Test GazeCNN")
    print("=" * 45)
    model = GazeCNN(img_size=128)
    count_parameters(model)

    dummy = torch.randn(4, 3, 128, 128)
    out   = model(dummy)
    print(f"\nUlaz:  {dummy.shape}")
    print(f"Output: {out.shape}  ([4, 3])")
    print(f"Range: [{out.min():.4f}, {out.max():.4f}]  ([0,1])")
    assert out.shape == (4, 3)
    assert 0 <= out.min() and out.max() <= 1
    print("\nTests passed!")