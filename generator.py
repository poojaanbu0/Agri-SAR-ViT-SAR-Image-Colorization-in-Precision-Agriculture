import torch
import torch.nn as nn

class TransformerBottleneck(nn.Module):
    """
    Implements the Swin/ViT Attention mechanism for global spatial reasoning[cite: 111, 143].
    """
    def __init__(self, dim, num_heads=8):
        super(TransformerBottleneck, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Flatten spatial dimensions for attention [cite: 175]
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = self.norm(attn_output + x_flat)
        
        # Restore spatial dimensions [cite: 176]
        return x_flat.permute(0, 2, 1).view(b, c, h, w)

class AgriSARViTGenerator(nn.Module):
    """
    Hybrid CNN-Transformer Generator for SAR-to-RGB translation[cite: 106, 121].
    """
    def __init__(self, in_channels=2, out_channels=3):
        super(AgriSARViTGenerator, self).__init__()
        
        # Encoder Blocks: Conv + BN + ReLU [cite: 107, 137]
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        
        # Transformer Bottleneck [cite: 111, 175]
        self.bottleneck = TransformerBottleneck(256, num_heads=8)
        
        # Decoder Blocks with Skip Connections [cite: 114, 144]
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(64 + 64, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        # Encoding path [cite: 174]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Global Reasoning [cite: 129]
        b = self.bottleneck(e3)
        
        # Decoding with Skip Connections [cite: 108, 145, 178]
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e2], dim=1))
        d3 = self.dec3(torch.cat([d2, e1], dim=1))
        return d3
