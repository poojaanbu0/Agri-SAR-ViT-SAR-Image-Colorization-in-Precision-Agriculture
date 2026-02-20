import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator for adversarial supervision[cite: 119, 147, 180].
    """
    def __init__(self, in_channels=5): # SAR (2) + RGB (3) = 5 [cite: 136]
        super(PatchGANDiscriminator, self).__init__()
        
        def disc_block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride, 1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            disc_block(64, 128),
            disc_block(128, 256),
            disc_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1) # Output 1-channel prediction map [cite: 180]
        )

    def forward(self, sar, rgb):
        # Concatenate SAR input and RGB for conditional GAN [cite: 179]
        x = torch.cat([sar, rgb], dim=1)
        return self.model(x)
