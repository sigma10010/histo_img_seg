import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Basic Building Blocks ---

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
        )

    def forward(self, x):
        return self.block(x)

# --- Skip Attention Block ---
class SkipAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(SkipAttention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

# --- UpBlock with Skip Attention ---
class UpBlockAttn(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.attn = SkipAttention(F_g=out_ch, F_l=skip_ch, F_int=out_ch // 2)
        self.conv = ConvBlock(in_ch=out_ch + skip_ch, out_ch=out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# --- ViT Transformer as Bottleneck ---
class ViTBottleneck(nn.Module):
    def __init__(self, in_channels=512, patch_size=8, emb_dim=1024, num_heads=8, depth=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)  # B, 1024, 32, 32 -> B, 1024, 1, 1

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1025, emb_dim))  # +1 for cls token

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, in_channels, patch_size, patch_size)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, emb_dim, H/patch, W/patch
        x = x.flatten(2).transpose(1, 2)  # B, N, D
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, 1:]  # remove cls token
        x = x.transpose(1, 2).view(B, -1, H // self.patch_size, W // self.patch_size)
        return self.reconstruct(x)

# --- Full TransUNet Model ---
class TransUNetWithAttention(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.vit = ViTBottleneck(512)

        self.up4 = UpBlockAttn(512, 512, 256)
        self.up3 = UpBlockAttn(256, 256, 128)
        self.up2 = UpBlockAttn(128, 128, 64)
        self.up1 = UpBlockAttn(64, 64, 64)

        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        e1 = self.enc1(x)         # 512
        e2 = self.enc2(self.pool1(e1))   # 256
        e3 = self.enc3(self.pool2(e2))   # 128
        e4 = self.enc4(self.pool3(e3))   # 64
        b = self.vit(self.pool4(e4))     # 32

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.out_conv(d1)
        out = self.activation(out)
        return out, torch.rand(1)

if __name__ == "__main__":
    model = TransUNetWithAttention(in_ch=3, out_ch=2)
    x = torch.randn(1, 3, 512, 512)
    out,_ = model(x)
    print(out.shape)