"""ViT architectures for models 9 and 11, matching their state_dict keys.

model9 : patch8, dim192, depth6, WITH cls-token, final `norm`, head on cls.
model11: patch16, dim256, depth8, NO cls-token, global-avg-pool + `fc_norm`.

num_heads is not encoded in the params (only the attention math depends on it);
we use the standard head_dim=64 convention (dim192->3 heads, dim256->4 heads).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img=64, patch=8, dim=192, depth=6, num_heads=3,
                 mlp_hidden=768, n_classes=200, use_cls=True, global_pool=False):
        super().__init__()
        self.use_cls = use_cls
        self.global_pool = global_pool
        self.patch = patch
        n_patches = (img // patch) ** 2
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, stride=patch)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        self.blocks = nn.ModuleList(
            [Block(dim, num_heads, mlp_hidden) for _ in range(depth)])
        if global_pool:
            self.fc_norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed.proj(x)                 # (B,dim,gh,gw)
        x = x.flatten(2).transpose(1, 2)             # (B,N,dim)
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x.mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        return self.head(x)


def build_vit(name, state):
    if name == "model9":
        m = ViT(img=64, patch=8, dim=192, depth=6, num_heads=3,
                mlp_hidden=768, use_cls=True, global_pool=False)
    elif name == "model11":
        m = ViT(img=64, patch=16, dim=256, depth=8, num_heads=4,
                mlp_hidden=1024, use_cls=False, global_pool=True)
    else:
        raise ValueError(name)
    missing, unexpected = m.load_state_dict(state, strict=False)
    return m.eval(), missing, unexpected
