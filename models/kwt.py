import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Vision Transformer implementation adapted and edited from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# Specific implementation details based on https://arxiv.org/pdf/2104.00769.pdf
# License: https://github.com/lucidrains/vit-pytorch/blob/main/LICENSE

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNormTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.depth = depth
        self.norm1 = nn.ModuleList([])
        self.norm2 = nn.ModuleList([])
        self.attn = nn.ModuleList([])
        self.ff = nn.ModuleList([])
        for layer in range(depth):
            self.norm1.append(nn.LayerNorm(dim))
            self.norm2.append(nn.LayerNorm(dim))
            self.attn.append(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ff.append(FeedForward(dim, mlp_dim, dropout=dropout))
    def forward(self, x):
        for i in range(self.depth):
            x = self.attn[i](self.norm1[i](x)) + x
            x = self.ff[i](self.norm2[i](x)) + x
        return x

class PostNormTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.depth = depth
        self.norm1 = nn.ModuleList([])
        self.norm2 = nn.ModuleList([])
        self.attn = nn.ModuleList([])
        self.ff = nn.ModuleList([])
        for layer in range(depth):
            self.norm1.append(nn.LayerNorm(dim))
            self.norm2.append(nn.LayerNorm(dim))
            self.attn.append(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ff.append(FeedForward(dim, mlp_dim, dropout=dropout))
    def forward(self, x):
        for i in range(self.depth):
            x = self.norm1[i](self.attn[i](x) + x)
            x = self.norm2[i](self.ff[i](x) + x)
        return x

class KWT(nn.Module):
    def __init__(self, *, img_x, img_y, patch_x, patch_y, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert img_x % patch_x == 0, 'Image dimensions must be divisible by the patch size.'
        assert img_y % patch_y == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_x // patch_x) * (img_y // patch_y)
        patch_dim = channels * patch_x * patch_y

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.lin_proj = nn.Linear(img_y, dim)

        self.g_feature = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.transformer = PostNormTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.lin_proj(img)
        b, T, _ = x.shape

        g_features = repeat(self.g_feature, '() n d -> b n d', b = b)
        x = torch.cat((g_features, x), dim=1)
        x = x + self.pos_embedding

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
