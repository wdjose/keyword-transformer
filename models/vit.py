import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Vision Transformer implementation adapted and edited from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# License: https://github.com/lucidrains/vit-pytorch/blob/main/README.md

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PostNormTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                Attention(dim, heads=heads, dim_head=dim_head, 
                    dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for norm1, norm2, attn, ff in self.layers:
            x = norm1(attn(x) + x)
            x = norm2(ff(x) + x)
        return x

class ViT(nn.Module):
    def __init__(self, *, img_x, img_y, patch_x, patch_y, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert img_x % patch_x == 0, 'Image dimensions must be divisible by the patch size.'
        assert img_y % patch_y == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_x // patch_x) * (img_y // patch_y)
        patch_dim = channels * patch_x * patch_y

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b (h p1) (w p2) -> b (h w) p1 p2', p1 = patch_x, p2 = patch_y),
        #     nn.Linear(patch_dim, dim),
        # )

        # self.patch_rearrange = Rearrange('b (h p1) (w p2) -> b (h w) p1 p2', p1 = patch_x, p2 = patch_y)
        self.lin_proj = nn.Linear(img_y, dim, bias = False)

        self.g_feature = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.transformer = PostNormTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # x = self.patch_rearrange(img)
        x = self.lin_proj(img)
        b, T, _ = x.shape

        g_features = repeat(self.g_feature, '() n d -> b n d', b = b)
        x = torch.cat((g_features, x), dim=1)
        x = x + self.pos_embedding

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
