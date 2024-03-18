import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=32, dropout=0.3):
        super(Former, self).__init__()
        mlp_dim = dim * 2
        self.layers = nn.ModuleList([])
        # dim_head = dim // heads
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


class Global2Local(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Global2Local, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)

        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        # x : [B, C, N]
        # z : [B, CN, C]

        b, m, _ = z.shape
        b, c, _ = x.shape
        x = x.transpose(1, 2).unsqueeze(1)
        # x : [B, 1, N, C]

        q = self.to_q(z).view(b, self.heads, m, c)
        k = self.to_k(x)
        v = self.to_v(x)
        # q : [B, 1, CN, C]
        # k : [B, 1, N, C]
        # v : [B, 1, N, C]
        
        dots = q @ k.transpose(2, 3) * self.scale
        # dots : [B, 1, CN, N]
        
        attn = self.attend(dots)
        # attn : [B, 1, CN, N]

        out = attn @ v
        # out : [B, 1, CN, C]
        
        out = rearrange(out, 'b h m c -> b m (h c)')
        # out : [B, CN, C]

        return v.squeeze(1).transpose(1, 2), z + self.to_out(out)


class Local2Global(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Local2Global, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        # x : [B, C, N]
        # z : [B, CN, C]

        b, m, _ = z.shape
        b, c, _ = x.shape
        
        q = self.to_q(x.transpose(1, 2).unsqueeze(1))
        # q : [B, 1, N, C]

        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        # k, v : [B, 1, CN, C]

        dots = q @ k.transpose(2, 3) * self.scale
        # dots : [B, 1, N, CN]
        
        attn = self.attend(dots)
        # attn : [B, 1, N, CN]

        out = attn @ v
        # out : [B, 1, N, C]

        out = rearrange(out, 'b h l c -> b l (h c)')
        # out : [B, N, C]

        out = self.to_out(out)
        # out : [B, N, C]

        out = out.permute(0, 2, 1)
        # out : [B, C, N]
        
        return x + out


class Aggregation_Transformer_Block(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Aggregation_Transformer_Block, self).__init__()
        self.G2L = Global2Local(dim, heads, channel, dropout)
        self.L2G = Local2Global(dim, heads, channel, dropout)
        self.former1 = Former(dim)
        self.former2 = Former(dim)

    def forward(self, x, z):
        new_x, z = self.G2L(x, z.permute(0, 2, 1))
        z = self.former1(z)
        new_x = self.L2G(new_x, z)
        agg = self.former2(new_x.permute(0, 2, 1)).permute(0, 2, 1)

        return agg


class Feature_Aggregation_Transformer(nn.Module):
    def __init__(self, in_channel):
        super(Feature_Aggregation_Transformer, self).__init__()
        self.in_channel = in_channel
        mid_feature = in_channel
        self.conva = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)

        self.ATB = Aggregation_Transformer_Block(mid_feature, 1, mid_feature)

        self.convn = nn.Conv1d(mid_feature, mid_feature, kernel_size=1, bias=False)
        self.convl = nn.Conv1d(mid_feature, mid_feature, kernel_size=1, bias=False)
        self.convd = nn.Sequential(
                nn.Conv1d(mid_feature * 2, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm1d(in_channel)
                )
        
        self.line_conv_att = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)

    def forward(self, global_tp, local_tp):
        local_att = self.line_conv_att(local_tp)

        local_inter = torch.sum(local_tp * F.softmax(local_att, dim=-1), dim=-1)
        local_inter = self.conva(local_inter)

        global_tp = self.ATB(global_tp, local_inter)

        return F.leaky_relu(global_tp, negative_slope=0.2)