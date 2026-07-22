import torch
from torch import nn
import torch.nn.functional as F

class LocalGlobalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, local_kernel_sizes=[3,5,7], global_kernel_size=11,
                 height=14, width=14, num_scales=3):
        super(LocalGlobalAttention, self).__init__()

        # 自动调整 num_heads 使其能整除 embed_dim
        if embed_dim % num_heads != 0:
            for nh in [8, 4, 2, 1]:
                if embed_dim % nh == 0:
                    num_heads = nh
                    break
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.num_scales = num_scales

        # 多尺度卷积
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=2**i, groups=embed_dim),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
            ) for i in range(num_scales)
        ])

        # 局部卷积注意力
        self.local_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=k, padding=k//2, groups=embed_dim),
                nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1, groups=self.num_heads)
            ) for k in local_kernel_sizes
        ])

        # 全局卷积注意力
        self.global_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=global_kernel_size, padding=global_kernel_size//2,
                      groups=embed_dim),
            nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1, groups=self.num_heads)
        )

        # 可学习位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, embed_dim, height, width))

        # 输出卷积
        self.out_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # 多尺度权重
        self.scale_attention = nn.Sequential(
            nn.Conv2d(embed_dim, num_scales+1, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.learnable_alpha_local = nn.Parameter(torch.tensor(0.5))
        self.learnable_alpha_global = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.size()

        # 分头与合并
        def split_heads(tensor):
            return tensor.view(B, self.num_heads, self.head_dim, -1)

        def combine_heads(tensor):
            return tensor.view(B, C, H, W)

        # 位置编码 resize
        pos_enc = F.interpolate(self.position_encoding, size=(H, W), mode='nearest')

        # 多尺度卷积特征融合
        multi_scale_features = [x]
        for scale_conv in self.multi_scale_convs:
            scaled = scale_conv(x)
            scaled = F.interpolate(scaled, size=(H, W), mode='nearest')
            scaled += x
            multi_scale_features.append(scaled)

        scale_weights = self.scale_attention(x)
        x_multi = sum(scale_weights[:, i:i+1, :, :] * multi_scale_features[i]
                      for i in range(self.num_scales+1))

        # 局部注意力
        local_outs = []
        for conv in self.local_convs:
            q, k, v = conv(x_multi + pos_enc).chunk(3, dim=1)
            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            energy = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
            attn = self.softmax(energy)
            local_out = torch.einsum('bhqk,bhvd->bhqd', attn, v)
            local_out = combine_heads(local_out) + x
            local_outs.append(local_out)
        local_out = sum(local_outs) / len(local_outs)

        # 全局注意力
        q, k, v = self.global_conv(x_multi + pos_enc).chunk(3, dim=1)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        energy = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        attn = self.softmax(energy)
        global_out = torch.einsum('bhqk,bhvd->bhqd', attn, v)
        global_out = combine_heads(global_out) + x

        # 融合局部和全局注意力
        out = self.learnable_alpha_local * local_out + self.learnable_alpha_global * global_out
        out = self.out_conv(out)
        return out
Attention = LocalGlobalAttention