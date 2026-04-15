import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        dx = dy = self.feature_dim // 3
        if dx % 2 == 1:
            dx -= 1
            dy -= 1
        dz = self.feature_dim - dx - dy
        div_term_x = torch.exp(
            torch.arange(0, dx, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dx)
        ).view(1, 1, -1)  # [1, 1, d//6]
        div_term_y = torch.exp(
            torch.arange(0, dy, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dy)
        ).view(1, 1, -1)  # [1, 1, d//6]
        div_term_z = torch.exp(
            torch.arange(0, dz, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / dz)
        ).view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term_x)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term_x)
        siny = torch.sin(y_position * div_term_y)
        cosy = torch.cos(y_position * div_term_y)
        sinz = torch.sin(z_position * div_term_z)
        cosz = torch.cos(z_position * div_term_z)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        return position_code.detach()


class RotaryPositionEncoding4D(RotaryPositionEncoding):
    """4D RoPE encoding (time, x, y, z).

    The head_dim is split into four roughly equal even-sized buckets (dt, dx,
    dy, dz) each encoding one axis.  Any remainder is absorbed into dz.

    Input:  TXYZ  [B, N, 4]  where TXYZ[...,0] is a temporal index (float,
            e.g. trajectory step 0,1,…,T-1) and TXYZ[...,1:4] are metric XYZ
            coordinates.
    Output: [B, N, feature_dim, 2]  stacked (cos, sin) ready for embed_rotary.
    """

    def __init__(self, feature_dim, pe_type="Rotary4D"):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, TXYZ):
        bsize, npoint, _ = TXYZ.shape
        t_pos = TXYZ[..., 0:1]
        x_pos = TXYZ[..., 1:2]
        y_pos = TXYZ[..., 2:3]
        z_pos = TXYZ[..., 3:4]

        dt = self.feature_dim // 4
        if dt % 2 == 1:
            dt -= 1
        remaining = self.feature_dim - dt
        dx = dy = remaining // 3
        if dx % 2 == 1:
            dx -= 1
            dy -= 1
        dz = self.feature_dim - dt - dx - dy

        def _freq(d):
            return torch.exp(
                torch.arange(0, d, 2, dtype=torch.float, device=TXYZ.device)
                * (-math.log(10000.0) / max(d, 2))
            ).view(1, 1, -1)

        sint = torch.sin(t_pos * _freq(dt))
        cost = torch.cos(t_pos * _freq(dt))
        sinx = torch.sin(x_pos * _freq(dx))
        cosx = torch.cos(x_pos * _freq(dx))
        siny = torch.sin(y_pos * _freq(dy))
        cosy = torch.cos(y_pos * _freq(dy))
        sinz = torch.sin(z_pos * _freq(dz))
        cosz = torch.cos(z_pos * _freq(dz))

        sint, cost, sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sint, cost, sinx, cosx, siny, cosy, sinz, cosz],
        )

        position_code = torch.stack(
            [
                torch.cat([cost, cosx, cosy, cosz], dim=-1),
                torch.cat([sint, sinx, siny, sinz], dim=-1),
            ],
            dim=-1,
        )
        return position_code.detach()


class PositionEmbeddingLearnedMLP(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, dim=3, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(dim, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
