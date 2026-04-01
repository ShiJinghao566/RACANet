"""Local anchor fusion module."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Modules import BasicConv2d


class LAFM(nn.Module):
    """Fuse one RGB/T stage and return an auxiliary loss."""

    def __init__(
        self,
        in_channels=64,
        embed_dim=32,
        anchor_kernel=3,
        local_window=3,
        lambda_p=1.0,
        eps=1e-6,
    ):
        super().__init__()
        if anchor_kernel % 2 == 0 or local_window % 2 == 0:
            raise ValueError("anchor_kernel and local_window must be odd.")

        self.embed_dim = embed_dim
        self.anchor_kernel = anchor_kernel
        self.local_window = local_window
        self.lambda_p = lambda_p
        self.eps = eps

        # Stage A: projection, prior, and reliability.
        self.rgb_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.t_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

        self.prior_head = nn.Sequential(
            BasicConv2d(embed_dim * 4, embed_dim, kernel_size=3, padding=1),
            nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True),
        )
        self.rgb_rel_head = nn.Sequential(
            BasicConv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True),
        )
        self.t_rel_head = nn.Sequential(
            BasicConv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True),
        )
        self.divergence_head = nn.Sequential(
            BasicConv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True),
        )

        # Stage C: anchor-guided local attention.
        self.query_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.scale = math.sqrt(embed_dim)

        # Output projection.
        self.j_to_out = BasicConv2d(embed_dim, in_channels, kernel_size=1)
        self.z_to_out = BasicConv2d(embed_dim, in_channels, kernel_size=1)
        self.guided_fuse = nn.Sequential(
            BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.out_fuse = nn.Sequential(
            BasicConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

    def _build_local_anchors(self, shared_feat, prior, rel_rgb, rel_t):
        """Build crowd-aware local anchors."""
        b, d, h, w = shared_feat.shape
        k = self.anchor_kernel

        weights = prior * (rel_rgb + rel_t) * 0.5  # [B,1,H,W]
        weighted_feat = shared_feat * weights

        feat_unfold = F.unfold(weighted_feat, kernel_size=k, padding=k // 2)
        feat_unfold = feat_unfold.view(b, d, k * k, h * w)
        anchor_num = feat_unfold.sum(dim=2)  # [B,D,HW]

        w_unfold = F.unfold(weights, kernel_size=k, padding=k // 2)
        w_sum = w_unfold.sum(dim=1, keepdim=True)  # [B,1,HW]

        anchors = (anchor_num / (w_sum + self.eps)).view(b, d, h, w)
        crowdness = F.avg_pool2d(prior, kernel_size=k, stride=1, padding=k // 2)
        return anchors, crowdness

    def _anchor_redistribute(self, shared_feat, anchors, crowdness):
        """Redistribute pixels over local anchors."""
        b, d, h, w = shared_feat.shape
        win = self.local_window

        q = self.query_proj(shared_feat).view(b, d, h * w).unsqueeze(2)  # [B,D,1,HW]
        k_map = self.key_proj(anchors)
        v_map = self.value_proj(anchors)

        k_unfold = F.unfold(k_map, kernel_size=win, padding=win // 2)
        v_unfold = F.unfold(v_map, kernel_size=win, padding=win // 2)
        k_unfold = k_unfold.view(b, d, win * win, h * w)
        v_unfold = v_unfold.view(b, d, win * win, h * w)

        crowd_unfold = F.unfold(crowdness, kernel_size=win, padding=win // 2)
        crowd_bias = self.lambda_p * torch.log(crowd_unfold.clamp_min(self.eps))

        score = (q * k_unfold).sum(dim=1) / self.scale + crowd_bias
        assign = torch.softmax(score, dim=1)

        redistributed = (assign.unsqueeze(1) * v_unfold).sum(dim=2)
        redistributed = redistributed.view(b, d, h, w)
        return redistributed

    def forward(self, feat_rgb, feat_t, top_down_feat=None):
        q_rgb = self.rgb_proj(feat_rgb)
        q_t = self.t_proj(feat_t)

        prior_in = torch.cat([q_rgb, q_t, torch.abs(q_rgb - q_t), q_rgb * q_t], dim=1)
        prior = torch.sigmoid(self.prior_head(prior_in))

        rel_rgb = torch.sigmoid(self.rgb_rel_head(torch.cat([q_rgb, prior], dim=1)))
        rel_t = torch.sigmoid(self.t_rel_head(torch.cat([q_t, prior], dim=1)))

        shared = rel_rgb * q_rgb + rel_t * q_t

        divergence = torch.sigmoid(self.divergence_head(torch.abs(q_rgb - q_t)))
        consistency_loss = ((1.0 - divergence) * torch.abs(rel_rgb - rel_t)).mean()

        anchors, crowdness = self._build_local_anchors(shared, prior, rel_rgb, rel_t)
        redistributed = self._anchor_redistribute(shared, anchors, crowdness)

        shared_out = self.j_to_out(shared)
        redist_out = self.z_to_out(redistributed)

        if top_down_feat is not None:
            top_down = F.interpolate(
                top_down_feat,
                size=shared.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            guided = self.guided_fuse(torch.cat([shared_out, top_down], dim=1))
        else:
            guided = shared_out

        fused = self.out_fuse(torch.cat([redist_out, guided], dim=1))
        aux = {
            "consistency_loss": consistency_loss,
            "prior": prior,
            "rel_rgb": rel_rgb,
            "rel_t": rel_t,
            "divergence": divergence,
        }
        return fused, aux
