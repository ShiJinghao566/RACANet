"""RACANet warm-up model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LAFM import LAFM
from models.Modules import BasicConv2d
from models.pvt_v2_encoders import pvt_v2_b3


class LocalSoftMatcher(nn.Module):
    """Perform local query-key soft matching."""

    def __init__(self, embed_dim=32, window_size=3):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        self.window_size = window_size
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.scale = math.sqrt(embed_dim)

    def forward(self, query_feat, context_feat):
        b, d, h, w = query_feat.shape
        k = self.window_size

        query = self.q_proj(query_feat).view(b, d, h * w).unsqueeze(2)  # [B,D,1,HW]
        key = self.k_proj(context_feat)

        key_unfold = F.unfold(key, kernel_size=k, padding=k // 2).view(b, d, k * k, h * w)
        value_unfold = F.unfold(context_feat, kernel_size=k, padding=k // 2).view(b, d, k * k, h * w)

        score = (query * key_unfold).sum(dim=1) / self.scale  # [B,K,HW]
        attn = torch.softmax(score, dim=1)

        aligned = (attn.unsqueeze(1) * value_unfold).sum(dim=2)
        aligned = aligned.view(b, d, h, w)
        return aligned


class RACANetWarmup(nn.Module):
    """Warm-up network."""

    def __init__(self, fusion_embed_dim=32, local_match_window=3):
        super().__init__()
        self.pvt_backbone_rgb = pvt_v2_b3(pretrained="pretrained_weights/pvt_v2_b3.pth")
        self.pvt_backbone_t = pvt_v2_b3(pretrained="pretrained_weights/pvt_v2_b3.pth")

        self.trans_rgb_32x = BasicConv2d(512, 64, 1)
        self.trans_rgb_16x = BasicConv2d(320, 64, 1)
        self.trans_rgb_8x = BasicConv2d(128, 64, 1)
        self.trans_rgb_4x = BasicConv2d(64, 64, 1)

        self.trans_t_32x = BasicConv2d(512, 64, 1)
        self.trans_t_16x = BasicConv2d(320, 64, 1)
        self.trans_t_8x = BasicConv2d(128, 64, 1)
        self.trans_t_4x = BasicConv2d(64, 64, 1)

        # Match names used in the full model for weight transfer.
        self.lafm_32x = LAFM(in_channels=64, embed_dim=fusion_embed_dim)
        self.lafm_16x = LAFM(in_channels=64, embed_dim=fusion_embed_dim)
        self.lafm_8x = LAFM(in_channels=64, embed_dim=fusion_embed_dim)
        self.lafm_4x = LAFM(in_channels=64, embed_dim=fusion_embed_dim)

        self.matcher_32x = LocalSoftMatcher(embed_dim=fusion_embed_dim, window_size=local_match_window)
        self.matcher_16x = LocalSoftMatcher(embed_dim=fusion_embed_dim, window_size=local_match_window)
        self.matcher_8x = LocalSoftMatcher(embed_dim=fusion_embed_dim, window_size=local_match_window)
        self.matcher_4x = LocalSoftMatcher(embed_dim=fusion_embed_dim, window_size=local_match_window)

    @staticmethod
    def _predict_candidate(stage_module, feat_rgb, feat_t):
        q_rgb = stage_module.rgb_proj(feat_rgb)
        q_t = stage_module.t_proj(feat_t)
        prior_in = torch.cat([q_rgb, q_t, torch.abs(q_rgb - q_t), q_rgb * q_t], dim=1)
        c_logits = stage_module.prior_head(prior_in)
        c_map = torch.sigmoid(c_logits)
        return q_rgb, q_t, c_logits, c_map

    def _forward_stage(self, stage_module, matcher_module, feat_rgb, feat_t):
        q_rgb, q_t, c_logits, c_map = self._predict_candidate(stage_module, feat_rgb, feat_t)
        q_t_aligned = matcher_module(q_rgb, q_t)  # r -> t
        q_r_aligned = matcher_module(q_t, q_rgb)  # t -> r
        return {
            "q_rgb": q_rgb,
            "q_t": q_t,
            "c_logits": c_logits,
            "c_map": c_map,
            "q_t_aligned": q_t_aligned,
            "q_r_aligned": q_r_aligned,
        }

    def forward(self, rgbt_inputs):
        rgb, t_img = rgbt_inputs[0], rgbt_inputs[1]

        rgb_4x = self.pvt_backbone_rgb.forward_stage1(rgb)
        rgb_8x = self.pvt_backbone_rgb.forward_stage2(rgb_4x)
        rgb_16x = self.pvt_backbone_rgb.forward_stage3(rgb_8x)
        rgb_32x = self.pvt_backbone_rgb.forward_stage4(rgb_16x)

        t_4x = self.pvt_backbone_t.forward_stage1(t_img)
        t_8x = self.pvt_backbone_t.forward_stage2(t_4x)
        t_16x = self.pvt_backbone_t.forward_stage3(t_8x)
        t_32x = self.pvt_backbone_t.forward_stage4(t_16x)

        trans_rgb_4x = self.trans_rgb_4x(rgb_4x)
        trans_rgb_8x = self.trans_rgb_8x(rgb_8x)
        trans_rgb_16x = self.trans_rgb_16x(rgb_16x)
        trans_rgb_32x = self.trans_rgb_32x(rgb_32x)

        trans_t_4x = self.trans_t_4x(t_4x)
        trans_t_8x = self.trans_t_8x(t_8x)
        trans_t_16x = self.trans_t_16x(t_16x)
        trans_t_32x = self.trans_t_32x(t_32x)

        stage32 = self._forward_stage(self.lafm_32x, self.matcher_32x, trans_rgb_32x, trans_t_32x)
        stage16 = self._forward_stage(self.lafm_16x, self.matcher_16x, trans_rgb_16x, trans_t_16x)
        stage8 = self._forward_stage(self.lafm_8x, self.matcher_8x, trans_rgb_8x, trans_t_8x)
        stage4 = self._forward_stage(self.lafm_4x, self.matcher_4x, trans_rgb_4x, trans_t_4x)

        return {
            "32x": stage32,
            "16x": stage16,
            "8x": stage8,
            "4x": stage4,
        }


def build_racanet_warmup(fusion_embed_dim=32, local_match_window=3):
    return RACANetWarmup(
        fusion_embed_dim=fusion_embed_dim,
        local_match_window=local_match_window,
    )
