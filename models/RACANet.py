"""RACANet with LAFM fusion."""

import torch
import torch.nn as nn
from models.Modules import BasicConv2d, FeatureFusionAndPrediction
from models.LAFM import LAFM
from models.pvt_v2_encoders import pvt_v2_b3



class RACANet(nn.Module):
    def __init__(
        self,
        fusion_embed_dim=32,
        anchor_kernel=3,
        local_window=3,
        lambda_p=1.0,
    ):
        super(RACANet, self).__init__()

        # Backbone.
        self.pvt_backbone_rgb = pvt_v2_b3 (pretrained="pretrained_weights/pvt_v2_b3.pth")
        self.pvt_backbone_t   = pvt_v2_b3 (pretrained="pretrained_weights/pvt_v2_b3.pth")
        
        # Project all stages to 64 channels.
        self.trans_rgb_32x = BasicConv2d(512, 64, 1)
        self.trans_rgb_16x = BasicConv2d(320, 64, 1)
        self.trans_rgb_8x  = BasicConv2d(128, 64, 1)
        self.trans_rgb_4x  = BasicConv2d(64,  64, 1)
        
        self.trans_t_32x = BasicConv2d(512, 64, 1)
        self.trans_t_16x = BasicConv2d(320, 64, 1)
        self.trans_t_8x  = BasicConv2d(128, 64, 1)
        self.trans_t_4x  = BasicConv2d(64,  64, 1)

        # LAFM fusion.
        self.lafm_32x = LAFM(
            in_channels=64,
            embed_dim=fusion_embed_dim,
            anchor_kernel=anchor_kernel,
            local_window=local_window,
            lambda_p=lambda_p,
        )
        self.lafm_16x = LAFM(
            in_channels=64,
            embed_dim=fusion_embed_dim,
            anchor_kernel=anchor_kernel,
            local_window=local_window,
            lambda_p=lambda_p,
        )
        self.lafm_8x = LAFM(
            in_channels=64,
            embed_dim=fusion_embed_dim,
            anchor_kernel=anchor_kernel,
            local_window=local_window,
            lambda_p=lambda_p,
        )
        self.lafm_4x = LAFM(
            in_channels=64,
            embed_dim=fusion_embed_dim,
            anchor_kernel=anchor_kernel,
            local_window=local_window,
            lambda_p=lambda_p,
        )
        
        # Density head.
        self.fdm = FeatureFusionAndPrediction()  
    
    def forward(self, rgbt_inputs, return_aux=False):
        """Return a downsampled density map for `[rgb, t]`."""

        rgb, t_img = rgbt_inputs[0], rgbt_inputs[1]
        
        # Backbone features.
        rgb_4x  = self.pvt_backbone_rgb.forward_stage1(rgb)
        rgb_8x  = self.pvt_backbone_rgb.forward_stage2(rgb_4x)
        rgb_16x = self.pvt_backbone_rgb.forward_stage3(rgb_8x)
        rgb_32x = self.pvt_backbone_rgb.forward_stage4(rgb_16x)
        
        t_4x  = self.pvt_backbone_t.forward_stage1(t_img)
        t_8x  = self.pvt_backbone_t.forward_stage2(t_4x)
        t_16x = self.pvt_backbone_t.forward_stage3(t_8x)
        t_32x = self.pvt_backbone_t.forward_stage4(t_16x)
        
        # Project all stages to 64 channels.
        trans_rgb_4x  = self.trans_rgb_4x(rgb_4x)
        trans_rgb_8x  = self.trans_rgb_8x(rgb_8x)
        trans_rgb_16x = self.trans_rgb_16x(rgb_16x)
        trans_rgb_32x = self.trans_rgb_32x(rgb_32x)
        
        trans_t_4x  = self.trans_t_4x(t_4x)
        trans_t_8x  = self.trans_t_8x(t_8x)
        trans_t_16x = self.trans_t_16x(t_16x)
        trans_t_32x = self.trans_t_32x(t_32x)

        # LAFM fusion.
        fused_32x, aux_32 = self.lafm_32x(trans_rgb_32x, trans_t_32x, top_down_feat=None)
        fused_16x, aux_16 = self.lafm_16x(trans_rgb_16x, trans_t_16x, top_down_feat=fused_32x)
        fused_8x, aux_8 = self.lafm_8x(trans_rgb_8x, trans_t_8x, top_down_feat=fused_16x)
        fused_4x, aux_4 = self.lafm_4x(trans_rgb_4x, trans_t_4x, top_down_feat=fused_8x)

        # Density prediction.
        final_pred = self.fdm(fused_4x, fused_8x, fused_16x, fused_32x)

        if return_aux:
            stage_losses = torch.stack(
                [
                    aux_32["consistency_loss"],
                    aux_16["consistency_loss"],
                    aux_8["consistency_loss"],
                    aux_4["consistency_loss"],
                ],
                dim=0,
            )
            return final_pred, {
                "stage_consistency_losses": stage_losses,
                "consistency_loss": stage_losses.sum(),
            }

        return final_pred


def build_racanet(
    fusion_embed_dim=32,
    anchor_kernel=3,
    local_window=3,
    lambda_p=1.0,
):
    model = RACANet(
        fusion_embed_dim=fusion_embed_dim,
        anchor_kernel=anchor_kernel,
        local_window=local_window,
        lambda_p=lambda_p,
    )
    return model
        

