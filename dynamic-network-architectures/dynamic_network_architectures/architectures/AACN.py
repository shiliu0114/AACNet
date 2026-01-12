from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
import numpy as np
# class AnatomicalAttentionSEBlock(nn.Module):
#     """
#     SDF-guided Anatomical Attention Squeeze-and-Excitation (SAA-SE) Block.

#     This module enhances the standard SE block by incorporating an anatomical prior
#     in the form of a Signed Distance Function (SDF). Instead of using Global
#     Average Pooling (GAP), which is spatially agnostic, it employs
#     Anatomical Weighted Pooling (AWP). AWP uses a spatial attention map derived
#     from the SDF to force the network to focus on features near anatomical
#     boundaries when computing channel-wise attention.

#     Args:
#         channels (int): Number of input channels.
#         reduction (int): Reduction ratio for the bottleneck in the excitation step.
#     """
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
#         # The Excitation part of the SE block remains the same.
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x: torch.Tensor, sdf_boundary_map: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
#         """
#         Forward pass of the SAA-SE block.

#         Args:
#             x (torch.Tensor): Input feature map. Shape: (B, C, D, H, W).
#             sdf_boundary_map (torch.Tensor): The pre-computed and fused boundary SDF map,
#                                              representing the distance to the nearest
#                                              tooth surface. Shape should be broadcastable
#                                              to x, e.g., (B, 1, D, H, W). It must be
#                                              resized to the same spatial dimensions as x
#                                              before being passed to this module.
#             tau (float): Temperature hyperparameter to control the sharpness of the
#                          spatial attention map. Default: 1.0.

#         Returns:
#             torch.Tensor: The recalibrated feature map. Shape: (B, C, D, H, W).
#         """
#         b, c, _, _, _ = x.size()

#         # Step 1: Generate Spatial Attention Map from the fused SDF boundary map.
#         # This map highlights regions near the anatomical surfaces (where SDF is close to 0).
#         # The map M_spatial will have values between (0, 1].
#         M_spatial = torch.exp(-torch.abs(sdf_boundary_map) / tau)

#         # Step 2: Anatomical Weighted Pooling (The core innovation).
#         # This replaces the standard Global Average Pooling.
#         # We compute the weighted average of features, guided by M_spatial.

#         # Numerator: Element-wise product of features and spatial weights, summed over spatial dimensions.
#         numerator = torch.sum(x * M_spatial, dim=(2, 3, 4))  # Shape: (B, C)

#         # Denominator: Sum of the spatial weights. Add a small epsilon for numerical stability.
#         denominator = torch.sum(M_spatial, dim=(2, 3, 4)) + 1e-6  # Shape: (B, 1)

#         # The new channel descriptor, z', which is aware of anatomical locations.
#         z_prime = numerator / denominator  # Shape: (B, C)

#         # Step 3: Excitation - Same as standard SE.
#         # Learns channel-wise dependencies from the anatomy-aware descriptor.
#         s = self.fc(z_prime)  # Shape: (B, C)
        
#         # Reshape for broadcasting.
#         s = s.view(b, c, 1, 1, 1) # Shape: (B, C, 1, 1, 1)

#         # Step 4: Recalibration - Same as standard SE.
#         # The original feature map is scaled by the learned channel attention weights.
#         x_recalibrated = x * s

#         return x_recalibrated
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomicalAttentionSEBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 reduction: int = 16,
                 norm_mode: str = "minmax",   # "minmax", "zscore", or "none"
                 use_adapter: bool = True,    # 1x1 conv adapter for sdf -> C channels
                 adapter_channels: int = None,
                 learnable_tau: bool = True,
                 learnable_scale_shift: bool = True
                 ):
        super().__init__()
        assert norm_mode in ("minmax", "zscore", "none")
        self.channels = channels
        self.norm_mode = norm_mode
        self.use_adapter = use_adapter
        self.adapter_channels = adapter_channels or channels
        self.learnable_tau = learnable_tau
        self.learnable_scale_shift = learnable_scale_shift

        # Excitation (SE)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # Optional adapter to map sdf map into channel space (learnable)
        if self.use_adapter:
            # map 1-channel sdf -> adapter_channels, then to channels if needed
            self.adapter = nn.Sequential(
                nn.Conv3d(1, self.adapter_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(self.adapter_channels),
                nn.ReLU(inplace=True)
            )
            if self.adapter_channels != self.channels:
                self.adapter_proj = nn.Conv3d(self.adapter_channels, self.channels, kernel_size=1, bias=False)
            else:
                self.adapter_proj = None
        else:
            self.adapter = None
            self.adapter_proj = None

        # learnable temperature tau (positive enforced via softplus)
        if self.learnable_tau:
            self._tau_param = nn.Parameter(torch.tensor(1.0))  # raw param
        else:
            self.register_buffer("_tau_const", torch.tensor(1.0))

        # optional learnable scale & shift applied to normalized sdf before exp
        if self.learnable_scale_shift:
            # initialized to identity (scale=1, shift=0)
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.shift = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scale", torch.tensor(1.0))
            self.register_buffer("shift", torch.tensor(0.0))

    def _get_tau(self):
        if self.learnable_tau:
            # ensure positive: softplus
            return F.softplus(self._tau_param) + 1e-6
        else:
            return self._tau_const

    def _normalize_sdf(self, sdf: torch.Tensor):
        """
        sdf: (B, 1, D, H, W) or broadcastable
        returns normalized sdf in roughly [0,1] (for minmax) or zscore
        """
        if self.norm_mode == "none":
            return sdf
        elif self.norm_mode == "minmax":
            # per-sample minmax (B-wise), keep spatial dims
            B = sdf.shape[0]
            sdf_flat = sdf.view(B, -1)
            min_val = sdf_flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
            max_val = sdf_flat.max(dim=1)[0].view(B, 1, 1, 1, 1)
            denom = (max_val - min_val)
            denom[denom == 0] = 1.0
            sdf_norm = (sdf - min_val) / denom
            return sdf_norm
        elif self.norm_mode == "zscore":
            B = sdf.shape[0]
            sdf_flat = sdf.view(B, -1)
            mean = sdf_flat.mean(dim=1).view(B, 1, 1, 1, 1)
            std = sdf_flat.std(dim=1).view(B, 1, 1, 1, 1)
            std[std == 0] = 1.0
            sdf_norm = (sdf - mean) / std
            # optionally clamp to [-3, 3] to avoid extreme values
            return torch.clamp(sdf_norm, -3.0, 3.0)
        else:
            return sdf

    def forward(self, x: torch.Tensor, sdf_boundary_map: torch.Tensor, tau: float = None,store_ms:bool = False) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        sdf_boundary_map: spatial map shape (B,1,D,H,W) or broadcastable
        """
        b, c, _, _, _ = x.shape

        # Ensure sdf has shape (B,1,D,H,W)
        if sdf_boundary_map.dim() == 4:
            sdf_boundary_map = sdf_boundary_map.unsqueeze(1)  # (B,1,D,H,W)

        # 1) normalize sdf per chosen mode
        sdf_norm = self._normalize_sdf(sdf_boundary_map)

        # 2) optional learned affine (scale + shift) applied to normalized sdf
        sdf_affine = sdf_norm * self.scale + self.shift  # these are learnable or buffers

        # 3) optional adapter: map sdf -> channels (so it can be combined or used as M_spatial)
        if self.use_adapter:
            # adapter expects shape (B,1,D,H,W)
            sdf_mapped = self.adapter(sdf_affine)
            if self.adapter_proj is not None:
                sdf_mapped = self.adapter_proj(sdf_mapped)  # now (B,C,D,H,W)
            # If adapter output channels == C, we can compute M_spatial from its magnitude
            # Here compute a single-channel spatial weight by pooling across channel dim
            M_spatial = torch.mean(torch.abs(sdf_mapped), dim=1, keepdim=True)  # (B,1,D,H,W)
            # normalize M_spatial to [0,1] per-sample for stability
            M_flat = M_spatial.view(b, -1)
            mn = M_flat.min(dim=1)[0].view(b,1,1,1,1)
            mx = M_flat.max(dim=1)[0].view(b,1,1,1,1)
            denom = mx - mn
            denom[denom == 0] = 1.0
            M_spatial = (M_spatial - mn) / denom
            # now M_spatial in [0,1]
        else:
            # No adapter: compute spatial weight directly from sdf_affine using exp
            cur_tau = tau if tau is not None else self._get_tau()
            # use absolute value since sdf is distance; ensure positive tau
            M_spatial = torch.exp(-torch.abs(sdf_affine) / (cur_tau + 1e-6))
            # optionally clamp small values
            M_spatial = torch.clamp(M_spatial, min=1e-6, max=1.0)

        # 4) Anatomical Weighted Pooling (AWP)
        numerator = torch.sum(x * M_spatial, dim=(2,3,4))          # (B, C)
        denominator = torch.sum(M_spatial, dim=(2,3,4)).clamp(min=1e-6)  # (B,1)
        z_prime = numerator / denominator                          # (B, C)

        # 5) Excitation
        s = self.fc(z_prime)   # (B, C)
        s = s.view(b, c, 1, 1, 1)

        # 6) Recalibration
        x_recalibrated = x * s
        # M_spatial is (B,1,D,H,W)

        ####调试
        # mp = M_spatial.detach().cpu()
        # print("M_spatial stats:", mp.min().item(), mp.mean().item(), mp.max().item())
        if store_ms:
            m_spatial_numpy = M_spatial.detach().cpu().numpy()
            # 从批处理中选择第一个样本进行保存
            # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
            first_sample_m_spatial = m_spatial_numpy[0, 0, :, :, :]

            # 使用np.save保存
            file_path = "sdf_feature/m_spatial_output.npy"
            np.save(file_path, first_sample_m_spatial)
        ####
        return x_recalibrated # also return M_spatial for monitoring/logging

# AGBR 模块
class AmbiguityGatedBoundaryRefiner(nn.Module):
    def __init__(self, in_channels, refine_channels=32):
        super().__init__()
        self.refiner = nn.Sequential(
        nn.Conv3d(in_channels, refine_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(refine_channels, refine_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(refine_channels, in_channels, 1)
        )
    def forward(self, features, P_upper, P_lower, alpha=1.0, tau=0.3):
        P_upper = F.interpolate(P_upper,size=features.shape[2:], mode='trilinear',align_corners=False)
        P_lower = F.interpolate(P_lower,size=features.shape[2:], mode='trilinear',align_corners=False)
        P_fg = (P_upper + P_lower) / 2.0
        A = 4 * P_fg * (1 - P_fg)
        A = torch.clamp(A, 0, 1)              # 限制到 [0,1]
        mask = ((A < 0.99) & (A > 0.94)).float()  # 创建掩码并转换为 float32  # 阈值化成二值 {0,1}
        # A = 1 - torch.abs(P_upper - P_lower) # 歧义度
        # A = torch.clamp(A, 0, 1)
        # mask = (A > tau).float()
        F_ref = self.refiner(features)
        out = features + alpha * mask * F_ref

        return out, mask


class AACNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
    ):
        super().__init__()

        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = ("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0")

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
        self.aab1 = AnatomicalAttentionSEBlock(channels=features_per_stage[1])
        self.aab2 = AnatomicalAttentionSEBlock(channels=features_per_stage[2])
        self.aab3 = AnatomicalAttentionSEBlock(channels=features_per_stage[3])
        self.aab4 = AnatomicalAttentionSEBlock(channels=features_per_stage[4])
        self.argb = AmbiguityGatedBoundaryRefiner(320)
        self.conv = nn.Conv3d(320, 320, kernel_size=[2,2,2],stride=[2,2,2], padding=0)
        # 1x1x1卷积，相当于通道级别的线性变换
        self.adapter = nn.Conv3d(2, 2, kernel_size=1)
    def forward(self, x, sdf):
        ##########
        x_in = x[:,0:1,:,:,:]
        x_in = x_in.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        # x_in = x_in[0, 0, :, :, :]
        # 使用np.save保存
        # file_path = "x_in.npy"
        # np.save(file_path, x_in)
        #####for argb
        P_upper = x[:,1:2,:,:,:]
        P_lower = x[:,2:3,:,:,:]
        #####预处理概率图和CBCT到同一个区间
        x[:,1:3,:,:,:] = self.adapter(x[:,1:3,:,:,:])  
        #####encoder
        skips = self.encoder(x)
        #####******skip connection*****######
        sdf1 = F.interpolate(sdf,size=skips[1].shape[2:],mode='trilinear',align_corners=False)
        sdf2 = F.interpolate(sdf,size=skips[2].shape[2:],mode='trilinear',align_corners=False)
        sdf3 = F.interpolate(sdf,size=skips[3].shape[2:],mode='trilinear',align_corners=False)
        sdf4 = F.interpolate(sdf,size=skips[4].shape[2:],mode='trilinear',align_corners=False)
        ##############
        skip1_before = skips[1]
        skip1_before = skip1_before.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        skip1_before = skip1_before[0, :, :, :, :]
        # 使用np.save保存
        file_path = "sdf_feature/skip1_before.npy"
        np.save(file_path, skip1_before)
        ##############****************************************************************
        skips[1] = self.aab1(skips[1],sdf1,store_ms=True)
        ##############****************************************************************
        skip1_after = skips[1]
        skip1_after = skip1_after.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        skip1_after = skip1_after[0, :, :, :, :]
        # 使用np.save保存
        file_path = "sdf_feature/skip1_after.npy"
        np.save(file_path, skip1_after)
        print("保存完毕")
        ##############
        skip2_before = skips[2]
        skip2_before = skip2_before.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        skip2_before = skip2_before[0, :, :, :, :]
        # 使用np.save保存
        file_path = "sdf_feature/skip2_before.npy"
        np.save(file_path, skip2_before)
        ##############****************************************************************
        skips[2] = self.aab2(skips[2],sdf2)
        ##############****************************************************************
        skip2_after = skips[2]
        skip2_after = skip2_after.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        skip2_after = skip2_after[0, :, :, :, :]
        # 使用np.save保存
        file_path = "sdf_feature/skip2_after.npy"
        np.save(file_path, skip2_after)
        ##############****************************************************************
        skip3_before = skips[3]
        skip3_before = skip3_before.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        skip3_before = skip3_before[0, :, :, :, :]
        # 使用np.save保存
        file_path = "sdf_feature/skip3_before.npy"
        np.save(file_path, skip3_before)
        ##############****************************************************************
        skips[3] = self.aab3(skips[3],sdf3)
        ##############****************************************************************
        skip3_after = skips[3]
        skip3_after = skip3_after.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        skip3_after = skip3_after[0, :, :, :, :]
        # 使用np.save保存
        file_path = "sdf_feature/skip3_after.npy"
        np.save(file_path, skip3_after)
        ##############****************************************************************
        skip4_before = skips[4]
        skip4_before = skip4_before.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        skip4_before = skip4_before[0, :, :, :, :]
        # 使用np.save保存
        file_path = "sdf_feature/skip4_before.npy"
        np.save(file_path, skip4_before)
        ##############****************************************************************
        skips[4] = self.aab4(skips[4],sdf4)
        ##############****************************************************************
        skip4_after = skips[4]
        skip4_after = skip4_after.detach().cpu().numpy()
        # 从批处理中选择第一个样本进行保存
        # m_spatial_numpy 的形状是 (B, 1, D, H, W)，我们保存 (D, H, W)
        skip4_after = skip4_after[0, :, :, :, :]
        # 使用np.save保存
        file_path = "sdf_feature/skip4_after.npy"
        np.save(file_path, skip4_after)
        ######******backneck*****#######
        out, mask = self.argb(skips[-2], P_upper, P_lower)
        skips[-1] = self.conv(out)
        del P_lower,P_upper,mask,out
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
