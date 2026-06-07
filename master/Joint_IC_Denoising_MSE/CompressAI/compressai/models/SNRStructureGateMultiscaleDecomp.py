import torch
import torch.nn as nn
from torch.nn import functional as F
from .MultiscaleDecomp import MultiscaleDecomp


class MultiScaleStructureGate(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def blur(self, x, kernel_size):
        pad = kernel_size // 2
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, x_gray, snr_mask):
        b3 = self.blur(x_gray, 3)
        b5 = self.blur(x_gray, 5)
        b9 = self.blur(x_gray, 9)

        h3 = torch.abs(x_gray - b3)
        h5 = torch.abs(x_gray - b5)
        h9 = torch.abs(x_gray - b9)

        d35 = torch.abs(h3 - h5)
        d59 = torch.abs(h5 - h9)

        u = torch.cat([snr_mask, h3, h5, h9, d35, d59], dim=1)
        return self.net(u)


class SNRStructureGateMultiscaleDecomp(MultiscaleDecomp):

    '''
    SNR-guided Multi-scale Structure-aware Fusion on top of MultiscaleDecomp.
    '''
    def __init__(self, N=192, opt=None, **kwargs):
        super().__init__(N=N, opt=opt, **kwargs)

        self.structure_gate_1 = MultiScaleStructureGate(hidden=16)
        self.structure_gate_2 = MultiScaleStructureGate(hidden=16)
        self.beta_structure = 0.3

    @classmethod
    def from_state_dict(cls, state_dict, opt=None):
        """Return a new model instance from `state_dict`."""
        N = state_dict["h_a.0.weight"].size(0)
        net = cls(N, opt)
        net.load_state_dict(state_dict, strict=False)
        return net

    def g_a_func(self, x, x_blur=None, denoise=False):
        x_gray = (
            x[:, 0:1, :, :] * 0.299
            + x[:, 1:2, :, :] * 0.587
            + x[:, 2:3, :, :] * 0.114
        )

        if denoise and x_blur is None:
            raise ValueError("x_blur must be provided when denoise=True")

        if x_blur is not None:
            mask = self.get_snr(x, x_blur)

        x = self.g_a_block1(x)
        if denoise:
            # x = self.denoise_module_1(x)
            local_feature_1 = self.local_feature_extraction_1(x)
            local_feature_1 = self.local_feature_extraction_1_CC(local_feature_1)
            non_local_feature_1 = self.non_local_feature_extraction_1(x)
            non_local_feature_1 = self.non_local_feature_extraction_1_CC(non_local_feature_1)
            h_feature_1 = non_local_feature_1.shape[2]
            w_feature_1 = non_local_feature_1.shape[3]
            mask_1 = F.interpolate(mask, size=[h_feature_1, w_feature_1], mode='nearest')
            non_local_feature_1_unfold = F.unfold(non_local_feature_1, kernel_size=4, dilation=1, stride=4, padding=0)
            non_local_feature_1_unfold = non_local_feature_1_unfold.permute(0, 2, 1)
            mask_unfold = F.unfold(mask_1, kernel_size=4, dilation=1, stride=4, padding=0)
            mask_unfold = mask_unfold.permute(0, 2, 1)
            mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
            mask_unfold[mask_unfold <= 0.5] = 0.0
            non_local_feature_1_unfold = self.transformer_enhance_1(non_local_feature_1_unfold, None,
                                                                    src_mask=mask_unfold)
            non_local_feature_1_unfold = non_local_feature_1_unfold.permute(0, 2, 1)
            non_local_feature_1_unfold = nn.Fold(
                output_size=(h_feature_1, w_feature_1),
                kernel_size=(4, 4),
                stride=4,
                padding=0,
                dilation=1)(non_local_feature_1_unfold)

            channel = non_local_feature_1.shape[1]

            x_gray_1 = F.interpolate(
                x_gray,
                size=mask_1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            structure_1 = self.structure_gate_1(x_gray_1, mask_1)

            mask_new_1 = mask_1 + self.beta_structure * (1.0 - mask_1) * structure_1
            mask_new_1 = torch.clamp(mask_new_1, 0.0, 1.0)

            mask_new_1 = mask_new_1.repeat(1, channel, 1, 1)

            non_local_feature_1_fold = (
                non_local_feature_1_unfold * (1.0 - mask_new_1)
                + local_feature_1 * mask_new_1
            )
            x = self.feature_fusion_1(x, non_local_feature_1_fold)
        y_inter = x

        x = self.g_a_block2(x)
        if denoise:
            # x = self.denoise_module_2(x)
            local_feature_2 = self.local_feature_extraction_2(x)
            local_feature_2 = self.local_feature_extraction_2_CC(local_feature_2)
            non_local_feature_2 = self.non_local_feature_extraction_2(x)
            non_local_feature_2 = self.non_local_feature_extraction_2_CC(non_local_feature_2)
            h_feature_2 = non_local_feature_2.shape[2]
            w_feature_2 = non_local_feature_2.shape[3]
            mask_2 = F.interpolate(mask, size=[h_feature_2, w_feature_2], mode='nearest')
            non_local_feature_2_unfold = F.unfold(non_local_feature_2, kernel_size=4, dilation=1, stride=4, padding=0)
            non_local_feature_2_unfold = non_local_feature_2_unfold.permute(0, 2, 1)
            mask_unfold_2 = F.unfold(mask_2, kernel_size=4, dilation=1, stride=4, padding=0)
            mask_unfold_2 = mask_unfold_2.permute(0, 2, 1)
            mask_unfold_2 = torch.mean(mask_unfold_2, dim=2).unsqueeze(dim=-2)
            mask_unfold_2[mask_unfold_2 <= 0.5] = 0.0
            non_local_feature_2_unfold = self.transformer_enhance_2(non_local_feature_2_unfold, None,
                                                                    src_mask=mask_unfold_2)
            non_local_feature_2_unfold = non_local_feature_2_unfold.permute(0, 2, 1)
            non_local_feature_2_unfold = nn.Fold(
                output_size=(h_feature_2, w_feature_2),
                kernel_size=(4, 4),
                stride=4,
                padding=0,
                dilation=1)(non_local_feature_2_unfold)

            channel = non_local_feature_2.shape[1]

            x_gray_2 = F.interpolate(
                x_gray,
                size=mask_2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            structure_2 = self.structure_gate_2(x_gray_2, mask_2)

            mask_new_2 = mask_2 + self.beta_structure * (1.0 - mask_2) * structure_2
            mask_new_2 = torch.clamp(mask_new_2, 0.0, 1.0)

            mask_new_2 = mask_new_2.repeat(1, channel, 1, 1)

            non_local_feature_2_fold = (
                non_local_feature_2_unfold * (1.0 - mask_new_2)
                + local_feature_2 * mask_new_2
            )
            x = self.feature_fusion_2(x, non_local_feature_2_fold)
        y = x

        return y_inter, y
