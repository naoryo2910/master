import torch
import torch.nn as nn
from torch.nn import functional as F

from .MultiscaleDecomp import MultiscaleDecomp


class MultiscaleDecompImportance(MultiscaleDecomp):

    '''
    MultiscaleDecomp with SNR + image importance fusion.
    '''
    def __init__(self, N=192, opt=None, **kwargs):
        super().__init__(N=N, opt=opt, **kwargs)
        self.alpha_snr = nn.Parameter(torch.tensor(0.5))
        self.alpha_imp = nn.Parameter(torch.tensor(0.5))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def g_a_func(self, x, x_blur=None, denoise=False):
        if denoise:
            if x_blur is None:
                snr_mask = torch.ones_like(x[:, 0:1, :, :])
            else:
                snr_mask = self.get_snr(x, x_blur)
            importance_map = self.get_importance_map(x)

        x = self.g_a_block1(x)
        if denoise:
            local_feature_1 = self.local_feature_extraction_1(x)
            local_feature_1 = self.local_feature_extraction_1_CC(local_feature_1)
            non_local_feature_1 = self.non_local_feature_extraction_1(x)
            non_local_feature_1 = self.non_local_feature_extraction_1_CC(non_local_feature_1)
            h_feature_1 = non_local_feature_1.shape[2]
            w_feature_1 = non_local_feature_1.shape[3]
            mask_1 = F.interpolate(snr_mask, size=[h_feature_1, w_feature_1], mode='nearest')
            imp_1 = F.interpolate(importance_map, size=[h_feature_1, w_feature_1], mode='nearest')

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
            fusion_weight_1 = self.get_fusion_weight(mask_1, imp_1).repeat(1, channel, 1, 1)
            non_local_feature_1_fold = local_feature_1 * fusion_weight_1 + non_local_feature_1_unfold * (1 - fusion_weight_1)
            x = self.feature_fusion_1(x, non_local_feature_1_fold)
        y_inter = x

        x = self.g_a_block2(x)
        if denoise:
            local_feature_2 = self.local_feature_extraction_2(x)
            local_feature_2 = self.local_feature_extraction_2_CC(local_feature_2)
            non_local_feature_2 = self.non_local_feature_extraction_2(x)
            non_local_feature_2 = self.non_local_feature_extraction_2_CC(non_local_feature_2)
            h_feature_2 = non_local_feature_2.shape[2]
            w_feature_2 = non_local_feature_2.shape[3]
            mask_2 = F.interpolate(snr_mask, size=[h_feature_2, w_feature_2], mode='nearest')
            imp_2 = F.interpolate(importance_map, size=[h_feature_2, w_feature_2], mode='nearest')

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
            fusion_weight_2 = self.get_fusion_weight(mask_2, imp_2).repeat(1, channel, 1, 1)
            non_local_feature_2_fold = local_feature_2 * fusion_weight_2 + non_local_feature_2_unfold * (1 - fusion_weight_2)
            x = self.feature_fusion_2(x, non_local_feature_2_fold)
        y = x

        return y_inter, y

    def get_importance_map(self, x):
        x_gray = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114

        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)
        edge_map = self.normalize_map(edge_map)

        kernel_size = 5
        local_mean = F.avg_pool2d(
            x_gray, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, count_include_pad=False
        )
        local_mean_sq = F.avg_pool2d(
            x_gray * x_gray, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, count_include_pad=False
        )
        texture_map = torch.clamp(local_mean_sq - local_mean * local_mean, min=0.0)
        texture_map = self.normalize_map(texture_map)

        importance_map = self.normalize_map(0.5 * edge_map + 0.5 * texture_map)
        return importance_map.float()

    def get_fusion_weight(self, snr_mask, importance_map):
        fusion_weight = self.alpha_snr * snr_mask + self.alpha_imp * importance_map
        return torch.clamp(fusion_weight, min=0.0, max=1.0)

    @classmethod
    def from_state_dict(cls, state_dict, opt=None):
        """Load old MultiscaleDecomp checkpoints while keeping new importance params."""
        N = state_dict["h_a.0.weight"].size(0)
        net = cls(N, opt)
        model_dict = net.state_dict()
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)
        return net

    @staticmethod
    def normalize_map(x):
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        x_flat = x.reshape(batch_size, -1)
        x_min = torch.min(x_flat, dim=1)[0].view(batch_size, 1, 1, 1)
        x_max = torch.max(x_flat, dim=1)[0].view(batch_size, 1, 1, 1)
        x_min = x_min.repeat(1, 1, height, width)
        x_max = x_max.repeat(1, 1, height, width)
        return (x - x_min) / (x_max - x_min + 1e-6)
