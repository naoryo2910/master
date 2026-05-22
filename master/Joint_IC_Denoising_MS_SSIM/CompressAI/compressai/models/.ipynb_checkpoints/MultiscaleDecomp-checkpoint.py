import torch
import torch.nn as nn
from torch.nn import functional as F
from .waseda import Cheng2020Anchor
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    conv1x1
)
from .transformer.Models import Encoder_patch66
from . import utils
import functools
import warnings

class MultiscaleDecomp(Cheng2020Anchor):
    def __init__(self, N=192, opt=None, **kwargs):
        super().__init__(N=N, **kwargs)
        self.g_a = None
        self.g_a_block1 = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
        )
        self.g_a_block2 = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )
        self.front_RBs = 1
        self.local_RBs = 2
        self.n_layers = 6
        self.n_head = 12
        self.trans_nf = N
        # self.denoise_module_1 = AttentionBlock(N)
        # self.denoise_module_2 = AttentionBlock(N)
        ResidualBlock_noBN_f = functools.partial(utils.ResidualBlock_noBN, nf=self.N)
        self.local_feature_extraction_1 = utils.make_layer(ResidualBlock_noBN_f, self.local_RBs)                # local feature 1
        self.local_feature_extraction_1_CC = conv3x3(self.N, self.trans_nf)
        self.non_local_feature_extraction_1 = utils.make_layer(ResidualBlock_noBN_f, self.front_RBs)             # non local feature 1
        self.non_local_feature_extraction_1_CC = conv3x3(self.N, self.trans_nf)

        self.local_feature_extraction_2 = utils.make_layer(ResidualBlock_noBN_f, self.local_RBs//2)              # local feature 2
        self.local_feature_extraction_2_CC = conv3x3(self.N, self.trans_nf)
        self.non_local_feature_extraction_2 = utils.make_layer(ResidualBlock_noBN_f, self.front_RBs)             # non local feature 2
        self.non_local_feature_extraction_2_CC = conv3x3(self.N, self.trans_nf)
        self.transformer_enhance_1 = Encoder_patch66(d_model=4*4*self.trans_nf, d_inner=4*4*2*self.trans_nf, n_layers=self.n_layers, n_head=self.n_head)
        self.transformer_enhance_2 = Encoder_patch66(d_model=4*4*self.trans_nf, d_inner=4*4*2*self.trans_nf, n_layers=self.n_layers, n_head=self.n_head)
        self.feature_fusion_1 = utils.Condition_feature_fusion_add(self.N, prior_nc=self.trans_nf, ks=3, nhidden=self.N*2)
        self.feature_fusion_2 = utils.Condition_feature_fusion_add(self.N, prior_nc=self.trans_nf, ks=3, nhidden=self.N*2)
        # self.feature_fusion_1 = utils.Condition_feature_fusion(self.N, prior_nc=self.trans_nf, ks=3, nhidden=self.N*2)
        # self.feature_fusion_2 = utils.Condition_feature_fusion(self.N, prior_nc=self.trans_nf, ks=3, nhidden=self.N*2)

    def g_a_func(self, x, x_blur=None, denoise=False):
        if x_blur != None:
            mask = self.get_snr(x, x_blur)
        x = self.g_a_block1(x)
        if denoise:
            # x = self.denoise_module_1(x)
            local_feature_1 = self.local_feature_extraction_1(x)
            # local_feature_1 = self.local_feature_extraction_1_CC(local_feature_1)
            non_local_feature_1 = self.non_local_feature_extraction_1(x)
            # non_local_feature_1 = self.non_local_feature_extraction_1_CC(non_local_feature_1)
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
            mask_1 = mask_1.repeat(1, channel, 1, 1)
            non_local_feature_1_fold = non_local_feature_1_unfold * (1 - mask_1) + local_feature_1 * mask_1
            x = self.feature_fusion_1(x, non_local_feature_1_fold)
        y_inter = x

        x = self.g_a_block2(x)
        if denoise:
            # x = self.denoise_module_2(x)
            local_feature_2 = self.local_feature_extraction_2(x)
            # local_feature_2 = self.local_feature_extraction_2_CC(local_feature_2)
            non_local_feature_2 = self.non_local_feature_extraction_2(x)
            # non_local_feature_2 = self.non_local_feature_extraction_2_CC(non_local_feature_2)
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
            mask_2 = mask_2.repeat(1, channel, 1, 1)
            non_local_feature_2_fold = non_local_feature_2_unfold * (1 - mask_2) + local_feature_2 * mask_2
            x = self.feature_fusion_2(x, non_local_feature_2_fold)
        y = x

        return y_inter, y

    def forward(self, x, x_blur=None, gt=None):
        # g_a for noisy input
        y_inter, y = self.g_a_func(x, x_blur=x_blur, denoise=True)

        # g_a for clean input
        if gt is not None:
            y_inter_gt, y_gt = self.g_a_func(gt)
        else:
            y_inter_gt, y_gt = None, None

        # h_a and h_s
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        # g_s
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "y_inter": y_inter,
            "y_inter_gt": y_inter_gt,
            "y": y,
            "y_gt": y_gt,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict, opt=None):
        """Return a new model instance from `state_dict`."""
        N = state_dict["h_a.0.weight"].size(0)
        net = cls(N, opt)
        net.load_state_dict(state_dict)
        return net

    def get_snr(self, x, x_blur):
        x_gray = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
        x_blur_gray = x_blur[:, 0:1, :, :] * 0.299 + x_blur[:, 1:2, :, :] * 0.587 + x_blur[:, 2:3, :, :] * 0.114
        noise = torch.abs(x_gray - x_blur_gray)
        mask = torch.div(x_blur_gray, noise + 0.0001)
        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)
        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        return mask

    def compress(self, x, x_blur=None):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        _, y = self.g_a_func(x, x_blur=x_blur, denoise=True)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
