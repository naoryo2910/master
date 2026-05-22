import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_4d_even(x):
    """Validate tensors used by Haar DWT.

    Haar DWT downsamples by a factor of 2, so both spatial dimensions must be
    even. The current training patches and test cropping are multiples of 64,
    which satisfy this requirement.
    """
    if x.dim() != 4:
        raise ValueError("expected a 4D tensor with shape [B, C, H, W]")
    h, w = x.shape[-2:]
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("H and W must be even for Haar DWT/IDWT")


def dwt_haar(x):
    """Haar DWT for tensors with shape [B, C, H, W].

    Returns LL, LH, HL, HH, each with shape [B, C, H/2, W/2].
    The scaling is orthonormal, so idwt_haar(dwt_haar(x)) reconstructs x.
    """
    _check_4d_even(x)

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x01 - x10 - x11) * 0.5
    hl = (x00 - x01 + x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def idwt_haar(ll, lh, hl, hh):
    """Inverse Haar DWT for four [B, C, H, W] subbands.

    If each input subband has shape [B, C, H/2, W/2], the output shape is
    [B, C, H, W].
    """
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("all Haar subbands must have the same shape")
    if ll.dim() != 4:
        raise ValueError("expected 4D Haar subbands with shape [B, C, H, W]")

    b, c, h, w = ll.shape
    x = ll.new_empty(b, c, h * 2, w * 2)
    x[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    x[:, :, 0::2, 1::2] = (ll + lh - hl - hh) * 0.5
    x[:, :, 1::2, 0::2] = (ll - lh + hl - hh) * 0.5
    x[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return x


def rgb_to_gray(x):
    """Return a grayscale tensor for 1-channel or RGB image tensors."""
    if x.size(1) == 1:
        return x
    if x.size(1) != 3:
        raise ValueError("expected a 1-channel or 3-channel image tensor")
    return (
        0.299 * x[:, 0:1, :, :]
        + 0.587 * x[:, 1:2, :, :]
        + 0.114 * x[:, 2:3, :, :]
    )


class _MaskHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class SubbandSNRNet(nn.Module):
    """Predict LL/LH/HL/HH SNR-like masks from a noisy image only.

    Input:
        noisy: [B, 1, H, W] or [B, 3, H, W], with even H and W.

    Output dictionary:
        mask_low: [B, 1, H/2, W/2], predicted from LL.
        mask_lh:  [B, 1, H/2, W/2], predicted from LH.
        mask_hl:  [B, 1, H/2, W/2], predicted from HL.
        mask_hh:  [B, 1, H/2, W/2], predicted from HH.

    All masks are sigmoid outputs in the [0, 1] range. Clean images are not
    used in forward(), so the module can run during inference.
    """

    def __init__(self, hidden_channels=32):
        super().__init__()
        self.low_head = _MaskHead(1, 1, hidden_channels=hidden_channels)
        self.high_head = _MaskHead(3, 3, hidden_channels=hidden_channels)

    def forward(self, noisy):
        noisy_gray = rgb_to_gray(noisy)
        ll_n, lh_n, hl_n, hh_n = dwt_haar(noisy_gray)

        mask_low = self.low_head(ll_n)
        high_input = torch.cat(
            [torch.abs(lh_n), torch.abs(hl_n), torch.abs(hh_n)],
            dim=1,
        )
        mask_lh, mask_hl, mask_hh = self.high_head(high_input).chunk(3, dim=1)

        return {
            "mask_low": mask_low,
            "mask_lh": mask_lh,
            "mask_hl": mask_hl,
            "mask_hh": mask_hh,
        }


def make_subband_snr_targets(noisy, clean, eps=1e-6):
    """Create subband-wise SNR targets from noisy and clean images.

    Clean is used only for teacher target generation, not for SubbandSNRNet
    inference. The target shapes match SubbandSNRNet outputs:
    [B, 1, H/2, W/2] for target_low, target_lh, target_hl, and target_hh.
    """
    noisy_gray = rgb_to_gray(noisy)
    clean_gray = rgb_to_gray(clean)

    ll_n, lh_n, hl_n, hh_n = dwt_haar(noisy_gray)
    ll_c, lh_c, hl_c, hh_c = dwt_haar(clean_gray)

    def make_target(clean_band, noisy_band):
        signal = torch.abs(clean_band)
        noise = torch.abs(noisy_band - clean_band)
        return torch.clamp(signal / (signal + noise + eps), min=0.0, max=1.0)

    return {
        "target_low": make_target(ll_c, ll_n),
        "target_lh": make_target(lh_c, lh_n),
        "target_hl": make_target(hl_c, hl_n),
        "target_hh": make_target(hh_c, hh_n),
    }


def subband_snr_loss(pred, target, lambda_hf=1.0, reduction="mean"):
    """L1 loss for LL and separate LH/HL/HH SNR masks.

    LH, HL, and HH are compared independently; they are not averaged into a
    single high-frequency mask.
    """
    l_low = F.l1_loss(pred["mask_low"], target["target_low"], reduction=reduction)
    l_lh = F.l1_loss(pred["mask_lh"], target["target_lh"], reduction=reduction)
    l_hl = F.l1_loss(pred["mask_hl"], target["target_hl"], reduction=reduction)
    l_hh = F.l1_loss(pred["mask_hh"], target["target_hh"], reduction=reduction)
    loss = l_low + lambda_hf * (l_lh + l_hl + l_hh)

    return {
        "subband_snr_loss": loss,
        "subband_snr_low_loss": l_low,
        "subband_snr_lh_loss": l_lh,
        "subband_snr_hl_loss": l_hl,
        "subband_snr_hh_loss": l_hh,
    }


def _self_test():
    x = torch.rand(2, 3, 256, 256)
    clean = torch.rand(2, 3, 256, 256)

    ll, lh, hl, hh = dwt_haar(x)
    recon = idwt_haar(ll, lh, hl, hh)
    max_recon_error = torch.max(torch.abs(recon - x)).item()

    net = SubbandSNRNet()
    pred = net(x)
    target = make_subband_snr_targets(x, clean)
    losses = subband_snr_loss(pred, target)

    print("DWT shapes:", ll.shape, lh.shape, hl.shape, hh.shape)
    print("IDWT shape:", recon.shape)
    print("max reconstruction error:", max_recon_error)
    for key, value in pred.items():
        print(key, value.shape, value.min().item(), value.max().item())
    print("subband_snr_loss:", losses["subband_snr_loss"].item())


if __name__ == "__main__":
    _self_test()
