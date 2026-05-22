import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
COMPRESSAI_ROOT = ROOT / "CompressAI"
sys.path.insert(0, str(COMPRESSAI_ROOT))

from compressai.models.MultiscaleDecomp import MultiscaleDecomp
from compressai.models.wavelet_subband_snr import (
    SubbandSNRNet,
    dwt_haar,
    idwt_haar,
    make_subband_snr_targets,
    subband_snr_loss,
)


def make_blur(x):
    return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)


def assert_finite(name, x):
    if not torch.isfinite(x).all():
        raise AssertionError(f"{name} contains NaN or Inf")


def assert_range_0_1(name, x):
    min_value = x.min().item()
    max_value = x.max().item()
    if min_value < 0.0 or max_value > 1.0:
        raise AssertionError(
            f"{name} is outside [0, 1]: min={min_value:.6f}, max={max_value:.6f}"
        )


def print_tensor(name, x):
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"min={x.min().item():.6f}, max={x.max().item():.6f}, "
        f"mean={x.mean().item():.6f}"
    )


def check_wavelet_module(noisy, clean):
    print("\n[1] Haar DWT / IDWT")
    ll, lh, hl, hh = dwt_haar(noisy)
    recon = idwt_haar(ll, lh, hl, hh)
    max_error = torch.max(torch.abs(recon - noisy)).item()
    print_tensor("LL", ll)
    print_tensor("LH", lh)
    print_tensor("HL", hl)
    print_tensor("HH", hh)
    print(f"IDWT reconstruction shape={tuple(recon.shape)}")
    print(f"IDWT max reconstruction error={max_error:.8f}")

    if recon.shape != noisy.shape:
        raise AssertionError(f"IDWT shape mismatch: {recon.shape} != {noisy.shape}")
    if not math.isfinite(max_error):
        raise AssertionError("IDWT reconstruction error is not finite")

    print("\n[2] SubbandSNRNet standalone forward")
    snr_net = SubbandSNRNet().to(noisy.device)
    pred = snr_net(noisy)
    target = make_subband_snr_targets(noisy, clean)
    loss_dict = subband_snr_loss(pred, target)

    expected_mask_shape = (noisy.size(0), 1, noisy.size(2) // 2, noisy.size(3) // 2)
    for key in ["mask_low", "mask_lh", "mask_hl", "mask_hh"]:
        value = pred[key]
        print_tensor(key, value)
        if tuple(value.shape) != expected_mask_shape:
            raise AssertionError(
                f"{key} shape mismatch: {tuple(value.shape)} != {expected_mask_shape}"
            )
        assert_finite(key, value)
        assert_range_0_1(key, value)

    for key in ["target_low", "target_lh", "target_hl", "target_hh"]:
        value = target[key]
        print_tensor(key, value)
        if tuple(value.shape) != expected_mask_shape:
            raise AssertionError(
                f"{key} shape mismatch: {tuple(value.shape)} != {expected_mask_shape}"
            )
        assert_finite(key, value)
        assert_range_0_1(key, value)

    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.8f}")
        assert_finite(key, value)


def check_multiscale_forward(noisy, clean, blur, n_channels, use_subband_snr):
    print(
        "\n[3] MultiscaleDecomp forward "
        f"(N={n_channels}, use_subband_snr={use_subband_snr})"
    )
    model = MultiscaleDecomp(N=n_channels).to(noisy.device)
    model.use_subband_snr = use_subband_snr
    model.eval()

    with torch.no_grad():
        out = model(noisy, x_blur=blur, gt=clean)

    required_keys = ["x_hat", "y_inter", "y_inter_gt", "y", "y_gt", "likelihoods"]
    for key in required_keys:
        if key not in out:
            raise AssertionError(f"missing output key: {key}")

    print_tensor("x_hat", out["x_hat"])
    print_tensor("y_inter", out["y_inter"])
    print_tensor("y", out["y"])
    assert_finite("x_hat", out["x_hat"])
    assert_finite("y_inter", out["y_inter"])
    assert_finite("y", out["y"])

    if out["x_hat"].shape != noisy.shape:
        raise AssertionError(f"x_hat shape mismatch: {out['x_hat'].shape} != {noisy.shape}")

    if use_subband_snr:
        subband_snr = out.get("subband_snr", None)
        if subband_snr is None:
            raise AssertionError("subband_snr output is missing")
        expected_mask_shape = (noisy.size(0), 1, noisy.size(2) // 2, noisy.size(3) // 2)
        for key in ["mask_low", "mask_lh", "mask_hl", "mask_hh"]:
            value = subband_snr[key]
            print_tensor(f"out.subband_snr.{key}", value)
            if tuple(value.shape) != expected_mask_shape:
                raise AssertionError(
                    f"out.subband_snr.{key} shape mismatch: "
                    f"{tuple(value.shape)} != {expected_mask_shape}"
                )
            assert_finite(f"out.subband_snr.{key}", value)
            assert_range_0_1(f"out.subband_snr.{key}", value)

    for key, value in out["likelihoods"].items():
        print_tensor(f"likelihoods.{key}", value)
        assert_finite(f"likelihoods.{key}", value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forward and shape checks for wavelet subband SNR integration."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only test wavelet_subband_snr.py without MultiscaleDecomp.",
    )
    parser.add_argument(
        "--check-baseline-branch",
        action="store_true",
        help="Also run MultiscaleDecomp with use_subband_snr=False.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.height % 64 != 0 or args.width % 64 != 0:
        raise ValueError("height and width should be multiples of 64 for this model")
    if args.channels not in [1, 3]:
        raise ValueError("--channels must be 1 or 3")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(0)
    noisy = torch.rand(
        args.batch_size, args.channels, args.height, args.width, device=device
    )
    clean = torch.rand_like(noisy)
    blur = make_blur(noisy)

    print(f"device={device}")
    print_tensor("noisy", noisy)
    print_tensor("clean", clean)
    print_tensor("blur", blur)

    check_wavelet_module(noisy, clean)

    if not args.skip_model:
        check_multiscale_forward(
            noisy,
            clean,
            blur,
            n_channels=args.N,
            use_subband_snr=True,
        )
        if args.check_baseline_branch:
            check_multiscale_forward(
                noisy,
                clean,
                blur,
                n_channels=args.N,
                use_subband_snr=False,
            )

    print("\nAll requested forward/shape checks passed.")


if __name__ == "__main__":
    main()
