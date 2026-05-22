# Wavelet Subband Forward Test

This memo describes the forward-only checks to run after the department server
is available again. These checks do not train the model and do not require
changes to `train.py` or `criterion.py`.

Run all commands from the experimental copy root:

```bash
cd /Users/satouryousuke/Desktop/vscode/master/Joint_IC_Denoising_MSE_wavelet_subband_exp
```

## 1. SubbandSNRNet Standalone Check

```bash
python test_wavelet_subband_forward.py --device cuda --skip-model
```

Expected checks:

- Haar DWT returns `LL`, `LH`, `HL`, `HH`.
- IDWT restores the original input shape.
- `SubbandSNRNet.forward()` runs with a noisy image only.
- `mask_low`, `mask_lh`, `mask_hl`, and `mask_hh` have shape
  `[B, 1, H/2, W/2]`.
- All masks are in the `[0, 1]` range.
- `make_subband_snr_targets(noisy, clean)` returns targets with the same shapes
  as the predicted masks.
- `subband_snr_loss(pred, target)` is finite.

## 2. MultiscaleDecomp Forward Check

```bash
python test_wavelet_subband_forward.py --device cuda
```

Expected checks:

- `MultiscaleDecomp.forward(noisy, x_blur=blur, gt=clean)` runs with
  `use_subband_snr=True`.
- `x_hat` has the same shape as the input noisy image.
- `y_inter`, `y`, and likelihood tensors are finite.
- The output dictionary includes `subband_snr`.
- `subband_snr["mask_low"]`, `subband_snr["mask_lh"]`,
  `subband_snr["mask_hl"]`, and `subband_snr["mask_hh"]` have shape
  `[B, 1, H/2, W/2]`.
- All subband masks are in the `[0, 1]` range.

## 3. Baseline Branch Check

```bash
python test_wavelet_subband_forward.py --device cuda --check-baseline-branch
```

Expected checks:

- The `use_subband_snr=True` forward path runs.
- The `use_subband_snr=False` forward path also runs.
- The `use_subband_snr=False` path uses the existing blur-difference
  `get_snr(x, x_blur)` branch.
- The baseline branch does not use the high-frequency residual branch.

## Optional CPU Check

If CUDA is unavailable, run the same checks on CPU:

```bash
python test_wavelet_subband_forward.py --device cpu --skip-model
python test_wavelet_subband_forward.py --device cpu
python test_wavelet_subband_forward.py --device cpu --check-baseline-branch
```

## Notes

- Default input shape is `[2, 3, 256, 256]`.
- Default model channel count is `N=128`.
- Height and width should be multiples of 64 for this model.
- Runtime tests are intentionally deferred on the local machine because PyTorch
  is not installed there.
