# 2026-05-22 Wavelet Subband SNR Step 1

## Scope

- Kept the original `Joint_IC_Denoising_MSE` directory unchanged.
- Created a new experimental copy:
  - `Joint_IC_Denoising_MSE_wavelet_subband_exp`
- Added the first standalone wavelet SNR module only.
- Did not connect the new module to `MultiscaleDecomp.py`, `train.py`, or `criterion.py` yet.

## Existing Code Locations Confirmed

- Current two-stage SNR-aware feature fusion:
  - `Joint_IC_Denoising_MSE/CompressAI/compressai/models/MultiscaleDecomp.py`
  - `MultiscaleDecomp.g_a_func()`
- Existing blur-difference SNR map generation:
  - `Joint_IC_Denoising_MSE/CompressAI/compressai/models/MultiscaleDecomp.py`
  - `MultiscaleDecomp.get_snr()`
- `blur_x` generation:
  - `SyntheticDataset.py`
  - `SiddDataset.py`
  - both use `cv2.blur(..., (5, 5))`

## Added File

- `Joint_IC_Denoising_MSE_wavelet_subband_exp/CompressAI/compressai/models/wavelet_subband_snr.py`

## Implemented

- `dwt_haar(x)`
- `idwt_haar(ll, lh, hl, hh)`
- `rgb_to_gray(x)`
- `SubbandSNRNet`
- `make_subband_snr_targets(noisy, clean)`
- `subband_snr_loss(pred, target, lambda_hf=1.0, reduction="mean")`
- `_self_test()` for standalone shape/forward checks when PyTorch is available.

## Verification

- Syntax check passed with:
  - `python3 -c "compile(open('CompressAI/compressai/models/wavelet_subband_snr.py').read(), 'wavelet_subband_snr.py', 'exec')"`
- Runtime forward test could not be executed in the current system Python because `torch` is not installed:
  - `ModuleNotFoundError: No module named 'torch'`

## Next Connection Work

- Add `SubbandSNRNet` to the experimental `MultiscaleDecomp`.
- Replace stage-1 blur-SNR mask usage with `mask_low`.
- Replace stage-2 single mask usage with separate `mask_lh`, `mask_hl`, `mask_hh`.
- Add concat + 1x1 Conv + learnable `gamma` for high-frequency residual feature fusion.
- Add optional subband SNR target/loss integration in `criterion.py` or the training loop.

## Static Review Update

- Added docstrings to `wavelet_subband_snr.py` documenting:
  - DWT input requirement: `[B, C, H, W]` with even `H` and `W`
  - IDWT output shape recovery
  - `SubbandSNRNet` output shapes
  - target shapes from `make_subband_snr_targets`
  - independent LH/HL/HH loss comparison
- Re-ran syntax-only compile check without importing PyTorch. It passed.
- PyTorch forward/runtime tests remain intentionally deferred until a torch-enabled environment is available.

## MultiscaleDecomp Connection Update

- Modified only the experimental copy:
  - `Joint_IC_Denoising_MSE_wavelet_subband_exp/CompressAI/compressai/models/MultiscaleDecomp.py`
- Did not modify `train.py` or `criterion.py`.
- Imported `SubbandSNRNet`.
- Added:
  - `self.use_subband_snr = True`
  - `self.subband_snr_net = SubbandSNRNet()`
  - `self.high_frequency_fusion = conv1x1(self.trans_nf * 3, self.N)`
  - `self.high_frequency_gamma = nn.Parameter(torch.tensor(0.1))`
- Stage 1 now uses `mask_low` for the existing local/non-local fusion path.
- Stage 2 now uses `mask_lh`, `mask_hl`, and `mask_hh` separately.
- Stage 2 high-frequency features are concatenated and integrated with a 1x1 Conv, then added to the main feature using learnable gamma.
- `forward()` now includes `subband_snr` in its output dictionary for later loss integration.
- `compress()` now also uses the subband-SNR path, so it no longer depends on `x_blur` when `use_subband_snr` is true.
- Syntax-only compile check for `MultiscaleDecomp.py` passed.
- PyTorch runtime forward check remains deferred.
