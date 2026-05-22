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

## MultiscaleDecomp Shape Review

- Stage 1:
  - `g_a_block1(x)` outputs `N` channels.
  - `local_feature_extraction_1_CC` outputs `trans_nf = N // 2` channels.
  - `non_local_feature_extraction_1_CC` outputs `trans_nf = N // 2` channels.
  - `mask_low` is resized to the stage-1 feature size.
  - The blended feature has `trans_nf` channels.
  - `feature_fusion_1` expects `prior_nc=self.trans_nf`, so the stage-1 connection is channel-consistent.
- Stage 2:
  - `g_a_block2(x)` outputs `N` channels.
  - `local_feature_extraction_2_CC` outputs `trans_nf = N // 2` channels.
  - `non_local_feature_extraction_2_CC` outputs `trans_nf = N // 2` channels.
  - Each of `mask_lh`, `mask_hl`, and `mask_hh` is resized to the stage-2 feature size.
  - Each high-frequency blended feature has `trans_nf` channels.
  - Concatenating LH/HL/HH blended features gives `3 * trans_nf` channels.
  - `high_frequency_fusion = conv1x1(self.trans_nf * 3, self.N)` maps this back to `N` channels.
  - The residual add `x + gamma * high_frequency_feature` is channel-consistent because both tensors have `N` channels.
- No shape mismatch was found in the static channel flow.

## use_subband_snr Branch Review

- `use_subband_snr=False`:
  - `g_a_func()` uses `mask = self.get_snr(x, x_blur)`.
  - Stage 1 uses the single blur-difference mask through the same local/non-local fusion path.
  - Stage 2 uses the single blur-difference mask and `feature_fusion_2`, matching the baseline-style path.
  - `high_frequency_condition`, `high_frequency_fusion`, and `high_frequency_gamma` are not used in this branch.
  - This branch requires `x_blur` when `denoise=True`; otherwise it raises a clear `ValueError`.
- `use_subband_snr=True`:
  - `g_a_func()` generates masks with `self.subband_snr_net(x)`.
  - Stage 1 uses `subband_masks["mask_low"]`.
  - Stage 2 uses `mask_lh`, `mask_hl`, and `mask_hh` separately.
  - LH/HL/HH are not averaged.
  - The three high-frequency denoise features are concatenated.
  - Added `high_frequency_condition = conv1x1(3 * trans_nf, trans_nf)` so the existing `feature_fusion_2` is still used.
  - After `feature_fusion_2`, `high_frequency_fusion = conv1x1(3 * trans_nf, N)` produces the high-frequency residual.
  - The residual is added with learnable `high_frequency_gamma`.
- Syntax-only compile check passed after this branch cleanup.

## Stage 2 Simplification Update

- Simplified the `use_subband_snr=True` stage-2 flow to avoid adding the same LH/HL/HH-derived features through both `feature_fusion_2` and the residual branch.
- Removed `high_frequency_condition`.
- Current stage-2 flow:
  - Base path:
    - Build one local/non-local fused feature.
    - Use `feature_fusion_2` to produce `x_base`.
  - HF residual path:
    - Build `F_LH`, `F_HL`, and `F_HH` separately.
    - Concatenate them.
    - Use `high_frequency_fusion = conv1x1(3 * trans_nf, N)`.
    - Add `gamma * F_HF` to `x_base`.
- Syntax-only compile check passed after this simplification.

## Forward Test Script

- Added:
  - `Joint_IC_Denoising_MSE_wavelet_subband_exp/test_wavelet_subband_forward.py`
- Purpose:
  - Run quick forward/shape checks on the torch-enabled server environment.
  - This script does not train and does not modify `train.py` or `criterion.py`.
- Checks included:
  - Haar DWT/IDWT shape recovery.
  - `SubbandSNRNet` mask shapes and `[0, 1]` range.
  - subband target shapes and `[0, 1]` range.
  - `subband_snr_loss()` finite values.
  - `MultiscaleDecomp.forward()` output keys and tensor shapes.
  - optional baseline branch check with `use_subband_snr=False`.
- Syntax-only compile check passed locally.
- Runtime forward check remains deferred until PyTorch is available.

## Server Test Procedure Memo

- Added:
  - `Joint_IC_Denoising_MSE_wavelet_subband_exp/README_wavelet_subband_forward_test.md`
- Run from:
  - `Joint_IC_Denoising_MSE_wavelet_subband_exp`
- Execution order after the server is available:
  1. `python test_wavelet_subband_forward.py --device cuda --skip-model`
  2. `python test_wavelet_subband_forward.py --device cuda`
  3. `python test_wavelet_subband_forward.py --device cuda --check-baseline-branch`
- The README also includes CPU fallback commands and expected checks for each step.
