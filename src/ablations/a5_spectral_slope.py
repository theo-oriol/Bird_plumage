from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# ── Settings ─────────────────────────────────────────────────────────────────
CODE = "A5"
BAND_LOW = 2.0
BAND_HIGH = 9.0
TARGET_ALPHA_224 = 1.49
TARGET_ALPHA_128 = 2.19
ALPHA_JITTER = 0.1  # +/- this, uniform, added to the reference target - see module docstring
ANCHOR_RADIUS = BAND_LOW  # the gain is anchored so it leaves this radius untouched


# ── Spectrum helpers ─────────────────────────────────────────────────────────
def _radial_index(size: int) -> np.ndarray:
    freqs = np.fft.fftshift(np.fft.fftfreq(size) * size)
    fy, fx = np.meshgrid(freqs, freqs, indexing="ij")
    radius = np.sqrt(fx**2 + fy**2)
    return np.round(radius).astype(int)


def _radial_average(magnitude: np.ndarray, radial_idx: np.ndarray, max_r: int) -> np.ndarray:
    sums = np.bincount(radial_idx.ravel(), weights=magnitude.ravel(), minlength=max_r + 1)
    counts = np.bincount(radial_idx.ravel(), minlength=max_r + 1)
    counts[counts == 0] = 1
    return sums / counts


# OLS log-log slope of a radial profile; NaN if the band has no energy (only logged)
def _alpha_from_profile(profile: np.ndarray, radii: np.ndarray, band=(BAND_LOW, BAND_HIGH)) -> float:
    in_band = (radii >= band[0]) & (radii <= band[1]) & (profile > 0)
    if not np.any(in_band):
        return float("nan")
    log_r = np.log(radii[in_band])
    log_a = np.log(profile[in_band])
    slope, _intercept = np.polyfit(log_r, log_a, 1)
    return float(-slope)


# ── Measure and apply ────────────────────────────────────────────────────────
# Radial-average slope of the G channel in the band, OLS in log-log space (cf. SHINE, Willenbockel 2010)
def measure_alpha(crop_uint8: np.ndarray, band=(BAND_LOW, BAND_HIGH)) -> float:
    g = crop_uint8[..., 1].astype(np.float64)
    size = g.shape[0]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    radial_idx = _radial_index(size)
    max_r = int(radial_idx.max())
    profile = _radial_average(spectrum, radial_idx, max_r)
    radii = np.arange(max_r + 1, dtype=np.float64)
    return _alpha_from_profile(profile, radii, band)


def apply(crop_uint8: np.ndarray, target_alpha: float) -> Tuple[np.ndarray, Dict[str, float]]:
    size = crop_uint8.shape[0]
    radial_idx = _radial_index(size)
    max_r = int(radial_idx.max())
    radii = np.arange(max_r + 1).astype(np.float64)

    g = crop_uint8[..., 1].astype(np.float64)
    g_fft = np.fft.fftshift(np.fft.fft2(g))
    magnitude = np.abs(g_fft)
    profile = _radial_average(magnitude, radial_idx, max_r)
    # Reuse the profile computed above instead of a second FFT
    alpha_before = _alpha_from_profile(profile, radii)

    anchor_r = int(round(ANCHOR_RADIUS))
    anchor_amplitude = profile[anchor_r] if profile[anchor_r] > 0 else 1.0
    with np.errstate(divide="ignore"):
        target_profile = anchor_amplitude * np.power(radii / anchor_r, -target_alpha)
    target_profile[0] = profile[0]  # never touch the DC term (mean brightness)

    gain_1d = np.ones_like(profile)
    nonzero = profile > 0
    gain_1d[nonzero] = target_profile[nonzero] / profile[nonzero]
    gain_2d = gain_1d[radial_idx]

    out = np.empty_like(crop_uint8, dtype=np.float64)
    for channel in range(3):
        channel_fft = np.fft.fftshift(np.fft.fft2(crop_uint8[..., channel].astype(np.float64)))
        regraded = channel_fft * gain_2d  # real, positive gain -> phase unchanged
        out[..., channel] = np.real(np.fft.ifft2(np.fft.ifftshift(regraded)))

    out_uint8 = np.clip(out, 0, 255).astype(np.uint8)
    meta = {
        "target_alpha": target_alpha,
        "alpha_before": alpha_before,
        "alpha_after": measure_alpha(out_uint8),
    }
    return out_uint8, meta


# ── Target slope ─────────────────────────────────────────────────────────────
# Reference slope for this crop size (measured on 120 specimens): centre of the jittered draw
def target_alpha_for_crop_source(source_size: int) -> float:
    if source_size == 224:
        return TARGET_ALPHA_224
    if source_size == 128:
        return TARGET_ALPHA_128
    raise ValueError(f"A5 only has a target for 128/224 source crops, got {source_size}")


# Jitter on the target slope, uniform in +/- ALPHA_JITTER: drawn for every crop
def draw_alpha_jitter(rng: np.random.Generator) -> float:
    return float(rng.uniform(-ALPHA_JITTER, ALPHA_JITTER))


# Target slope for this crop size: reference + jitter
def target_alpha(source_size: int, jitter: float) -> float:
    return target_alpha_for_crop_source(source_size) + jitter


# ── Run as a script ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Blocky upsampled noise has a measurable slope (white noise is flat)
    base = rng.integers(0, 256, size=(28, 28, 3)).astype(np.uint8)
    crop = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))

    target = target_alpha(224, draw_alpha_jitter(rng))
    out, meta = apply(crop, target)

    print(f"A5: reference target (224px crop) = {target_alpha_for_crop_source(224)}  drawn target = {target:.3f}")
    print(f"alpha before: {meta['alpha_before']:.2f}  ->  after: {meta['alpha_after']:.2f}")
