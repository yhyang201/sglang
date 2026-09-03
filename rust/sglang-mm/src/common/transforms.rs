//! Reusable image transform primitives.
//!
//! Model-specific processors compose these to build their preprocessing
//! pipelines. All functions operate on flat RGB byte arrays (HWC layout).
//!
//! Not every primitive is wired into a compiled-in processor yet; they are
//! kept available for upcoming model integrations.
#![allow(dead_code)]

use super::par;

/// Normalize u8 RGB pixels to f32 in a single pass: `(pixel/255 - mean) / std`.
///
/// Writes into `out` which must have length `h * w * 3`.
pub fn normalize_rgb_f32(
    rgb: &[u8],
    h: usize,
    w: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    out: &mut [f32],
) {
    debug_assert_eq!(rgb.len(), h * w * 3);
    debug_assert_eq!(out.len(), h * w * 3);
    let inv255 = 1.0f32 / 255.0;
    for i in 0..h * w {
        for c in 0..3 {
            let raw = rgb[i * 3 + c] as f32 * inv255;
            out[i * 3 + c] = (raw - mean[c]) / std[c];
        }
    }
}

/// Pad an HWC image to a grid-aligned size, filling padded pixels with `pad_value`.
///
/// Returns the padded buffer and the new (height, width).
pub fn pad_to_grid(
    rgb_f32: &[f32],
    h: usize,
    w: usize,
    channels: usize,
    grid_h: usize,
    grid_w: usize,
    pad_value: &[f32],
) -> (Vec<f32>, usize, usize) {
    let new_h = h.div_ceil(grid_h) * grid_h;
    let new_w = w.div_ceil(grid_w) * grid_w;
    let mut out = vec![0.0f32; new_h * new_w * channels];
    // Fill with pad value
    for i in 0..new_h * new_w {
        for c in 0..channels {
            out[i * channels + c] = pad_value[c];
        }
    }
    // Copy original data
    for y in 0..h {
        let src_start = y * w * channels;
        let dst_start = y * new_w * channels;
        out[dst_start..dst_start + w * channels]
            .copy_from_slice(&rgb_f32[src_start..src_start + w * channels]);
    }
    (out, new_h, new_w)
}

/// Reshape a padded HWC image into patches of shape `[num_patches, ph, pw, C]`.
///
/// `h` and `w` must be divisible by `ph` and `pw` respectively.
pub fn extract_patches_hwc(
    data: &[f32],
    h: usize,
    w: usize,
    channels: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let nph = h / ph;
    let npw = w / pw;
    let patch_size = ph * pw * channels;
    let mut out = vec![0.0f32; nph * npw * patch_size];
    for i in 0..nph {
        for j in 0..npw {
            let patch_idx = i * npw + j;
            for y in 0..ph {
                let src_y = i * ph + y;
                let src_start = (src_y * w + j * pw) * channels;
                let dst_start = patch_idx * patch_size + y * pw * channels;
                out[dst_start..dst_start + pw * channels]
                    .copy_from_slice(&data[src_start..src_start + pw * channels]);
            }
        }
    }
    out
}

/// Compute the patch grid dimensions for a given image size and patch size.
#[inline]
pub fn patch_grid(h: usize, w: usize, patch_h: usize, patch_w: usize) -> (usize, usize) {
    (h.div_ceil(patch_h), w.div_ceil(patch_w))
}

/// Python-`round()` (round-half-to-even), which `round_by_factor` relies on.
fn round_half_even(x: f64) -> f64 {
    if (x - x.trunc()).abs() == 0.5 {
        (x / 2.0).round() * 2.0
    } else {
        x.round()
    }
}

/// The dynamic-resolution resize shared by the Qwen-VL and MiniMax-VL HF
/// processors: dims divisible by `factor`, total pixels within
/// `[min_pixels, max_pixels]`, aspect ratio preserved as closely as possible.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    const MAX_RATIO: f64 = 200.0;
    let (h, w) = (height as f64, width as f64);
    if height == 0 || width == 0 {
        return Err("empty image".into());
    }
    let ratio = h.max(w) / h.min(w);
    if ratio > MAX_RATIO {
        return Err(format!(
            "absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        ));
    }
    let f = factor as f64;
    let mut h_bar = ((round_half_even(h / f) * f) as usize).max(factor);
    let mut w_bar = ((round_half_even(w / f) * f) as usize).max(factor);
    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = ((h / beta / f).floor() * f) as usize;
        w_bar = ((w / beta / f).floor() * f) as usize;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() * f) as usize;
        w_bar = ((w * beta / f).ceil() * f) as usize;
    }
    // The downscale branch floors without a lower clamp (as Python does), so a
    // very thin image against a small `max_pixels` can floor a side to 0.
    // Python then fails inside PIL's resize; here it would reach the resize
    // coefficient math (overflow panic in debug, garbage in release) and the
    // downstream grid division, so reject it as a request error.
    if h_bar == 0 || w_bar == 0 {
        return Err(format!(
            "smart_resize: {height}x{width} degenerates to {h_bar}x{w_bar} at \
             max_pixels={max_pixels}; image is too thin for this pixel budget"
        ));
    }
    Ok((h_bar, w_bar))
}

/// u8 → normalized f32 lookup for a torchvision `rescale_and_normalize` with
/// both stages on: the rescale folds into mean/std first
/// (`_fuse_mean_std_and_rescale_factor`), i.e. `(v - mean*255) / (std*255)`
/// rounded per f32 op exactly as ATen rounds them.
pub fn normalize_lut_fused(mean: f32, std: f32) -> [f32; 256] {
    let (mean, std) = (mean * 255.0, std * 255.0);
    core::array::from_fn(|v| (v as f32 - mean) / std)
}

/// Patchify in the HF VL flatten order shared by the Qwen-VL and MiniMax-VL
/// image processors: patches ordered `(gh/m, gw/m, m, m)`, features
/// `(C, tps, ps, ps)`, a still image's temporal copies duplicated. `lut`
/// performs the u8 → normalized-f32 mapping inline. `h`/`w` must be divisible
/// by `patch_size * merge_size` (what `smart_resize` guarantees); parallel
/// over merged-block rows.
pub fn patchify_merged_blocks(
    rgb: &[u8],
    h: usize,
    w: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    lut: &[[f32; 256]; 3],
) -> Vec<f32> {
    let (ps, m, tps) = (patch_size, merge_size, temporal_patch_size);
    let (gh, gw) = (h / ps, w / ps);
    let dim = 3 * tps * ps * ps;
    let block_row = gw * m * dim; // one merged-block row of patches
    let mut out = vec![0.0f32; gh * gw * dim];

    par::for_chunks_mut(&mut out, block_row, |i, chunk| {
        let mut p = 0;
        for j in 0..gw / m {
            for mh in 0..m {
                for mw in 0..m {
                    let y0 = (i * m + mh) * ps;
                    let x0 = (j * m + mw) * ps;
                    let patch = &mut chunk[p * dim..(p + 1) * dim];
                    for c in 0..3 {
                        let ch = &mut patch[c * tps * ps * ps..];
                        for py in 0..ps {
                            let src = ((y0 + py) * w + x0) * 3 + c;
                            for px in 0..ps {
                                ch[py * ps + px] = lut[c][rgb[src + px * 3] as usize];
                            }
                        }
                        // Temporal copies of a still are duplicates.
                        let (t0, rest) = ch.split_at_mut(ps * ps);
                        for t in 0..tps - 1 {
                            rest[t * ps * ps..(t + 1) * ps * ps].copy_from_slice(t0);
                        }
                    }
                    p += 1;
                }
            }
        }
    });
    out
}
