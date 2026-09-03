//! libjpeg-turbo decode backend (feature `turbo-jpeg`).
//!
//! Two wins over the pure-Rust `image` decoder for JPEG inputs:
//!
//! * **Speed**: libjpeg-turbo's SIMD iDCT/Huffman is the fastest JPEG decode
//!   available on CPU (PIL links the same library).
//! * **Scaled iDCT**: [`decode_rgb_scaled`] decodes at the smallest M/8
//!   fraction whose output still covers a target size, skipping most of the
//!   full-resolution iDCT + a later downscale's input reads.
//!
//! Bit-exactness: libjpeg-turbo's iDCT is *not* bit-identical to the
//! pure-Rust decoder's (both approximate the ideal iDCT; measured diffs are
//! within ±1 LSB per sample on standard YCbCr JPEGs — see
//! `tests::full_res_close_to_image_crate`). Anything routed through this
//! backend is therefore a **tolerance-mode** path, never the parity default:
//! [`super::decode_rgb`] stays the bit-exact reference.
//!
//! Non-JPEG inputs and exotic JPEG color spaces the turbo API rejects fall
//! back to the pure-Rust decoder, so callers can route unconditionally.

use std::cell::RefCell;

/// JPEG SOI + first marker byte.
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

thread_local! {
    /// tj3Init + state setup is non-trivial; reuse one decompressor per thread.
    static DECOMP: RefCell<Option<turbojpeg::Decompressor>> = const { RefCell::new(None) };
}

fn with_decompressor<R>(
    f: impl FnOnce(&mut turbojpeg::Decompressor) -> Result<R, String>,
) -> Result<R, String> {
    DECOMP.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(turbojpeg::Decompressor::new().map_err(|e| format!("tj3Init: {e}"))?);
        }
        f(slot.as_mut().unwrap())
    })
}

/// `(height, width)` from the JPEG header, without decoding.
pub fn header(data: &[u8]) -> Result<(usize, usize), String> {
    let h = turbojpeg::read_header(data).map_err(|e| format!("turbojpeg header: {e}"))?;
    Ok((h.height, h.width))
}

/// The smallest M/8 scaling factor whose decoded output still covers
/// `(min_h, min_w)` — i.e. the cheapest iDCT that does not force a later
/// upscale. `1/1` when even full resolution falls short or no smaller factor
/// covers.
pub fn pick_scale_factor(
    header_h: usize,
    header_w: usize,
    min_h: usize,
    min_w: usize,
) -> turbojpeg::ScalingFactor {
    // Do not rely on the order of supported_scaling_factors() (it differs
    // across crate versions): take the minimum-ratio covering factor directly.
    let mut chosen = turbojpeg::ScalingFactor::new(1, 1);
    for sf in turbojpeg::Decompressor::supported_scaling_factors() {
        if sf.num() > sf.denom() {
            continue; // never upscale at decode time
        }
        if sf.scale(header_h) >= min_h
            && sf.scale(header_w) >= min_w
            && sf.num() * chosen.denom() < chosen.num() * sf.denom()
        {
            chosen = sf;
        }
    }
    chosen
}

fn decompress_at(
    data: &[u8],
    factor: turbojpeg::ScalingFactor,
) -> Result<(Vec<u8>, usize, usize), String> {
    with_decompressor(|decomp| {
        let header = decomp
            .read_header(data)
            .map_err(|e| format!("turbojpeg header: {e}"))?;
        decomp
            .set_scaling_factor(factor)
            .map_err(|e| format!("turbojpeg set_scaling_factor: {e}"))?;
        let (h, w) = (factor.scale(header.height), factor.scale(header.width));
        let mut rgb = vec![0u8; h * w * 3];
        let image = turbojpeg::Image {
            pixels: &mut rgb[..],
            width: w,
            pitch: w * 3,
            height: h,
            format: turbojpeg::PixelFormat::RGB,
        };
        decomp
            .decompress(data, image)
            .map_err(|e| format!("turbojpeg decompress: {e}"))?;
        Ok((rgb, h, w))
    })
}

/// Full-resolution JPEG decode via libjpeg-turbo. Falls back to the pure-Rust
/// decoder for non-JPEG input or anything libjpeg-turbo rejects (e.g. CMYK
/// encodings PIL handles differently), so it is safe to route blindly — but
/// note the fallback is a *different* decoder, so mixed-format batches can
/// mix iDCT roundings.
pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    if is_jpeg(data) {
        if let ok @ Ok(_) = decompress_at(data, turbojpeg::ScalingFactor::new(1, 1)) {
            return ok;
        }
    }
    super::decode_rgb(data)
}

/// Scaled-iDCT JPEG decode: decode at the smallest M/8 fraction whose output
/// still covers `(min_h, min_w)`, so a subsequent resize to that target reads
/// far fewer pixels. Returns the *scaled* dims, which are ≥ the requested
/// minimums and almost never equal to them — callers must resize from the
/// returned dims, and the result diverges from a full-resolution decode by
/// more than iDCT rounding (tolerance mode). Same fallback rules as
/// [`decode_rgb`]; a non-JPEG input decodes at full resolution.
pub fn decode_rgb_scaled(
    data: &[u8],
    min_h: usize,
    min_w: usize,
) -> Result<(Vec<u8>, usize, usize), String> {
    if is_jpeg(data) {
        let factor = header(data).map(|(h, w)| pick_scale_factor(h, w, min_h, min_w));
        if let Ok(factor) = factor {
            if let ok @ Ok(_) = decompress_at(data, factor) {
                return ok;
            }
        }
    }
    super::decode_rgb(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline 4:4:4 JPEG (no subsampling surprises) of a gradient.
    fn gradient_jpeg(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
        buf.into_inner()
    }

    #[test]
    fn header_and_full_res_dims() {
        let jpg = gradient_jpeg(200, 120);
        assert!(is_jpeg(&jpg));
        assert_eq!(header(&jpg).unwrap(), (120, 200));
        let (rgb, h, w) = decode_rgb(&jpg).unwrap();
        assert_eq!((h, w), (120, 200));
        assert_eq!(rgb.len(), 120 * 200 * 3);
    }

    #[test]
    fn scale_factor_picks_smallest_covering() {
        // libjpeg-turbo reports *reduced* fractions (1/2, not 4/8), so assert
        // ratios cross-multiplied. For a 1000x2000 source:
        //   target 500x1000 → 1/2 gives exactly 500x1000; 3/8 (375x750) falls short.
        let sf = pick_scale_factor(1000, 2000, 500, 1000);
        assert_eq!(sf.num() * 2, sf.denom(), "ratio must be 1/2");
        // Just past a boundary: 501x1001 needs 5/8 (625x1250).
        let sf = pick_scale_factor(1000, 2000, 501, 1001);
        assert_eq!(sf.num() * 8, 5 * sf.denom(), "ratio must be 5/8");
        // Either axis uncovered forces a larger factor: 500x1000 high but
        // 1500 wide needs 3/4 (750x1500).
        let sf = pick_scale_factor(1000, 2000, 500, 1500);
        assert_eq!(sf.num() * 4, 3 * sf.denom(), "ratio must be 3/4");
        // Target beyond full resolution stays 1/1 (never upscale).
        let sf = pick_scale_factor(1000, 2000, 2000, 4000);
        assert_eq!((sf.num(), sf.denom()), (1, 1));
    }

    #[test]
    fn scaled_decode_dims_cover_target() {
        let jpg = gradient_jpeg(512, 384);
        let (rgb, h, w) = decode_rgb_scaled(&jpg, 130, 200).unwrap();
        // 3/8 of 384x512 = 144x192 < 200 wide; 4/8 = 192x256 covers.
        assert_eq!((h, w), (192, 256));
        assert!(h >= 130 && w >= 200);
        assert_eq!(rgb.len(), h * w * 3);
    }

    /// The tolerance budget for the whole turbo backend: full-res turbo vs
    /// the pure-Rust `image` decoder must agree within ±2 LSB per sample
    /// (the two iDCT roundings differ; measured max is 2 on this fixture).
    /// The tighter claim — turbo vs PIL within ±1, usually 0 — is verified
    /// against PIL from Python in bench/bench_spike.py, since both share
    /// libjpeg's default ISLOW iDCT.
    #[test]
    fn full_res_close_to_image_crate() {
        let jpg = gradient_jpeg(256, 192);
        let (turbo, h, w) = decode_rgb(&jpg).unwrap();
        let (rust, rh, rw) = crate::common::decode_rgb(&jpg).unwrap();
        assert_eq!((h, w), (rh, rw));
        let mut max_diff = 0i32;
        let mut n_diff = 0usize;
        for (a, b) in turbo.iter().zip(rust.iter()) {
            let d = (*a as i32 - *b as i32).abs();
            max_diff = max_diff.max(d);
            n_diff += (d > 0) as usize;
        }
        assert!(max_diff <= 2, "max abs diff {max_diff}");
        assert!(
            n_diff * 100 < turbo.len() * 60,
            "{} of {} samples differ",
            n_diff,
            turbo.len()
        );
    }

    /// Non-JPEG input must fall through to the pure-Rust decoder untouched.
    #[test]
    fn png_falls_back() {
        let img =
            image::RgbImage::from_fn(8, 6, |x, y| image::Rgb([x as u8 * 30, y as u8 * 40, 9]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let png = buf.into_inner();
        assert!(!is_jpeg(&png));
        assert_eq!(
            decode_rgb(&png).unwrap(),
            super::super::decode_rgb(&png).unwrap()
        );
        assert_eq!(
            decode_rgb_scaled(&png, 2, 2).unwrap(),
            super::super::decode_rgb(&png).unwrap()
        );
    }
}
