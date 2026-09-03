//! MiniMax M3 VL family server-pipeline image processor.
//!
//! Pure-Rust equivalent of the HF `MiniMaxM3VLImageProcessor` pipeline the
//! Python `MiniMaxM3VLProcessor` drives (transformers in-tree
//! `transformers.models.minimax_m3_vl`, identical to the checkpoint's remote
//! code): `smart_resize` → torchvision uint8 antialias bicubic → fused
//! rescale+normalize → patchify into `[grid_h*grid_w, C*tps*ps*ps]` (HF
//! flatten order: patches by `(gh/m, gw/m, m, m)`, features by `(C, tps, ps,
//! ps)`, temporal copies duplicated for stills).
//!
//! Unlike the Qwen family there is exactly one HF image processor for M3 —
//! the torchvision one — so the resize kernel is fixed to
//! [`resize::Resample::AtenU8`]; no `resample` knob exists to get wrong. The
//! LLM side uses plain 1-D RoPE (the 3-D rope lives inside the ViT, driven by
//! `image_grid_thw`), so the family keeps the default `positions`. Prompt
//! expansion wraps each image's token run in start/end special tokens, with
//! the item's offsets covering the inner run only (the Python
//! `get_mm_items_offset` convention).
//!
//! All parameters come from the runtime spec; nothing is hardcoded per model.
//! `min_pixels` is worth a note: the HF processor never exposes it as an
//! attribute — it is the `smart_resize` function default `4 * 28 * 28` — so
//! the Python side passes it explicitly.

use crate::common::{resize, token_layout, transforms};
use crate::pipeline::{
    DecodedMedia, Geometry, MmFamilyProcessor, ProcessedItem, Tensor, TensorData, TokenLayout,
};

/// Resolved processor params, deserialized from the Python-side spec JSON
/// (unknown fields like `family` are ignored here).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct MiniMaxM3Spec {
    pub image_token_id: i32,
    pub image_start_token_id: i32,
    pub image_end_token_id: i32,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

pub struct MiniMaxM3Processor {
    spec: MiniMaxM3Spec,
    /// Per-channel u8 → normalized-f32 lookup for the fused
    /// rescale+normalize; see [`transforms::normalize_lut_fused`].
    lut: [[f32; 256]; 3],
}

impl MiniMaxM3Processor {
    pub fn new(spec: MiniMaxM3Spec) -> Result<Self, String> {
        if spec.patch_size == 0 || spec.merge_size == 0 || spec.temporal_patch_size == 0 {
            return Err("minimax_m3 spec: sizes must be positive".into());
        }
        let lut = core::array::from_fn(|c| {
            transforms::normalize_lut_fused(spec.image_mean[c], spec.image_std[c])
        });
        Ok(Self { spec, lut })
    }

    pub fn from_spec_json(json: &str) -> Result<Self, String> {
        let spec: MiniMaxM3Spec =
            serde_json::from_str(json).map_err(|e| format!("minimax_m3 spec: {e}"))?;
        Self::new(spec)
    }

    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / (self.spec.merge_size * self.spec.merge_size)
    }

    /// One decoded image → `pixel_values` + `image_grid_thw`, shared by the
    /// server pipeline (`process_item`) and the Python parity/preprocess
    /// bindings.
    fn process_image(&self, rgb: &[u8], h: usize, w: usize) -> Result<ProcessedItem, String> {
        let (th, tw) = transforms::smart_resize(
            h,
            w,
            self.spec.patch_size * self.spec.merge_size,
            self.spec.min_pixels,
            self.spec.max_pixels,
        )?;
        let resized;
        let data = if (th, tw) != (h, w) {
            resized = resize::resize_rgb(rgb, h, w, th, tw, resize::Resample::AtenU8);
            &resized
        } else {
            rgb
        };
        let (gh, gw) = (th / self.spec.patch_size, tw / self.spec.patch_size);
        // `smart_resize` guarantees both: dims are positive and divisible by
        // `patch_size * merge_size`. `patchify_merged_blocks` indexes on that
        // (and the `dim` division below needs a non-empty grid), so fail
        // loudly rather than panic if a future spec change breaks the
        // guarantee.
        if gh == 0 || gw == 0 || gh % self.spec.merge_size != 0 || gw % self.spec.merge_size != 0 {
            return Err(format!(
                "minimax_m3: patch grid {gh}x{gw} is empty or not a multiple of \
                 merge_size {}",
                self.spec.merge_size
            ));
        }
        let pixel_values = transforms::patchify_merged_blocks(
            data,
            th,
            tw,
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.temporal_patch_size,
            &self.lut,
        );
        let dim = pixel_values.len() / (gh * gw);
        Ok(ProcessedItem {
            feature: Tensor {
                shape: vec![gh * gw, dim],
                data: TensorData::F32(pixel_values),
            },
            aux: vec![(
                "image_grid_thw".to_string(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(vec![1, gh as i64, gw as i64]),
                },
            )],
            geometry: Geometry::Grid([1, gh as u32, gw as u32]),
        })
    }
}

impl MmFamilyProcessor for MiniMaxM3Processor {
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String> {
        let DecodedMedia::Image { rgb, height, width } = media;
        self.process_image(rgb, *height, *width)
    }

    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String> {
        let counts = items
            .iter()
            .map(|Geometry::Grid(grid)| self.tokens_per_image(grid))
            .collect::<Vec<_>>();
        token_layout::layout_by_placeholder_wrapped(
            input_ids,
            self.spec.image_token_id,
            self.spec.image_start_token_id,
            self.spec.image_end_token_id,
            &counts,
        )
    }
}

// --- Python bindings (parity tests drive the exact server pipeline) ---

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::*;
    use crate::pipeline::TensorData;

    /// `(pixel_values flat f32, (t, h, w))` for one preprocessed image.
    type PyProcessedImage<'py> = (Bound<'py, PyArray1<f32>>, (u32, u32, u32));

    /// `(pixel_values flat f32, (t, h, w))` before numpy conversion.
    type ProcessedImageTensors = (Vec<f32>, (u32, u32, u32));

    fn unpack(item: ProcessedItem) -> Result<ProcessedImageTensors, String> {
        let Geometry::Grid([t, h, w]) = item.geometry;
        let TensorData::F32(pixel_values) = item.feature.data else {
            return Err("minimax_m3: expected f32 feature".into());
        };
        Ok((pixel_values, (t, h, w)))
    }

    /// Run the full native image path on encoded image bytes:
    /// decode → smart_resize → antialias bicubic → normalize → patchify.
    /// Returns `(pixel_values flat f32, (t, h, w))`.
    #[pyfunction]
    fn preprocess<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        spec_json: &str,
    ) -> PyResult<PyProcessedImage<'py>> {
        let proc = MiniMaxM3Processor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let (pixel_values, grid) = py
            .detach(move || {
                let (rgb, height, width) = crate::common::decode_rgb(&data)?;
                proc.process_image(&rgb, height, width).and_then(unpack)
            })
            .map_err(PyValueError::new_err)?;
        Ok((pixel_values.into_pyarray(py), grid))
    }

    /// The same pipeline from already-decoded pixels (the live Python
    /// integration: sglang loads images to PIL before the processor runs, and
    /// starting from the same pixels keeps the two paths bit-identical
    /// regardless of image format). One result per input image, parallelized
    /// across the batch. `arrays` are HWC u8 RGB.
    #[pyfunction]
    fn preprocess_arrays<'py>(
        py: Python<'py>,
        arrays: Vec<PyReadonlyArray3<'py, u8>>,
        spec_json: &str,
    ) -> PyResult<Vec<PyProcessedImage<'py>>> {
        let proc = MiniMaxM3Processor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let mut decoded = Vec::with_capacity(arrays.len());
        for arr in arrays {
            let shape = arr.shape();
            let (h, w, c) = (shape[0], shape[1], shape[2]);
            if c != 3 {
                return Err(PyValueError::new_err(format!(
                    "expected HWC RGB array with 3 channels, got {c}"
                )));
            }
            let rgb = arr
                .as_slice()
                .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?
                .to_vec();
            decoded.push((rgb, h, w));
        }
        let results = py
            .detach(move || {
                crate::common::par::try_map(&decoded, |(rgb, h, w)| {
                    proc.process_image(rgb, *h, *w).and_then(unpack)
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok(results
            .into_iter()
            .map(|(pv, grid)| (pv.into_pyarray(py), grid))
            .collect())
    }

    #[pyfunction]
    fn smart_resize_py(
        height: usize,
        width: usize,
        factor: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> PyResult<(usize, usize)> {
        transforms::smart_resize(height, width, factor, min_pixels, max_pixels)
            .map_err(PyValueError::new_err)
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "minimax_m3")?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        m.add_function(wrap_pyfunction!(preprocess_arrays, &m)?)?;
        m.add_function(wrap_pyfunction!(smart_resize_py, &m)?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::token_layout::apply_layout;
    use crate::pipeline::MmFamilyProcessor;

    fn spec() -> MiniMaxM3Spec {
        MiniMaxM3Spec {
            image_token_id: 1,
            image_start_token_id: 90,
            image_end_token_id: 91,
            patch_size: 2,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 4,
            max_pixels: 1 << 30,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        }
    }

    /// Geometry values generated from the HF `MiniMaxM3VLImageProcessor`'s
    /// `smart_resize` (transformers 5.12.1, factor=28, min_pixels=3136,
    /// max_pixels=451584).
    #[test]
    fn smart_resize_matches_m3_reference() {
        let sr = |h, w| transforms::smart_resize(h, w, 28, 3136, 451584).unwrap();
        assert_eq!(sr(640, 480), (644, 476));
        assert_eq!(sr(1024, 683), (812, 532));
        assert_eq!(sr(50, 40), (84, 56));
        assert_eq!(sr(300, 301), (308, 308));
        assert_eq!(sr(1365, 2048), (532, 812));
        // Downscale branch: 4000x3000 exceeds 672*672 → floor_by_factor.
        assert_eq!(sr(4000, 3000), (756, 560));
        // Upscale branch: tiny image below min_pixels → ceil_by_factor.
        assert_eq!(sr(20, 20), (56, 56));
        // No-op when already on grid within the pixel budget.
        assert_eq!(sr(672, 672), (672, 672));
        // Banker's rounding ties: 42/28 = 1.5 → 2; 70/28 = 2.5 → 2 (not 3).
        assert_eq!(sr(42, 1024), (56, 1036));
        assert_eq!(sr(70, 1024), (56, 1036));
        // Extreme aspect ratio rejected.
        assert!(transforms::smart_resize(10000, 10, 28, 3136, 451584).is_err());
    }

    /// The fused LUT folds the 1/255 rescale into mean/std, exactly as
    /// `rescale_and_normalize` does for the HF processor (values computed from
    /// torchvision on the fused constants).
    #[test]
    fn normalize_lut_matches_fused_reference() {
        let lut = transforms::normalize_lut_fused(0.48145466, 0.26862954);
        assert_eq!(
            lut[0],
            (0.0f32 - 0.48145466f32 * 255.0) / (0.26862954f32 * 255.0)
        );
        assert_eq!(
            lut[255],
            (255.0f32 - 0.48145466f32 * 255.0) / (0.26862954f32 * 255.0)
        );
        assert!((lut[128] - 0.07633601).abs() < 1e-7, "{}", lut[128]);
        // Zero mean/unit std degenerates to v/255.
        let identity = transforms::normalize_lut_fused(0.0, 1.0);
        assert_eq!(identity[128], 128.0f32 / 255.0);
    }

    /// Patch order, channel layout and temporal duplication of one item, on a
    /// hand-computable image. Same HF flatten contract as the Qwen family's
    /// `patchify_layout_matches_hf_order`.
    #[test]
    fn process_item_layout_matches_hf_order() {
        // 4x8 image, ps=2, m=2, tps=2 → gh=2, gw=4, dim=3*2*2*2=24.
        // Pixel value encodes its (y, x): v = y*16 + x*2 (fits u8).
        let (h, w) = (4usize, 8usize);
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = (y * 16 + x * 2 + c) as u8;
                }
            }
        }
        let proc = MiniMaxM3Processor::new(spec()).unwrap();
        let item = proc
            .process_item(&DecodedMedia::Image {
                rgb,
                height: h,
                width: w,
            })
            .unwrap();
        let TensorData::F32(pv) = item.feature.data else {
            panic!("expected f32 feature")
        };
        let dim = 24; // 3 * tps * ps * ps
        assert_eq!(item.feature.shape, [8, dim]);
        assert_eq!(pv.len(), 2 * 4 * dim);
        let Geometry::Grid(grid) = item.geometry;
        assert_eq!(grid, [1, 2, 4]);

        // Patch order (gh/m=1, gw/m=2, m, m): patch 0 = block(0,0) offset (0,0),
        // patch 1 = (0,0)+(0,1) → x0=2, patch 2 = (0,0)+(1,0) → y0=2,
        // patch 4 = block(0,1) → x0=4. lut with mean 0/std 1 is v/255.
        let lut = |y: usize, x: usize, c: usize| ((y * 16 + x * 2 + c) as f32) / 255.0;
        assert_eq!(pv[dim], lut(0, 2, 0));
        assert_eq!(pv[2 * dim], lut(2, 0, 0));
        assert_eq!(pv[4 * dim], lut(0, 4, 0));
        // Temporal duplicate: t=1 block equals t=0 block.
        let ps2 = 4; // ps*ps
        assert_eq!(pv[dim + ps2], pv[dim]);
        // Channel 1 block of patch 0 → same pixel, c=1.
        assert_eq!(pv[2 * ps2], lut(0, 0, 1)); // c stride = tps*ps*ps = 8
    }

    /// The prompt expands `[text, IMG, text]` into
    /// `[text, START, IMG × n, END, text]` with offsets over the inner run —
    /// the Python `get_mm_items_offset` convention the scheduler consumes.
    #[test]
    fn layout_wraps_image_span() {
        let proc = MiniMaxM3Processor::new(spec()).unwrap();
        // grid [1, 4, 6], m=2 → 24/4 = 6 tokens.
        let layout = proc
            .layout(&[7, 1, 8], &[Geometry::Grid([1, 4, 6])])
            .unwrap();
        let expanded = apply_layout(&[7, 1, 8], &layout, 1).unwrap();
        assert_eq!(expanded.input_ids, vec![7, 90, 1, 1, 1, 1, 1, 1, 91, 8]);
        assert_eq!(expanded.offsets, vec![(2, 7)]);

        // Placeholder/item count mismatch is a request error.
        assert!(proc.layout(&[7, 8], &[Geometry::Grid([1, 4, 6])]).is_err());
    }

    /// A thin image against a small `max_pixels` floors one side to 0; the
    /// family must reject it, never panic on a worker thread.
    #[test]
    fn degenerate_target_is_rejected_not_panicked() {
        let mut spec = spec();
        spec.patch_size = 14;
        spec.min_pixels = 3136;
        spec.max_pixels = 3136;
        let proc = MiniMaxM3Processor::new(spec).unwrap();
        let err = proc
            .process_item(&DecodedMedia::Image {
                rgb: vec![0u8; 10 * 2000 * 3],
                height: 10,
                width: 2000,
            })
            .err()
            .expect("degenerate geometry must be an Err, never a panic");
        assert!(err.contains("smart_resize"), "unexpected error: {err}");
    }

    /// The server's message layer gates modalities on what a family declares,
    /// so a family gaining video/audio support must not silently inherit the
    /// images-only default.
    #[test]
    fn minimax_m3_declares_images_only() {
        let caps = MiniMaxM3Processor::new(spec()).unwrap().capabilities();
        assert!(!caps.video && !caps.audio);
    }

    /// Zero-sized patch geometry in the spec is a config error, not a panic.
    #[test]
    fn invalid_spec_rejected() {
        let mut spec = spec();
        spec.patch_size = 0;
        assert!(MiniMaxM3Processor::new(spec).is_err());
    }
}
