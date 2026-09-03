//! Qwen VL family (Qwen2-VL / 2.5-VL / 3-VL / 3.5) server-pipeline image processor.
//!
//! Pure-Rust equivalent of the HF `Qwen2VLImageProcessor` pipeline the Python
//! `QwenVLImageProcessor` drives: `smart_resize` → bicubic resize → rescale +
//! normalize → patchify into `[grid_h*grid_w, C*tps*ps*ps]` (HF flatten order:
//! patches by `(gh/m, gw/m, m, m)`, features by `(C, tps, ps, ps)`, temporal
//! copies duplicated for stills) — plus the image-only M-RoPE fast path.
//! All parameters come from the runtime spec; nothing is hardcoded per model.

use crate::common::{par, resize, token_layout, transforms};
use crate::pipeline::{
    DecodedMedia, Geometry, MmFamilyProcessor, PositionOutput, ProcessedItem, Tensor, TensorData,
    TokenLayout,
};

// The dynamic-resize geometry is shared with the other VL families
// (`common::transforms`); re-exported so existing users keep resolving it here.
pub use crate::common::transforms::smart_resize;

/// The fused resize+normalize+patchify path is bitwise identical to the
/// unfused chain (see [`QwenVlProcessor::patchify_fused`]), so it is the
/// default. `SGL_MM_RS_FUSED=0` forces the unfused chain, kept for A/B
/// benches and debugging.
fn fused_enabled() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| !matches!(std::env::var("SGL_MM_RS_FUSED"), Ok(v) if v == "0"))
}

/// One media item's placement for M-RoPE: inclusive token range + patch grid.
pub struct MropeItem {
    pub start: u32,
    pub end: u32,
    pub grid: [u32; 3],
}

/// Resolved processor params, deserialized from the Python-side spec JSON
/// (unknown fields like `family` are ignored here).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct QwenVlSpec {
    pub image_token_id: i32,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    #[serde(default)]
    pub resample: Resampler,
}

/// The HF image processor the pipeline must match bit-exactly. Defaults to the
/// one a default server runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resampler {
    /// `Qwen2VLImageProcessor` / `…Fast` — torchvision on a uint8 tensor.
    #[default]
    AtenU8,
    /// `Qwen2VLImageProcessorPil`, behind `--disable-fast-image-processor`.
    Pil,
}

impl From<Resampler> for resize::Resample {
    fn from(r: Resampler) -> Self {
        match r {
            Resampler::AtenU8 => resize::Resample::AtenU8,
            Resampler::Pil => resize::Resample::Pil(resize::Filter::Bicubic),
        }
    }
}

pub struct QwenVlProcessor {
    spec: QwenVlSpec,
    /// Per-channel u8 → normalized-f32 lookup; see [`normalize_lut`].
    lut: [[f32; 256]; 3],
}

/// `1 / rescale_factor`; `resolve_spec` rejects any other factor.
const INV_RESCALE: f32 = 255.0;

/// u8 → normalized f32, rounded as the mirrored processor rounds. The slow one
/// rescales then normalizes; the fast one folds the rescale into mean/std first
/// (`_fuse_mean_std_and_rescale_factor`), which differs on 128 of the 256 inputs.
fn normalize_lut(resample: Resampler, mean: f32, std: f32) -> [f32; 256] {
    match resample {
        Resampler::Pil => core::array::from_fn(|v| (v as f32 / INV_RESCALE - mean) / std),
        Resampler::AtenU8 => {
            let (mean, std) = (mean * INV_RESCALE, std * INV_RESCALE);
            core::array::from_fn(|v| (v as f32 - mean) / std)
        }
    }
}

impl QwenVlProcessor {
    pub fn new(spec: QwenVlSpec) -> Result<Self, String> {
        if spec.patch_size == 0 || spec.merge_size == 0 || spec.temporal_patch_size == 0 {
            return Err("qwen_vl spec: sizes must be positive".into());
        }
        let lut = core::array::from_fn(|c| {
            normalize_lut(spec.resample, spec.image_mean[c], spec.image_std[c])
        });
        Ok(Self { spec, lut })
    }

    pub fn from_spec_json(json: &str) -> Result<Self, String> {
        let spec: QwenVlSpec =
            serde_json::from_str(json).map_err(|e| format!("qwen_vl spec: {e}"))?;
        Self::new(spec)
    }

    fn factor(&self) -> usize {
        self.spec.patch_size * self.spec.merge_size
    }

    /// HF flatten: patches ordered `(gh/m, gw/m, m, m)`, features `(C, tps,
    /// ps, ps)`; parallel over merged-block rows.
    fn patchify(&self, rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
        transforms::patchify_merged_blocks(
            rgb,
            h,
            w,
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.temporal_patch_size,
            &self.lut,
        )
    }

    /// Fused resize + normalize + patchify: a [`resize::RowProducer`] yields
    /// the resized u8 rows one at a time and each is scattered straight into
    /// the patch layout through the LUT — the full resized u8 image is never
    /// materialized. Bitwise identical to `resize::resize_rgb` followed by
    /// [`Self::patchify`] (the row producer shares its fixed-point math with
    /// the two-pass resize; test `fused_matches_unfused_bitwise`).
    fn patchify_fused(&self, rgb: &[u8], h: usize, w: usize, th: usize, tw: usize) -> Vec<f32> {
        let (ps, m, tps) = (
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.temporal_patch_size,
        );
        let (gh, gw) = (th / ps, tw / ps);
        let dim = 3 * tps * ps * ps;
        let block_row = gw * m * dim; // one merged-block row of patches
        let mut out = vec![0.0f32; gh * gw * dim];

        par::in_pool(|| {
            let producer = resize::RowProducer::new(rgb, h, w, th, tw, self.spec.resample.into());
            par::for_chunks_mut(&mut out, block_row, |i, chunk| {
                let mut rowbuf = vec![0u8; tw * 3];
                for mh in 0..m {
                    for py in 0..ps {
                        let yy = (i * m + mh) * ps + py;
                        producer.row(yy, &mut rowbuf);
                        for j in 0..gw / m {
                            for mw in 0..m {
                                let x0 = (j * m + mw) * ps;
                                let p = j * m * m + mh * m + mw;
                                let patch = &mut chunk[p * dim..(p + 1) * dim];
                                for c in 0..3 {
                                    let lut = &self.lut[c];
                                    let (t0, rest) = patch
                                        [c * tps * ps * ps..(c + 1) * tps * ps * ps]
                                        .split_at_mut(ps * ps);
                                    let src = x0 * 3 + c;
                                    for px in 0..ps {
                                        let v = lut[rowbuf[src + px * 3] as usize];
                                        t0[py * ps + px] = v;
                                        // Temporal copies of a still are
                                        // duplicates: store every slot while
                                        // the value is at hand.
                                        for t in 1..tps {
                                            rest[(t - 1) * ps * ps + py * ps + px] = v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        });
        out
    }
}

impl QwenVlProcessor {
    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / (self.spec.merge_size * self.spec.merge_size)
    }
}

/// Spike-bench output: the processed item plus stage timings and the dims the
/// decode backend actually produced (`decoded_*` differs from the header dims
/// under scaled iDCT).
pub struct TimedProcess {
    pub item: ProcessedItem,
    pub decode_ns: u64,
    pub post_ns: u64,
    pub decoded_h: usize,
    pub decoded_w: usize,
}

impl QwenVlProcessor {
    /// `process_item` from encoded bytes with a selectable decode backend and
    /// post-decode structure — the spike bench's A/B driver. `backend`:
    /// * `"baseline"`: `image`-crate decode, unfused resize → patchify
    ///   (the pre-spike chain);
    /// * `"fused"`: `image`-crate decode, fused resize+normalize+patchify —
    ///   bitwise identical to `"baseline"`;
    /// * `"turbo"` / `"turbo_fused"`: libjpeg-turbo scaled-iDCT decode with
    ///   the same post-decode split. **Tolerance mode**: the scaled iDCT
    ///   changes the resize input, and even full-resolution libjpeg output is
    ///   not bit-identical to the pure-Rust decoder.
    pub fn preprocess_timed(&self, data: &[u8], backend: &str) -> Result<TimedProcess, String> {
        let (turbo, fused) = match backend {
            "baseline" => (false, false),
            "fused" => (false, true),
            "turbo" => (true, false),
            "turbo_fused" => (true, true),
            other => {
                return Err(format!(
                    "unknown backend {other:?}; expected \"baseline\", \"fused\", \
                     \"turbo\" or \"turbo_fused\""
                ));
            }
        };
        let t0 = std::time::Instant::now();
        let (rgb, dh, dw, th, tw) = if turbo {
            self.turbo_decode_scaled(data)?
        } else {
            let (rgb, h, w) = crate::common::decode_rgb(data)?;
            let (th, tw) = smart_resize(
                h,
                w,
                self.factor(),
                self.spec.min_pixels,
                self.spec.max_pixels,
            )?;
            (rgb, h, w, th, tw)
        };
        let decode_ns = t0.elapsed().as_nanos() as u64;

        let t1 = std::time::Instant::now();
        let (gh, gw) = (th / self.spec.patch_size, tw / self.spec.patch_size);
        if gh == 0 || gw == 0 || gh % self.spec.merge_size != 0 || gw % self.spec.merge_size != 0 {
            return Err(format!(
                "qwen_vl: patch grid {gh}x{gw} is empty or not a multiple of \
                 merge_size {}",
                self.spec.merge_size
            ));
        }
        let pixel_values = if (th, tw) == (dh, dw) {
            self.patchify(&rgb, th, tw)
        } else if fused {
            self.patchify_fused(&rgb, dh, dw, th, tw)
        } else {
            let resized = resize::resize_rgb(&rgb, dh, dw, th, tw, self.spec.resample.into());
            self.patchify(&resized, th, tw)
        };
        let post_ns = t1.elapsed().as_nanos() as u64;

        let dim = pixel_values.len() / (gh * gw);
        Ok(TimedProcess {
            item: ProcessedItem {
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
            },
            decode_ns,
            post_ns,
            decoded_h: dh,
            decoded_w: dw,
        })
    }

    /// Decode via the turbo backend: header → `smart_resize` target →
    /// scaled-iDCT decode covering the target. Non-JPEG input (scaled iDCT is
    /// JPEG-only) and anything libjpeg-turbo rejects decode at full
    /// resolution through the usual path.
    #[cfg(feature = "turbo-jpeg")]
    fn turbo_decode_scaled(
        &self,
        data: &[u8],
    ) -> Result<(Vec<u8>, usize, usize, usize, usize), String> {
        if crate::common::turbo::is_jpeg(data) {
            let (oh, ow) = crate::common::turbo::header(data)?;
            let (th, tw) = smart_resize(
                oh,
                ow,
                self.factor(),
                self.spec.min_pixels,
                self.spec.max_pixels,
            )?;
            let (rgb, dh, dw) = crate::common::turbo::decode_rgb_scaled(data, th, tw)?;
            Ok((rgb, dh, dw, th, tw))
        } else {
            let (rgb, h, w) = crate::common::decode_rgb(data)?;
            let (th, tw) = smart_resize(
                h,
                w,
                self.factor(),
                self.spec.min_pixels,
                self.spec.max_pixels,
            )?;
            Ok((rgb, h, w, th, tw))
        }
    }

    #[cfg(not(feature = "turbo-jpeg"))]
    fn turbo_decode_scaled(
        &self,
        _data: &[u8],
    ) -> Result<(Vec<u8>, usize, usize, usize, usize), String> {
        Err("turbo-jpeg feature not built; re-run with --features turbo-jpeg".into())
    }
}

impl MmFamilyProcessor for QwenVlProcessor {
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String> {
        let DecodedMedia::Image { rgb, height, width } = media;
        let (h, w) = (*height, *width);
        let (th, tw) = smart_resize(
            h,
            w,
            self.factor(),
            self.spec.min_pixels,
            self.spec.max_pixels,
        )?;
        let (gh, gw) = (th / self.spec.patch_size, tw / self.spec.patch_size);
        // `smart_resize` guarantees both: dims are positive and divisible by
        // `patch_size * merge_size`. `patchify` indexes on that (and the `dim`
        // division below needs a non-empty grid), so fail loudly rather than
        // panic if a future spec change breaks the guarantee.
        if gh == 0 || gw == 0 || gh % self.spec.merge_size != 0 || gw % self.spec.merge_size != 0 {
            return Err(format!(
                "qwen_vl: patch grid {gh}x{gw} is empty or not a multiple of \
                 merge_size {}",
                self.spec.merge_size
            ));
        }
        let pixel_values = if (th, tw) == (h, w) {
            self.patchify(rgb, th, tw)
        } else if fused_enabled() {
            self.patchify_fused(rgb, h, w, th, tw)
        } else {
            let resized = resize::resize_rgb(rgb, h, w, th, tw, self.spec.resample.into());
            self.patchify(&resized, th, tw)
        };
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

    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String> {
        let counts = items
            .iter()
            .map(|Geometry::Grid(grid)| self.tokens_per_image(grid))
            .collect::<Vec<_>>();
        token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }

    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let mrope_items = offsets
            .iter()
            .zip(items)
            .map(|(&(start, end), Geometry::Grid(grid))| MropeItem {
                start,
                end,
                grid: *grid,
            })
            .collect::<Vec<_>>();
        let (positions, delta) = mrope_image_only(input_len, &mrope_items, self.spec.merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
    }
}

/// Image-only M-RoPE fast path (the image branch of
/// `MRotaryEmbedding.get_rope_index`, identical across Qwen generations):
/// text runs sequentially on all three rows; each image spans `(t, h/m, w/m)`
/// index grids; positions advance by `max(t, h/m, w/m)` past an image.
/// Returns flattened row-major `[3, input_len]` positions and the delta
/// (`max + 1 - input_len`). `items` must be in prompt order.
pub fn mrope_image_only(
    input_len: usize,
    items: &[MropeItem],
    merge_size: usize,
) -> Result<(Vec<i64>, i64), String> {
    let len = input_len;
    let mut pos = vec![0i64; 3 * len];
    let fill_text = |st: usize, n: usize, base: i64, pos: &mut [i64]| {
        for k in 0..n {
            let v = base + k as i64;
            pos[st + k] = v;
            pos[len + st + k] = v;
            pos[2 * len + st + k] = v;
        }
    };
    let mut st = 0usize;
    let mut next_pos = 0i64;
    for item in items {
        let (start, end) = (item.start as usize, item.end as usize);
        if start < st || end >= len {
            return Err(format!(
                "mrope: item range ({start},{end}) out of order/bounds"
            ));
        }
        fill_text(st, start - st, next_pos, &mut pos);
        next_pos += (start - st) as i64;

        let t = item.grid[0] as usize;
        let gh = item.grid[1] as usize / merge_size;
        let gw = item.grid[2] as usize / merge_size;
        if t * gh * gw != end - start + 1 {
            return Err("mrope: token span does not match grid".into());
        }
        for ti in 0..t {
            for hi in 0..gh {
                for wi in 0..gw {
                    let idx = start + (ti * gh + hi) * gw + wi;
                    pos[idx] = next_pos + ti as i64;
                    pos[len + idx] = next_pos + hi as i64;
                    pos[2 * len + idx] = next_pos + wi as i64;
                }
            }
        }
        next_pos += (t.max(gh).max(gw)) as i64;
        st = end + 1;
    }
    if st < len {
        fill_text(st, len - st, next_pos, &mut pos);
    }
    let max = pos.iter().copied().max().unwrap_or(-1);
    Ok((pos, max + 1 - len as i64))
}

/// The qwen scheduler-drain shape, extracted from the generic driver
/// [`Output`](crate::driver::Output). Shared by `sglang-server`'s MM worker
/// and the parity binding so the mapping can't drift; replaced by a generic
/// named-tensor handoff once a second family needs a different shape.
pub struct QwenDrain {
    pub input_ids: Vec<i32>,
    /// All items' `pixel_values`, concatenated in prompt order.
    pub features: Vec<f32>,
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

pub fn pack_drain(output: crate::driver::Output) -> Result<QwenDrain, String> {
    use crate::pipeline::PositionOutput;

    let PositionOutput::MRope { positions, delta } = output.positions else {
        return Err("qwen_vl drain: expected M-RoPE positions".into());
    };
    let mut features = Vec::new();
    let mut grids = Vec::with_capacity(output.items.len());
    let mut hashes = Vec::with_capacity(output.items.len());
    for item in output.items {
        let TensorData::F32(pixel_values) = item.feature.data else {
            return Err("qwen_vl drain: expected f32 feature".into());
        };
        features.extend(pixel_values);
        let grid = item
            .aux
            .into_iter()
            .find_map(|(name, tensor)| match (name.as_str(), tensor.data) {
                ("image_grid_thw", TensorData::I64(v)) => Some(v),
                _ => None,
            })
            .ok_or("qwen_vl drain: missing image_grid_thw")?;
        grids.push([grid[0] as u32, grid[1] as u32, grid[2] as u32]);
        hashes.push(item.hash);
    }
    Ok(QwenDrain {
        input_ids: output.input_ids,
        features,
        grids,
        hashes,
        offsets: output.offsets,
        mrope: positions,
        mrope_delta: delta,
    })
}

// --- Python bindings (parity tests drive the exact server pipeline) ---

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::*;
    use crate::pipeline::TensorData;

    /// `(pixel_values flat f32, (t, h, w))` for one preprocessed image.
    type PyProcessedImage<'py> = (Bound<'py, PyArray1<f32>>, (u32, u32, u32));
    /// Full Rust pipeline output at the scheduler boundary:
    /// `(input_ids, features, grids, hashes, offsets, mrope, mrope_delta)`.
    type PyNativeOutput<'py> = (
        Vec<i32>,
        Bound<'py, PyArray1<f32>>,
        Vec<(u32, u32, u32)>,
        Vec<u64>,
        Vec<(u32, u32)>,
        Bound<'py, PyArray1<i64>>,
        i64,
    );

    /// Run the full native image path on encoded image bytes:
    /// decode → smart_resize → bicubic → normalize → patchify.
    /// Returns `(pixel_values flat f32, (t, h, w))`.
    #[pyfunction]
    fn preprocess<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        spec_json: &str,
    ) -> PyResult<PyProcessedImage<'py>> {
        let proc = QwenVlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let out = py
            .detach(move || {
                let (rgb, height, width) = crate::common::decode_rgb(&data)?;
                proc.process_item(&DecodedMedia::Image { rgb, height, width })
            })
            .map_err(PyValueError::new_err)?;
        let Geometry::Grid([t, h, w]) = out.geometry;
        let TensorData::F32(pixel_values) = out.feature.data else {
            return Err(PyValueError::new_err("qwen_vl: expected f32 feature"));
        };
        Ok((pixel_values.into_pyarray(py), (t, h, w)))
    }

    #[pyfunction]
    fn smart_resize_py(
        height: usize,
        width: usize,
        factor: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> PyResult<(usize, usize)> {
        smart_resize(height, width, factor, min_pixels, max_pixels).map_err(PyValueError::new_err)
    }

    /// `(positions flat [3*input_len], delta)` for image-only requests;
    /// `items` = [(start, end_inclusive, t, h, w), ...] in prompt order.
    #[pyfunction]
    fn mrope_image_only_py<'py>(
        py: Python<'py>,
        input_len: usize,
        items: Vec<(u32, u32, u32, u32, u32)>,
        merge_size: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, i64)> {
        let items: Vec<MropeItem> = items
            .into_iter()
            .map(|(start, end, t, h, w)| MropeItem {
                start,
                end,
                grid: [t, h, w],
            })
            .collect();
        let (pos, delta) =
            mrope_image_only(input_len, &items, merge_size).map_err(PyValueError::new_err)?;
        Ok((pos.into_pyarray(py), delta))
    }

    /// One image source: a `str` (data:/base64/file/http, resolved by
    /// `common::fetch`) or raw encoded `bytes`.
    #[derive(FromPyObject)]
    enum PyImageSource {
        Str(String),
        Bytes(Vec<u8>),
    }

    /// Drive the same typed native Qwen request pipeline used by
    /// `sglang-server` (whose message layer owns the wire-payload parsing).
    #[pyfunction]
    #[pyo3(signature = (input_ids, images, spec_json))]
    fn process_mm<'py>(
        py: Python<'py>,
        input_ids: Option<Vec<i32>>,
        images: Vec<PyImageSource>,
        spec_json: String,
    ) -> PyResult<PyNativeOutput<'py>> {
        let images = images
            .into_iter()
            .map(|source| match source {
                PyImageSource::Str(s) => crate::driver::ImageSource::String(s),
                PyImageSource::Bytes(b) => crate::driver::ImageSource::Bytes(b),
            })
            .collect();
        let input = crate::driver::MmInput {
            text: None,
            input_ids,
            images,
        };
        let drain = py
            .detach(move || {
                let family = crate::registry::pipeline_from_spec(&spec_json)?;
                let output = crate::driver::process(family.as_ref(), input, |_| {
                    Err("native parity API requires input_ids".into())
                })?;
                pack_drain(output)
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            drain.input_ids,
            drain.features.into_pyarray(py),
            drain.grids.into_iter().map(|[t, h, w]| (t, h, w)).collect(),
            drain.hashes,
            drain.offsets,
            drain.mrope.into_pyarray(py),
            drain.mrope_delta,
        ))
    }

    /// Spike-bench entry: `(pixel_values, (t,h,w), decode_ns, post_ns,
    /// decoded_h, decoded_w)` for one image under `backend` ∈ {"baseline",
    /// "fused", "turbo", "turbo_fused"}; see
    /// [`QwenVlProcessor::preprocess_timed`].
    type PreprocessTimedOut<'py> = (
        Bound<'py, PyArray1<f32>>,
        (u32, u32, u32),
        u64,
        u64,
        usize,
        usize,
    );

    #[pyfunction]
    fn preprocess_timed<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        spec_json: &str,
        backend: &str,
    ) -> PyResult<PreprocessTimedOut<'py>> {
        let proc = QwenVlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let out = py
            .detach(move || proc.preprocess_timed(&data, backend))
            .map_err(PyValueError::new_err)?;
        let Geometry::Grid([t, h, w]) = out.item.geometry;
        let TensorData::F32(pixel_values) = out.item.feature.data else {
            return Err(PyValueError::new_err("qwen_vl: expected f32 feature"));
        };
        Ok((
            pixel_values.into_pyarray(py),
            (t, h, w),
            out.decode_ns,
            out.post_ns,
            out.decoded_h,
            out.decoded_w,
        ))
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "qwen_vl")?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        m.add_function(wrap_pyfunction!(preprocess_timed, &m)?)?;
        m.add_function(wrap_pyfunction!(smart_resize_py, &m)?)?;
        m.add_function(wrap_pyfunction!(mrope_image_only_py, &m)?)?;
        m.add_function(wrap_pyfunction!(process_mm, &m)?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> QwenVlSpec {
        QwenVlSpec {
            image_token_id: 1,
            patch_size: 2,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 4,
            max_pixels: 1 << 30,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            resample: Resampler::default(),
        }
    }

    /// The fused and unfused normalize forms are not interchangeable: with
    /// mean = std = 0.5 they disagree on 128 of the 256 u8 inputs, so picking
    /// the wrong one silently costs bit-exactness with the HF processor.
    #[test]
    fn normalize_lut_differs_per_resampler() {
        let pil = normalize_lut(Resampler::Pil, 0.5, 0.5);
        let aten = normalize_lut(Resampler::AtenU8, 0.5, 0.5);
        assert_eq!(pil.iter().zip(aten).filter(|(p, a)| *p != a).count(), 128);
        // Both still span [-1, 1] — this is rounding, not a scale error.
        for lut in [pil, aten] {
            assert_eq!(lut[0], -1.0);
            assert_eq!(lut[255], 1.0);
        }
    }

    #[test]
    fn smart_resize_matches_python_reference() {
        // Values from the Python `smart_resize` (qwen_vl.py) run offline.
        assert_eq!(
            smart_resize(1365, 2048, 28, 3136, 12845056).unwrap(),
            (1372, 2044)
        );
        assert_eq!(
            smart_resize(100, 100, 28, 3136, 12845056).unwrap(),
            (112, 112)
        );
        // Downscale branch: 4000x3000 exceeds 1280*28*28 → floor_by_factor.
        assert_eq!(
            smart_resize(3000, 4000, 28, 3136, 1003520).unwrap(),
            (840, 1148)
        );
        // Upscale branch: tiny image below min_pixels → ceil_by_factor.
        assert_eq!(smart_resize(20, 20, 28, 3136, 12845056).unwrap(), (56, 56));
        // Qwen3.5 factors (patch 16 * merge 2, min 65536, max 16777216).
        assert_eq!(
            smart_resize(1365, 2048, 32, 65536, 16777216).unwrap(),
            (1376, 2048)
        );
        // Banker's rounding tie: 48/32 = 1.5 rounds to 2 (even), not 1.
        assert_eq!(smart_resize(4000, 48, 32, 4, 1 << 30).unwrap(), (4000, 64));
        // Extreme aspect ratio rejected.
        assert!(smart_resize(10000, 10, 28, 3136, 12845056).is_err());
    }

    /// A thin image against a small `max_pixels` floors one side to 0. That
    /// used to reach the resize coefficient math and panic on a worker thread
    /// (`attempt to multiply with overflow`) instead of rejecting the request.
    #[test]
    fn degenerate_target_is_rejected_not_panicked() {
        // Aspect ratio 200 is exactly at MAX_RATIO, so it passes that guard;
        // 10 / beta then floors to 0 with factor 28.
        assert!(smart_resize(10, 2000, 28, 3136, 3136).is_err());

        let mut spec = spec();
        spec.patch_size = 14;
        spec.min_pixels = 3136;
        spec.max_pixels = 3136;
        let proc = QwenVlProcessor::new(spec).unwrap();
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
    fn qwen_declares_images_only() {
        let caps = QwenVlProcessor::new(spec()).unwrap().capabilities();
        assert!(!caps.video && !caps.audio);
    }

    #[test]
    fn patchify_layout_matches_hf_order() {
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
        let proc = QwenVlProcessor::new(spec()).unwrap();
        let pv = proc.patchify(&rgb, h, w);
        let dim = 24; // 3 * tps * ps * ps
        assert_eq!(pv.len(), 2 * 4 * dim);

        // Patch order (gh/m=1, gw/m=2, m, m): patch 0 = block(0,0) offset (0,0),
        // patch 1 = (0,0)+(0,1) → x0=2, patch 2 = (0,0)+(1,0) → y0=2,
        // patch 4 = block(0,1) → x0=4.
        let lut = |y: usize, x: usize, c: usize| ((y * 16 + x * 2 + c) as f32) / 255.0;
        // patch 1, channel 0, t=0, (py=0, px=0) → pixel (0, 2).
        assert_eq!(pv[dim], lut(0, 2, 0));
        // patch 2, channel 0, t=0, (0,0) → pixel (2, 0).
        assert_eq!(pv[2 * dim], lut(2, 0, 0));
        // patch 4, channel 0 → pixel (0, 4).
        assert_eq!(pv[4 * dim], lut(0, 4, 0));
        // Temporal duplicate: t=1 block equals t=0 block.
        let ps2 = 4; // ps*ps
        assert_eq!(pv[dim + ps2], pv[dim]);
        // Channel 1 block of patch 0 → same pixel, c=1.
        assert_eq!(pv[2 * ps2], lut(0, 0, 1)); // c stride = tps*ps*ps = 8
    }

    #[test]
    fn mrope_image_only_matches_reference() {
        // 3 text tokens, image of grid [1, 4, 6] (m=2 → 2x3 = 6 tokens), 2 text.
        // input: [T T T I I I I I I T T], len 11.
        let items = [MropeItem {
            start: 3,
            end: 8,
            grid: [1, 4, 6],
        }];
        let (pos, delta) = mrope_image_only(11, &items, 2).unwrap();
        let len = 11;
        // Text prefix 0..3: all rows 0,1,2.
        for k in 0..3 {
            assert_eq!(
                (pos[k], pos[len + k], pos[2 * len + k]),
                (k as i64, k as i64, k as i64)
            );
        }
        // Image tokens: t=0, h in 0..2, w in 0..3, +3 offset.
        assert_eq!((pos[3], pos[len + 3], pos[2 * len + 3]), (3, 3, 3));
        assert_eq!((pos[4], pos[len + 4], pos[2 * len + 4]), (3, 3, 4));
        assert_eq!((pos[6], pos[len + 6], pos[2 * len + 6]), (3, 4, 3));
        // Text tail resumes at 3 + max(1,2,3) = 6.
        assert_eq!((pos[9], pos[len + 9], pos[2 * len + 9]), (6, 6, 6));
        assert_eq!((pos[10], pos[len + 10], pos[2 * len + 10]), (7, 7, 7));
        // delta = max + 1 - len = 7 + 1 - 11.
        assert_eq!(delta, -3);
    }

    /// The fused resize+normalize+patchify must be bitwise identical to
    /// `resize_rgb` followed by `patchify` — f32 `Vec` equality is a bit
    /// comparison. Covers both resamplers, both-axes / h-only / w-only /
    /// upscale / 1-px cases, plus a random sweep, and the no-resize case
    /// against `patchify` directly.
    #[test]
    fn fused_matches_unfused_bitwise() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for resampler in [Resampler::AtenU8, Resampler::Pil] {
            let mut spec = spec();
            spec.resample = resampler;
            let proc = QwenVlProcessor::new(spec).unwrap();
            let cases = [
                (37usize, 53usize, 8usize, 12usize),
                (53, 37, 8, 8),
                (16, 16, 8, 12), // vertical only
                (8, 12, 8, 24),  // horizontal only
                (8, 12, 16, 16), // upscale
                (4, 4, 8, 8),    // upscale both axes
                (100, 173, 24, 40),
                (1, 1, 4, 4),
                (5, 500, 4, 124),
            ];
            for (h, w, th, tw) in cases {
                let rgb: Vec<u8> = (0..h * w * 3).map(|_| (rand() % 256) as u8).collect();
                let resized = resize::resize_rgb(&rgb, h, w, th, tw, resampler.into());
                let expect = proc.patchify(&resized, th, tw);
                let got = proc.patchify_fused(&rgb, h, w, th, tw);
                assert_eq!(expect, got, "{h}x{w}->{th}x{tw} under {resampler:?}");
            }
            // No-resize: fused against `patchify` on the raw buffer.
            let rgb: Vec<u8> = (0..8 * 12 * 3).map(|_| (rand() % 256) as u8).collect();
            assert_eq!(
                proc.patchify(&rgb, 8, 12),
                proc.patchify_fused(&rgb, 8, 12, 8, 12)
            );
            // Random sweep: targets stay multiples of patch*merge (=4).
            for _ in 0..40 {
                let (h, w) = ((rand() % 96 + 1) as usize, (rand() % 96 + 1) as usize);
                let (th, tw) = (
                    (rand() % 24 + 1) as usize * 4,
                    (rand() % 24 + 1) as usize * 4,
                );
                let rgb: Vec<u8> = (0..h * w * 3).map(|_| (rand() % 256) as u8).collect();
                let resized = resize::resize_rgb(&rgb, h, w, th, tw, resampler.into());
                let expect = proc.patchify(&resized, th, tw);
                let got = proc.patchify_fused(&rgb, h, w, th, tw);
                assert_eq!(expect, got, "{h}x{w}->{th}x{tw} under {resampler:?}");
            }
        }
    }
}
