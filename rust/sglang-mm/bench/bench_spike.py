#!/usr/bin/env python3
"""Spike bench + parity driver for the turbo-jpeg / fused-kernel work.

Loads the freshly built `_multimodal` extension straight from the cargo
target dir (no sglang install needed) and reports, per fixture:

  * decode parity: PIL vs pure-Rust `image` decoder vs libjpeg-turbo
    (per-sample abs-diff stats — the turbo backend's tolerance budget);
  * fused parity: `baseline` vs `fused` (and `turbo` vs `turbo_fused`) must be
    bitwise identical — the fused kernel keeps the crate's fixed-point weights;
  * 4-way bench: baseline / turbo / fused / turbo_fused, µs per image, with
    decode / post-decode stage splits from the Rust side.

Thread count is controlled by SGL_MM_RS_THREADS (read once per process by the
crate's pool); the shell driver re-runs this script per value.

Fixtures are generated programmatically (same `make_photo_like` recipe as
bench/bench_parity.py) and cached under bench/fixtures_spike/.
"""

import argparse
import importlib.util
import io
import json
import os
import sys
import time
from importlib.machinery import ExtensionFileLoader

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(ROOT, "..", "..", "target", "release", "libsglang_mm_core.so")
FIXTURE_DIR = os.path.join(ROOT, "fixtures_spike")

# Qwen2.5-VL server default (matches bench_hf.py of the reference branch).
SPEC_JSON = json.dumps(
    {
        "family": "qwen_vl",
        "image_token_id": 151655,
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "min_pixels": 56 * 56 * 4,
        "max_pixels": 28 * 28 * 1280,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    }
)

# (label, width, height, bench iters)
SIZES = [
    ("512x512", 512, 512, 30),
    ("1024x768", 1024, 768, 20),
    ("2048x1536", 2048, 1536, 10),
    ("4096x3072", 4096, 3072, 5),
]

BACKENDS = ["baseline", "turbo", "fused", "turbo_fused"]


def load_ext():
    loader = ExtensionFileLoader("_multimodal", SO)
    spec = importlib.util.spec_from_loader("_multimodal", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def make_photo_like(h, w, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            127 + 100 * np.sin(yy / 97.0) * np.cos(xx / 131.0),
            127 + 100 * np.cos(yy / 61.0) * np.sin(xx / 89.0),
            127 + 100 * np.sin((xx + yy) / 149.0),
        ],
        axis=-1,
    )
    noise = rng.normal(0, 12, (h // 8 + 1, w // 8 + 1, 3))
    noise = np.kron(noise, np.ones((8, 8, 1)))[:h, :w]
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def encode(arr, fmt):
    buf = io.BytesIO()
    Image.fromarray(arr).save(
        buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {})
    )
    return buf.getvalue()


def build_fixtures():
    """[(label, fmt, bytes)], generated once and cached on disk."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    out = []
    for i, (label, w, h, iters) in enumerate(SIZES):
        arr_path = os.path.join(FIXTURE_DIR, f"{label}.npy")
        if os.path.exists(arr_path):
            arr = np.load(arr_path)
        else:
            arr = make_photo_like(h, w, seed=100 + i)
            np.save(arr_path, arr)
        for fmt, ext in [("JPEG", "jpg"), ("PNG", "png")]:
            path = os.path.join(FIXTURE_DIR, f"{label}.{ext}")
            if os.path.exists(path):
                data = open(path, "rb").read()
            else:
                data = encode(arr, fmt)
                open(path, "wb").write(data)
            out.append((f"{label}.{ext}", fmt, data, arr, iters))
    return out


def diff_stats(a, b):
    d = a.astype(np.int16) - b.astype(np.int16)
    ad = np.abs(d)
    return {
        "max": int(ad.max()),
        "mean": float(ad.mean()),
        "pct_diff": float((ad > 0).mean() * 100),
        "exact": bool(ad.max() == 0),
    }


def decode_parity(mm, fixtures):
    print("\n=== Decode parity (per-sample |diff| on decoded RGB) ===")
    print(f"{'fixture':<22} {'pair':<22} {'max':>4} {'mean':>8} {'%diff':>7}  exact")
    for label, fmt, data, arr, _ in fixtures:
        pil = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        h, w, rust_raw = mm.common.image_decode_rgb(data)
        rust = np.frombuffer(rust_raw, np.uint8).reshape(h, w, 3)
        pairs = [("rust-image vs PIL", rust, pil)]
        if hasattr(mm.common, "turbo_decode_rgb"):
            th, tw, turbo_raw = mm.common.turbo_decode_rgb(data)
            turbo = np.frombuffer(turbo_raw, np.uint8).reshape(th, tw, 3)
            pairs += [
                ("turbo vs PIL", turbo, pil),
                ("turbo vs rust-image", turbo, rust),
            ]
        for name, a, b in pairs:
            s = diff_stats(a, b)
            print(
                f"{label:<22} {name:<22} {s['max']:>4} {s['mean']:>8.4f} "
                f"{s['pct_diff']:>6.2f}%  {s['exact']}"
            )


def fused_parity(mm, fixtures):
    print("\n=== Fused-kernel parity (bitwise vs unfused chain) ===")
    ok = True
    for label, fmt, data, arr, _ in fixtures:
        base = mm.qwen_vl.preprocess_timed(data, SPEC_JSON, "baseline")[0]
        fused = mm.qwen_vl.preprocess_timed(data, SPEC_JSON, "fused")[0]
        exact = np.array_equal(base, fused)
        line = f"  {label:<22} baseline==fused: {exact}"
        if hasattr(mm.common, "turbo_decode_rgb"):
            tbase = mm.qwen_vl.preprocess_timed(data, SPEC_JSON, "turbo")[0]
            tfused = mm.qwen_vl.preprocess_timed(data, SPEC_JSON, "turbo_fused")[0]
            texact = np.array_equal(tbase, tfused)
            line += f"   turbo==turbo_fused: {texact}"
            ok &= texact
        print(line)
        ok &= exact
    print(f"  ALL_BITWISE_EQUAL={ok}")
    return ok


def hf_compare(mm, fixtures):
    """Tolerance picture against the HF processor (PIL decode + torchvision)."""
    try:
        import torch  # noqa: F401
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )
    except Exception as e:
        print(f"\n=== HF comparison skipped ({e}) ===")
        return
    proc = Qwen2VLImageProcessor(
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=56 * 56 * 4,
        max_pixels=28 * 28 * 1280,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    print("\n=== vs HF Qwen2VLImageProcessor (pixel_values abs diff) ===")
    print(f"{'fixture':<22} {'backend':<14} {'max':>9} {'mean':>9} {'exact':>6}")
    for label, fmt, data, arr, _ in fixtures:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        hf = proc(images=img, return_tensors="pt")["pixel_values"].numpy()
        for backend in BACKENDS:
            if backend.startswith("turbo") and not hasattr(
                mm.common, "turbo_decode_rgb"
            ):
                continue
            got = mm.qwen_vl.preprocess_timed(data, SPEC_JSON, backend)[0]
            if got.size != hf.size:
                print(f"{label:<22} {backend:<14} SIZE {got.size} vs {hf.size}")
                continue
            ad = np.abs(got.reshape(hf.shape) - hf)
            print(
                f"{label:<22} {backend:<14} {ad.max():>9.5f} {ad.mean():>9.6f} "
                f"{bool(ad.max() == 0)!s:>6}"
            )


def bench(mm, fixtures):
    threads = os.environ.get("SGL_MM_RS_THREADS", "default(8-cap)")
    print(f"\n=== Bench (SGL_MM_RS_THREADS={threads}) — per-image µs ===")
    print(
        f"{'fixture':<22} {'backend':<14} {'decode_us':>10} {'post_us':>10} "
        f"{'total_us':>10} {'vs_base':>8}   decoded -> target"
    )
    rows = []
    for label, fmt, data, arr, iters in fixtures:
        base_total = None
        for backend in BACKENDS:
            if backend.startswith("turbo") and not hasattr(
                mm.common, "turbo_decode_rgb"
            ):
                continue
            # warmup (pool spin-up, TLS, first-touch pages)
            for _ in range(2):
                mm.qwen_vl.preprocess_timed(data, SPEC_JSON, backend)
            dec = post = 0
            th = tw = dh = dw = None
            t_start = time.perf_counter()
            for _ in range(iters):
                pv, (t, gh, gw), dec_ns, post_ns, dh, dw = mm.qwen_vl.preprocess_timed(
                    data, SPEC_JSON, backend
                )
                dec += dec_ns
                post += post_ns
            wall = (time.perf_counter() - t_start) / iters
            dec_us = dec / iters / 1000
            post_us = post / iters / 1000
            total_us = dec_us + post_us
            if backend == "baseline":
                base_total = total_us
            speedup = base_total / total_us if base_total else 1.0
            rows.append(
                (label, backend, dec_us, post_us, total_us, speedup, wall * 1e6)
            )
            print(
                f"{label:<22} {backend:<14} {dec_us:>10.1f} {post_us:>10.1f} "
                f"{total_us:>10.1f} {speedup:>7.2f}x   {dh}x{dw} -> {gw * 14}x{gh * 14}"
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    mm = load_ext()
    fixtures = build_fixtures()
    turbo = hasattr(mm.common, "turbo_decode_rgb")
    print(f"extension: {SO}")
    print(f"turbo-jpeg backend present: {turbo}")
    print(f"SGL_MM_RS_THREADS={os.environ.get('SGL_MM_RS_THREADS', '<unset>')}")

    if not args.skip_parity:
        decode_parity(mm, fixtures)
        ok = fused_parity(mm, fixtures)
        hf_compare(mm, fixtures)
        if not ok:
            print("\nFUSED PARITY FAILURE — refusing to bench")
            sys.exit(1)
    if not args.skip_bench:
        bench(mm, fixtures)


if __name__ == "__main__":
    main()
