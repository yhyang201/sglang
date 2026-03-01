# Helios Video Generation with SGLang

## Example 1: Short Video (33 frames, ~1.4s)

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 sglang generate \
  --model-path /root/models/Helios-Base \
  --prompt "A cat walking on the beach at sunset, cinematic lighting, high quality" \
  --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
  --height 384 \
  --width 640 \
  --num-frames 33 \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --output-path outputs/
```

## Example 2: Long Video (99 frames, ~4s, Helios default settings)

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 sglang generate \
  --model-path /root/models/Helios-Base \
  --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. Medium shot focusing on the train window and the rushing scenery beyond." \
  --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
  --height 384 \
  --width 640 \
  --num-frames 99 \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --output-path outputs/
```

## Reference Parameters (from Helios-main/infer_helios.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| height | 384 | Output video height |
| width | 640 | Output video width |
| num_frames | 99 | Number of output frames |
| num_inference_steps | 50 | Denoising steps |
| guidance_scale | 5.0 | CFG guidance scale |
| seed | 42 | Random seed |
| fps | 24 | Output video FPS |
| num_latent_frames_per_chunk | 9 | Latent frames per chunk (Stage 1) |
| history_sizes | [16, 2, 1] | Multi-term memory history sizes |
| zero_steps | 1 | CFG Zero Star initial steps |

## Supported Resolutions

| Resolution | Aspect Ratio |
|------------|-------------|
| 640 x 384 | ~5:3 (landscape) |
| 384 x 640 | ~3:5 (portrait) |
| 832 x 480 | ~16:9 |
| 480 x 832 | ~9:16 |

## Model Variants

| Model | Path | Description |
|-------|------|-------------|
| Helios-Base | `BestWishYsh/Helios-Base` | Best quality |
| Helios-Distilled | `BestWishYsh/Helios-Distilled` | Best efficiency |

## Performance (H200, CPU offload)

| Frames | Chunks | Denoising Time | Total Time | Peak GPU Memory |
|--------|--------|----------------|------------|-----------------|
| 33 | 1 | ~134s | ~138s | ~32 GB |
| 99 | 3 | ~392s | ~398s | ~32 GB |
