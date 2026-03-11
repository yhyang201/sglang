# Disaggregated Diffusion Pipeline

Split a monolithic text-to-video/image pipeline into independent **Encoder**, **Denoiser**, and **Decoder** roles, each running on its own GPU(s). A central **DiffusionServer** routes requests through the pipeline.

## Quick Start

Disaggregation is controlled by a single flag: `--disagg-role`. Each component is launched independently, just like LLM PD disaggregation.

| `--disagg-role` | What it runs |
|----------------|--------------|
| `monolithic` | (Default) Standard single-server mode |
| `encoder` | Encoder role instance (text/image encoding) |
| `denoising` | Denoiser role instance (DiT forward) |
| `decoder` | Decoder role instance (VAE decode) |
| `server` | DiffusionServer head node + HTTP server (no GPU) |

### Single-Machine Example (Verified)

The following commands have been tested end-to-end on an 8×H200 machine with
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. Each role runs on a separate GPU via
`--base-gpu-id`; the `server` head node requires no GPU.

```bash
# Terminal 1: Encoder (GPU 0)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role encoder \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19000 \
    --num-gpus 1 --base-gpu-id 0

# Terminal 2: Denoiser (GPU 1)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role denoising \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19001 \
    --num-gpus 1 --base-gpu-id 1

# Terminal 3: Decoder (GPU 2)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role decoder \
    --disagg-server-addr tcp://127.0.0.1:19655 \
    --scheduler-port 19002 \
    --num-gpus 1 --base-gpu-id 2

# Terminal 4: DiffusionServer head (no GPU, receives HTTP requests)
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --disagg-role server \
    --encoder-urls  "tcp://127.0.0.1:19000" \
    --denoiser-urls "tcp://127.0.0.1:19001" \
    --decoder-urls  "tcp://127.0.0.1:19002" \
    --host 0.0.0.0 --port 22000 \
    --scheduler-port 19655

# Send request (video generation)
curl http://127.0.0.1:22000/v1/videos \
    -H "Content-Type: application/json" \
    -d '{"model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "prompt": "A curious raccoon exploring a garden, cinematic", "size": "832x480"}'
```

> **Tested result (8×H200, relay mode):**
> Encoder 2.3 s (TextEncoding) → Denoiser 312.8 s (50 steps, layerwise offload) → Decoder 7.1 s (VAE decode).
> Total ~322 s for 81-frame 1024×1024 video.

> **Tip:** `--base-gpu-id` controls which physical GPU the role uses.
> Encoder and Decoder can share a GPU (e.g. both `--base-gpu-id 0`) to save resources,
> but make sure the combined GPU memory is sufficient.

### Multi-Machine Example

The exact same CLI pattern — just replace `127.0.0.1` with actual IPs and add
P2P flags for RDMA direct transfer:

```bash
# Machine A (10.0.0.1): Encoder
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role encoder \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19000 \
    --num-gpus 1 \
    --disagg-p2p-mode --disagg-p2p-hostname 10.0.0.1 --disagg-ib-device mlx5_0

# Machine B (10.0.0.2): Denoiser (4 GPUs with SP)
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role denoising \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19001 \
    --num-gpus 4 --denoiser-sp 4 --denoiser-ulysses 2 --denoiser-ring 2 \
    --disagg-p2p-mode --disagg-p2p-hostname 10.0.0.2 --disagg-ib-device mlx5_0

# Machine C (10.0.0.3): Decoder
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role decoder \
    --disagg-server-addr tcp://10.0.0.4:19655 \
    --scheduler-port 19002 \
    --num-gpus 1 \
    --disagg-p2p-mode --disagg-p2p-hostname 10.0.0.3 --disagg-ib-device mlx5_0

# Machine D (10.0.0.4): DiffusionServer head
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --disagg-role server \
    --encoder-urls  "tcp://10.0.0.1:19000" \
    --denoiser-urls "tcp://10.0.0.2:19001" \
    --decoder-urls  "tcp://10.0.0.3:19002" \
    --host 0.0.0.0 --port 30000 \
    --scheduler-port 19655 \
    --disagg-dispatch-policy max_free_slots \
    --disagg-p2p-mode
```

> ZMQ handles startup order gracefully — instances and head can start in any order.

## Multiple Instances per Role

Use semicolons in `--*-urls` to register multiple instances:

```bash
# 2 encoders + 2 denoisers (4-GPU SP each) + 1 decoder
sglang serve --model-path ... --disagg-role server \
    --encoder-urls  "tcp://10.0.0.1:35000;tcp://10.0.0.2:35000" \
    --denoiser-urls "tcp://10.0.0.3:35000;tcp://10.0.0.4:35000" \
    --decoder-urls  "tcp://10.0.0.5:35000"
```

## Port Convention

Result endpoints are derived deterministically from the head node's `--scheduler-port` (default: 5655):

| Socket | Port |
|--------|------|
| DS frontend (ROUTER) | `scheduler_port` |
| Encoder result (PULL) | `scheduler_port + 1` |
| Denoiser result (PULL) | `scheduler_port + 2` |
| Decoder result (PULL) | `scheduler_port + 3` |

Role instances derive their result endpoint automatically from `--disagg-server-addr`. No manual endpoint configuration needed.

## P2P Transfer Mode & Mooncake

By default, tensor data relays through DiffusionServer. For direct RDMA transfers:

```bash
pip install mooncake-transfer-engine
```

| Flag | Default | Description |
|------|---------|-------------|
| `--disagg-p2p-mode` | `False` | Enable P2P transfer |
| `--disagg-p2p-hostname` | `127.0.0.1` | RDMA-reachable hostname/IP of this instance |
| `--disagg-ib-device` | `None` | InfiniBand device (e.g., `mlx5_0`, `mlx5_roce0`) |
| `--disagg-transfer-pool-size` | 256 MiB | Pinned memory pool per instance |

Set `--disagg-p2p-hostname` to the actual IP on each machine. For multi-machine, `--disagg-ib-device` specifies the RDMA NIC.

## Per-Role Parallelism

| Flag | Description |
|------|-------------|
| `--encoder-tp` / `--encoder-sp` / `--encoder-ulysses` / `--encoder-ring` | Encoder parallelism |
| `--denoiser-tp` / `--denoiser-sp` / `--denoiser-ulysses` / `--denoiser-ring` | Denoiser parallelism |
| `--decoder-tp` / `--decoder-sp` / `--decoder-ulysses` / `--decoder-ring` | Decoder parallelism |

If not specified, parallelism is auto-derived from `--num-gpus`.

## Other Options

| Flag | Default | Description |
|------|---------|-------------|
| `--disagg-timeout` | `600` | Timeout (seconds) for pending requests |
| `--disagg-dispatch-policy` | `round_robin` | `round_robin` or `max_free_slots` |

## Python API

For programmatic single-machine deployment, `launch_pool_disagg_server()` is still available:

```python
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server

server_args = ServerArgs.from_kwargs(
    model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    denoiser_sp=4, denoiser_ulysses=2, denoiser_ring=2,
    disagg_p2p_mode=True, disagg_ib_device="mlx5_0",
)

launch_pool_disagg_server(
    server_args,
    encoder_gpus=[[0]],
    denoiser_gpus=[[1, 2, 3, 4], [5, 6, 7, 8]],
    decoder_gpus=[[0]],
)
```

## Architecture

```
Client -> HTTP Server (port 30000)
              |
         DiffusionServer (ROUTER, scheduler_port)
              |
    +---------+---------+
    | PUSH              | PULL (scheduler_port + 1/2/3)
    v                   |
  Encoder[0..N-1]   result
  Denoiser[0..M-1]  result
  Decoder[0..K-1]   result -> Client
```
