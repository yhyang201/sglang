# Disaggregated Diffusion Pipeline

Split a monolithic text-to-video/image pipeline into independent **Encoder**, **Denoiser**, and **Decoder** roles, each running on its own GPU(s). This reduces per-GPU memory usage and enables heterogeneous parallelism (e.g., encoder on 1 GPU, denoiser on 4 GPUs with sequence parallelism).

## Deployment

Use `launch_pool_disagg_server()` to launch N:M:K role instances orchestrated by a central DiffusionServer:

```python
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server

server_args = ServerArgs.from_kwargs(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
)

# 2 encoders, 3 denoisers, 2 decoders
launch_pool_disagg_server(
    server_args,
    encoder_gpus=[[0], [4]],        # encoder 0 on GPU 0, encoder 1 on GPU 4
    denoiser_gpus=[[1], [2], [3]],   # 3 single-GPU denoisers
    decoder_gpus=[[0], [4]],         # share GPUs with encoders
)
```

### Multi-GPU Denoisers

```python
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server

server_args = ServerArgs.from_kwargs(
    model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    denoiser_sp=4,
    denoiser_ulysses=2,
    denoiser_ring=2,
)

# 1 encoder (GPU 0), 2 denoisers (each using 4 GPUs with SP=4), 1 decoder (GPU 0)
launch_pool_disagg_server(
    server_args,
    encoder_gpus=[[0]],
    denoiser_gpus=[[1, 2, 3, 4], [5, 6, 7, 8]],
    decoder_gpus=[[0]],
)
```

When a role uses multiple GPUs, only rank 0 communicates with DiffusionServer via ZMQ. All ranks participate in `execute_forward()` via NCCL collectives — the pipeline internally handles tensor sharding/broadcasting.

## Per-Role Parallelism

Each role can have its own TP (tensor parallelism) and SP (sequence parallelism) configuration. This is useful when the denoiser needs multi-GPU parallelism but the encoder/decoder do not.

| Flag | Description |
|------|-------------|
| `--encoder-tp` | Tensor parallelism for encoder |
| `--encoder-sp` | Sequence parallelism for encoder |
| `--encoder-ulysses` | Ulysses SP degree for encoder |
| `--encoder-ring` | Ring SP degree for encoder |
| `--denoiser-tp` | Tensor parallelism for denoiser |
| `--denoiser-sp` | Sequence parallelism for denoiser |
| `--denoiser-ulysses` | Ulysses SP degree for denoiser |
| `--denoiser-ring` | Ring SP degree for denoiser |
| `--decoder-tp` | Tensor parallelism for decoder |
| `--decoder-sp` | Sequence parallelism for decoder |
| `--decoder-ulysses` | Ulysses SP degree for decoder |
| `--decoder-ring` | Ring SP degree for decoder |

If not specified, parallelism is auto-derived from the GPU count for each role.

## P2P Transfer Mode

By default, tensor data is relayed through DiffusionServer (relay mode). Enable P2P mode for direct RDMA transfers between role instances, bypassing the DiffusionServer data path:

```python
server_args = ServerArgs.from_kwargs(
    model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    disagg_p2p_mode=True,
    disagg_transfer_pool_size=512 * 1024 * 1024,  # 512 MiB pinned memory pool
    disagg_p2p_hostname="192.168.1.10",            # RDMA-reachable hostname
)
```

| Flag | Default | Description |
|------|---------|-------------|
| `--disagg-p2p-mode` | `False` | Enable P2P transfer (RDMA or mock fallback) |
| `--disagg-transfer-pool-size` | `268435456` (256 MiB) | Pinned memory pool size per instance |
| `--disagg-p2p-hostname` | `127.0.0.1` | Hostname advertised for RDMA connections |

P2P mode uses a 6-step DS-brokered handshake: `staged → alloc → allocated → push → pushed → ready`. DiffusionServer still handles control-plane routing; only the data plane is P2P.

## Other Options

| Flag | Default | Description |
|------|---------|-------------|
| `--disagg-timeout` | `600` | Timeout (seconds) for pending requests |
| `--disagg-dispatch-policy` | `round_robin` | Dispatch policy: `round_robin` or `max_free_slots` |

## Architecture Overview

```
Pool mode (N:M:K):
  Client → HTTP Server
              ↓
         DiffusionServer (ROUTER)
              ↓ PUSH          ↑ PULL
         Encoder[0..N-1]  →  DiffusionServer  →  Denoiser[0..M-1]  →  DiffusionServer  →  Decoder[0..K-1]
              ↑                                                                                    ↓
              └────────────────────────────── result ──────────────────────────────────────────────┘
```

## Full Example: 8-GPU Wan2.1-14B Deployment

```python
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server

server_args = ServerArgs.from_kwargs(
    model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    denoiser_sp=4,
    denoiser_ulysses=2,
    denoiser_ring=2,
    disagg_dispatch_policy="max_free_slots",
    disagg_p2p_mode=True,
    disagg_transfer_pool_size=512 * 1024 * 1024,
)

launch_pool_disagg_server(
    server_args,
    encoder_gpus=[[0]],
    denoiser_gpus=[[1, 2, 3, 4], [5, 6, 7, 8]],  # 2 × SP=4 denoisers
    decoder_gpus=[[0]],
)
```

Once the server is running, send requests via the standard OpenAI-compatible API:

```bash
curl http://localhost:30000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "prompt": "A curious raccoon exploring a garden",
    "n": 1
  }'
```
