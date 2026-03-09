# Diffusion Disaggregation Implementation Plan

RFC: https://github.com/sgl-project/sglang/issues/19512

## Background

- **Core idea**: Split monolithic T2V pipeline into EncoderRole / DenoisingRole / DecoderRole
- **Primary value**: Memory savings (offload encoder weights from denoising GPU), enabling SP for denoising, heterogeneous parallelism
- **Constraint**: BS=1 only; Denoising dominates 99.9%+ of total time

## Phase 1: Role Abstraction & Weight Separation (Single Machine)

**Goal**: Each Role only loads its own weights. Validates correctness, gets memory savings immediately.

**Files to create/modify**:

| File | Action | Description |
|------|--------|-------------|
| `runtime/disaggregation/roles.py` | **New** | `RoleType` enum (ENCODER/DENOISING/DECODER), `RoleConfig` dataclass |
| `runtime/pipelines_core/composed_pipeline_base.py` | **Modify** | Add `role` parameter; filter `_required_config_modules` by role; `create_pipeline_stages()` only creates stages for assigned role |
| `runtime/server_args.py` | **Modify** | Add `--role` arg (encoder/denoising/decoder/monolithic), `--peer-addr` for cross-role communication |
| `runtime/pipelines_core/stages/base.py` | **Modify** | Add `role_affinity` property to each stage class (which role it belongs to) |

**Key design**:
```
# Stage -> Role mapping (derived from existing stage classes)
ENCODER:   InputValidation, TextEncoding, ImageEncoding, LatentPrep, TimestepPrep
DENOISING: DenoisingStage (+ CausalDenoisingStage, HeliosDenoisingStage)
DECODER:   DecodingStage
```

Each pipeline's `create_pipeline_stages()` already uses `add_stage()` -- we filter stages based on `server_args.role`. The pipeline loads only modules needed by its active stages via the existing `_required_config_modules` mechanism.

**Validation**: Run monolithic vs encoder+denoising+decoder (same GPU, sequential) and compare outputs bit-for-bit.

## Phase 2: Multi-Process Deployment (Single Machine, IPC)

**Goal**: Run Encoder/Denoising/Decoder as separate processes on the same machine, communicating via shared memory or ZMQ.

**Files to create/modify**:

| File | Action | Description |
|------|--------|-------------|
| `runtime/disaggregation/__init__.py` | **New** | Module init |
| `runtime/disaggregation/transfer_buffer.py` | **New** | `TransferTensorBuffer` (pinned memory pool), `TransferMetaBuffer` (metadata) -- start with simple fixed-slot allocator, buddy system later |
| `runtime/disaggregation/ipc_channel.py` | **New** | IPC channel abstraction: `send_tensors()` / `recv_tensors()` using CUDA IPC or torch multiprocessing shared memory |
| `runtime/disaggregation/role_connector.py` | **New** | `EncoderToDenoiserConnector`, `DenoiserToDecoderConnector` -- serializes/deserializes `Req` intermediate state (embeddings, latents, metadata) |
| `runtime/managers/scheduler.py` | **Modify** | Add disagg-aware scheduling: if role is encoder, after execution send result to denoiser; if denoiser, send to decoder |
| `runtime/launch_server.py` | **Modify** | Add `launch_disagg_server()` that spawns encoder/denoising/decoder processes with their own GPUs |

**Transfer payload between roles**:
```
Encoder -> Denoiser:  prompt_embeds, negative_prompt_embeds, pooled_embeds,
                      latents, timesteps, masks (~50-200 MB for video)
Denoiser -> Decoder:  latents (~10-50 MB for video)
```

**Key design choices**:
- Start with ZMQ + `torch.save()`/`torch.load()` serialization (simple, correct)
- Optimize later with CUDA IPC shared memory for same-machine, RDMA for cross-machine
- Each role process is a full `Scheduler + GPUWorker`, just with filtered stages

## Phase 3: DiffusionServer -- Request Routing & Orchestration

**Goal**: A coordinator that receives client requests and routes them through the Encoder -> Denoiser -> Decoder pipeline across role instances.

**Files to create/modify**:

| File | Action | Description |
|------|--------|-------------|
| `runtime/disaggregation/diffusion_server.py` | **New** | Central request router: receives HTTP/ZMQ requests, dispatches to encoder instances, tracks request state machine, routes results to next role |
| `runtime/disaggregation/request_state.py` | **New** | `RequestState` enum + `RequestTracker` class for state machine management |
| `runtime/disaggregation/dispatch_policy.py` | **New** | Dispatch policies: `MaxFreeSlotsFirst`, `RoundRobin` -- selects which role instance handles a request |
| `runtime/entrypoints/http_server.py` | **Modify** | Add disagg mode routes; DiffusionServer replaces direct Scheduler for disagg deployments |

**Request flow**:
```
Client -> DiffusionServer -> Encoder[i] -> (transfer) -> Denoiser[j] -> (transfer) -> Decoder[k] -> Client
```

**State machine** (simplified since BS=1):
```
Pending -> EncoderRunning -> EncoderDone -> DenoisingRunning -> DenoisingDone -> DecoderRunning -> Done
```

## Phase 4: Enable Independent Parallelism Per Role

**Goal**: Each role can have its own TP/SP/CFG degree.

**Files to modify**:

| File | Action | Description |
|------|--------|-------------|
| `runtime/server_args.py` | **Modify** | Per-role parallelism: `--encoder-tp`, `--denoiser-tp`, `--denoiser-sp`, `--decoder-tp` |
| `runtime/launch_server.py` | **Modify** | Each role process group gets its own TP/SP size and NCCL communicator |
| `runtime/distributed/parallel_state.py` | **Modify** | Support per-role group initialization |

**Example deployment**:
```
Encoder: TP=1 (small model, efficient on 1 GPU)
Denoiser: SP=4 (4 GPUs, sequence parallel for 4K video)
Decoder: TP=1 (fast enough on 1 GPU)
```

## Phase 5: Cross-Machine Transfer (RDMA)

**Goal**: Enable encoder and denoiser to run on different machines.

**Files to create/modify**:

| File | Action | Description |
|------|--------|-------------|
| `runtime/disaggregation/transfer_engine.py` | **New** | Adapt `MooncakeTransferEngine` from `srt/disaggregation/mooncake/conn.py` for diffusion tensor transfer |
| `runtime/disaggregation/transfer_buffer.py` | **Modify** | Add buddy-system allocator (split/coalesce/defrag) for dynamic slot management |
| `runtime/disaggregation/transfer_manager.py` | **New** | `send_event_loop` / `receive_event_loop` threads, `transfer_slice` for SP-aware transfer |

**Reuse from existing PD disagg** (~70%):
- `MooncakeTransferEngine` -- RDMA engine wrapper
- `BaseKVConn` pattern -- connection interface
- Metadata buffer / poll patterns from `kv_events.py`

## Phase 6: 3-Stream Pipeline Overlap (Optional/Future)

**Goal**: Overlap H2D / Compute / D2H within each role for pipelining.

**Reality check**: With 50-step denoising and BS=1, this saves <0.1%. Only valuable if:
- Step count drops significantly (e.g., consistency models with 1-4 steps)
- Decoder becomes heavier (future high-res VAE)

**Defer this unless step counts decrease substantially.**

## Timeline Summary

```
Phase 1 (Role Abstraction)     -> 1-2 weeks   | Foundation; immediate memory savings
Phase 2 (Multi-Process IPC)    -> 2-3 weeks   | Proves disagg works end-to-end
Phase 3 (DiffusionServer)      -> 2-3 weeks   | Multi-instance orchestration
Phase 4 (Per-Role Parallelism) -> 1-2 weeks   | Unlocks SP for denoiser (4K video)
Phase 5 (Cross-Machine RDMA)   -> 2-3 weeks   | Cluster-level deployment
Phase 6 (3-Stream Overlap)     -> Defer       | <0.1% benefit currently
```
