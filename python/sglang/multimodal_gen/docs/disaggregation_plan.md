# Diffusion Disaggregation Implementation Plan

RFC: https://github.com/sgl-project/sglang/issues/19512

## Background

- **Core idea**: Split monolithic T2V pipeline into EncoderRole / DenoisingRole / DecoderRole
- **Primary value**: Memory savings (offload encoder weights from denoising GPU), enabling SP for denoising, heterogeneous parallelism
- **Constraint**: BS=1 only; Denoising dominates 99.9%+ of total time

---

## Implementation Progress

### Commit 1 — Phase 1: Role Abstraction & Weight Separation ✅

> Corresponds to [Original Plan Phase 1](#original-plan-phase-1-role-abstraction--weight-separation-single-machine)

`ff99f9a2c` — `[Diffusion] Add disaggregation Phase 1: role-based weight separation`

- `runtime/disaggregation/roles.py` — `RoleType` enum, `get_module_role()`, `filter_modules_for_role()`
- `runtime/pipelines_core/composed_pipeline_base.py` — Filter `_required_config_modules` and stages by `role_affinity`
- `runtime/server_args.py` — `--disagg-role` (encoder/denoising/decoder/monolithic)
- `runtime/pipelines_core/stages/base.py` — `role_affinity` property on stage classes

### Commit 2 — Phase 2: ZMQ Zero-Copy IPC ✅

> Corresponds to [Original Plan Phase 2](#original-plan-phase-2-multi-process-deployment-single-machine-ipc)

`74f9bee52` — `[Diffusion] Add disaggregation Phase 2: ZMQ zero-copy IPC`

- `runtime/disaggregation/tensor_transport.py` — `TensorWrapper` (zero-copy via `__buffer__`), `send_tensors()` / `recv_tensors()` over ZMQ multipart
- `runtime/disaggregation/role_connector.py` — `RoleConnectorSender` / `RoleConnectorReceiver` with per-transition field lists
- `runtime/managers/scheduler.py` — Disagg-aware event loops for denoiser/decoder; encoder sends to denoiser and waits for decoder result

**Deviation from plan**: Used ZMQ zero-copy multipart instead of `transfer_buffer.py` + `ipc_channel.py`. The plan noted "start with ZMQ" as the simple-first approach; CUDA IPC / RDMA deferred to later phases.

### Commit 3 — Phase 3: CLI Integration & Launch Orchestration ✅

> Partially covers [Original Plan Phase 2](#original-plan-phase-2-multi-process-deployment-single-machine-ipc) (launch_server) and [Original Plan Phase 3](#original-plan-phase-3-diffusionserver----request-routing--orchestration) (basic single-instance routing)

`9d12bed9b` — `[Diffusion] Add disaggregation Phase 3: CLI integration and launch orchestration`

- `runtime/server_args.py` — `--disagg-mode`, `--encoder-gpus`, `--denoiser-gpus`, `--decoder-gpus`
- `runtime/launch_server.py` — `launch_disagg_server()` auto-spawns 3 role process groups with per-role `ServerArgs` (via `from_kwargs` to trigger `__post_init__` parallelism recalculation)
- `runtime/entrypoints/cli/serve.py` — CLI entry point dispatches to `launch_disagg_server`
- `runtime/managers/scheduler.py` — Denoiser/decoder receive tensors directly on GPU; decoder preserves raw output tensor for return to encoder
- `test/server/test_disagg_server.py` — Server-level E2E test

**Deviation from plan**: Routing logic is embedded in the encoder scheduler (not a separate DiffusionServer). Supports single-instance only (1 encoder + 1 denoiser + 1 decoder). Multi-instance orchestration deferred.

### Commit 5 — Phase 3 Completion: Multi-Instance Orchestration (Pool-Based) ✅

> Completes [Original Plan Phase 3](#original-plan-phase-3-diffusionserver----request-routing--orchestration) — RFC-compliant N:M:K pool-based pipeline orchestration

**DiffusionServer as global pipeline orchestrator** — dispatches at every role transition:
```
Client → DiffusionServer → Encoder[i] → DiffusionServer → Denoiser[j] → DiffusionServer → Decoder[k] → DiffusionServer → Client
```

- `runtime/disaggregation/request_state.py` — `RequestState` enum (9 states: Pending → EncoderRunning → EncoderDone → DenoisingRunning → ... → Done/Failed/TimedOut), `RequestRecord`, `RequestTracker` (thread-safe lifecycle manager with per-role instance assignment)
- `runtime/disaggregation/dispatch_policy.py` — `DispatchPolicy` ABC, `RoundRobin`, `MaxFreeSlotsFirst`, `PoolDispatcher` (wraps 3 independent per-role policies), `create_dispatch_policy()` factory
- `runtime/disaggregation/diffusion_server.py` — `DiffusionServer` (ROUTER frontend + PUSH/PULL per-role pools, state-machine-driven dispatch at each role transition, zero-copy multipart relay for tensor data, per-role dispatch policies, timeout handling, error propagation)
- `runtime/disaggregation/role_connector.py` — Added `pack_encoder_output()`, `pack_denoiser_output()`, `build_req_from_frames()` for pool-mode serialization
- `runtime/managers/scheduler.py` — Added `_pool_mode_event_loop()` (unified for all roles), `_pool_mode_encoder_step()`, `_pool_mode_denoiser_step()`, `_pool_mode_decoder_step()`, `_init_pool_mode_sockets()`
- `runtime/launch_server.py` — `launch_pool_disagg_server()` spawns N:M:K independent role instances and wires DiffusionServer
- `runtime/server_args.py` — `--disagg-dispatch-policy`, `disagg_pool_mode`, `pool_work_endpoint`, `pool_result_endpoint`
- `test/unit/test_request_state.py` — 14 tests for state machine transitions, tracking, snapshots
- `test/unit/test_dispatch_policy.py` — 13 tests for RoundRobin, MaxFreeSlotsFirst, PoolDispatcher, factory
- `test/unit/test_diffusion_server.py` — 5 tests including full pipeline flow (Client→Encoder→Denoiser→Decoder→Client) with live ZMQ and tensor relay

**Socket topology**:
```
DiffusionServer:
  ROUTER (bind) ← HTTP server DEALER connects
  PUSH (connect) → Encoder[i] PULL (bind)     × N
  PUSH (connect) → Denoiser[j] PULL (bind)    × M
  PUSH (connect) → Decoder[k] PULL (bind)     × K
  PULL (bind) ← Encoder PUSH (connect)        × 1 shared
  PULL (bind) ← Denoiser PUSH (connect)       × 1 shared
  PULL (bind) ← Decoder PUSH (connect)        × 1 shared
```

**Backward compatible**: Chain mode (`--disagg-mode`) with 1:1:1 instances still works as before.

### Commit 7 — Phase 5: Capacity-Aware Dispatch with FreeBufferSlots & TTA Queues ✅

> Covers [RFC §3 TODO 1.2](#phase-5-diffusionserver--capacity-aware-dispatch-rfc-3-todo-12) — admission control based on buffer capacity

`[Diffusion] Add disaggregation Phase 5: capacity-aware dispatch with FreeBufferSlots and TTA queues`

- `runtime/disaggregation/request_state.py` — Added 3 WAITING states: `ENCODER_WAITING`, `DENOISING_WAITING`, `DECODER_WAITING` (request in TTA queue awaiting slot). States can be skipped when capacity is immediately available (PENDING → ENCODER_RUNNING directly)
- `runtime/disaggregation/dispatch_policy.py` — Added `select_with_capacity(free_slots) → int | None` to `DispatchPolicy` ABC, `RoundRobin` (skips full instances round-robin), `MaxFreeSlotsFirst` (picks highest free_slots, returns None when all zero). `PoolDispatcher` gains `select_*_with_capacity()` methods
- `runtime/disaggregation/diffusion_server.py` — Full capacity-aware rewrite:
  - `FreeBufferSlots`: per-instance counters (`encoder_capacity` / `denoiser_capacity` / `decoder_capacity`, configurable)
  - `TryToAdd` (TTA) queues: 3 dequeues (`_encoder_tta` / `_denoiser_tta` / `_decoder_tta`) with typed entries
  - Dispatch flow: `select_*_with_capacity()` → if slot available: decrement + dispatch; if None: transition to WAITING + enqueue TTA
  - Completion callback: result handler increments `FreeBufferSlots` + calls `_drain_*_tta()` to dispatch queued requests
  - Timeout cleanup: frees slots for timed-out requests + purges TTA entries
  - `get_stats()` reports `free_slots`, `tta_depth` per role
- `test/unit/test_request_state.py` — 18 tests (+4): WAITING lifecycle, fail from WAITING, skip WAITING, timeout from WAITING
- `test/unit/test_dispatch_policy.py` — 19 tests (+6): `select_with_capacity` for RoundRobin (skip full, return None, cycle available) and MaxFreeSlotsFirst (pick most free, return None, tie-break)
- `test/unit/test_diffusion_server.py` — 8 tests (+3): TTA queuing and drain, FreeBufferSlots decrement/increment through full pipeline, capacity=1 end-to-end

**Dispatch flow with admission control**:
```
Client request arrives:
  FreeBufferSlots[encoder] > 0?
    YES → decrement slot, dispatch to Encoder[i], state = ENCODER_RUNNING
    NO  → enqueue in encoder TTA, state = ENCODER_WAITING

Encoder result arrives:
  increment FreeBufferSlots[encoder][i]
  drain encoder TTA (dispatch queued requests if slots now available)
  FreeBufferSlots[denoiser] > 0?
    YES → decrement slot, relay to Denoiser[j], state = DENOISING_RUNNING
    NO  → enqueue in denoiser TTA, state = DENOISING_WAITING
  (same pattern for denoiser→decoder and decoder→client)
```

### Commit 6 — Phase 6: TransferBuffer with Buddy-System Allocator ✅

> Covers [RFC §2 TODO 3.1–3.3](#phase-6-transferbuffer--pinned-memory-pool-rfc-2-todo-3133) — pinned memory pool infrastructure

`[Diffusion] Add disaggregation Phase 6: TransferBuffer with buddy-system allocator`

- `runtime/disaggregation/transfer_allocator.py` — `BuddyAllocator`: power-of-2 buddy-system memory allocator with split/allocate/free/coalesce, thread-safe, `count_free_slots()` for reporting capacity to DiffusionServer
- `runtime/disaggregation/transfer_buffer.py` — `TransferMetaBuffer` (thread-safe dict-based metadata store) + `TransferTensorBuffer` (contiguous pinned-memory pool backed by BuddyAllocator, supports async D2H/H2D via CUDA streams, batch write/read with manifest)
- `test/unit/test_transfer_allocator.py` — 25 tests: init, alloc/split, free/coalesce, slot counting, thread safety, realistic 256 MiB pool with 64 MiB slots
- `test/unit/test_transfer_buffer.py` — 22 tests: meta buffer CRUD, tensor buffer alloc/free, single/multi-tensor I/O, bfloat16 support, list-of-tensors, Wan2.1-realistic 60 MB encoder output end-to-end

**Key interfaces**:
- `BuddyAllocator.allocate(size, request_id) → offset | None` — find smallest power-of-2 block, split as needed
- `BuddyAllocator.free(offset)` — release + recursively coalesce with buddy
- `BuddyAllocator.count_free_slots(slot_size) → int` — how many allocations of given size can fit (feeds `FreeBufferSlots`)
- `TransferTensorBuffer.allocate(size, request_id) → SlotHandle | None` — allocate pinned memory slot
- `TransferTensorBuffer.write_tensors_from_gpu(handle, tensors, stream) → manifest` — batch D2H copy with layout descriptor
- `TransferTensorBuffer.read_tensors_from_manifest(handle, manifest, device, stream) → tensors` — batch H2D read
- `TransferTensorBuffer.free_slots_count(typical_size) → int` — available capacity for DiffusionServer

**Design notes**:
- Pool is a single `torch.empty(pool_size, dtype=uint8, pin_memory=True)` allocation — contiguous, RDMA-registerable (Phase 7)
- `SlotHandle` tracks offset + size + tensor views; passed to TransferManager for network send (Phase 7)
- Manifest format records per-tensor `{offset, shape, dtype}` within the slot — enables receiver to reconstruct without re-serializing
- 512-byte alignment between tensors for cache line efficiency
- Role-specific sizing: encoder pool smaller (fast execution), denoiser pool larger (slow, more concurrent buffering needed)

### Commit 4 — Phase 4: Async Pipelining, Timeouts & Observability ✅

> Not in original plan — added based on functional gaps identified after Phase 3

`b1afa5643` — `[Diffusion] Add disaggregation Phase 4: async pipelining, timeouts, and observability`

**P0 — Async request pipelining**:
- Encoder is non-blocking: encode → fire-and-forget to denoiser → stash identity in `_pending_disagg` → immediately accept next request
- `_poll_disagg_results()` drains decoder results via `zmq.NOBLOCK` each loop iteration
- Multiple requests can be in-flight across roles simultaneously

**P1 — Timeout & error propagation**:
- `--disagg-timeout` flag (default 600s); `_check_disagg_timeouts()` returns errors for stale requests
- `RoleConnectorReceiver.recv(timeout_ms=...)` via `zmq.RCVTIMEO` prevents permanent blocking on upstream crash
- Denoiser error forwarding: sets `_disagg_error` marker → decoder forwards to encoder

**P2 — Batch draining**:
- Denoiser/decoder `_disagg_event_loop()` drains queued requests via `try_recv()` after first blocking recv
- Processes batch sequentially (true GPU batching requires pipeline stage refactoring)

**P3 — Observability**:
- `runtime/disaggregation/metrics.py` — `DisaggMetrics` class: completed/failed/in-flight/timed-out counts, latency (last/avg/max), throughput (60s rolling RPS), queue depth
- `GET /stats` HTTP endpoint queries encoder metrics via `GetDisaggStatsReq`

---

## Remaining Work (RFC-Aligned)

Below maps directly to RFC sections and TODO items. See [RFC](https://github.com/sgl-project/sglang/issues/19512) for full context.

### RFC Coverage Summary

| RFC Section | TODO | Status | Target Phase |
|-------------|------|--------|--------------|
| 1. Role-Based Decomposition | — | ✅ Done | Commit 1 |
| 3. DiffusionServer — Server Core | 1.1 | ✅ Done | Commit 5 |
| 3. DiffusionServer — Dispatch Loop | 1.3 | ✅ Done | Commit 5 |
| 3. DiffusionServer — State Tracker (FreeBufferSlots/TTA) | 1.2 | ✅ Done | Commit 7 |
| 3. DiffusionServer — Callback (slot increment on completion) | 1.2 | ✅ Done | Commit 7 |
| 2. TransferBuffer — MetaBuffer | 3.1 | ✅ Done | Commit 6 |
| 2. TransferBuffer — TensorBuffer + Buddy Allocator | 3.2, 3.3 | ✅ Done | Commit 6 |
| 2. TransferManager — Engine Setup (RDMA/RPC) | 2.1 | ❌ | Phase 7 |
| 2. TransferManager — Receive Loop | 2.2 | ❌ | Phase 7 |
| 2. TransferManager — Send Loop + transfer_slice | 2.3 | ❌ | Phase 7 |
| 2. TransferringQueue / SwappingQueue | 4.4 | ❌ | Phase 7 |
| 4. E2E Workflow — P2P data transfer (bypass DS) | — | ❌ | Phase 7 |
| 5. CUDA Stream Overlap — 3-stream scheduling | 4.2 | ❌ | Phase 8 |
| 5. CUDA Stream Overlap — Request-level trigger | 4.3 | ❌ | Phase 8 |
| 4. DisaggUtils — Role Weights | 4.1 | ✅ Done | Commit 1 |
| Future — etcd P2P Routing | — | ❌ | Future |

---

### Done: Phase 5 — Capacity-Aware Dispatch (RFC §3 TODO 1.2) ✅

> Completed in Commit 7. See [Commit 7](#commit-7--phase-5-capacity-aware-dispatch-with-freebufferslots--tta-queues-) for details.

FreeBufferSlots per instance + TryToAdd (TTA) queues per role + completion callbacks that drain TTA. 45 tests passing. Capacity is configurable via `encoder_capacity` / `denoiser_capacity` / `decoder_capacity` constructor args (will be wired to TransferBuffer slot counts in Phase 7).

---

### Commit 8 — Phase 5.5: Per-Role Parallelism CLI Args ✅

> Covers [Original Plan Phase 4](#original-plan-phase-4-enable-independent-parallelism-per-role) — independent of RFC data transfer, high practical value

**Goal**: Each role can have its own TP/SP/Ulysses/Ring degree (e.g., `--denoiser-gpus 1 2 3 4 --denoiser-sp 4`).

`[Diffusion] Add disaggregation Phase 5.5: per-role parallelism CLI args`

- `runtime/server_args.py` — Added 12 per-role parallelism fields: `encoder_tp`, `encoder_sp`, `encoder_ulysses`, `encoder_ring`, `denoiser_tp`, `denoiser_sp`, `denoiser_ulysses`, `denoiser_ring`, `decoder_tp`, `decoder_sp`, `decoder_ulysses`, `decoder_ring`. All default to None (auto-derive from num_gpus). Added `get_role_parallelism(role_type) → dict` helper method.
- `runtime/server_args.py` — 12 new CLI args: `--encoder-tp`, `--encoder-sp`, `--encoder-ulysses`, `--encoder-ring`, `--denoiser-tp`, `--denoiser-sp`, `--denoiser-ulysses`, `--denoiser-ring`, `--decoder-tp`, `--decoder-sp`, `--decoder-ulysses`, `--decoder-ring`
- `runtime/launch_server.py` — `launch_disagg_server()` and `launch_pool_disagg_server()` use `get_role_parallelism()` to pass per-role TP/SP/Ulysses/Ring overrides when building per-role `ServerArgs`. When None, `_adjust_parallelism()` auto-derives from `num_gpus` as before.
- `test/unit/test_server_args_unit.py` — 7 new tests: defaults are None, encoder/denoiser/decoder overrides, monolithic returns all None, mixed roles independent, CLI arg parsing

**Example usage**:
```bash
# Encoder TP=1, Denoiser SP=4 (ulysses), Decoder TP=1
sglang serve --model-path ... --disagg-mode \
  --encoder-gpus 0 --denoiser-gpus 1 2 3 4 --decoder-gpus 0 \
  --denoiser-sp 4 --denoiser-ulysses 4
```

**Remaining for full Phase 4 completion**:
- `distributed/parallel_state.py` support for per-role NCCL group initialization (if needed beyond current auto-derive)
- Validate multi-GPU denoiser with Wan2.1 on 4+ GPUs

---

### Done: Phase 6 — TransferBuffer — Pinned Memory Pool (RFC §2 TODO 3.1–3.3) ✅

> Completed in Commit 6. See [Commit 6](#commit-6--phase-6-transferbuffer-with-buddy-system-allocator-) for details.

`BuddyAllocator` + `TransferMetaBuffer` + `TransferTensorBuffer` with pinned memory pool, async D2H/H2D, manifest-based batch I/O. 47 tests passing. Provides `free_slots_count()` for Phase 5 `FreeBufferSlots` integration and `pool_data_ptr` for Phase 7 RDMA registration.

---

### Phase 7: TransferManager + P2P Data Transfer (RFC §2 TODO 2.1–2.3, §4)

> Current: all tensor data routes through DiffusionServer (ZMQ relay). RFC requires **P2P direct transfer** between role instances, with DiffusionServer only on the control plane.

**Goal**: Role instances transfer tensor data directly to each other via RDMA/RPC, DiffusionServer only dispatches metadata.

**Files to create/modify**:

| File | Action | Description |
|------|--------|-------------|
| `runtime/disaggregation/transfer_engine.py` | **New** | Wrap `MooncakeTransferEngine` (reuse from `srt/disaggregation/mooncake/`) for RDMA/RPC |
| `runtime/disaggregation/transfer_manager.py` | **New** | `TransferManager` with `receive_event_loop` and `send_event_loop` |
| `runtime/disaggregation/diffusion_server.py` | **Modify** | DS sends only metadata (no tensor relay); role instances do P2P |
| `runtime/managers/scheduler.py` | **Modify** | Add `TransferringQueue`, `SwappingQueue`, `Transfer.Poll` per request |

**P2P workflow (RFC §4 sequence)**:
```
1. Encoder finishes forward → swap out tensors to local TransferBuffer (D2H)
2. Encoder → DiffusionServer: metadata only (request_id, encoder's IP:Port, DP/FSDP info)
3. DiffusionServer selects Denoiser[j] → sends metadata to Denoiser[j]
4. Denoiser[j] allocates TransferBuffer slot → sends (IP:Port, buffer_address) back to Encoder
5. Encoder initiates RDMA/RPC transfer directly to Denoiser[j]'s buffer
6. Transfer complete → Encoder notifies Denoiser[j] + DiffusionServer
7. Denoiser[j] swaps tensor from TransferBuffer to GPU (H2D) → starts compute
```

**TransferManager internals**:
- `receive_event_loop`: listens for incoming buffer addresses from downstream, transfer completion notifications from upstream. Caches `Instance_ID → Rank_ID → IP:Port + BufferAddress` routing
- `send_event_loop`: thread pool + task queue for concurrent transfers. Supports `transfer_slice` to handle FSDP (encoder) → SP (denoiser) tensor redistribution
- Reuse ~70% from existing PD disagg: `MooncakeTransferEngine`, `BaseKVConn` pattern, metadata buffer/poll from `kv_events.py`

**Queue management (RFC TODO 4.4)**:
- `TransferringQueue`: requests waiting for network data to arrive
- `SwappingQueue`: requests with data arrived, ready for H2D transfer
- `Transfer.Poll`: per-request state tracking (analogous to `KV.Poll` in LLM PD disagg)

---

### Phase 8: CUDA Stream Overlap Scheduling (RFC §5 TODO 4.2–4.3)

> Current: synchronous execution per role. RFC specifies 3-stream pipelining for overlap.

**Goal**: Overlap H2D / Compute / D2H within each role to hide transfer latency.

**Files to modify**:

| File | Action | Description |
|------|--------|-------------|
| `runtime/managers/scheduler.py` | **Modify** | Replace synchronous event loop with 3-stream overlap loop (RFC §5 pseudocode) |

**3-stream design**:
```
Stream 1 (swap_in):   H2D transfer for Batch i+1  (from SwappingQueue → GPU)
Stream 2 (compute):   Forward pass for Batch i     (on GPU)
Stream 3 (swap_out):  D2H transfer for Batch i-1   (GPU → TransferBuffer)
```

**Request-level network trigger** (RFC TODO 4.3):
- After D2H to TransferBuffer, use non-blocking `torch.cuda.Event.query()` per request
- When any single request's D2H completes, immediately hand off to TransferManager for network send
- Decoupled from batch boundaries — avoids pipeline bubbles

**Reality check**: With 50-step denoising and BS=1, overlap benefit is <0.1%. This becomes valuable when:
- Step count drops (consistency models, 1–4 steps)
- Multi-request batching is enabled
- Encoder/decoder become heavier (future high-res models)

---

### Future: Decentralized P2P Routing via etcd (RFC Future Map)

Replace centralized DiffusionServer with decentralized architecture:
- **Control plane (etcd)**: service discovery, role registration with leases
- **Data plane (P2P)**: upstream roles perform edge load-balancing using local topology caches
- **Piggybacked state sync**: downstream nodes piggyback `FreeBufferSlots` in network ACKs

Not planned for near-term. Prerequisite: Phase 7 (P2P transfers) must be stable first.

---

## Original Plan (Reference)

The sections below preserve the original plan for reference. See [Implementation Progress](#implementation-progress) above for what was actually built.

### Original Plan Phase 1: Role Abstraction & Weight Separation (Single Machine)

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

### Original Plan Phase 2: Multi-Process Deployment (Single Machine, IPC)

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

### Original Plan Phase 3: DiffusionServer -- Request Routing & Orchestration

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

### Original Plan Phase 4: Enable Independent Parallelism Per Role

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

### Original Plan Phase 5: Cross-Machine Transfer (RDMA)

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

### Original Plan Phase 6: 3-Stream Pipeline Overlap (Optional/Future)

**Goal**: Overlap H2D / Compute / D2H within each role for pipelining.

**Reality check**: With 50-step denoising and BS=1, this saves <0.1%. Only valuable if:
- Step count drops significantly (e.g., consistency models with 1-4 steps)
- Decoder becomes heavier (future high-res VAE)

**Defer this unless step counts decrease substantially.**

---

## Timeline Summary

```
RFC §  Original Plan                   Actual / Planned
────── ──────────────────────────────  ──────────────────────────────────────────
§1     Phase 1 (Role Abstraction)      Commit 1 ✅
§4.1   Phase 1 (Role Weights)          Commit 1 ✅
—      Phase 2 (Multi-Process IPC)     Commit 2 ✅  (ZMQ multipart)
§3     Phase 3 (DiffusionServer)       Commit 3 ✅  (CLI + single-instance)
§3     Phase 3 (Pool-Based N:M:K)      Commit 5 ✅  (multi-instance orchestration)
—      — (Async + Observability)        Commit 4 ✅
§3     Phase 5 (Capacity-Aware DS)     Commit 7 ✅ (FreeBufferSlots, TTA, callbacks)
—      Phase 5.5 (Per-Role Parallel)   Commit 8 ✅ (CLI args + launcher wiring)
§2     Phase 6 (TransferBuffer)        Commit 6 ✅ (buddy allocator + pinned pool)
§2,§4  Phase 7 (TransferManager+P2P)   TODO — RDMA/RPC, P2P direct transfer, queues
§5     Phase 8 (3-Stream Overlap)      TODO — H2D/Compute/D2H pipelining
Future — (etcd P2P Routing)            Future — decentralized control plane
```

### Dependency Graph

```
Phase 5 (Capacity-Aware DS) ✅ ──┐
Phase 6 (TransferBuffer) ✅ ─────┼──→ Phase 7 (TransferManager + P2P)
                                 │        └──→ Phase 8 (3-Stream Overlap)
Phase 5.5 (Per-Role Parallelism) ✅ ┘   ← independent, can parallel
             └──→ Phase 7 (transfer_slice needs SP/FSDP info)
```
