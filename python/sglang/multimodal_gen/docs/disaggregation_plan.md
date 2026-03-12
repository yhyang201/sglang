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
- `runtime/server_args.py` — `--disagg-role` (encoder/denoiser/decoder/monolithic)
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
| 2. TransferManager — Engine Setup (RDMA/RPC) | 2.1 | ✅ Done | Commit 9 |
| 2. TransferManager — Receive Loop | 2.2 | ✅ Done | Commit 10 |
| 2. TransferManager — Send Loop + transfer_slice | 2.3 | 🔶 Partial (send done, transfer_slice deferred — see Route B) | Commit 10 |
| 2. TransferringQueue / SwappingQueue | 4.4 | 🔶 Deferred (BS=1, no multi-request batching yet) | — |
| Multi-Rank Pool Mode (Route A: broadcast) | — | ✅ Done | Commit 11 |
| 4. E2E Workflow — P2P data transfer (bypass DS) | — | ✅ Done | Commit 10 |
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

- `runtime/server_args.py` — Added 6 per-role parallelism fields: `encoder_tp`, `denoiser_tp`, `denoiser_sp`, `denoiser_ulysses`, `denoiser_ring`, `decoder_tp`. All default to None (auto-derive from num_gpus). Added `get_role_parallelism(role_type) → dict` helper method. Encoder and decoder only support TP override; denoiser supports all four.
- `runtime/server_args.py` — 6 new CLI args: `--encoder-tp`, `--denoiser-tp`, `--denoiser-sp`, `--denoiser-ulysses`, `--denoiser-ring`, `--decoder-tp`
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

### Commit 9 — Phase 7a: TransferEngine + TransferManager + P2P Protocol ✅

> Covers [RFC §2 TODO 2.1](#phase-7-transfermanager--p2p-data-transfer-rfc-2-todo-2123-4) — Engine setup, transfer abstraction, P2P protocol

`[Diffusion] Add disaggregation Phase 7a: TransferEngine, TransferManager, and P2P protocol`

- `runtime/disaggregation/transfer_engine.py` — **New**: `BaseTransferEngine` ABC, `MooncakeDiffusionEngine` (wraps `srt` `MooncakeTransferEngine` for RDMA), `MockTransferEngine` (in-process ctypes.memmove for unit testing), `create_transfer_engine()` factory with auto-fallback
- `runtime/disaggregation/p2p_protocol.py` — **New**: P2P control message definitions (`P2PStagedMsg`, `P2PAllocMsg`, `P2PAllocatedMsg`, `P2PPushMsg`, `P2PPushedMsg`, `P2PReadyMsg`, `P2PDoneMsg`, `P2PRegisterMsg`), `encode_p2p_msg()` / `decode_p2p_msg()` / `is_p2p_message()` helpers, `P2P_MAGIC` discriminator
- `runtime/disaggregation/transfer_manager.py` — **New**: `DiffusionTransferManager` per-instance P2P coordinator: `stage_tensors()` (D2H), `push_to_peer()` (RDMA), `allocate_receive_slot()`, `load_tensors()` (H2D), slot lifecycle management
- `runtime/disaggregation/diffusion_server.py` — **Modified**: Added `p2p_mode` flag, `_P2PRequestState` tracking, `_P2PTTAEntry`, peer registry, P2P message handlers (`_handle_p2p_register`, `_handle_p2p_staged`, `_handle_p2p_allocated`, `_handle_p2p_pushed`, `_handle_p2p_done`), P2P TTA drain, refactored result handlers via `_handle_role_result` dispatcher
- `runtime/server_args.py` — Added `--disagg-p2p-mode` CLI arg
- `runtime/launch_server.py` — Wires `p2p_mode` to DiffusionServer constructor
- `test/unit/test_transfer_engine.py` — 9 tests: MockTransferEngine data copy, session management, factory
- `test/unit/test_transfer_manager.py` — 23 tests: staging, receive, full P2P cycle, concurrent transfers, protocol encode/decode
- `test/unit/test_diffusion_server.py` — 14 tests (+5): P2P init, registration, staged→alloc flow, full E2E handshake

**P2P handshake protocol (DS-brokered):**
```
1. Encoder → DS:      p2p_staged   {request_id, data_size, manifest, session_id, slot_offset}
2. DS → Denoiser:     p2p_alloc    {request_id, data_size, source_role}
3. Denoiser → DS:     p2p_allocated {request_id, session_id, pool_ptr, slot_offset}
4. DS → Encoder:      p2p_push     {request_id, dest_session_id, dest_addr, transfer_size}
5. Encoder → DS:      p2p_pushed   {request_id}
6. DS → Denoiser:     p2p_ready    {request_id, manifest, slot_offset, scalar_fields}
```

---

### Commit 10 — Phase 7b: Scheduler P2P Integration ✅

> Covers [RFC §2 TODO 2.2–2.3, §4](#phase-7-transfermanager--p2p-data-transfer-rfc-2-todo-2123-4) — instance-side P2P event loop

`[Diffusion] Add disaggregation Phase 7b: scheduler P2P integration`

**Goal**: Integrate TransferManager into scheduler pool mode event loop. Each role instance handles P2P messages alongside compute work.

- `runtime/managers/scheduler.py` — **Modified**: Added P2P-aware pool mode event loop with `__p2p__` frame discriminator. New methods:
  - `_init_p2p_transfer_manager()`: Creates TransferTensorBuffer + BaseTransferEngine + DiffusionTransferManager, sends `p2p_register` to DS
  - `_is_p2p_frames()` / `_handle_p2p_message()`: Static discriminator + dispatcher
  - `_handle_p2p_alloc()`: Allocates receive slot, sends `p2p_allocated`
  - `_handle_p2p_push_cmd()`: RDMA pushes staged data to peer, sends `p2p_pushed`, frees staged slot
  - `_handle_p2p_ready()`: Loads tensors from buffer (H2D), reconstructs Req, dispatches to role compute
  - `_build_req_from_p2p()`: Reconstructs Req from scalar + tensor dicts (via `object.__new__` to skip validation)
  - `_p2p_denoiser_compute()`: Run denoising → stage output → send `p2p_done` with staged info for decoder
  - `_p2p_decoder_compute()`: Run decoding → pack result frames → send `p2p_done` with result
  - `_pool_mode_encoder_p2p_stage()`: Stage encoder output to TransferBuffer → send `p2p_staged`
  - Updated `_pool_mode_event_loop()`: Pre-receives frames, checks for P2P, routes to P2P or relay handlers
  - Updated step methods to accept pre-received `frames` parameter
- `runtime/server_args.py` — Added `--disagg-transfer-pool-size` (default 256 MiB), `--disagg-p2p-hostname` (default 127.0.0.1)
- `test/unit/test_scheduler_p2p.py` — **New**: 16 tests covering frame detection, alloc/allocated, push/pushed, Req reconstruction, encoder staging, message dispatch, full E2E data transfer, init + registration

**Instance-side P2P event loop**:
```python
while running:
    frames = pool_work_pull.recv_multipart()
    if is_p2p_message(frames):
        msg = decode_p2p_msg(frames)
        if msg_type == "p2p_alloc":    → allocate_receive_slot → send p2p_allocated
        elif msg_type == "p2p_push":   → push_to_peer (RDMA) → send p2p_pushed
        elif msg_type == "p2p_ready":  → load_tensors (H2D) → compute → send p2p_done
    else:
        # Existing relay-mode compute work
```

---

### Commit 11 — Phase 7c: Multi-Rank Pool Mode (Route A — Broadcast Full Tensor) ✅

> Covers multi-rank support for pool mode event loop. Each role can use SP/TP/CFG parallelism with multiple GPUs.

`[Diffusion] Add disaggregation Phase 7c: multi-rank pool mode support`

**Goal**: When a role uses multiple GPUs (e.g., denoiser with SP=4), only rank 0 communicates via ZMQ with DiffusionServer. All ranks participate in `execute_forward()` via NCCL collectives.

**Route A (implemented)**: Broadcast full tensors to all ranks, let pipeline shard internally.
- Encoder output (e.g., prompt_embeds) is broadcast in full; the pipeline's `shard_latents_for_sp()` handles sharding.
- TP mode: all ranks receive the same full tensor; model internally handles weight partitioning.
- Simple, correct, works with all models and parallelism strategies.

- `runtime/managers/scheduler.py` — **Modified**:
  - `_init_pool_mode_sockets()`: Gated on `self.gpu_id == 0` — non-rank-0 has no ZMQ sockets
  - `_init_p2p_transfer_manager()`: Gated on `self.gpu_id == 0` — non-rank-0 has no TransferManager
  - `_pool_mode_recv_work()`: **New** — rank 0 receives from ZMQ, broadcasts to all ranks via `broadcast_pyobj` over SP/CFG/TP CPU groups
  - `_pool_mode_event_loop()`: Rewritten — uses `_pool_mode_recv_work()`, routes P2P frames to rank-0 `_handle_p2p_message()` or non-rank-0 `_handle_p2p_non_rank0()`
  - `_handle_p2p_non_rank0()`: **New** — skip alloc/push (rank-0-only), enter compute on p2p_ready
  - `_handle_p2p_ready_non_rank0()`: **New** — build minimal Req from scalars, call execute_forward (pipeline NCCL handles tensor sync)
  - Step methods (`_pool_mode_encoder_step`, `_pool_mode_denoiser_step`, `_pool_mode_decoder_step`): All ZMQ send calls gated on `self._pool_result_push is not None`
  - Cleanup gated on socket/manager existence
- `test/unit/test_scheduler_p2p.py` — Added 9 multi-rank gating tests: init skip, send gating for all 3 roles, P2P flag logic, non-rank-0 P2P handling

**Multi-rank event loop flow**:
```
All ranks:
  frames = _pool_mode_recv_work()       # rank 0 ZMQ recv + broadcast
  if P2P message:
    rank 0  → _handle_p2p_message()     # alloc/push/ready (full handling)
    rank !0 → _handle_p2p_non_rank0()   # only p2p_ready → execute_forward
  else (relay mode):
    all ranks → step method              # execute_forward (NCCL sync)
    rank 0 only → send result via ZMQ
```

---

### Deferred: Route B — `transfer_slice` Optimization

> [RFC §2 TODO 2.3](#phase-7-transfermanager--p2p-data-transfer-rfc-2-todo-2123-4) — SP-aware tensor slicing during transfer

**What**: Instead of broadcasting the full tensor and having the pipeline shard it, `transfer_slice` would send only the slice each SP rank needs directly via RDMA.

**Why deferred**:
1. **Model-specific shard patterns**: Each model family has a different sharding dimension and strategy:
   - Video (Wan, HunyuanVideo): `[B,C,T,H,W]` shard on dim=2 (temporal)
   - Flux: `[B,C,H',W']` shard on dim=2
   - SDXL: `[B,H*W,C]` shard on dim=1
   - LTX-2: `[B,S,D]` shard on dim=1 by frame
   - ZImage: `[B,C,T,H,W]` shard on dim=3 with H/W swap
   This would require exposing shard patterns from pipeline configs into the transfer layer.
2. **Pipeline interface changes**: `shard_latents_for_sp()` and `gather_latents_for_sp()` are pipeline-config methods. Transfer layer would need to call these, creating a coupling between transport and pipeline logic.
3. **NCCL broadcast is fast**: For intra-machine (same NVLink/PCIe domain), broadcast overhead is negligible vs. the denoising compute. The optimization matters most for cross-machine RDMA where bandwidth is limited.
4. **TP doesn't benefit**: TP needs the full tensor on every rank — no slicing possible.

**When to implement**: When cross-machine RDMA deployments show transfer-to-compute ratio > 5%, or when consistency models reduce step counts to 1–4 (making transfer time proportionally significant).

---

### Deferred: TransferringQueue / SwappingQueue (RFC §4 TODO 4.4)

Request-level queue management for pipelining multiple in-flight transfers. Currently unnecessary because:
- BS=1 constraint means at most 1 request per role at a time
- Transfer + compute are synchronous within the event loop
- Queuing becomes valuable with multi-request batching (future work)

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
§2     Phase 7a (Engine+Manager+P2P)    Commit 9 ✅ (TransferEngine, TransferManager, protocol)
§2,§4  Phase 7b (Scheduler P2P)        Commit 10 ✅ (P2P event loop + handlers)
—      Phase 7c (Multi-Rank Pool)      Commit 11 ✅ (Route A: broadcast full tensor)
§2     Route B (transfer_slice)        Deferred — model-specific shard patterns, NCCL fast enough
§4     Queues (Transferring/Swapping)  Deferred — BS=1 constraint, no multi-request batching
§5     Phase 8 (3-Stream Overlap)      TODO — H2D/Compute/D2H pipelining
Future — (etcd P2P Routing)            Future — decentralized control plane
```

### Dependency Graph

```
Phase 5 (Capacity-Aware DS) ✅ ──┐
Phase 6 (TransferBuffer) ✅ ─────┼──→ Phase 7a (Engine+Manager+P2P) ✅
                                 │        └──→ Phase 7b (Scheduler P2P) ✅
Phase 5.5 (Per-Role Parallelism) ✅ ┘        └──→ Phase 7c (Multi-Rank Pool) ✅
                                                      ├──→ Route B (transfer_slice) [deferred]
                                                      └──→ Phase 8 (3-Stream Overlap)
```
