# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import dataclasses
import multiprocessing as mp
import os
import signal
import sys
import threading

import psutil
import uvicorn

from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
from sglang.multimodal_gen.runtime.managers.gpu_worker import run_scheduler_process
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    prepare_server_args,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import configure_logger, logger


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def launch_server(server_args: ServerArgs, launch_http_server: bool = True):
    """
    Args:
        launch_http_server: False for offline local mode
    """
    configure_logger(server_args)

    # Start a new server with multiple worker processes
    logger.info("Starting server...")

    num_gpus = server_args.num_gpus
    processes = []

    # Pipes for master to talk to slaves
    task_pipes_to_slaves_w = []
    task_pipes_to_slaves_r = []
    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        task_pipes_to_slaves_r.append(r)
        task_pipes_to_slaves_w.append(w)

    # Pipes for slaves to talk to master
    result_pipes_from_slaves_w = []
    result_pipes_from_slaves_r = []
    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        result_pipes_from_slaves_r.append(r)
        result_pipes_from_slaves_w.append(w)

    # Launch all worker processes
    master_port = server_args.master_port or (server_args.master_port + 100)
    scheduler_pipe_readers = []
    scheduler_pipe_writers = []

    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_writers.append(writer)
        if i == 0:  # Master worker
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    i,  # rank
                    master_port,
                    server_args,
                    writer,
                    None,  # No task pipe to read from master
                    None,  # No result pipe to write to master
                    task_pipes_to_slaves_w,
                    result_pipes_from_slaves_r,
                ),
                name=f"sglang-diffusionWorker-{i}",
                daemon=True,
            )
        else:  # Slave workers
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    i,  # rank
                    master_port,
                    server_args,
                    writer,
                    None,  # No task pipe to read from master
                    None,  # No result pipe to write to master
                    task_pipes_to_slaves_r[i - 1],
                    result_pipes_from_slaves_w[i - 1],
                ),
                name=f"sglang-diffusionWorker-{i}",
                daemon=True,
            )
        scheduler_pipe_readers.append(reader)
        process.start()
        processes.append(process)

    # Wait for all workers to be ready
    scheduler_infos = []
    for writer in scheduler_pipe_writers:
        writer.close()

    # Close unused pipe ends in parent process
    for p in task_pipes_to_slaves_w:
        p.close()
    for p in task_pipes_to_slaves_r:
        p.close()
    for p in result_pipes_from_slaves_w:
        p.close()
    for p in result_pipes_from_slaves_r:
        p.close()

    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            processes[i].join()
            logger.error(f"Exit code: {processes[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)
        reader.close()

    logger.debug("All workers are ready")

    if launch_http_server:
        logger.info("Starting FastAPI server.")
        if server_args.webui:
            logger.info("Launch FastAPI server in another process because of webui.")
            http_server_process = mp.Process(
                target=launch_http_server_only,
                args=(server_args,),
                name=f"sglang-diffusion-webui",
                daemon=True,
            )
            http_server_process.start()
        else:
            launch_http_server_only(server_args)

    return processes


def launch_pool_disagg_server(
    server_args: ServerArgs,
    encoder_gpus: list[list[int]],
    denoiser_gpus: list[list[int]],
    decoder_gpus: list[list[int]],
    launch_http_server: bool = True,
):
    """Launch a pool-based disaggregated server with N:M:K independent role instances.

    DiffusionServer orchestrates the full pipeline, dispatching at every
    role transition (Encoder → Denoiser → Decoder).

    Args:
        server_args: Base server configuration
        encoder_gpus: List of GPU ID lists, one per encoder instance.
            e.g., [[0], [2]] for 2 encoder instances on GPUs 0 and 2.
        denoiser_gpus: List of GPU ID lists, one per denoiser instance.
            e.g., [[1], [3]] for 2 denoiser instances.
        decoder_gpus: List of GPU ID lists, one per decoder instance.
            e.g., [[0], [2]] for 2 decoder instances (can share with encoder).
        launch_http_server: Whether to launch the HTTP server.

    Example:
        launch_pool_disagg_server(server_args,
            encoder_gpus=[[0], [2]],
            denoiser_gpus=[[1], [3]],
            decoder_gpus=[[0], [2]],
        )
    """
    from sglang.multimodal_gen.runtime.disaggregation.diffusion_server import (
        DiffusionServer,
    )
    from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
    from sglang.multimodal_gen.runtime.utils.common import is_port_available

    configure_logger(server_args)

    num_encoders = len(encoder_gpus)
    num_denoisers = len(denoiser_gpus)
    num_decoders = len(decoder_gpus)
    logger.info(
        "Starting pool disagg server: %d encoder(s), %d denoiser(s), %d decoder(s)...",
        num_encoders,
        num_denoisers,
        num_decoders,
    )

    host = server_args.host or "127.0.0.1"

    def find_port(start):
        port = start
        while not is_port_available(port):
            port += 1
        return port

    # Allocate endpoints
    port_cursor = server_args.scheduler_port + 3000

    # Per-instance work endpoints (instance binds PULL, DS connects PUSH)
    encoder_work_endpoints = []
    for i in range(num_encoders):
        p = find_port(port_cursor)
        encoder_work_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    denoiser_work_endpoints = []
    for i in range(num_denoisers):
        p = find_port(port_cursor)
        denoiser_work_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    decoder_work_endpoints = []
    for i in range(num_decoders):
        p = find_port(port_cursor)
        decoder_work_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    # Per-role-type result endpoints (DS binds PULL, instances connect PUSH)
    encoder_result_ep = f"tcp://{host}:{find_port(port_cursor)}"
    port_cursor += 1
    denoiser_result_ep = f"tcp://{host}:{find_port(port_cursor)}"
    port_cursor += 1
    decoder_result_ep = f"tcp://{host}:{find_port(port_cursor)}"
    port_cursor += 1

    logger.info(
        "Pool endpoints allocated: %d work + 3 result endpoints",
        num_encoders + num_denoisers + num_decoders,
    )

    # Launch all role instances
    all_processes = []

    role_configs = [
        (RoleType.ENCODER, encoder_gpus, encoder_work_endpoints, encoder_result_ep),
        (
            RoleType.DENOISING,
            denoiser_gpus,
            denoiser_work_endpoints,
            denoiser_result_ep,
        ),
        (RoleType.DECODER, decoder_gpus, decoder_work_endpoints, decoder_result_ep),
    ]

    for role_type, gpu_lists, work_eps, result_ep in role_configs:
        for inst_idx, gpu_ids in enumerate(gpu_lists):
            num_role_gpus = len(gpu_ids)

            # Per-role parallelism: use explicit overrides if set, else None (auto-derive)
            role_par = server_args.get_role_parallelism(role_type)

            role_overrides = {
                "disagg_role": role_type,
                "disagg_pool_mode": True,
                "pool_work_endpoint": work_eps[inst_idx],
                "pool_result_endpoint": result_ep,
                "num_gpus": num_role_gpus,
                "warmup": role_type == RoleType.ENCODER,
                "scheduler_port": find_port(port_cursor),
                "master_port": find_port(port_cursor + 100),
                # Per-role parallelism (None = auto-derive from num_gpus)
                "tp_size": role_par["tp_size"],
                "sp_degree": role_par["sp_degree"],
                "ulysses_degree": role_par["ulysses_degree"],
                "ring_degree": role_par["ring_degree"],
            }
            port_cursor = role_overrides["master_port"] + 100

            base_dict = {
                f.name: getattr(server_args, f.name)
                for f in dataclasses.fields(server_args)
            }
            base_dict.update(role_overrides)
            base_dict.pop("pipeline_config", None)
            role_args = ServerArgs.from_kwargs(**base_dict)

            pool_ctx = mp.get_context("spawn")

            for rank_idx in range(num_role_gpus):
                reader, writer = pool_ctx.Pipe(duplex=False)
                gpu_id = gpu_ids[rank_idx]

                process = pool_ctx.Process(
                    target=_run_disagg_role_process,
                    args=(gpu_id, rank_idx, rank_idx, role_args, writer, [], []),
                    name=f"sglang-pool-{role_type.value}-{inst_idx}-r{rank_idx}",
                    daemon=True,
                )
                process.start()
                all_processes.append(process)

                try:
                    data = reader.recv()
                except EOFError:
                    logger.error(
                        "Pool %s[%d] rank %d is dead.",
                        role_type.value,
                        inst_idx,
                        rank_idx,
                    )
                    raise
                if data.get("status") != "ready":
                    raise RuntimeError(
                        f"Pool {role_type.value}[{inst_idx}] rank {rank_idx} "
                        "failed to initialize."
                    )
                reader.close()

            logger.info(
                "Pool %s[%d] ready on GPU(s) %s (work=%s)",
                role_type.value.upper(),
                inst_idx,
                gpu_ids,
                work_eps[inst_idx],
            )

    logger.info("All pool role instances ready")

    # Start DiffusionServer
    frontend_endpoint = f"tcp://{host}:{server_args.scheduler_port}"

    p2p_mode = getattr(server_args, "disagg_p2p_mode", False)
    diffusion_server = DiffusionServer(
        frontend_endpoint=frontend_endpoint,
        encoder_work_endpoints=encoder_work_endpoints,
        denoiser_work_endpoints=denoiser_work_endpoints,
        decoder_work_endpoints=decoder_work_endpoints,
        encoder_result_endpoint=encoder_result_ep,
        denoiser_result_endpoint=denoiser_result_ep,
        decoder_result_endpoint=decoder_result_ep,
        dispatch_policy_name=server_args.disagg_dispatch_policy,
        timeout_s=float(server_args.disagg_timeout),
        p2p_mode=p2p_mode,
    )
    diffusion_server.start()

    if launch_http_server:
        logger.info(
            "Starting FastAPI server (connected to DiffusionServer at port %d).",
            server_args.scheduler_port,
        )
        launch_http_server_only(server_args)

    return all_processes


def _run_disagg_role_process(
    gpu_id: int,
    _local_rank: int,
    rank: int,
    server_args: ServerArgs,
    pipe_writer: mp.connection.Connection,
    task_pipes: list,
    result_pipes: list,
):
    """Entry point for a disagg role process.

    Uses the physical GPU index (gpu_id) as local_rank so that
    torch.cuda.set_device(local_rank) selects the correct GPU.
    This avoids relying on CUDA_VISIBLE_DEVICES remapping, which
    may not work if CUDA was pre-initialized in the parent process.
    """
    run_scheduler_process(
        local_rank=gpu_id,
        rank=rank,
        master_port=server_args.master_port,
        server_args=server_args,
        pipe_writer=pipe_writer,
        task_pipe_r=None,
        result_pipe_w=None,
        task_pipes_to_slaves=task_pipes,
        result_pipes_from_slaves=result_pipes,
    )


def launch_http_server_only(server_args):
    # set for endpoints to access global_server_args
    set_global_server_args(server_args)
    app = create_app(server_args)
    uvicorn.run(
        app,
        use_colors=True,
        log_level=server_args.log_level,
        host=server_args.host,
        port=server_args.port,
        reload=False,
    )


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
