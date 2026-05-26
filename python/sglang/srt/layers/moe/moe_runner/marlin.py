from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )

MARLIN_MOE_WORKSPACE: Optional[torch.Tensor] = None


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # GPTQ specific (Optional)
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific (Optional)
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # Optional
    expert_map: Optional[torch.Tensor] = None


@register_fused_func("none", "marlin")
def fused_experts_none_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    global MARLIN_MOE_WORKSPACE
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    assert runner_config.activation == "silu", "Only SiLU activation is supported."

    if (
        MARLIN_MOE_WORKSPACE is None
        or MARLIN_MOE_WORKSPACE.device != hidden_states.device
    ):
        MARLIN_MOE_WORKSPACE = marlin_make_workspace(
            hidden_states.device, max_blocks_per_sm=4
        )

    marlin_hidden_states = hidden_states
    # Avoid aliasing the MoE input buffer until Marlin output semantics are
    # fully validated across shared-expert and overlap paths.
    marlin_inplace = False
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):
        # MXFP4(E8M0) Marlin kernels are only numerically valid on the bf16
        # activation path. The fp16 + E8M0 path is intentionally not generated
        # in sgl-kernel, so upcast activations here and cast the result back.
        marlin_hidden_states = hidden_states.to(torch.bfloat16)
        marlin_inplace = False

    output = fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        w1=quant_info.w13_qweight,
        w2=quant_info.w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        gating_output=topk_output.router_logits,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        expert_map=quant_info.expert_map,
        g_idx1=quant_info.w13_g_idx,
        g_idx2=quant_info.w2_g_idx,
        sort_indices1=quant_info.w13_g_idx_sort_indices,
        sort_indices2=quant_info.w2_g_idx_sort_indices,
        w1_zeros=quant_info.w13_qzeros,
        w2_zeros=quant_info.w2_qzeros,
        workspace=MARLIN_MOE_WORKSPACE,
        num_bits=quant_info.weight_bits,
        is_k_full=quant_info.is_k_full,
        inplace=marlin_inplace,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        clamp_limit=runner_config.swiglu_limit,
    ).to(hidden_states.dtype)

    return StandardCombineInput(
        hidden_states=output,
    )


def _run_marlin_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    """Shared Marlin kernel invocation for DeepEP paths."""
    global MARLIN_MOE_WORKSPACE
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    assert runner_config.activation == "silu", "Only SiLU activation is supported."

    if (
        MARLIN_MOE_WORKSPACE is None
        or MARLIN_MOE_WORKSPACE.device != hidden_states.device
    ):
        MARLIN_MOE_WORKSPACE = marlin_make_workspace(
            hidden_states.device, max_blocks_per_sm=4
        )

    marlin_hidden_states = hidden_states
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):
        marlin_hidden_states = hidden_states.to(torch.bfloat16)

    M = marlin_hidden_states.shape[0]
    # gating_output is only used for a shape assertion; pass a dummy tensor.
    gating_output = torch.empty(
        M, 1, device=hidden_states.device, dtype=hidden_states.dtype
    )

    return fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        w1=quant_info.w13_qweight,
        w2=quant_info.w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        gating_output=gating_output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=quant_info.expert_map,
        g_idx1=quant_info.w13_g_idx,
        g_idx2=quant_info.w2_g_idx,
        sort_indices1=quant_info.w13_g_idx_sort_indices,
        sort_indices2=quant_info.w2_g_idx_sort_indices,
        w1_zeros=quant_info.w13_qzeros,
        w2_zeros=quant_info.w2_qzeros,
        workspace=MARLIN_MOE_WORKSPACE,
        num_bits=quant_info.weight_bits,
        is_k_full=quant_info.is_k_full,
        inplace=False,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        clamp_limit=runner_config.swiglu_limit,
    ).to(hidden_states.dtype)


def _deepep_ll_to_flat(
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert DeepEP LL padded per-expert layout to flat (M, 1) topk format.

    Returns (topk_ids, topk_weights) both shaped (num_experts * expected_m, 1).
    Padding tokens get weight 0.0 so they contribute nothing to output.
    """
    num_experts = masked_m.shape[0]
    device = hidden_states.device

    # topk_ids: token at position i*expected_m + j → expert i
    topk_ids = (
        torch.arange(num_experts, device=device)
        .unsqueeze(1)
        .expand(-1, expected_m)
        .reshape(-1, 1)
    )

    # topk_weights: 1.0 for valid tokens, 0.0 for padding
    token_pos = torch.arange(expected_m, device=device).unsqueeze(0)  # (1, expected_m)
    valid_mask = token_pos < masked_m.unsqueeze(1)  # (num_experts, expected_m)
    topk_weights = valid_mask.float().reshape(-1, 1)

    return topk_ids, topk_weights


def _deepep_normal_to_flat(
    num_recv_tokens_per_expert: List[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert DeepEP Normal expert-grouped layout to flat (M, 1) topk format.

    Returns (topk_ids, topk_weights) both shaped (sum(num_recv), 1).
    Each token-expert pair gets weight 1.0; actual weights applied by combine.
    """
    counts = torch.tensor(num_recv_tokens_per_expert, device=device, dtype=torch.long)
    total = counts.sum().item()

    topk_ids = torch.repeat_interleave(
        torch.arange(len(num_recv_tokens_per_expert), device=device), counts
    ).unsqueeze(1)

    topk_weights = torch.ones(total, 1, device=device, dtype=torch.float32)

    return topk_ids, topk_weights


@register_fused_func("deepep", "marlin")
def fused_experts_deepep_to_marlin(
    dispatch_output: Union[DeepEPLLDispatchOutput, DeepEPNormalDispatchOutput],
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> Union[DeepEPLLCombineInput, DeepEPNormalCombineInput]:
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )

    if DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
        hidden_states = dispatch_output.hidden_states
        topk_ids, topk_weights = _deepep_ll_to_flat(
            hidden_states, dispatch_output.masked_m, dispatch_output.expected_m
        )

        output = _run_marlin_moe(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            topk_ids,
            topk_weights,
            quant_info,
            runner_config,
        )

        return DeepEPLLCombineInput(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        hidden_states = dispatch_output.hidden_states
        topk_ids, topk_weights = _deepep_normal_to_flat(
            dispatch_output.num_recv_tokens_per_expert, hidden_states.device
        )

        output = _run_marlin_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            quant_info,
            runner_config,
        )

        return DeepEPNormalCombineInput(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    else:
        raise ValueError(
            f"fused_experts_deepep_to_marlin: unsupported dispatch format "
            f"{dispatch_output.format}"
        )
