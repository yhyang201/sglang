from typing import Iterable, Optional, Tuple

import torch
from transformers import LlamaConfig

from sglang.srt.distributed import get_moe_tensor_parallel_rank
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gpt_oss import GptOssSparseMoeBlock
from sglang.srt.models.llama_eagle3 import (
    LlamaDecoderLayer,
    LlamaForCausalLMEagle3,
    LlamaModel,
)
from sglang.srt.utils import add_prefix


class GptOssDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)
        # Use the default settings for gptoss
        config.swiglu_limit = 7.0
        self.mlp = GptOssSparseMoeBlock(layer_id, config, quant_config, prefix)


class GptOssModel(LlamaModel):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.midlayer = GptOssDecoderLayer(config, 0, quant_config, prefix)


class GptOssForCausalLMEagle3(LlamaForCausalLMEagle3):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)
        self.model = GptOssModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        loaded_params = set()
        params_dict = dict(self.named_parameters())
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping_fused(
            ckpt_gate_up_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_gate_up_proj_bias_name="gate_up_proj_bias",
            ckpt_down_proj_bias_name="down_proj_bias",
        )

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(param_name)
                break
            else:
                original_name = name
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if "experts" in name:
                        print(f"{weight_name=} {name=}")
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    name = "model." + name
                    if name not in params_dict:
                        name = original_name
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    if "bias" not in name:
                        loaded_weight = loaded_weight.transpose(-2, -1)
                    if "w2_weight_bias" in name and get_moe_tensor_parallel_rank() != 0:
                        loaded_weight = loaded_weight.zero_()

                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                    )
                    loaded_params.add(name)
                    break
                else:
                    # Handle regular parameters
                    param_name = name if name in params_dict else f"model.{name}"
                    print(f"param_name: {param_name}")
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(param_name)
                    else:
                        raise ValueError(
                            f"Parameter {param_name} not found in params_dict"
                        )


EntryClass = GptOssForCausalLMEagle3
