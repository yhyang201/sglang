# import torch
# from torch import nn
# import torch.nn.functional as F

# from sglang.srt.layers.linear import (
#     ColumnParallelLinear,
#     RowParallelLinear,
# )
# from sglang.srt.utils import add_prefix
# from transformers.activations import ACT2FN
# from typing import Optional, Type, List
# import torch
# import torch.nn as nn
# from sglang.srt.layers.radix_attention import RadixAttention
# from sglang.srt.layers.vocab_parallel_embedding import (
#     ParallelLMHead,
#     VocabParallelEmbedding,
# )
# from sglang.srt.layers.rotary_embedding import get_rope
# from sglang.srt.layers.dp_attention import is_dp_attention_enabled
# from sglang.srt.layers.activation import QuickGELU
# from sglang.srt.layers.attention.vision import VisionAttention
# from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
# from sglang.srt.layers.logits_processor import LogitsProcessor
# from sglang.srt.layers.quantization.base_config import QuantizationConfig
# from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
# from sglang.srt.managers.mm_utils import general_mm_embed_routine
# from sglang.srt.layers.layernorm import RMSNorm
# from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
# from sglang.srt.model_executor.forward_batch_info import ForwardBatch
# from sglang.srt.model_loader.weight_utils import default_weight_loader
# from sglang.srt.models.utils import WeightsMapper, compute_cu_seqlens_from_grid_numpy
# from sglang.srt.utils import add_prefix, make_layers


# class GlmImageVisionMLP(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: int = None,
#         act_layer: Type[nn.Module] = QuickGELU,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ):
#         super().__init__()
#         self.fc1 = ColumnParallelLinear(
#             in_features,
#             hidden_features,
#             quant_config=quant_config,
#             prefix=add_prefix("fc1", prefix),
#         )
#         self.act = act_layer()
#         self.fc2 = RowParallelLinear(
#             hidden_features,
#             in_features,
#             quant_config=quant_config,
#             prefix=add_prefix("fc2", prefix),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x, _ = self.fc1(x)
#         x = self.act(x)
#         x, _ = self.fc2(x)
#         return x


# class GlmImageVisionAttention(nn.Module):
#     def __init__(self, config) -> None:
#         super().__init__()
#         self.dim = config.hidden_size
#         self.num_heads = config.num_heads
#         self.head_dim = self.dim // self.num_heads
#         self.num_key_value_groups = 1  # needed for eager attention
#         self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
#         self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
#         self.scaling = self.head_dim**-0.5
#         self.config = config
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = False

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#     ) -> torch.Tensor:
#         seq_length = hidden_states.shape[0]
#         query_states, key_states, value_states = (
#             self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
#         )
#         query_states = query_states.transpose(0, 1).unsqueeze(0)
#         key_states = key_states.transpose(0, 1).unsqueeze(0)
#         value_states = value_states.transpose(0, 1).unsqueeze(0)

#         attention_interface: Callable = eager_attention_forward
#         if self.config._attn_implementation != "eager":
#             attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

#         if "flash" in self.config._attn_implementation:
#             # Flash Attention: Use cu_seqlens for variable length attention
#             max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
#             attn_output, _ = attention_interface(
#                 self,
#                 query_states,
#                 key_states,
#                 value_states,
#                 attention_mask=None,
#                 scaling=self.scaling,
#                 dropout=0.0 if not self.training else self.attention_dropout,
#                 cu_seq_lens_q=cu_seqlens,
#                 cu_seq_lens_k=cu_seqlens,
#                 max_length_q=max_seqlen,
#                 max_length_k=max_seqlen,
#                 is_causal=False,
#             )
#         else:
#             # Other implementations: Process each chunk separately
#             lengths = cu_seqlens[1:] - cu_seqlens[:-1]
#             splits = [
#                 torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
#             ]

#             attn_outputs = [
#                 attention_interface(
#                     self,
#                     q,
#                     k,
#                     v,
#                     attention_mask=None,
#                     scaling=self.scaling,
#                     dropout=0.0 if not self.training else self.attention_dropout,
#                     is_causal=False,
#                 )[0]
#                 for q, k, v in zip(*splits)
#             ]
#             attn_output = torch.cat(attn_outputs, dim=1)

#         attn_output = attn_output.reshape(seq_length, -1).contiguous()
#         attn_output = self.proj(attn_output)
#         return attn_output


# class GlmImageVisionPatchEmbed(nn.Module):
#     def __init__(self, config) -> None:
#         super().__init__()
#         self.patch_size = config.patch_size
#         self.in_channels = config.in_channels
#         self.embed_dim = config.hidden_size
#         kernel_size = [self.patch_size, self.patch_size]
#         self.proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

#     def forward(self, hidden_states) -> torch.Tensor:
#         target_dtype = self.proj.weight.dtype
#         hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
#         hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
#         return hidden_states


# class GlmImageVisionEmbeddings(nn.Module):
#     def __init__(self, config) -> None:
#         super().__init__()
#         self.config = config
#         self.embed_dim = config.hidden_size
#         self.image_size = config.image_size
#         self.patch_size = config.patch_size

#         self.num_patches = (self.image_size // self.patch_size) ** 2
#         self.num_positions = self.num_patches
#         self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
#         self.interpolated_method = "bilinear"

#     def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords) -> torch.Tensor:
#         # Get position embedding parameters
#         pos_embed_weight = self.position_embedding.weight
#         hidden_size = pos_embed_weight.shape[1]
#         device = pos_embed_weight.device

#         # Convert inputs to tensors if needed
#         if isinstance(lengths, list):
#             lengths = torch.tensor(lengths, device=device, dtype=torch.long)

#         # Prepare 2D position embedding
#         orig_size_sq = pos_embed_weight.shape[0]
#         orig_size = int(orig_size_sq**0.5)
#         pos_embed_2d = (
#             pos_embed_weight.view(orig_size, orig_size, hidden_size)
#             .permute(2, 0, 1)
#             .unsqueeze(0)
#             .to(device=device, dtype=torch.float32)
#         )

#         # Calculate target dimensions for each patch
#         target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(
#             device=device, dtype=torch.float32
#         )
#         target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(
#             device=device, dtype=torch.float32
#         )

#         # Normalize coordinates to [-1, 1] range for grid_sample
#         norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
#         norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

#         # Create sampling grid
#         grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

#         # Perform bicubic interpolation
#         interpolated_embed_fp32 = F.grid_sample(
#             pos_embed_2d, grid, mode=self.interpolated_method, align_corners=False, padding_mode="border"
#         )

#         # Reshape and convert back to original dtype
#         adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
#         adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(embeddings.device)

#         # Add adapted position encoding to embeddings
#         embeddings = embeddings + adapted_pos_embed
#         return embeddings


# class GlmImageVisionBlock():
#     def __init__(self, config) -> None:
#         super().__init__()
#         self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         # self.attn = GlmImageVisionAttention(config)
#         self.attn = VisionAttention(config)
#         self.mlp = GlmImageVisionMLP(config)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#     ) -> torch.Tensor:
#         residual = hidden_states

#         hidden_states = self.norm1(hidden_states)
#         hidden_states = self.attn(
#             hidden_states,
#             cu_seqlens=cu_seqlens,
#         )
#         hidden_states = residual + hidden_states

#         residual = hidden_states
#         hidden_states = self.norm2(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         return hidden_states


# def rotate_half(x):
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)

#     # Keep half or full tensor for later concatenation
#     rotary_dim = cos.shape[-1]
#     q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
#     k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

#     # Apply rotary embeddings on the first half or full tensor
#     q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
#     k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

#     # Concatenate back to full shape
#     q_embed = torch.cat([q_embed, q_pass], dim=-1)
#     k_embed = torch.cat([k_embed, k_pass], dim=-1)
#     return q_embed, k_embed


# class GlmImageTextMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.config = config
#         self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
#         self.activation_fn = ACT2FN[config.hidden_act]

#     def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
#         up_states = self.gate_up_proj(hidden_states)

#         gate, up_states = up_states.chunk(2, dim=-1)
#         up_states = up_states * self.activation_fn(gate)

#         return self.down_proj(up_states)

# class GlmImageTextAttention(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         num_kv_heads: int,
#         head_dim: Optional[int] = None,
#         layer_id: int = 0,
#         rope_theta: float = 1000000,
#         rope_scaling: Optional[Dict[str, Any]] = None,
#         max_position_embeddings: int = 131072,
#         quant_config: Optional[QuantizationConfig] = None,
#         dual_chunk_attention_config: Optional[dict[str, Any]] = None,
#         partial_rotary_factor: float = 0.5,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         tp_size = get_tensor_model_parallel_world_size()
#         self.total_num_heads = num_heads
#         assert self.total_num_heads % tp_size == 0
#         self.num_heads = self.total_num_heads // tp_size
#         self.total_num_kv_heads = num_kv_heads
#         if self.total_num_kv_heads >= tp_size:
#             # Number of KV heads is greater than TP size, so we partition
#             # the KV heads across multiple tensor parallel GPUs.
#             assert self.total_num_kv_heads % tp_size == 0
#         else:
#             # Number of KV heads is less than TP size, so we replicate
#             # the KV heads across multiple tensor parallel GPUs.
#             assert tp_size % self.total_num_kv_heads == 0
#         self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
#         if head_dim is not None:
#             self.head_dim = head_dim
#         else:
#             self.head_dim = hidden_size // self.total_num_heads
#         self.q_size = self.num_heads * self.head_dim
#         self.kv_size = self.num_kv_heads * self.head_dim
#         self.scaling = self.head_dim**-0.5
#         self.rope_theta = rope_theta
#         self.max_position_embeddings = max_position_embeddings
#         self.partial_rotary_factor = partial_rotary_factor

#         self.qkv_proj = QKVParallelLinear(
#             hidden_size,
#             self.head_dim,
#             self.total_num_heads,
#             self.total_num_kv_heads,
#             bias=True,
#             quant_config=quant_config,
#             prefix=add_prefix("qkv_proj", prefix),
#         )
#         self.o_proj = RowParallelLinear(
#             self.total_num_heads * self.head_dim,
#             hidden_size,
#             bias=False,
#             quant_config=quant_config,
#             prefix=add_prefix("o_proj", prefix),
#         )

#         self.rotary_emb = get_rope(
#             self.head_dim,
#             rotary_dim=self.head_dim,
#             max_position=max_position_embeddings,
#             base=rope_theta,
#             rope_scaling=rope_scaling,
#             dual_chunk_attention_config=dual_chunk_attention_config,
#             partial_rotary_factor=partial_rotary_factor,
#             is_neox_style=False,
#         )
#         self.attn = RadixAttention(
#             self.num_heads,
#             self.head_dim,
#             self.scaling,
#             num_kv_heads=self.num_kv_heads,
#             layer_id=layer_id,
#             quant_config=quant_config,
#             prefix=add_prefix("attn", prefix),
#         )

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         forward_batch: ForwardBatch,
#     ) -> torch.Tensor:
#         qkv, _ = self.qkv_proj(hidden_states)
#         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
#         q, k = self.rotary_emb(positions, q, k)
#         attn_output = self.attn(q, k, v, forward_batch)
#         output, _ = self.o_proj(attn_output)
#         return output

# class GlmImageTextDecoderLayer(nn.Module):
#     def __init__(self, config, quant_config: Optional[QuantizationConfig] = layer_idx, layer_id: int = 0, prefix: str = ""):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.self_attn = GlmImageTextAttention(config, layer_idx)
#         self.mlp = GlmImageTextMLP(config)
#         self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_self_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         forward_batch: ForwardBatch,
#         input_embeds: torch.Tensor = None,
#     ) -> torch.Tensor:

#         if input_embeds is None:
#             hidden_states = self.embed_tokens(input_ids)
#         else:
#             hidden_states = input_embeds
#         residual = None

#         for layer in self.layers[self.start_layer : self.end_layer]:
#             hidden_states, residual = layer(
#                 positions,
#                 hidden_states,
#                 forward_batch,
#                 residual,
#             )

#         if hidden_states.shape[0] != 0:
#             if residual is None:
#                 hidden_states = self.norm(hidden_states)
#             else:
#                 hidden_states, _ = self.norm(hidden_states, residual)

#         return hidden_states


# class GlmImageVQVAEVectorQuantizer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_embeddings = config.num_embeddings
#         self.embedding_dim = config.embed_dim
#         self.beta = getattr(config, "beta", 0.25)

#         self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

#     def forward(self, hidden_state: torch.Tensor):
#         hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
#         hidden_state_flattened = hidden_state.view(-1, self.embedding_dim)

#         # L2 normalize
#         hidden_state = F.normalize(hidden_state, p=2, dim=-1)
#         hidden_state_flattened = F.normalize(hidden_state_flattened, p=2, dim=-1)
#         embedding = F.normalize(self.embedding.weight, p=2, dim=-1)

#         # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
#         distances = (
#             torch.sum(hidden_state_flattened**2, dim=1, keepdim=True)
#             + torch.sum(embedding**2, dim=1)
#             - 2 * torch.einsum("bd,dn->bn", hidden_state_flattened, embedding.transpose(0, 1))
#         )

#         min_encoding_indices = torch.argmin(distances, dim=1)
#         hidden_state_quant = embedding[min_encoding_indices].view(hidden_state.shape)

#         # compute loss for embedding
#         loss = torch.mean((hidden_state_quant.detach() - hidden_state) ** 2) + self.beta * torch.mean(
#             (hidden_state_quant - hidden_state.detach()) ** 2
#         )

#         # preserve gradients
#         hidden_state_quant = hidden_state + (hidden_state_quant - hidden_state).detach()

#         # reshape back to match original input shape
#         hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

#         return hidden_state_quant, loss, min_encoding_indices


# class GlmImageVQVAE(nn.Module):
#     def __init__(self, config):
#         super().__init__(config)
#         self.quantize = GlmImageVQVAEVectorQuantizer(config)
#         self.quant_conv = torch.nn.Conv2d(config.latent_channels, config.embed_dim, 1)
#         self.post_quant_conv = torch.nn.Conv2d(config.embed_dim, config.latent_channels, 1)
#         self.eval()  # GlmImage's VQ model is frozen
#         self.post_init()

#     def encode(self, hidden_states):
#         hidden_states = self.quant_conv(hidden_states)
#         quant, emb_loss, indices = self.quantize(hidden_states)
#         return quant, emb_loss, indices


# class GlmImageVisionModel(nn.Module):
#     def __init__(self, config) -> None:
#         super().__init__(config)
#         self.spatial_merge_size = config.spatial_merge_size
#         self.patch_size = config.patch_size

#         self.embeddings = GlmImageVisionEmbeddings(config)
#         self.patch_embed = GlmImageVisionPatchEmbed(config)

#         head_dim = config.hidden_size // config.num_heads

#         self.blocks = nn.ModuleList([GlmImageVisionBlock(config) for _ in range(config.depth)])

#         self.head_dim = head_dim

#     def rot_pos_emb(self, grid_thw):
#         pos_ids = []
#         for t, h, w in grid_thw:
#             hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
#             hpos_ids = hpos_ids.reshape(
#                 h // self.spatial_merge_size,
#                 self.spatial_merge_size,
#                 w // self.spatial_merge_size,
#                 self.spatial_merge_size,
#             )
#             hpos_ids = hpos_ids.permute(0, 2, 1, 3)
#             hpos_ids = hpos_ids.flatten()

#             wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
#             wpos_ids = wpos_ids.reshape(
#                 h // self.spatial_merge_size,
#                 self.spatial_merge_size,
#                 w // self.spatial_merge_size,
#                 self.spatial_merge_size,
#             )
#             wpos_ids = wpos_ids.permute(0, 2, 1, 3)
#             wpos_ids = wpos_ids.flatten()
#             pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
#         pos_ids = torch.cat(pos_ids, dim=0)
#         return pos_ids

#     def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.patch_embed(pixel_values)
#         image_type_ids = self.rot_pos_emb(grid_thw)

#         cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
#             dim=0,
#             dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
#         )
#         cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
#         seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
#         hidden_states = self.embeddings(
#             hidden_states,
#             seqlens,
#             grid_thw,
#             image_type_ids[:, 0].to(hidden_states.device),
#             image_type_ids[:, 1].to(hidden_states.device),
#         )

#         # Transformer blocks (no position_embeddings needed, already added above)
#         for blk in self.blocks:
#             hidden_states = blk(
#                 hidden_states,
#                 cu_seqlens=cu_seqlens,
#             )
#         return hidden_states

# class GlmImageTextModel(nn.Module):
#     def __init__(
#         self,
#         config: Glm4Config,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = VocabParallelEmbedding(
#             config.vocab_size,
#             config.hidden_size,
#             quant_config=quant_config,
#             enable_tp=not is_dp_attention_enabled(),
#             prefix=add_prefix("embed_tokens", prefix),
#         )
#         self.layers, self.start_layer, self.end_layer = make_layers(
#             config.num_hidden_layers,
#             lambda idx, prefix: GlmImageTextDecoderLayer(
#                 layer_id=idx,
#                 config=config,
#                 quant_config=quant_config,
#                 prefix=prefix,
#             ),
#             prefix=add_prefix("layers", prefix),
#         )
#         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = get_rope(
#             self.head_dim,
#             rotary_dim=self.head_dim,
#             max_position=config.max_position_embeddings,
#             base=config.rope_theta, # maybe wrong
#             rope_scaling=config.rope_scaling,
#             partial_rotary_factor=config.partial_rotary_factor,
#             is_neox_style=False,
#         )


#     def forward(
#         self,
#         input_ids: torch.LongTensor | None = None,
#         attention_mask: torch.Tensor | None = None,
#         position_ids: torch.LongTensor | None = None,
#         past_key_values: Cache | None = None,
#         inputs_embeds: torch.FloatTensor | None = None,
#         use_cache: bool | None = None,
#         cache_position: torch.LongTensor | None = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> tuple | BaseModelOutputWithPast:

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         # the hard coded `3` is for temporal, height and width.
#         if position_ids is None:
#             position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
#         elif position_ids.ndim == 2:
#             position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)


#         if position_ids.ndim == 3 and position_ids.shape[0] == 4:
#             text_position_ids = position_ids[0]
#             position_ids = position_ids[1:]
#         else:
#             # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
#             text_position_ids = None

#         hidden_states = inputs_embeds
#         position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

#         for decoder_layer in self.layers:
#             layer_outputs = decoder_layer(
#                 hidden_states,
#                 attention_mask=causal_mask,
#                 position_ids=text_position_ids,
#                 past_key_values=past_key_values,
#                 cache_position=cache_position,
#                 position_embeddings=position_embeddings,
#                 **kwargs,
#             )
#             hidden_states = layer_outputs

#         hidden_states = self.norm(hidden_states)

#         return hidden_states


# class GlmImageModel():

#     def __init__(self, config):
#         super().__init__(config)
#         self.visual = GlmImageVisionModel()
#         self.language_model = GlmImageTextModel._from_config(config.text_config)
#         self.vqmodel = GlmImageVQVAE._from_config(config.vq_config)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         forward_batch: ForwardBatch,
#         input_embeds: torch.Tensor = None,
#     ) -> torch.Tensor:
#         if input_embeds is None:
#             hidden_states = self.embed_tokens(input_ids)
#         else:
#             hidden_states = input_embeds
#         residual = None

#         for i in range(self.start_layer, self.end_layer):
#             layer = self.layers[i]
#             hidden_states, residual = layer(
#                 positions,
#                 hidden_states,
#                 forward_batch,
#                 residual,
#             )

#         if hidden_states.shape[0] != 0:
#             if residual is None:
#                 hidden_states = self.norm(hidden_states)
#             else:
#                 hidden_states, _ = self.norm(hidden_states, residual)

#         return hidden_states


#     def get_image_tokens(
#         self,
#         hidden_states: torch.FloatTensor,
#         image_grid_thw: torch.LongTensor,
#     ) -> torch.LongTensor:
#         hidden_size = hidden_states.shape[-1]
#         split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
#         hidden_states_list = torch.split(hidden_states, split_sizes, dim=0)

#         all_image_toks = []
#         for i, hs in enumerate(hidden_states_list):
#             grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
#             hs = hs.view(grid_t, grid_h, grid_w, hidden_size)
#             hs = hs.permute(0, 3, 1, 2).contiguous()
#             _, _, image_toks = self.vqmodel.encode(hs)
#             all_image_toks.append(image_toks)
#         return torch.cat(all_image_toks, dim=0)


# class GlmImageForConditionalGeneration(nn.Module):
#     def __init__(
#         self,
#         config,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()

#         self.model = GlmImageModel(config)
#         self.lm_head = ParallelLMHead(config.text_config.vocab_size, config.text_config.hidden_size, quant_config=quant_config, prefix=add_prefix("lm_head", prefix))

#     def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
#         pixel_values = torch.cat([item.feature for item in items], dim=0).type(
#             self.visual.dtype
#         )
#         image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
#         assert pixel_values.dim() == 2, pixel_values.dim()
#         assert image_grid_thw.dim() == 2, image_grid_thw.dim()
#         image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#         return image_embeds

#     def get_image_tokens(self, hidden_states: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None):
#         return self.model.get_image_tokens(hidden_states, image_grid_thw)

#     def forward(
#         self,
#         input_ids: torch.LongTensor | None = None,
#         attention_mask: torch.Tensor | None = None,
#         position_ids: torch.LongTensor | None = None,
#         past_key_values: Cache | None = None,
#         inputs_embeds: torch.FloatTensor | None = None,
#         labels: torch.LongTensor | None = None,
#         pixel_values: torch.Tensor | None = None,
#         image_grid_thw: torch.LongTensor | None = None,
#         cache_position: torch.LongTensor | None = None,
#         logits_to_keep: int | torch.Tensor = 0,
#     ):
#         outputs = self.model(
#             input_ids=input_ids,
#             pixel_values=pixel_values,
#             image_grid_thw=image_grid_thw,
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             cache_position=cache_position,
#         )

#         hidden_states = outputs[0]

#         # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(hidden_states[:, slice_indices, :])

#         return GlmImageCausalLMOutputWithPast(
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             rope_deltas=outputs.rope_deltas,
#         )

#     @torch.no_grad()
#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         forward_batch: ForwardBatch,
#         input_embeds=None,
#     ):
#         if self.is_mrope_enabled:
#             positions = forward_batch.mrope_positions

#         if not (
#             forward_batch.forward_mode.is_decode()
#             or not forward_batch.contains_image_inputs()
#         ):
#             if self.is_mrope_enabled:
#                 assert positions.ndim == 2 and positions.size(0) == 3, (
#                     "multimodal section rotary embedding requires "
#                     f"(3, seq_len) positions, but got {positions.size()}"
#                 )

#         hidden_states = general_mm_embed_routine(
#             input_ids=input_ids,
#             forward_batch=forward_batch,
#             language_model=self.model,
#             multimodal_model=self,
#             positions=positions,
#         )

#         return self.logits_processor(
#             input_ids,
#             hidden_states,
#             self.lm_head,
#             forward_batch,
#         )


class GlmImageForConditionalGeneration:
    def __init__(self, config):
        self.config = config
        print(f"802 {self.config=}")

    pass


Entryclass = GlmImageForConditionalGeneration
