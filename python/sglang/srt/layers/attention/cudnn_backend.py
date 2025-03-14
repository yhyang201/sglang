from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import cudnn

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend


if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

# Helper functions
def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")

def compute_default_stride(shape):
    ndim = len(shape)
    stride = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return stride

class CuDNNAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.cudnn_dtype = convert_to_cudnn_type(model_runner.model_config.dtype)
        self.device = model_runner.device

        self.max_context_len = model_runner.model_config.context_len

        self.extending_cudnn_graph = None
        self.extending_cudnn_graph_param = None
        self.extending_key = None
        self.extending_workspace = None

        self.extending_batch_size = 1
        # Currently not in use, actual batch_size = 1
        # A larger value may cause out of memory issues
        self.extending_seq_len = model_runner.server_args.chunked_prefill_size
        # This value determines the S dimension of query when building the cuDNN graph
        # We observed that there can be some performance degradation if the actual query length
        # is less than this value. However, to avoid rebuilding the graph multiple times,
        # we choose the maximum possible value here

        # This is also why we build separate graphs for extend and decode operations:
        # If decoding_batch were to use the extending_graph, there would be cases where
        # the actual query length (1) is much smaller than self.extending_seq_len,
        # which would lead to performance degradation

        self.decoding_batch_size = 32
        # Currently not in use, actual batch_size = 1
        self.decoding_cudnn_graph = None
        self.decoding_cudnn_graph_param = None
        self.decoding_key = None
        self.decoding_workspace = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def create_cudnn_graph(
        self, 
        query: torch.Tensor, 
        container_k_gpu: torch.Tensor, 
        container_v_gpu: torch.Tensor, 
        is_causal: bool,
        Sq: int,
        Skv: int,
    ):

        _, Hq, _, Dq = query.shape
        nbkv, Hk, _, Dk = container_k_gpu.shape
        _, Hv, _, Dv = container_v_gpu.shape
        
        graph = cudnn.pygraph(
            io_data_type=self.cudnn_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
            
        q = graph.tensor(dim=(1, Hq, Sq, Dq), stride = compute_default_stride((1, Hq, Sq, Dq)), data_type=self.cudnn_dtype)
        container_k = graph.tensor(dim=(nbkv, Hk, 1, Dk), stride=compute_default_stride((nbkv, Hk, 1, Dk)), data_type=self.cudnn_dtype)
        container_v = graph.tensor(dim=(nbkv, Hv, 1, Dv), stride=compute_default_stride((nbkv, Hv, 1, Dv)), data_type=self.cudnn_dtype)
        page_table_k = graph.tensor(dim=(1, 1, Skv, 1), stride=compute_default_stride((1, 1, Skv, 1)), data_type=cudnn.data_type.INT32)
        page_table_v = graph.tensor(dim=(1, 1, Skv, 1), stride=compute_default_stride((1, 1, Skv, 1)), data_type=cudnn.data_type.INT32)
        attn_scale = graph.tensor(dim=(1, 1, 1, 1), stride=compute_default_stride((1, 1, 1, 1)), data_type=self.cudnn_dtype)
        seq_len_q = graph.tensor(dim=(1, 1, 1, 1), stride=compute_default_stride((1, 1, 1, 1)), data_type=cudnn.data_type.INT32)
        seq_len_kv = graph.tensor(dim=(1, 1, 1, 1), stride=compute_default_stride((1, 1, 1, 1)), data_type=cudnn.data_type.INT32)

        o, _ = graph.sdpa(
            name="sdpa_decode" if Sq == 1 else "sdpa_extend",
            q=q,
            k=container_k,  
            v=container_v,  
            is_inference=True,
            use_padding_mask=True, 
            attn_scale=attn_scale,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            diagonal_band_right_bound=0 if is_causal else None,
            diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT if is_causal else cudnn.diagonal_alignment.TOP_LEFT,
            paged_attention_k_table=page_table_k,  
            paged_attention_v_table=page_table_v,  
        )

        o.set_output(True).set_dim((1, Hq, Sq, Dq)).set_stride(compute_default_stride((1, Hq, Sq, Dq)))

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        graph_param = (
            q,
            container_k,
            container_v,
            page_table_k,
            page_table_v,
            attn_scale,
            seq_len_q,
            seq_len_kv,
            o,
        )
        workspace = torch.empty(graph.get_workspace_size(), device=self.device, dtype=torch.uint8)

        return graph, graph_param, workspace

    def _run_cudnn_sdpa_extend(
        self,
        query: torch.Tensor,
        container_k_gpu: torch.Tensor,
        container_v_gpu: torch.Tensor,
        output: torch.Tensor,
        page_table_k_gpu: torch.Tensor,
        page_table_v_gpu: torch.Tensor,
        attn_scale: float,
        seq_len_q_gpu: torch.Tensor,
        seq_len_kv_gpu: torch.Tensor,
        is_causal: bool,
    ):
        # cuDNN only support int32
        seq_len_q_gpu = seq_len_q_gpu.to(torch.int32)   
        seq_len_kv_gpu = seq_len_kv_gpu.to(torch.int32)

        attn_scale_gpu = torch.tensor(attn_scale, device=self.device, dtype=query.dtype).reshape(1, 1, 1, 1)

        _, Hq, Sq, Dq = query.shape
        nbkv, Hk, _, Dk = container_k_gpu.shape
        _, Hv, _, Dv = container_v_gpu.shape

        assert query.shape[2] <= Sq
        
        Sq = self.extending_seq_len
        Skv = nbkv

        curr_key = (Hq, Dq, nbkv, Hk, Dk, Hv, Dv, is_causal)
    
        # Rebuild the graph if the current key differs from the previously built graph's key
        # Not certain if this case actually occurs, but this approach ensures maximum correctness
        if self.extending_cudnn_graph is None or curr_key != self.extending_key: 
            
            self.extending_key = curr_key
            self.extending_cudnn_graph, self.extending_cudnn_graph_param, self.extending_workspace = self.create_cudnn_graph(
                query, 
                container_k_gpu, 
                container_v_gpu, 
                is_causal,
                Sq,
                Skv,
            )

        query_stride = torch.empty((1, Hq, Sq, Dq), device=self.device, dtype=query.dtype)
        query_stride[:, :, :query.shape[2], :] = query
        page_table_k_stride = torch.empty((1, 1, Skv, 1), device=self.device, dtype=torch.int32)
        page_table_v_stride = torch.empty((1, 1, Skv, 1), device=self.device, dtype=torch.int32)
        page_table_k_stride[:, :, :page_table_k_gpu.shape[2], :] = page_table_k_gpu
        page_table_v_stride[:, :, :page_table_v_gpu.shape[2], :] = page_table_v_gpu
        output_stride = torch.empty((1, Hq, Sq, Dq), device=self.device, dtype=output.dtype)

        q, container_k, container_v, page_table_k, page_table_v, attn_scale, seq_len_q, seq_len_kv, o = self.extending_cudnn_graph_param
        variant_pack = {
            q: query_stride,
            container_k: container_k_gpu,
            container_v: container_v_gpu,
            page_table_k: page_table_k_stride,
            page_table_v: page_table_v_stride,
            attn_scale: attn_scale_gpu,
            seq_len_q: seq_len_q_gpu,
            seq_len_kv: seq_len_kv_gpu,
            o: output_stride,
        }

        self.extending_cudnn_graph.execute(variant_pack, self.extending_workspace)
        torch.cuda.synchronize()

        output[:] = output_stride[:, :, :output.shape[2], :]

        return output
    
    def _run_cudnn_sdpa_decode(
        self,
        query: torch.Tensor,
        container_k_gpu: torch.Tensor,
        container_v_gpu: torch.Tensor,
        output: torch.Tensor,
        page_table_k_gpu: torch.Tensor,
        page_table_v_gpu: torch.Tensor,
        attn_scale: float,
        seq_len_q_gpu: torch.Tensor,
        seq_len_kv_gpu: torch.Tensor,
        is_causal: bool,
    ):
        # cuDNN only support int32
        seq_len_q_gpu = seq_len_q_gpu.to(torch.int32)   
        seq_len_kv_gpu = seq_len_kv_gpu.to(torch.int32)

        attn_scale_gpu = torch.tensor(attn_scale, device=self.device, dtype=query.dtype).reshape(1, 1, 1, 1)

        _, Hq, Sq, Dq = query.shape
        nbkv, Hk, _, Dk = container_k_gpu.shape
        _, Hv, _, Dv = container_v_gpu.shape

        Sq = 1
        Skv  = nbkv

        curr_key = (Hq, Dq, nbkv, Hk, Dk, Hv, Dv, is_causal)

        if self.decoding_cudnn_graph is None or curr_key != self.decoding_key: 
            
            self.decoding_key = curr_key
            self.decoding_cudnn_graph, self.decoding_cudnn_graph_param, self.decoding_workspace = self.create_cudnn_graph(
                query, 
                container_k_gpu, 
                container_v_gpu, 
                is_causal,
                Sq,
                Skv,
            )

        
        page_table_k_stride = torch.empty((1, 1, Skv, 1), device=self.device, dtype=torch.int32)
        page_table_v_stride = torch.empty((1, 1, Skv, 1), device=self.device, dtype=torch.int32)
        page_table_k_stride[:, :, :page_table_k_gpu.shape[2], :] = page_table_k_gpu
        page_table_v_stride[:, :, :page_table_v_gpu.shape[2], :] = page_table_v_gpu
        output_stride = torch.empty((1, Hq, Sq, Dq), device=self.device, dtype=output.dtype)


        q, container_k, container_v, page_table_k, page_table_v, attn_scale, seq_len_q, seq_len_kv, o = self.decoding_cudnn_graph_param
        variant_pack = {
            q: query,
            container_k: container_k_gpu,
            container_v: container_v_gpu,
            page_table_k: page_table_k_gpu,
            page_table_v: page_table_v_gpu,
            attn_scale: attn_scale_gpu,
            seq_len_q: seq_len_q_gpu,
            seq_len_kv: seq_len_kv_gpu,
            o: output_stride,
        }

        self.decoding_cudnn_graph.execute(variant_pack, self.decoding_workspace)
        torch.cuda.synchronize()

        output[:] = output_stride[:, :, :output.shape[2], :]

        return output


    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        causal=False,
    ):

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]
        
        query = query.movedim(0, query.dim() - 2)
     
        start_q, start_kv = 0, 0
        container_k_gpu = k_cache.view(k_cache.shape[0], k_cache.shape[1], 1, k_cache.shape[2])
        container_v_gpu = v_cache.view(v_cache.shape[0], v_cache.shape[1], 1, v_cache.shape[2])
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]
            seq_len_kv = seq_lens[seq_idx]

            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            assert prefill_seq_len_q + extend_seq_len_q == seq_len_kv

            per_req_query = query[:, start_q:end_q, :].unsqueeze(0)
            
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]

            # index for each token in the sequence.
            page_table_k_gpu = per_req_tokens.reshape(1, 1, seq_len_kv, 1)
            page_table_v_gpu = per_req_tokens.reshape(1, 1, seq_len_kv, 1)
            seq_len_q_gpu = extend_seq_len_q.reshape(1, 1, 1, 1)
            seq_len_kv_gpu = seq_len_kv.reshape(1, 1, 1, 1)
            per_req_out = torch.empty_like(per_req_query)
            
            per_req_out = self._run_cudnn_sdpa_extend(
                per_req_query,
                container_k_gpu,
                container_v_gpu,
                per_req_out,
                page_table_k_gpu,
                page_table_v_gpu, 
                scaling,
                seq_len_q_gpu,
                seq_len_kv_gpu,
                causal,
            ).squeeze(0).movedim(query.dim() - 2, 0)

            output[start_q:end_q, : , :] = per_req_out
            start_q, start_kv = end_q, end_kv
        
        return output

    
    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        causal=False,
    ):
        
        query = query.movedim(0, query.dim() - 2)
        
        start_q, start_kv = 0, 0
        container_k_gpu = k_cache.view(k_cache.shape[0], k_cache.shape[1], 1, k_cache.shape[2])
        container_v_gpu = v_cache.view(v_cache.shape[0], v_cache.shape[1], 1, v_cache.shape[2])

        for seq_idx in range(seq_lens.shape[0]):
            
            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :].unsqueeze(0)
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]

            page_table_k_gpu = per_req_tokens.reshape(1, 1, seq_len_kv, 1)
            page_table_v_gpu = per_req_tokens.reshape(1, 1, seq_len_kv, 1)
            seq_len_q_gpu = torch.tensor(1, device=self.device).reshape(1, 1, 1, 1)
            seq_len_kv_gpu = seq_len_kv.reshape(1, 1, 1, 1)
            per_req_out = torch.empty_like(per_req_query)
            
            per_req_out = self._run_cudnn_sdpa_decode(
                per_req_query,
                container_k_gpu,
                container_v_gpu,
                per_req_out,
                page_table_k_gpu, 
                page_table_v_gpu, 
                scaling,
                seq_len_q_gpu,
                seq_len_kv_gpu,
                causal,
            ).squeeze(0).movedim(query.dim() - 2, 0)

            output[start_q:end_q, : , :] = per_req_out
            start_q, start_kv = end_q, end_kv
        
        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        
        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            causal=not layer.is_cross_attention,
        )

        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
 
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
  
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            causal=True,
        )

        return o