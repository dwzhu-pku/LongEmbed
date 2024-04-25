import random
import math
import torch
import numpy as np
import torch.nn as nn

from typing import Optional, Tuple, List, Union
from torch.nn import DataParallel

from utils import logger
from modify_utils import modify_method_of_instance

def get_rope_scaling_config(rope_scale_factor: float) -> Optional[dict]:
    if rope_scale_factor <= 1.0:
        return None
    else:
        return {
            'type': 'linear',
            'factor': rope_scale_factor,
        }

def replace_with_xformers():
    import xformers.ops as xops

    from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
    from transformers.models.bert.modeling_bert import BertSelfAttention

    def custom_llama_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = None
        attn_output = xops.memory_efficient_attention(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
            attn_bias=xops.LowerTriangularMask()
        ).reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    logger.info("Replacing llama attention with xformers attention")
    LlamaAttention.forward = custom_llama_forward

    from transformers.models.mistral.modeling_mistral import MistralAttention, repeat_kv, apply_rotary_pos_emb

    def custom_mistral_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = None

        attn_factor = math.log(kv_seq_len, 4096)
        attn_factor = max(1.0, attn_factor)  # Ensure a minimum value of 1
        query_states *= attn_factor

        # attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
        # attn_output = xops.memory_efficient_attention(
        #     query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
        #     attn_bias=attention_mask
        # ).reshape(bsz, q_len, self.hidden_size)

        attn_output = xops.memory_efficient_attention(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
            attn_bias=xops.LowerTriangularMask()
        ).reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    MistralAttention.forward = custom_mistral_forward
    logger.info('Replacing mistral attention with xformers attention')

    def custom_bert_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer) # bsz, n_heads, seq_len, head_dim
        bsz, n_heads, seq_len, head_dim = query_layer.shape

        # get each seq len
        tmp_attention_mask = attention_mask.squeeze()
        if tmp_attention_mask.dim() == 1:
            tmp_attention_mask = tmp_attention_mask.unsqueeze(0)
        each_seq_len = torch.sum(tmp_attention_mask == 0, dim=-1)
        original_len = torch.tensor(512)

        attn_factors = torch.log(each_seq_len) / torch.log(original_len)
        attn_factors = torch.clamp(attn_factors, min=1.0)  # Ensure a minimum value of 1
        attn_factors = attn_factors.view(-1, 1, 1, 1)
        query_layer *= attn_factors

        attention_mask = attention_mask.expand(bsz, n_heads, seq_len, seq_len).to(dtype=query_layer.dtype)
        attn_output = xops.memory_efficient_attention(
            query_layer.transpose(1, 2), key_layer.transpose(1, 2), value_layer.transpose(1, 2),
            attn_bias=attention_mask, p=(self.dropout.p if self.training else 0)
        ).reshape(bsz, seq_len, n_heads * head_dim)

        if output_attentions is True:
            raise NotImplementedError('output_attentions is not supported for xformers attention')

        return (attn_output,)

    BertSelfAttention.forward = custom_bert_attn_forward
    logger.info('Replacing bert attention with xformers attention')


def use_self_extend(args, loaded_model):
    import self_extend_patch as SE
    from functools import partial
    mistral_self_extend_forward = partial(SE.mistral_flash_self_extend_forward, group_size_1=args.group_size_1, group_size_2=args.group_size_2, scale_base=4096)
    e5rope_self_extend_forward = partial(SE.e5rope_self_extend_forward, group_size_1=args.group_size_1, group_size_2=args.group_size_2)
    modify_method_of_instance(loaded_model, "MistralFlashAttention2", "forward", mistral_self_extend_forward)
    modify_method_of_instance(loaded_model, "MistralFlashAttention2", "_flash_attention_forward", SE.flash_attention2_forward_with_window_size)

    modify_method_of_instance(loaded_model, "E5RopeSelfAttention", "forward", e5rope_self_extend_forward)
    logger.info('Patching self-extend for mistral and e5rope')


class ReplicateOnceDataParallel(DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)
        assert len(self.device_ids) > 1
        self.replicas = self.replicate(self.module, self.device_ids)

    @torch.no_grad()
    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        outputs = self.parallel_apply(self.replicas[:len(inputs)], inputs, kwargs)
        return self.gather(outputs, self.output_device)


def get_torch_dtype(args) -> torch.dtype:
    if args.fp16:
        return torch.float16
    elif args.bf16:
        return torch.bfloat16
    else:
        return torch.float32