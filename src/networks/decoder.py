# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import IntEnum
from typing import OrderedDict

import torch
import torch.nn as nn
from einops import repeat


class HELPER_TOKEN(IntEnum):
    PAD = 0
    START = 1
    PART = 2
    STOP = 3
    NOT_USED = 4
    NOT_USED_1 = 5
    NUM = 6


def make_autoregressive_mask(size, device=None):
    # Generates an upper-triangular matrix of -inf, with zeros on diag.
    # Example size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]
    return torch.triu(torch.ones(size, size, device=device) * float("-inf"), diagonal=1)


class SceneScriptDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_attn_heads,
        dim_feedforward,
        num_bins,
        max_num_tokens,
        max_num_type_tokens,
        num_decoder_layers,
    ):
        """
        Args:
            d_model: int. Dimension of model.
            num_attn_heads: int. Number of attention heads.
            dim_feedforward: int. Dimension of feedforward network.
            num_bins: int. Number of discretized bins.
            max_num_tokens: int. Maximum number of tokens.
            max_num_type_tokens: int. Maximum number of type tokens.
            num_decoder_layers: int. Number of decoder layers.
        """
        super().__init__()
        self.d_model = d_model
        self.max_num_tokens = max_num_tokens

        # Embeddings
        self.position_embedding = nn.Embedding(max_num_tokens, d_model)
        self.value_embedding = nn.Embedding(num_bins + HELPER_TOKEN.NUM, d_model)
        self.type_embedding = nn.Embedding(max_num_type_tokens, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            num_attn_heads,
            dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, nn.LayerNorm(d_model)
        )

        # Decoding to bins

        self.tail = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(d_model, 2 * d_model)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(2 * d_model, d_model)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(d_model, num_bins + HELPER_TOKEN.NUM)),
        ]))

        self.register_buffer("causal_mask", make_autoregressive_mask(max_num_tokens))

    def embed_position(self, seq_value):
        """Apply positional embedding.

        Args:
            seq_value: [B, T] torch.LongTensor. In range [0, num_bins + HELPER_TOKEN.NUM).

        Returns:
            pos_emb: [B, T, d_model] torch.FloatTensor.
        """
        B, T = seq_value.shape
        device = seq_value.device

        # Target embedding
        t = torch.arange(T, device=device)
        pos_emb = repeat(self.position_embedding(t), "t d -> b t d", b=B)
        # print(pos_emb)
        return pos_emb
    
    def prepare_inputs_for_generation(self, **inputs):
        # print(inputs)
        pass 

    def forward(self, context, context_mask, seq_value, seq_type):
        """
        Args:
            context: [B, context_length, d_model] torch.FloatTensor.
            context_mask: [B, context_length] torch.BoolTensor. True means ignore.
            seq_value: [B, T] torch.LongTensor. In range [0, num_bins + HELPER_TOKEN.NUM).
            seq_type: [B, T] torch.LongTensor. In range [0, max_num_type_tokens)

        Returns:
            [B, T, num_bins + HELPER_TOKEN.NUM] torch.FloatTensor.
        """
        B, T = seq_value.shape[:2]

        decoder_input = (
            self.embed_position(seq_value)
            + self.value_embedding(seq_value)
            + self.type_embedding(seq_type)
        )

        # Get causal_mask
        assert T <= self.max_num_tokens
        # print(self.causal_mask)
        causal_mask = repeat(self.causal_mask[:T, :T], "T Y -> B T Y", B=B)

        # transformer
        decoder_out = self.transformer_decoder(
            tgt=decoder_input,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
            memory=context,
            memory_mask=None,
            memory_key_padding_mask=context_mask,
        )  # [B, T, d_model]
        logits = self.tail(decoder_out)  # [B, T, num_bins + HELPER_TOKEN.NUM]

        labels = seq_value  # [B, T]

        log_probs = -nn.functional.log_softmax(logits, dim=-1)  # [B, T, num_bins + HELPER_TOKEN.NUM]
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)  # [B, T, 1]

        padding_mask = labels.eq(HELPER_TOKEN.PAD)  # [B, T, 1]
        labels = torch.clamp(labels, min=0)  # [B, T, 1]
        nll_loss = log_probs.gather(dim=-1, index=labels)  # [B, T, 1]
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)  # [B, T, 1]

        nll_loss.masked_fill_(padding_mask, 0.0)  # [B, T, 1]
        smoothed_loss.masked_fill_(padding_mask, 0.0)  # [B, T, 1]

        num_active_elements = padding_mask.numel() - padding_mask.long().sum()  # scalar
        nll_loss = nll_loss.sum() / num_active_elements  # scalar
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])  # scalar
        loss = (1 - 0.1) * nll_loss + 0.1 * smoothed_loss  # scalar

        return {"logits": logits, "loss": loss}
