from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask
from torch.utils import checkpoint


class CrossAttentionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.0,
            batch_first=True,
    ):
        nn.Module.__init__(self)

        # cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # normalization
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ):
        """
        tgt: [B, T, C]
        memory: [B, S, C]
        """

        # Cross Attention
        tgt2, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Feedforward
        tgt2 = self.linear2(
            self.dropout_ff(self.activation(self.linear1(tgt)))
        )

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class CrossAttentionDecoder(nn.TransformerDecoder):
    def __init__(self, use_checkpoint=False, **kwargs):
        super().__init__(**kwargs)
        self.use_checkpoint = use_checkpoint

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: Optional[bool] = None,
            memory_is_causal: bool = False,
    ) -> Tensor:
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    mod,
                    output,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    tgt_is_causal,
                    memory_is_causal,
                )
            else:
                output = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_is_causal=tgt_is_causal,
                    memory_is_causal=memory_is_causal,
                )

        if self.norm is not None:
            output = self.norm(output)

        return output
