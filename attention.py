"""
Based on https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/attention.py
"""

from typing import Optional
from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
        layer_idx: int | None = None,
        streaming: bool = False,
        cache_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.streaming = streaming
        
        self.cache_len = cache_len
        self.batch_size = batch_size
        self.height = height
        self.width = width
        
        if self.streaming:
            if self.cache_len is None:
                raise ValueError(
                    "cache_len must be provided when streaming is True."
                )

            self.cache_batch_dim = self.batch_size * self.height * self.width

            self.register_buffer(
                "cache_k",
                torch.zeros(
                    self.cache_batch_dim,
                    self.heads,
                    self.cache_len,
                    self.head_dim,
                ),
                persistent=False
            )

            self.register_buffer(
                "cache_v",
                torch.zeros(
                    self.cache_batch_dim,
                    self.heads,
                    self.cache_len,
                    self.head_dim,
                ),
                persistent=False,
            )

            self.current_cache_idx: int = 0
            self.total_tokens_processed: int = 0

    def reset_cache(self):
        """Resets the cache and counters for a new streaming sequence."""
        if self.streaming:
            self.current_cache_idx = 0
            self.total_tokens_processed = 0
            self.cache_k.zero_()
            self.cache_v.zero_()

    def forward(self, x: torch.Tensor, last_frame: bool =False):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        effective_bsz = B * H * W
        _time_seq_dim_for_rope = 2
        
        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        
        if self.streaming:
            assert effective_bsz <= self.cache_batch_dim, (
                f"Current effective batch size {effective_bsz} (B*H*W) exceeds "
                f"pre-allocated max_cache_batch_dim {self.cache_batch_dim}. Call reset_cache() if dimensions changed "
                f"or re-initialize with larger max values."
            )
            
            q = self.rotary_emb.rotate_queries_or_keys(
                q,
                freqs=self.rotary_emb.freqs,
                seq_dim=_time_seq_dim_for_rope,
                offset=self.total_tokens_processed,
            )
            
            k = self.rotary_emb.rotate_queries_or_keys(
                k,
                freqs=self.rotary_emb.freqs,
                seq_dim=_time_seq_dim_for_rope,
                offset=self.total_tokens_processed,
            )
            
            self.cache_k[
                :effective_bsz,
                :,
                self.current_cache_idx : self.current_cache_idx + 1,
                :,
            ] = k.detach()
            
            self.cache_v[
                :effective_bsz,
                :,
                self.current_cache_idx : self.current_cache_idx + 1,
                :,
            ] = v.detach()
            
            num_valid_in_cache = min(self.total_tokens_processed, self.cache_len)

            k = self.cache_k[:, :, :num_valid_in_cache + 1, :]
            v = self.cache_v[:, :, :num_valid_in_cache + 1, :]
            
            if last_frame:
                self.current_cache_idx = (self.current_cache_idx + 1) % self.cache_len
                self.total_tokens_processed += 1
        else:
            q = self.rotary_emb.rotate_queries_or_keys(q, freqs=self.rotary_emb.freqs,
                                                       seq_dim=_time_seq_dim_for_rope, offset=0)
            k = self.rotary_emb.rotate_queries_or_keys(k, freqs=self.rotary_emb.freqs,
                                                       seq_dim=_time_seq_dim_for_rope, offset=0)
            
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        x = F.scaled_dot_product_attention(
            query=q, key=k, value=v, is_causal=self.is_causal and not self.streaming
        )
        
        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)
        # linear proj
        x = self.to_out(x)
        return x


class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        freqs = self.rotary_emb.get_axial_freqs(H, W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # prepare for attn
        q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x
