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
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func   # main kernel


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
        self.new_frame = True
        
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
        
        self.new_frame = last_frame
        
        return x


# class SpatialAxialAttention(nn.Module):
#     """
#     Axial-spatial self-attention with FlashAttention-2 fallback.
#     The tensor shapes now match the expected (batch, heads, seq, d)
#     convention, so you get a proper (seq × seq) attention matrix.
#     """
#     def __init__(self, dim: int, heads: int, dim_head: int,
#                  rotary_emb: RotaryEmbedding, layer_idx: int | None = None):
#         super().__init__()
#         self.layer_idx  = layer_idx
#         self.heads      = heads
#         self.head_dim   = dim_head
#         self.inner_dim  = heads * dim_head

#         self.to_qkv = nn.Linear(dim, 3 * self.inner_dim, bias=False)
#         self.to_out = nn.Linear(self.inner_dim, dim, bias=True)

#         self.rotary_emb = rotary_emb
#         self.first_frame = True

#     def forward(self, x: torch.Tensor, last_frame=False) -> torch.Tensor:
#         """
#         x: [B, T, H, W, D]  →  returns same shape
#         """
#         B, T, H, W, _ = x.shape

#         # ---------- 1. project to q/k/v ----------
#         q, k, v = self.to_qkv(x).chunk(3, dim=-1)                # [B,T,H,W,heads*dim]
#         q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
#         k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
#         v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

#         # ---------- 2. rotary on 2-D grid ----------
#         freqs = self.rotary_emb.get_axial_freqs(H, W)            # [H,W,d]
#         q = apply_rotary_emb(freqs, q)
#         k = apply_rotary_emb(freqs, k)

#         # ---------- 3. flatten spatial grid ----------
#         q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T)
#         k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T)
#         v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T)
#         seq_len = H * W                                           # S = H·W

#         # ---------- 4. attention ----------
#         # Uncomment ONE of the two blocks below
#         # -- A. FlashAttention-2 (fast, memory-efficient) --------
#         # q_f = q.permute(0, 2, 1, 3).contiguous()                # (B*,S,h,d)
#         # k_f = k.permute(0, 2, 1, 3).contiguous()
#         # v_f = v.permute(0, 2, 1, 3).contiguous()
#         # out = flash_attn_qkvpacked_func(
#         #     torch.stack((q_f, k_f, v_f), dim=2), causal=False)   # (B*,S,h,d)
#         # out = out.permute(0, 2, 1, 3)                           # back to (B*,h,S,d)

#         # -- B. Plain soft-max attention (easier to debug) -------
#         scale = self.head_dim ** -0.5
#         attn   = torch.matmul(q, k.transpose(-2, -1)) * scale     # (B*,h,S,S)
#         attn   = attn.softmax(dim=-1)
        
#         # --------------------------------------------------------
        
#         if self.first_frame:
#             self.top_k_indices = # computation
#             out    = attn @ v                                         # (B*,h,S,d)
#             attn_top_k = attn[:, :, self.top_k_indices, :]
#             out_top_k = attn_top_k @ v
#             self.out_not_top_k = out - out_top_k
        
#         else:
#             out = out_top_k + # compute attn_top_k in an efficient way

#         # ---------- 5. restore original layout ----------
#         out = rearrange(out, "(B T) h (H W) d -> B T H W (h d)",
#                         B=B, T=T, H=H, W=W)

#         self.first_frame = last_frame
        
#         return self.to_out(out.to(x.dtype))

class SpatialAxialAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int,
                 rotary_emb: RotaryEmbedding, layer_idx: int | None = None, top_k_keys: int = 144):
        super().__init__()
        self.layer_idx     = layer_idx
        self.heads         = heads
        self.head_dim      = dim_head
        self.inner_dim     = heads * dim_head
        self.top_k_keys    = top_k_keys

        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=True)
        self.rotary_emb = rotary_emb
        
        self.first_frame = True

        self.register_buffer('top_k_indices', None, persistent=False)
        self.register_buffer('out_not_top_k', None, persistent=False)

    def _compute_out_top_k(self, q_flat, k_flat, v_flat, indices_flat, scale):
        B_h_flat, S, d_head = q_flat.shape
        
        batch_indices = torch.arange(B_h_flat, device=q_flat.device)[:, None, None]
        k_gathered = k_flat[batch_indices, indices_flat] 
        v_gathered = v_flat[batch_indices, indices_flat] 
        
        attn_scores = (q_flat.unsqueeze(2) @ k_gathered.transpose(-2, -1)) * scale
        
        attn_probs = attn_scores.squeeze(2).softmax(dim=-1)
        
        out_top_k_flat = (attn_probs.unsqueeze(2) @ v_gathered).squeeze(2)
        
        return out_top_k_flat

    def forward(self, x: torch.Tensor, last_frame: bool = False) -> torch.Tensor:
        B, T, H, W, _ = x.shape
        
        q_grid, k_grid, v_grid = self.to_qkv(x).chunk(3, dim=-1)
        q_grid = rearrange(q_grid, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k_grid = rearrange(k_grid, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v_grid = rearrange(v_grid, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        freqs = self.rotary_emb.get_axial_freqs(H, W).to(q_grid.device)
        q_grid = apply_rotary_emb(freqs, q_grid)
        k_grid = apply_rotary_emb(freqs, k_grid)

        q = rearrange(q_grid, "b h H W d -> b h (H W) d")
        k = rearrange(k_grid, "b h H W d -> b h (H W) d")
        v = rearrange(v_grid, "b h H W d -> b h (H W) d")
        
        BT, heads, S, d_head = q.shape

        q_flat = q.reshape(BT * heads, S, d_head)
        k_flat = k.reshape(BT * heads, S, d_head)
        v_flat = v.reshape(BT * heads, S, d_head)

        scale = self.head_dim ** -0.5

        if self.first_frame:
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = attn @ v 
            
            _, self.top_k_indices = torch.topk(attn, self.top_k_keys, dim=-1, sorted=False)
            indices_flat = self.top_k_indices.reshape(BT * heads, S, self.top_k_keys)

            out_top_k_flat = self._compute_out_top_k(q_flat, k_flat, v_flat, indices_flat, scale)
            out_top_k = out_top_k_flat.reshape(BT, heads, S, d_head)

            self.out_not_top_k = out - out_top_k
            
        else:
            if self.top_k_indices is None or self.out_not_top_k is None:
                raise RuntimeError("Cache is not populated. You must run a `first_frame` pass before subsequent frames.")

            indices_flat = self.top_k_indices.reshape(BT * heads, S, self.top_k_keys)
            out_top_k_flat = self._compute_out_top_k(q_flat, k_flat, v_flat, indices_flat, scale)
            out_top_k = out_top_k_flat.reshape(BT, heads, S, d_head)

            out = out_top_k + self.out_not_top_k

        out = rearrange(out, "(B T) h (H W) d -> B T H W (h d)", B=B, T=T, H=H, W=W)
        
        # If this was the last frame, the next one is a new first frame
        self.first_frame = last_frame
        
        return self.to_out(out.to(x.dtype))
