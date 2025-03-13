import torch
import torch.nn as nn
import math
import triton
import triton.language as tl

@triton.jit
def flash_attn_v2_forward(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr,
    # Matrix dimensions
    batch_size, seq_len, num_heads, head_dim,
    # Strides for accessing tensors
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_head_dim_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_head_dim_stride,
    # Scale factor
    scale,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Causal flag
    IS_CAUSAL: tl.constexpr,
):
    """
    Simplified Flash Attention V2 forward pass
    """
    # Program ID
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    m_id = tl.program_id(2)

    # Starting row index
    start_m = m_id * BLOCK_SIZE_M

    q_batch_offset = batch_id * q_batch_stride
    q_head_offset = head_id * q_head_stride
    k_batch_offset = batch_id * k_batch_stride
    k_head_offset = head_id * k_head_stride
    v_batch_offset = batch_id * v_batch_stride
    v_head_offset = head_id * v_head_stride
    o_batch_offset = batch_id * o_batch_stride
    o_head_offset = head_id * o_head_stride

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    # Create row indices and mask
    row_indices = start_m + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_indices < seq_len

    # Loading Q block once - I'll reuse it for all K,V blocks
    q_block = tl.load(
        q_ptr + q_batch_offset + q_head_offset +
        row_indices[:, None] * q_seq_stride +
        tl.arange(0, BLOCK_SIZE_K)[None, :] * q_head_dim_stride,
        mask=row_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim),
        other=0.0
    )

    # Process blocks of K and V
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        col_indices = start_n + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_indices < seq_len

        if IS_CAUSAL:
            causal_mask = row_indices[:, None] >= col_indices[None, :]

        # Loading K block
        k_block = tl.load(
            k_ptr + k_batch_offset + k_head_offset +
            col_indices[:, None] * k_seq_stride +
            tl.arange(0, BLOCK_SIZE_K)[None, :] * k_head_dim_stride,
            mask=col_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim),
            other=0.0
        )

        scores = tl.dot(q_block, tl.trans(k_block)) * scale

        if IS_CAUSAL:
            scores = tl.where(causal_mask, scores, float('-inf'))

        # Compute new max for stable softmax
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_i_new)

        # Update max values
        m_i = m_i_new

        # Compute softmax values with updated max
        p = tl.exp(scores - m_i[:, None])

        # Load V block
        v_block = tl.load(
            v_ptr + v_batch_offset + v_head_offset +
            col_indices[:, None] * v_seq_stride +
            tl.arange(0, BLOCK_SIZE_K)[None, :] * v_head_dim_stride,
            mask=col_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim),
            other=0.0
        )

        # Update sum of exponentials
        l_i_new = alpha * l_i + tl.sum(p, axis=1)

        # Update weighted sum
        acc_new = alpha[:, None] * acc + tl.dot(p, v_block)

        # Update accumulators
        l_i = l_i_new
        acc = acc_new

    # Normalize output
    out = acc / l_i[:, None]

    # Store output
    tl.store(
        o_ptr + o_batch_offset + o_head_offset +
        row_indices[:, None] * o_seq_stride +
        tl.arange(0, BLOCK_SIZE_K)[None, :] * o_head_dim_stride,
        out,
        mask=row_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim)
    )


class FlashAttentionV2(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.0,
        causal=False,
        block_size=64
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.causal = causal
        self.block_size = block_size

        # Linear projections
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

        # Make sure head_dim is a power of 2
        if not (self.head_dim & (self.head_dim - 1) == 0):
            raise ValueError(f"Head dimension ({self.head_dim}) must be a power of 2")

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Prepare output tensor
        output = torch.empty_like(q)

        # Scaling factor
        scale = 1.0 / math.sqrt(self.head_dim)

        # Calculate grid dimensions
        grid = (
            batch_size,
            self.num_heads,
            triton.cdiv(seq_len, self.block_size)
        )

        # Round head_dim up to the nearest power of 2 if needed
        block_k = self.head_dim
        if block_k & (block_k - 1) != 0:
            block_k = 1
            while block_k < self.head_dim:
                block_k *= 2

        # Launch kernel
        flash_attn_v2_forward[grid](
            q, k, v, output,
            batch_size, seq_len, self.num_heads, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            scale,
            BLOCK_SIZE_M=self.block_size,
            BLOCK_SIZE_N=self.block_size,
            BLOCK_SIZE_K=block_k,
            IS_CAUSAL=self.causal,
        )

        # Reshape output back
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        # Apply dropout
        if self.dropout > 0.0 and self.training:
            output = torch.nn.functional.dropout(output, p=self.dropout)

        # Final linear projection
        output = self.W_o(output)

        return output
