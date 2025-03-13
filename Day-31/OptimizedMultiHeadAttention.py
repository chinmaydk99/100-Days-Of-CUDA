import torch
import torch.nn as nn
import math
import triton
import triton.language as tl

@triton.jit
def optimized_attention_kernel(
      q_ptr, k_ptr, v_ptr, o_ptr,
      batch_size, seq_len, num_heads, head_dim,
      q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
      k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride,
      v_batch_stride, v_head_stride, v_seq_stride, v_head_dim_stride,
      o_batch_stride, o_head_stride, o_seq_stride, o_head_dim_stride,
      scale, # 1 / square_root(d_k)
      BLOCK_SIZE_M: tl.constexpr, # Query Block Size
      BLOCK_SIZE_N: tl.constexpr, # Key Block Size
      BLOCK_SIZE_DMODEL : tl.constexpr, # Head Dimension Block Size,
      USE_CAUSAL_MASK : tl.constexpr
      ):

      batch_id = tl.program_id(0)
      head_id = tl.program_id(1)
      seq_start =  tl.program_id(2) * BLOCK_SIZE_M

      q_head_offset = head_id * q_head_stride
      k_head_offset = head_id * k_head_stride
      v_head_offset = head_id * v_head_stride

      q_batch_offset = batch_id * q_batch_stride
      k_batch_offset = batch_id * k_batch_stride
      v_batch_offset = batch_id * v_batch_stride

      o_head_offset = head_id * o_head_stride
      o_batch_offset = batch_id * o_batch_stride

      # Initializing accumulators
      m_i = tl.zeros([BLOCK_SIZE_M], dtype = tl.float32) - float('inf')
      l_i = tl.zeros([BLOCK_SIZE_M], dtype = tl.float32)
      acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_DMODEL], dtype = tl.float32)

      q_block_mask = (seq_start + tl.arange(0, BLOCK_SIZE_M)) < seq_len

      # Processing key blocks
      for key_start in range(0, seq_len, BLOCK_SIZE_N):
          k_block_mask = (key_start + tl.arange(0, BLOCK_SIZE_N)) < seq_len

          if USE_CAUSAL_MASK:
            causal_mask = tl.arange(0, BLOCK_SIZE_M)[:, None] + seq_start >= tl.arange(0, BLOCK_SIZE_N)[None, :] + key_start # Process only tokens that occur before the given query

          # Loading Query Block [BLOCK_M, BLOCK_DMODEL]
          q_block_ptr = q_ptr + q_batch_offset + q_head_offset + (seq_start + tl.arange(0, BLOCK_SIZE_M)[:, None])* q_seq_stride + (key_start + tl.arange(0, BLOCK_SIZE_DMODEL)[None, :]) * q_head_dim_stride

          q_block = tl.load(q_block_ptr, mask=q_block_mask[:, None] & (tl.arange(0, BLOCK_SIZE_DMODEL)[None, :] < head_dim), other=0.0)

          # Loading Key Block [BLOCK_N, BLOCK_DMODEL]
          k_block_ptr = k_ptr + k_batch_offset + k_head_offset + (key_start + tl.arange(0,BLOCK_SIZE_N)[:, None])* k_seq_stride + (key_start + tl.arange(0, BLOCK_SIZE_DMODEL)[None, :]) * k_head_dim_stride

          k_block = tl.load(k_block_ptr, mask=k_block_mask[:, None] & (tl.arange(0, BLOCK_SIZE_DMODEL)[None, :] < head_dim), other=0.0)

          # Computing attention scores
          scores = tl.dot(q_block, tl.trans(k_block)) * scale

          if USE_CAUSAL_MASK:
            scores = tl.where(causal_mask, scores, float("-inf")) # This will be zeroed our during softmax

          # Stable Softmax Computation
          # 1. Computing Max for Numerical Stability
          m_ij = tl.max(scores, axis = 1)

          # 2. Updating Running Max
          m_i_new = tl.maximum(m_i, m_ij)

          # 3. Computing Exponentials with the updated max
          exp_scores = tl.exp(scores - m_i_new[:, None])

          # 4. Compute Scaling factor for previous computations
          alpha = tl.exp(m_i - m_i_new)

          # 5. Updating normalization factor
          l_i_new = alpha * l_i + tl.sum(exp_scores, axis = 1)

          # Loading Value block [BLOCK_N, BLOCK_DMODEL]
          v_block_ptr = v_ptr + v_batch_offset + v_head_offset + (key_start + tl.arange(0, BLOCK_SIZE_N)[:, None])* v_seq_stride + (key_start + tl.arange(0, BLOCK_SIZE_DMODEL)[None, :]) * v_head_dim_stride

          v_block = tl.load(v_block_ptr, mask=k_block_mask[:, None] & (tl.arange(0, BLOCK_SIZE_DMODEL)[None, :] < head_dim), other=0.0)

          acc = acc * alpha[:, None] + tl.dot(exp_scores, v_block)

          m_i = m_i_new
          l_i = l_i_new

      acc /= l_i[:, None]

      o_block_ptr = o_ptr + o_batch_offset + o_head_offset + (seq_start + tl.arange(0, BLOCK_SIZE_M)[:, None])* o_seq_stride + (tl.arange(0, BLOCK_SIZE_DMODEL)[None, :]) * o_head_dim_stride

      tl.store(o_block_ptr, acc, mask=q_block_mask[:, None] & (tl.arange(0, BLOCK_SIZE_DMODEL)[None, :] < head_dim))


class TritonAttentionOptimized(nn.Module):
    def __init__(self, d_model, num_heads, causal = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)

        self.W_o = nn.Linear(d_model, d_model, bias = False)

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

        BLOCK_M = 16
        BLOCK_N = 16

        # Round head_dim up to the nearest power of 2 for BLOCK_DMODEL
        BLOCK_DMODEL = 1
        while BLOCK_DMODEL < self.head_dim:
            BLOCK_DMODEL *= 2

        grid = (batch_size, self.num_heads, triton.cdiv(seq_len, BLOCK_M))

        optimized_attention_kernel[grid](
            q, k, v, output,
            batch_size, seq_len, self.num_heads, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            scale,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_DMODEL=BLOCK_DMODEL,
            USE_CAUSAL_MASK=self.causal,
        )

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(output)

        return output