import torch
import torch.nn as nn
import math
import triton
import triton.language as tl

@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    batch_size, seq_len, num_heads, head_dim,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_head_dim_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_head_dim_stride,
    scale, # 1 / square_root(d_k)
    BLOCK_SIZE: tl.constexpr
    ):

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx =  tl.program_id(2)

    # Computing pointer offsets
    # Navigate to correct starting positions
    q_batch_offset = batch_idx * q_batch_stride
    q_head_offset = head_idx * q_head_stride
    q_seq_offset = seq_idx * q_seq_stride

    # No sequence offset for K and V since each query block will see all the key blocks
    k_batch_offset = batch_idx * k_batch_stride
    k_head_offset = head_idx * k_head_stride

    v_batch_offset = batch_idx * v_batch_stride
    v_head_offset = head_idx * v_head_stride

    o_batch_offset = batch_idx * o_batch_stride
    o_head_offset = head_idx * o_head_stride
    o_seq_offset = seq_idx * o_seq_stride

    # Loading query vector for this sequence position
    q_ptrs = q_ptr + q_batch_offset + q_head_offset + q_seq_offset + tl.arange(0, BLOCK_SIZE) * q_head_dim_stride # This loads data even if it is in non contiguous locations in memory
    q = tl.load(q_ptrs, mask = tl.arange(0, BLOCK_SIZE) < head_dim, other = 0.0)

    # Initialise accumulator for weighted sum. One score for each key token
    acc = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

    softmax_denominator = 0.0

    for k_seq_idx in range(seq_len):
        k_seq_offset = k_seq_idx * k_seq_stride
        k_ptrs =  k_ptr + k_batch_offset + k_head_offset + k_seq_offset + tl.arange(0, BLOCK_SIZE) * k_head_dim_stride
        k = tl.load(k_ptrs, mask = tl.arange(0, BLOCK_SIZE) < head_dim, other = 0.0)

        score = tl.sum(q * k) * scale
        attention_weight = tl.exp(score)

        softmax_denominator += attention_weight

        v_seq_offset = k_seq_idx * v_seq_stride
        v_ptrs = v_ptr + v_batch_offset + v_head_offset + v_seq_offset + tl.arange(0, BLOCK_SIZE) * v_head_dim_stride

        v = tl.load(v_ptrs, mask = tl.arange(0, BLOCK_SIZE) < head_dim, other = 0.0)

        acc += attention_weight * v

    acc /= softmax_denominator

    output_ptrs = o_ptr + o_batch_offset + o_head_offset + o_seq_offset + tl.arange(0, BLOCK_SIZE) * o_head_dim_stride
    tl.store(output_ptrs, acc, mask = tl.arange(0, BLOCK_SIZE) < head_dim)


class TritonAttentionNaive(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)

        self.W_o = nn.Linear(d_model, d_model, bias = False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        o = torch.empty_like(q)

        scale = 1.0 / math.sqrt(self.head_dim)

        grid = (batch_size, self.num_heads, seq_len)

        block_size = 1
        while block_size < self.head_dim:
            block_size *= 2

        attention_kernel[grid](
            q, k, v, o,
            batch_size, seq_len, self.num_heads, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            scale,
            BLOCK_SIZE = block_size
        )

        #  [batch_size, num_heads , seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, d_model]
        o = o.permute(0,2,1,3).contiguous().view(batch_size, seq_len, self.d_model)
        o = self.W_o(o)

        return o