import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def gqa_forward(
    q_ptr, k_ptr, v_ptr, o_ptr,
    batch_size, seq_len, num_q_heads, num_kv_heads, head_dim,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_head_dim_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_head_dim_stride,
    scale,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    IS_CAUSAL: tl.constexpr):
    
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_id = tl.program_id(2)
    
    start_m = seq_id * BLOCK_SIZE_M
    
    head_kv_id = head_id // num_queries_per_kv
    
    q_batch_offset = batch_id * q_batch_stride
    k_batch_offset = batch_id * k_batch_stride
    v_batch_offset = batch_id * v_batch_stride
    o_batch_offset = batch_id * o_batch_stride
    
    q_head_offset = head_id * q_head_stride
    k_head_offset = head_kv_id * k_head_stride
    v_head_offset = head_kv_id * v_head_stride
    o_head_offset = head_id * o_head_stride
    
    # Initialise accumulators
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
    
    row_indices = start_m + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_indices < seq_len
    
    # Loading a query block of size [BLOCK_SIZE_M, BLOCK_SIZE_K]
    q_block = tl.load(
        q_ptr + q_batch_offset + q_head_offset + 
        row_indices[:, None] * q_seq_stride + 
        tl.arange(0, BLOCK_SIZE_K)[None, :] * q_head_dim_stride,
        mask=row_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim),
        other=0.0
    )
    
    # Processing blocks of K and V
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        col_indices = start_n + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_indices < seq_len
        
        if IS_CAUSAL:
            causal_mask = row_indices[:, None] >= col_indices[None, :]
        
        # Loading a key block of size [BLOCK_SIZE_N, BLOCK_SIZE_K]
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
        
        # Stable Online Softmax
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
        
        # Scaling factor
        alpha = tl.exp(m_i - m_i_new)
        
        # Updating the max value
        m_i = m_i_new
        
        p = tl.exp(scores - m_i[:, None])
        
        v_block = tl.load(
            v_ptr + v_batch_offset + v_head_offset + 
            col_indices[:, None] * v_seq_stride + 
            tl.arange(0, BLOCK_SIZE_K)[None, :] * v_head_dim_stride,
            mask=col_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim),
            other=0.0
        )
        
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc_new = alpha[:, None] * acc + tl.dot(p, v_block)
        
        l_i = l_i_new
        acc = acc_new
    
    # Scaling by normalization factor
    out = acc / l_i[:, None]
    
    tl.store(
        o_ptr + o_batch_offset + o_head_offset +
        row_indices[:, None] * o_seq_stride +
        tl.arange(0, BLOCK_SIZE_K)[None, :] * o_head_dim_stride,
        out,
        mask=row_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim)
    )

class Triton_GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.1, causal=False, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.q_proj_dim = d_model
        self.kv_proj_dim = self.num_kv_heads * self.head_dim

        self.causal = causal
        self.block_size = block_size

        self.W_q = nn.Linear(d_model, self.q_proj_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.kv_proj_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.kv_proj_dim, bias=False)
        self.W_o = nn.Linear(self.q_proj_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        output = torch.empty_like(q)

        scale = 1.0 / math.sqrt(self.head_dim)
        
        grid = (
            batch_size,                                    
            self.num_heads,                              
            triton.cdiv(seq_len, self.block_size)          
        )
        
        # Ensure block_k is a power of 2
        block_k = 1
        while block_k < self.head_dim:
            block_k *= 2
        
        # Launch kernel
        gqa_forward[grid](
            q, k, v, output,
            batch_size, seq_len, self.num_heads, self.num_kv_heads, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            scale,
            BLOCK_SIZE_M=self.block_size,
            BLOCK_SIZE_N=self.block_size,
            BLOCK_SIZE_K=block_k,
            num_queries_per_kv=self.num_queries_per_kv,
            IS_CAUSAL=self.causal,
        )
        
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, seq_len, self.d_model)
        
        if self.dropout.p > 0.0 and self.training:
            output = self.dropout(output)
        
        output = self.W_o(output)
        
        return output