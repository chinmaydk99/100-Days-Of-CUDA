import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads = None, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads # Defaults to multi head attention
        self.head_dim = d_model // num_heads
        
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Dimension for each projection
        self.q_proj_dim  = d_model
        self.kv_proj_dim = self.num_kv_heads * self.head_dim

        # Projections 
        self.W_q = nn.Linear(self.q_proj_dim, self.q_proj_dim, bias=False)
        self.W_k = nn.Linear(self.q_proj_dim, self.kv_proj_dim, bias=False)
        self.W_v = nn.Linear(self.q_proj_dim, self.kv_proj_dim, bias=False)
        self.W_o = nn.Linear(self.q_proj_dim, self.q_proj_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.shape

        # Applying Linear Projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Repeating k and v for grouped query attention
        # [batch_size, seq_len, num_kv_heads, head_dim] -> [batch_size, seq_len, num_heads,  head_dim]
        # We repeat each key and value head num_queries_per_kv number of times

        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        
        # [batch_size, seq_len, num_heads, head_him] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #
        context = torch.matmul(attn_weights, v)

        #[batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads,  head_dim]
        context = context.transpose(1,2)

        context = context.contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(context)

        return output

class MultiQueryAttention(GroupedQueryAttention):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__(d_model, num_heads, num_kv_heads=1, dropout=dropout)
    
    def forward(self,x, mask = None):
        return super().forward(x, mask)