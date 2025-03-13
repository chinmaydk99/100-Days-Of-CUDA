import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TorchAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # Initialise Projection matrices
        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)

        # Output Projection
        self.W_o = nn.Linear(d_model, d_model, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # x is of shape [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.W_q(x) # [batch_size, seq_len, d_model]
        k = self.W_k(x) # [batch_size, seq_len, d_model]
        v = self.W_v(x) # [batch_size, seq_len, d_model]

        # Reshaping for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # Computing attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [batch_size, num_heads, seq_len, seq_len]

        # Mask for causal attention
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim = -1)
        # Softmax along dim = -1 to determine how much attention along each key dimension

        # Applying dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Applying attention weights to values
        # [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_weights, v)

        # Concatenating heads
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2)
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_model]
        context = context.contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output

def create_causal_mask(seq_len):
    # Lower triangular matrix so that query tokens don't have access to keys that come after them in the sequence
    mask = torch.tril(torch.ones((seq_len, seq_len)))
    return mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]