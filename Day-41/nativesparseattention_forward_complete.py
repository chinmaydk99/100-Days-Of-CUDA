
import torch
import triton
import triton.language as tl
from torch.autograd import Function
from einops import rearrange, repeat, reduce
import einx
import math

# !pip install jaxtyping einx

TRITON_BLOCK_SIZE = 128

from torch import Tensor
from jaxtyping import Float, Int, Bool

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes:str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(Float)
Int   = TorchTyping(Int)
Bool  = TorchTyping(Bool)

def divisible_by(a, b):
    return a % b == 0

@triton.jit
def sparse_attn_forward_causal_and_sparse(
    q, k, v,
    kv_block_indices,
    kv_block_mask,
    out_ptr,
    lse_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_outb, stride_outh, stride_outm,
    stride_kvb_b, stride_kvb_h, stride_kvb_m,
    stride_logsumexp_b,
    kv_heads,
    q_seq_len, kv_seq_len,
    rounded_q_seqlen,
    headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM : tl.constexpr,
    EVEN_M :tl.constexpr,
    EVEN_N : tl.constexpr,
    EVEN_HEADDIM : tl.constexpr,
    BLOCKSIZE : tl.constexpr,
    SEL_BLOCK : tl.constexpr,
    QUERY_HEAD_GROUPS : tl.constexpr,
    QUERY_EXPAND_DIM : tl.constexpr,
    NUM_SEL_KV_BLOCKS : tl.constexpr,
    NUM_BLOCKS_PER_SEL :tl.constexpr,
    INCLUDE_BLOCK_CAUSAL : tl.constexpr,
    SLIDING: tl.constexpr):

    start_m = tl.program_id(0) # Query block index
    offs_batch_head = tl.program_id(1)

    offs_batch = offs_batch_head // kv_heads
    offs_head = offs_batch_head % kv_heads

    # Query head indices
    offs_qh = offs_head * QUERY_HEAD_GROUPS + tl.arange(0, QUERY_HEAD_GROUPS) # Each kv block is used/shared by QUERY_HEAD_GROUP number of query blocks

    offs_m = start_m * BLOCKSIZE + tl.arange(0, BLOCKSIZE) # Tokens processed in current block

    offs_d = tl.arange(0, BLOCK_HEADDIM) # Feature dimension indices

    q_ptrs = (
        q +
        offs_batch* stride_qb +
        offs_qh[None, :, None] * stride_qh +
        offs_m[:, None, None] * stride_qm +
        offs_d[None, None, : ]
        )   # [BLOCKSIZE, QUERY_HEAD_GROUPS, BLOCK_HEADDIM]


    # Maximum
    m_i = tl.zeros([BLOCKSIZE, QUERY_HEAD_GROUPS], dtype = tl.float32) - float("inf")

    lse = (
        lse +
        offs_batch * stride_logsumexp_b +
        offs_qh[None, :] * rounded_q_seqlen +
        offs_m[:, None]
    ) # [BLOCKSIZE, QUERY_HEAD_GROUPS]

    lse_i = tl.zeros([BLOCKSIZE, QUERY_HEAD_GROUPS], dtype = tl.float32)

    out_ptrs = (
        out_ptr +
        offs_batch * stride_outb +
        offs_qh[None, :, None] * stride_outh +
        offs_m[:, None, None] * stride_outm +
        offs_d[None, None, : ]
    ) # [BLOCKSIZE, QUERY_HEAD_GROUPS, BLOCK_HEADDIM]

    acc_o = tl.zeros([BLOCKSIZE, QUERY_HEAD_GROUPS, BLOCK_HEADDIM])

    # Accounting forn cases where BLOCKSIZE > q_seqlen BLOCK_HEADDIM > headdim
    if EVEN: # If q_seqlen is a multiple of BLOCKSIZE
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs,
                        mask = offs_d[None, None, :] < headdim,
                        other = 0.0)

    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs,
                        mask = offs_m[:, None , None] < q_seq_len,
                        other = 0.0)
        else:
            q = tl.load(q_ptrs,
                        mask = (offs_m[:, None, None] < q_seq_len) & (offs_d[None, None, :] < headdim),
                        other = 0.0)

    q = q.reshape(BLOCKSIZE * QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

    # Computing attention via Causal Blocks
    if INCLUDE_BLOCK_CAUSAL:
        # Key Indices to slide over
        start_n = (
            start_m * BLOCKSIZE  + tl.arange(0, SEL_BLOCK) + (SEL_BLOCK - BLOCKSIZE)
        )

        # Either attend over just one KV block(static block attention) or 2 blocks(sliding window attention)
        if SLIDING:
            num_kv_blocks = 2
        else:
            num_kv_blocks = 1

        for kv_block_offset_ind in range(num_kv_blocks):
            offset = kv_block_offset_ind * -SEL_BLOCK
            offset_n = start_n + offset   # For sliding window we create 2 offsets one that attends to previous block and one for current block. For non-sliding just the current block

            # Loading  [BLOCKSIZE, BLOCK_HEADDIM] chunks of K and V
            k_ptrs = (
                k +
                offs_batch * stride_kb +
                offs_head * stride_kh +
                offset_n[:, None] * stride_kn +
                offs_d[None, : ]
            )  # [SEL_BLOCK, BLOCK_HEADDIM]

            v_ptrs = (
                v +
                offs_batch * stride_vb +
                offs_head * stride_vh +
                offset_n[:, None] * stride_vn +
                offs_d[None, : ]
            )

            if EVEN_M and EVEN_N:
                if EVEN_HEADDIM:
                    k = tl.load(
                        k_ptrs,
                        mask = offset_n[:, None] >= 0, # Prevent negative indexing
                        other = 0.0
                    )

                else:
                    k = tl.load(
                        k_ptrs,
                        mask = (offset_n[:, None] >= 0) & (offs_d[None, :] < headdim),
                        other = 0.0
                    )
            else:
                if EVEN_HEADDIM:
                    k = tl.load(
                        k_ptrs,
                        mask = (offset_n[:, None] >= 0) & (offset_n[:, None] < kv_seq_len),
                        other = 0.0
                    )
                else:
                    k = tl.load(
                        k_ptrs,
                        mask = offset_n[:, None] >= 0 & (offset_n[:, None] < kv_seq_len) & (offs_d[None, :] < headdim),
                        other = 0.0
                    )

            qk = tl.zeros([BLOCKSIZE * QUERY_HEAD_GROUPS, SEL_BLOCK], dtype = tl.float32)

            qk += tl.dot(q, tl.trans(k))  # [BLOCKSIZE * QUERY_HEAD_GROUPS, SEL_BLOCK]

            qk = qk.reshape(BLOCKSIZE, QUERY_HEAD_GROUPS, SEL_BLOCK)

            # Block attention mask: In case of non sliding window attention, query tokens must attend to keys only within the current blocks

            if BLOCK != SEL_BLOCK and not SLIDING:
                block_diagonal_mask = (
                    offset_n[None, None, :] >= 0.0 &
                    ((offset_n[None, None, :] // SEL_BLOCK) == (offs_m[:, None, None] // SEL_BLOCK)) # shape [BLOCKSIZE, 1, SEL_BLOCK]
                )
                qk += tl.where(block_diagonal_mask,  0, float("-inf"))

            if not EVEN_N:
                qk += tl.where(offset_n[None , :] < kv_seq_len, 0, float("-inf"))

            qk = qk.reshape(BLOCKSIZE, QUERY_HEAD_GROUPS, SEL_BLOCK)

            causal_mask = offs_m[:, None, None] >= offset_n[None, None, :] # [BLOCKSIZE, 1, SEL_BLOCK]

            # In case of sliding window attention, we use a modified version of the earlier block attention mask by restricting that query tokens can attend to previous block  key tokens as long as they aren't more than SEL_BLOCK tokens apart (sliding window size we hac chosen)

            if SLIDING:
                causal_mask &= (
                    (offset_n[None, None, :] >= 0.0) &
                    ((offs_m[:, None, None] - offs_n[None, None, :] <= SEL_BLOCK))
                )

            qk += tl.where(causal_mask, 0, float("-inf"))

            m_ij = tl.maximum(tl.max(qk,2) * softmax_scale, lse_i) # Taking max along SEL_BLOCK dim.
            # Shape (BLOCKSIZE, QUERY_HEAD_GROUPS)

            p = tl.exp(qk * softmax_scale - m_ij[:, :, None]) # [BLOCKSIZE, QUERY_HEAD_GROUPS, SEL_BLOCK]

            l_ij = tl.sum(p, 2) # Again summing along SEL_BLOCK

            acc_o_scale = tl.exp(m_i - m_ij) #[BLOCKSIZE, QUERY_HEAD_GROUPS]
            acc_o *= acc_o_scale[:, :, None] # [BLOCKSIZE, QUERY_HEAD_GROUPS, 1]

            # Load V blocks
            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    v = tl.load(
                        v_ptrs,
                        mask = offset_n[:, None] >= 0,
                        other = 0.0
                    )
                else:
                    v = tl.load(
                        v_ptrs,
                        mask = offset_n[:, None] >= 0 & offs_d[None, :] < headdim,
                        other = 0.0
                    )
            else:
                if EVEN_HEADDIM:
                    v = tl.load(
                        v_ptrs,
                        mask = (offset_n[:, None] >= 0) & (offset_n[:, None] < kv_seq_len),
                        other = 0.0
                    )
                else:
                    v = tl.load(
                        v_ptrs,
                        mask = offset_n[:, None] >= 0 & (offset_n[:, None] < kv_seq_len) & (offs_d[None,:] < headdim),
                        other = 0.0
                    )

            p = p.reshape(BLOCKSIZE*QUERY_HEAD_GROUPS, SEL_BLOCK)

            causal_o = tl.dot(p,v) # [BLOCKSIZE * QUERY_HEAD_GROUPS, SEL_BLOCK]

            acc_o += causal_o.reshape(BLOCKSIZE, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

            # Update the online softmax parameters
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)


    # Selected KV Block attention

    kv_block_indices_ptrs = (
        kv_block_indices +
        offs_batch * stride_kvb_b +
        offs_head * stride_kvb_h +
        offs_m * stride_kvb_m
    ) # Shape - [BLOCKSIZE]

    kv_block_mask_ptrs = (
        kv_block_mask +
        offs_batch * stride_kvb_b +
        offs_head * stride_kvb_h +
        offs_m * stride_kvb_m
    ) # Shape - [BLOCKSIZE]

    q = q.reshape(BLOCKSIZE, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)
    q = tl.expand_dims(q, 2) # (BLOCKSIZE, QUERY_HEAD_GROUPS, 1 , BLOCK_HEADDIM)

    q = tl.broadcast_to(q, (BLOCKSIZE, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM)) # Repeating along QUERY_EXPAND_DIM axis
    # This is done so that each query is matched against QUERY_EXPAND_DIM keys and the attention scores are then averaged

    q = q.reshape(BLOCKSIZE, 16, BLOCK_HEADDIM)

    #################################### Modify
    # Iterate through the selected KV blocks ie the selected KV block for each query token
    for off_sel_kv_block in range(NUM_SEL_KV_BLOCKS):

        block_indices = tl.load(
            kv_block_indices_ptrs + off_sel_kv_block,
            mask = offs_m < seqlen_q,
            other = 0
        )

        block_masks = tl.load(
            kv_block_mask_ptrs + off_sel_kv_block,
            mask = offs_m < seqlen_q,
            other = False
        )

        for off_blocks_per_sel in range(NUM_BLOCKS_PER_SEL):

            blocks_offs_n = (
                block_indices[:, None] * (BLOCK * NUM_BLOCKS_PER_SEL) +
                tl.arange(0, BLOCK)[None, :] + (off_blocks_per_sel * BLOCK)
            )

            block_k_ptrs = (
                K +
                off_b * stride_kb +
                off_h * stride_kh +
                blocks_offs_n[:, :, None] * stride_kn +
                offs_d[None, None, :]
            )

            block_v_ptrs = (
                V +
                off_b * stride_vb +
                off_h * stride_vh +
                blocks_offs_n[:, :, None] * stride_vn +
                offs_d[None, None, :]
            )

            # load k of shape (m, n, d), sparsely selected by each query

            k_block = tl.load(
                block_k_ptrs,
                mask = blocks_offs_n[:, :, None] < seqlen_k,
                other = 0.
            )

            # similarities

            block_qk = tl.zeros([BLOCK, 16, BLOCK], dtype = tl.float32)
            sel_qk = tl.zeros([BLOCK, QUERY_HEAD_GROUPS, BLOCK], dtype = tl.float32)

            k_block = k_block.reshape(BLOCK, BLOCK, BLOCK_HEADDIM)
            k_block = k_block.permute(0, 2, 1)

            block_qk += tl.dot(q, k_block)
            block_qk = block_qk.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK)
            block_qk = tl.reduce(block_qk, 2, reduce_avg)

            sel_qk += block_qk
            sel_qk += tl.where(block_masks[:, None, None], 0, float("-inf"))

            # attention

            m_ij = tl.maximum(tl.max(sel_qk, 2) * softmax_scale, lse_i)
            block_p = tl.exp(sel_qk * softmax_scale - m_ij[:, :, None])

            l_ij = tl.sum(block_p, 2)

            # renormalize the running output

            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, :, None]

            # aggregate values

            v_block = tl.load(
                block_v_ptrs,
                mask = blocks_offs_n[:, :, None] < seqlen_k,
                other = 0.
            )

            v_block = tl.reshape(v_block, (BLOCK, BLOCK, BLOCK_HEADDIM))

            block_p = block_p.to(v_block.dtype)
            p_expanded = block_p.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK)
            p_expanded = tl.expand_dims(p_expanded, 2)
            p_expanded = tl.broadcast_to(p_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK))
            p_expanded = p_expanded.reshape(BLOCK, 16, BLOCK)

            block_acc_o = tl.dot(p_expanded, v_block)
            block_acc_o = block_acc_o.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM)
            block_acc_o = tl.reduce(block_acc_o, 2, reduce_avg)

            acc_o += block_acc_o

            # -- update statistics

            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    # normalize accumulated out

    acc_o_scale = tl.exp(m_i - lse_i)
    acc_o *= acc_o_scale[:, :, None]

    # write back lse

    lse_i = lse_i.reshape(BLOCK, QUERY_HEAD_GROUPS)
    tl.store(lse_ptrs, lse_i)

    # write to output

    acc_o = acc_o.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask = offs_d[None, None, :] < headdim
            )
    else:
        if EVEN_HEADDIM:
            tl.store(
                out_ptrs,
                acc_o,
                mask = offs_m[:, None, None] < seqlen_q
            )
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim)
            )

@triton.heuristics(
    dict(
        EVEN_M = lambda args : divisible_by(args["q_seq_len"], args["BLOCKSIZE"]),
        EVEN_N = lambda args : divisible_by(args["kv_seq_len"], args["BLOCKSIZE"]),
        EVEN_HEADDIM = lambda args : divisible_by(args["dim"], args["BLOCK_HEADDIM"]),
        QUERY_EXPAND_DIM = lambda args : 16 // args["QUERY_HEAD_GROUPS"]
    )
)
@triton.jit
def sparse_attn_forward_kernel(
      q, k, v,
      kv_block_indices,
      kv_block_mask,
      out, sliding_out,
      logsumexp, sliding_logsumexp,
      softmax_scale,
      stride_qb, stride_qh, stride_qm,
      stride_kb, stride_kh, stride_kn,
      stride_vb, stride_vh, stride_vn,
      stride_outb, stride_outh, stride_outm,
      stride_kvb_b, stride_kvb_h, stride_kvb_m,
      stride_logsumexp_b,
      kv_heads, # Number of key/value attention heads
      q_seq_len, kv_seq_len,
      rounded_q_seqlen, # Query sequence length rounded up to a multiple of block size
      headdim,  # Dimension of each attention head
      CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, # Keys for triton's kernel cache
      BLOCK_HEADDIM: tl.constexpr,
      EVEN_M : tl.constexpr,
      EVEN_N : tl.constexpr,
      EVEN_HEADDIM : tl.constexpr,
      BLOCKSIZE : tl.constexpr,
      SEL_BLOCK : tl.constexpr,
      QUERY_HEAD_GROUPS : tl.constexpr,
      QUERY_EXPAND_DIM : tl.constexpr,
      NUM_SEL_KV_BLOCKS :tl.constexpr,
      NUm_BLOCKS_PER_SEL :tl.constexpr,
      INCLUDE_BLOCK_CAUSAL = tl.constexpr,
      COMBINE_SLIDING_WINDOW_OUT = tl.constexpr):

      if COMBINE_SLIDING_WINDOW_OUT:
          sliding = tl.program_id(2) == 0 # Two passes here, one for sliding window attention, second for selected KV blocks
          out_ptr = sliding_out if sliding else out
          lse_ptr = sliding_logsumexp if sliding else logsumexp
          num_sel_kv_blocks = 0 if sliding else NUM_SEL_KV_BLOCKS
      else:
          out_ptr = out
          lse_ptr = logsumexp
          num_sel_kv_blocks = NUM_SEL_KV_BLOCKS

      sparse_attn_forward_causal_and_sparse(
          q, k, v,
          kv_block_indices,
          kv_block_mask,
          out_ptr,
          lse_ptr,
          softmax_scale,
          stride_qb, stride_qh, stride_qm,
          stride_kb, stride_kh, stride_kn,
          stride_vb, stride_vh, stride_vn,
          stride_outb, stride_outh, stride_outm,
          stride_kvb_b, stride_kvb_h, stride_kvb_m,
          stride_logsumexp_b,
          kv_heads,
          q_seq_len, kv_seq_len,
          rounded_q_seqlen,
          headdim,
          CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
          BLOCK_HEADDIM,
          EVEN_M, EVEN_N,
          EVEN_HEADDIM,
          BLOCKSIZE,
          SEL_BLOCK,
          QUERY_HEAD_GROUPS,
          QUERY_EXPAND_DIM,
          num_sel_kv_blocks,
          NUM_BLOCKS_PER_SEL,
          INCLUDE_BLOCK_CAUSAL,
          sliding
      )

def native_sparse_attention_forward(
    q, k, v,
    kv_block_indices, kv_block_mask,
    block_size = 128,
    include_block_causal = True,
    combine_sliding_window_out = False
  ):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch, q_heads, q_seq_len, dim, device = *q.shape, q.device
    _, kv_heads, kv_seq_len, _ = *k.shape, k.device

    assert q_heads % kv_heads == 0

    head_groups = q_heads // kv_heads

    assert block_size % 16 == 0

    # Calculating number of subblocks that can fit into a single selection block
    # Potential Optimisation
    num_subblocks = block_size // 16
    if num_subblocks > 1:
        # Block indices hold global values of blocks that have been selected
        # We need to account for the subblocks. Ex if kv_block_indices are [2,5], and there are 4 subblocks , kv_block_indices will now start from [8,20] - (8,9,10,11)  and (20,21,22,23)
        kv_block_indices = einx.add('... sel_dim r -> ... (sel_dim r)', kv_block_indices * num_subblocks, torch.arange(num_subblocks, device = device))
        kv_block_mask = einx.repeat(kv_block_mask, '... sel_dim -> ... (sel_dim r)', r = num_subblocks)

    num_selected_subblocks = kv_block_indices.shape[-1]

    assert kv_block_indices.shape == kv_block_mask.shape
    assert k.shape == (batch, kv_heads, kv_seq_len, dim)
    assert v.shape == (batch, kv_heads, kv_seq_len, dim)
    assert dim <= 128, "Only supports head dimesnions till 128 (Needs Optimisation)"
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [torch.float16, torch.bfloat16], "Only supports fp16 and bf16"
    assert all([t.cuda for t in(q,k,v)])


    softmax_scale = dim ** -0.5

    rounded_q_seqlen = math.ceil(q_seq_len / TRITON_BLOCK_SIZE) * TRITON_BLOCK_SIZE # Rounding up query sequence length to be a multiple of triton block size

    logsumexp = torch.empty([batch, q_heads, rounded_q_seqlen], device = device, dtype = torch.float32)
    sliding_logsumexp = torch.empty([batch, q_heads, rounded_q_seqlen], device = device, dtype = torch.float32)

    # Outputs for attention and sliding window attention
    o = torch.empty_like(q)
    sliding_o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)
    num_warps =  4 if dim <= 64 else 8

    grid = lambda META: (
        triton.cdiv(q_seq_len, META['BLOCKSIZE']),
        batch * kv_heads, # As per paper in each block we load queries corresponding to kv heads to optimise memory access
        2 if combine_sliding_window_out else 1
    )

    sparse_attn_forward_kernel[grid](
        q, k, v,
        kv_block_indices, kv_block_mask,
        o, sliding_o,
        logsumexp, sliding_logsumexp,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        kv_block_indices.stride(0), kv_block_indices.stride(1), kv_block_indices.stride(2), logsumexp.stride(0),
        kv_heads, # Number of key/value attention heads
        q_seq_len, kv_seq_len,
        rounded_q_seqlen, # Query sequence length rounded up to a multiple of block size
        dim,  # Dimension of each attention head
        q_seq_len // 32, kv_seq_len // 32, # Keys for triton's kernel cache
        BLOCK_HEADDIM, # Padded Head Dimension
        BLOCKSIZE = 16,
        SEL_BLOCK = block_size,
        QUERY_HEAD_GROUPS = head_groups,
        NUM_SEL_KV_BLOCKS = num_selected_subblocks, # Number of selected subblocks per query
        NUM_BLOCKS_PER_SEL = num_subblocks, # Number of subblocks per selected block
        INCLUDE_BLOCK_CAUSAL = include_block_causal,
        COMBINE_SLIDING_WINDOW_OUT = combine_sliding_window_out,
        num_warps = num_warps,
        num_stages = 1
    )

    return o, sliding_o, logsumexp, sliding_logsumexp

class NativeSparseAttention(Function):
    @classmethod
    def forward(
        self,
        ctx, # For backward pass
        full_q, full_k, full_v, # Query, Key , Values in full precision
        block_size,
        selected_block_indices, # Blocks selected for attention
        fmask, # Mask for selected block indices
        sel_scale, # Scaling factors for selection gradient in backward pass
        include_block_causal, # Causal attention or not
        block_dk_dv_use_dot, # Optimisation parameter for the backward pass
        combine_sliding_window_out # Return sliding window attention result along with the compression + selection result
        ):

        dtype  = full_q.dtype
        q_heads, kv_heads = full_q.shape[1], full_k.shape[1]
        assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads" # In case of MQA or GQA

        head_groups = q_heads // kv_heads

        full_q, full_k, full_v = full_q.half(), full_k.half(), full_v.half()

        output, sliding_output, logsumexp, sliding_logsumexp  = native_sparse_attention_forward(
            full_q, full_k, full_v,
            selected_block_indices,
            fmask,
            block_size = block_size,
            include_block_causal = include_block_causal,
            combine_sliding_window_out = combine_sliding_window_out
        )

        ctx.save_for_backward(full_q, full_k, full_v, selected_block_indices, fmask, output, sliding_output, logsumexp, sliding_logsumexp)

        # Checking if gradient is needed during backward pass for block index selection operation
        if sel_scale is not None:
            assert (sel_scale == 1.).all(), "Since the selection operation is non differentiable for now we are assuming a straight through estimator with scale factor 1. grad_in = 1 * grad_out"

        ctx._saved_variables = (
            block_size,
            head_groups,
            fmask,
            sel_scale,
            include_block_causal,
            block_dk_dv_use_dot,
            combine_sliding_window_out
        )

        return output.type(dtype), sliding_output.type(dtype), logsumexp, sliding_logsumexp

_native_sparse_attention = NativeSparseAttention.apply

def native_sparse_attention(
    full_q : Float['b q_heads seq_len d'],
    full_k : Float['b kv_heads seq_len d'],
    full_v : Float['b kv_heads seq_len d'],
    block_size : int,
    selected_block_indices : Int['b q_heads seq_len sel_dim'] | Int['b kv_heads seq_len sel_dim'],
    fmask: Bool['b q_heads seq_len sel_dim'] | Bool['b kv_heads seq_len sel_dim'],
    sel_scale : Float['b q_heads seq_len sel_dim'] | Float['b kv_heads seq_len sel_dim'] | None = None,
    include_block_causal = True,
    return_logsumexp = False,
    block_dk_dv_use_dot = False,
    combine_sliding_window_out = False
  ):
    seq_len = full_q.shape[-2]

    q_heads, kv_heads, sel_heads = full_q.shape[1], full_k.shape[1], selected_block_indices.shape[1]

    assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads" # In case of MQA or GQA
    assert sel_heads in (q_heads, kv_heads), "selected_block_indices must be of shape (b, q_heads, seq_len, sel_dim) or (b, kv_heads, seq_len, sel_dim)"

    assert block_size >= 16, "Subblock size must be atleast 16 for now"

    # If dimensionality of selection indices doesn't match kv_heads, we need to expand the key and value tensors

    if kv_heads != sel_heads: # Which means sel_heads = q_heads due to earlier assertion
        full_k, full_v = tuple(repeat(t, 'b h ... -> b (h gh) ...', gh = q_heads // kv_heads) for t in (full_k, full_v))
        # [batch_size, kv_heads, seq_len, head_dim] -> [batch_size, q_heads, seq_len, head_dim]

    output, sliding_output, logsumexp, sliding_logsumexp = _native_sparse_attention(
        full_q, full_k, full_v,
        block_size,
        selected_block_indices,
        fmask,
        sel_scale,
        include_block_causal,
        block_dk_dv_use_dot,
        combine_sliding_window_out)

    if combine_sliding_window_out:
        output = (output, sliding_output)

    if not return_logsumexp:
        return output

    logsumexp = logsumexp[..., :seq_len]
    sliding_logsumexp = sliding_logsumexp[..., :seq_len]

    if combine_sliding_window_out:
        logsumexp = (logsumexp, sliding_logsumexp)

    return output, logsumexp

