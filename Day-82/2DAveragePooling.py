import triton
import triton.language as tl
import torch

@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    B, C, H, W,
    OH, OW, KH, KW,
    SH, SW,
    stride_b, stride_c, stride_h, stride_w,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    channel_block = tl.program_id(1)

    c_offsets = tl.arange(0, BLOCK_C)
    c_idxs = channel_block * BLOCK_C + c_offsets

    for oh in range(OH):
        for ow in range(OW):
            h_start = oh * SH
            w_start = ow * SW

            acc = tl.zeros([BLOCK_C], dtype=tl.float32)
            for kh in range(KH):
                for kw in range(KW):
                    h = h_start + kh
                    w = w_start + kw

                    mask = (c_idxs < C) & (h < H) & (w < W)
                    input_offset = (batch * stride_b +
                                    c_idxs * stride_c +
                                    h * stride_h +
                                    w * stride_w)
                    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                    acc += input_val

            avg_val = acc / (KH * KW)
            out_offset = (batch * stride_ob +
                          c_idxs * stride_oc +
                          oh * stride_oh +
                          ow * stride_ow)
            tl.store(output_ptr + out_offset, avg_val, mask=c_idxs < C)
