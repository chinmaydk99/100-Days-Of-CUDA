import triton
import triton.language as tl
import torch

@triton.jit
def avg_pool3d_kernel(
    input_ptr, output_ptr,
    B, C, D, H, W,
    OD, OH, OW, KD, KH, KW,
    SD, SH, SW,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    stride_ob, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    channel_block = tl.program_id(1)

    c_offsets = tl.arange(0, BLOCK_C)
    c_idxs = channel_block * BLOCK_C + c_offsets

    for od in range(OD):
        for oh in range(OH):
            for ow in range(OW):
                d_start = od * SD
                h_start = oh * SH
                w_start = ow * SW

                acc = tl.zeros([BLOCK_C], dtype=tl.float32)

                for kd in range(KD):
                    for kh in range(KH):
                        for kw in range(KW):
                            d = d_start + kd
                            h = h_start + kh
                            w = w_start + kw

                            mask = (c_idxs < C) & (d < D) & (h < H) & (w < W)
                            input_offset = (
                                batch * stride_b +
                                c_idxs * stride_c +
                                d * stride_d +
                                h * stride_h +
                                w * stride_w
                            )
                            val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                            acc += val

                pooled = acc / (KD * KH * KW)
                output_offset = (
                    batch * stride_ob +
                    c_idxs * stride_oc +
                    od * stride_od +
                    oh * stride_oh +
                    ow * stride_ow
                )
                tl.store(output_ptr + output_offset, pooled, mask=c_idxs < C)
