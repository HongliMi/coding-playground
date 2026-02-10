import os
# 指定 Triton 缓存目录到 scratch 空间（避免 home 目录空间不足）
os.environ["TRITON_CACHE_DIR"] = "/home/scratch.peiyuanz_gpu/mhl/.triton_cache"

import torch
import triton
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda
import json
import numpy as np
import os
import sys

# Add path for Triton reference kernel
sys.path.insert(0, "/home/scratch.peiyuanz_gpu/mhl/Personal_workspace/flash-linear-attention")
from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_intra_sub_chunk, IS_GATHER_SUPPORTED


# ===========================================================================
# TF32 MMA Inline PTX (m16n8k8)
# ===========================================================================
@dsl_user_op
def mma_tf32_m16n8k8(
    a0, a1, a2, a3,      # A: 4 TF32 registers
    b0, b1,              # B: 2 TF32 registers
    c0, c1, c2, c3,      # C accumulator: 4 FP32 registers
    *, loc=None, ip=None
):
    """TF32 MMA: D = A * B + C, shape m16n8k8"""
    a0_bits = llvm.bitcast(T.i32(), a0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a1_bits = llvm.bitcast(T.i32(), a1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a2_bits = llvm.bitcast(T.i32(), a2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a3_bits = llvm.bitcast(T.i32(), a3.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b0_bits = llvm.bitcast(T.i32(), b0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b1_bits = llvm.bitcast(T.i32(), b1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [a0_bits, a1_bits, a2_bits, a3_bits, b0_bits, b1_bits,
         c0.ir_value(loc=loc, ip=ip), c1.ir_value(loc=loc, ip=ip),
         c2.ir_value(loc=loc, ip=ip), c3.ir_value(loc=loc, ip=ip)],
        """{
            mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {$0, $1, $2, $3},
                {$4, $5, $6, $7},
                {$8, $9},
                {$10, $11, $12, $13};
        }""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    
    d0 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


# ===========================================================================
# Manual load/store for TF32 MMA (from mma_tf32_16x16.py)
# ===========================================================================
@dsl_user_op
def load_A_tf32(
    sA: cute.Tensor,  # (16, 8) FP32 in SMEM, row-major
    lane_id,
    *, loc=None, ip=None
):
    """
    Load A matrix (16x8) from SMEM to registers for TF32 MMA.
    
    Register layout (from PTX ISA):
    - group_id = lane_id / 4 (0-7)
    - tid_in_group = lane_id % 4 (0-3)
    - a0 = A[group_id,     tid_in_group]
    - a1 = A[group_id + 8, tid_in_group]
    - a2 = A[group_id,     tid_in_group + 4]
    - a3 = A[group_id + 8, tid_in_group + 4]
    """
    group_id = lane_id // 4
    tid_in_group = (lane_id % 4) * 2
    
    a0 = cutlass.Float32(sA[group_id, tid_in_group])
    a1 = cutlass.Float32(sA[group_id + 8, tid_in_group])
    a2 = cutlass.Float32(sA[group_id, tid_in_group + 1])
    a3 = cutlass.Float32(sA[group_id + 8, tid_in_group + 1])
    
    return a0, a1, a2, a3


@dsl_user_op
def load_B_tf32_from_rowmajor(
    sB: cute.Tensor,  # (8, 8) FP32 in SMEM, row-major
    lane_id,
    *, loc=None, ip=None
):
    """
    Load B matrix (8x8) from row-major SMEM to registers for TF32 MMA.
    MMA expects B in col-major, so we load transposed.
    
    For B in row-major: B[row, col]
    MMA wants col-major view, so we read with swapped indices.
    """
    group_id = lane_id // 4
    tid_in_group = (lane_id % 4) * 2
    
    # Load with transposed indices (swap row/col)
    b0 = cutlass.Float32(sB[group_id, tid_in_group])
    b1 = cutlass.Float32(sB[group_id, tid_in_group + 1])
    
    return b0, b1


@dsl_user_op
def store_C_tf32(
    sC: cute.Tensor,  # (16, 8) FP32 in SMEM, row-major
    c0, c1, c2, c3,   # 4 FP32 registers
    lane_id,
    *, loc=None, ip=None
):
    """
    Store C/D matrix (16x8) from registers to SMEM.
    
    Register layout:
    - group_id = lane_id / 4 (0-7)
    - tid_in_group = lane_id % 4 (0-3)
    - c0 -> C[group_id, tid_in_group * 2]
    - c1 -> C[group_id, tid_in_group * 2 + 1]
    - c2 -> C[group_id + 8, tid_in_group * 2]
    - c3 -> C[group_id + 8, tid_in_group * 2 + 1]
    """
    group_id = lane_id // 4
    tid_in_group = lane_id % 4
    
    sC[group_id, tid_in_group * 2] = c0
    sC[group_id, tid_in_group * 2 + 1] = c1
    sC[group_id + 8, tid_in_group * 2] = c2
    sC[group_id + 8, tid_in_group * 2 + 1] = c3



# ===========================================================================
# Helper: 16x16 diagonal block inversion (FP32 output)
# ===========================================================================
@dsl_user_op
def _invert_16x16_halfwarp_fp32(
    sA_in: cute.Tensor,   # 2D fp32 - input (unit lower triangular)
    sA_out: cute.Tensor,  # 2D fp32 - output (inverted)
    diag_offset,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    """Invert a 16x16 unit lower triangular block, write FP32 result."""
    my_row = lane_id % 16
    halfwarp_base = (lane_id // 16) * 16
    
    row_off = diag_offset
    col_off = diag_offset
    
    # Registers for computation (FP32)
    rInv = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    rA = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    
    rInv[0] = cutlass.Float32(1.0)
    for x in range(1, 16):
        rInv[x] = cutlass.Float32(0.0)
    for x in range(16):
        rA[x] = cutlass.Float32(0.0)
    
    # Compute inverse using anti-diagonal sweeps
    # d=1
    col1 = my_row - 1
    valid1 = cutlass.Float32(col1 >= 0)
    a_val1 = cutlass.Float32(sA_in[row_off + my_row, col_off + col1]) * valid1
    rA[1] = a_val1
    rInv[1] = -a_val1
    
    # d=2
    col2 = my_row - 2
    valid2 = cutlass.Float32(col2 >= 0)
    a_val2 = cutlass.Float32(sA_in[row_off + my_row, col_off + col2]) * valid2
    rA[2] = a_val2
    inv_from_row1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 1)
    sum2 = rA[1] * inv_from_row1
    rInv[2] = (-a_val2 - sum2) * valid2
    
    # d=3
    col3 = my_row - 3
    valid3 = cutlass.Float32(col3 >= 0)
    a_val3 = cutlass.Float32(sA_in[row_off + my_row, col_off + col3]) * valid3
    rA[3] = a_val3
    inv_d3_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 2)
    inv_d3_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 1)
    sum3 = rA[2] * inv_d3_k1 + rA[1] * inv_d3_k2
    rInv[3] = (-a_val3 - sum3) * valid3
    
    col4 = my_row - 4
    valid4 = cutlass.Float32(col4 >= 0)
    a_val4 = cutlass.Float32(sA_in[row_off + my_row, col_off + col4]) * valid4
    rA[4] = a_val4
    inv_d4_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 3)
    inv_d4_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 2)
    inv_d4_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 1)
    sum4 = rA[3] * inv_d4_k1 + rA[2] * inv_d4_k2 + rA[1] * inv_d4_k3
    rInv[4] = (-a_val4 - sum4) * valid4
    
    col5 = my_row - 5
    valid5 = cutlass.Float32(col5 >= 0)
    a_val5 = cutlass.Float32(sA_in[row_off + my_row, col_off + col5]) * valid5
    rA[5] = a_val5
    inv_d5_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 4)
    inv_d5_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 3)
    inv_d5_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 2)
    inv_d5_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 1)
    sum5 = rA[4] * inv_d5_k1 + rA[3] * inv_d5_k2 + rA[2] * inv_d5_k3 + rA[1] * inv_d5_k4
    rInv[5] = (-a_val5 - sum5) * valid5
    
    col6 = my_row - 6
    valid6 = cutlass.Float32(col6 >= 0)
    a_val6 = cutlass.Float32(sA_in[row_off + my_row, col_off + col6]) * valid6
    rA[6] = a_val6
    inv_d6_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 5)
    inv_d6_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 4)
    inv_d6_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 3)
    inv_d6_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 2)
    inv_d6_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 1)
    sum6 = rA[5] * inv_d6_k1 + rA[4] * inv_d6_k2 + rA[3] * inv_d6_k3 + rA[2] * inv_d6_k4 + rA[1] * inv_d6_k5
    rInv[6] = (-a_val6 - sum6) * valid6
    
    col7 = my_row - 7
    valid7 = cutlass.Float32(col7 >= 0)
    a_val7 = cutlass.Float32(sA_in[row_off + my_row, col_off + col7]) * valid7
    rA[7] = a_val7
    inv_d7_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 6)
    inv_d7_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 5)
    inv_d7_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 4)
    inv_d7_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 3)
    inv_d7_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 2)
    inv_d7_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 1)
    sum7 = rA[6] * inv_d7_k1 + rA[5] * inv_d7_k2 + rA[4] * inv_d7_k3 + rA[3] * inv_d7_k4 + rA[2] * inv_d7_k5 + rA[1] * inv_d7_k6
    rInv[7] = (-a_val7 - sum7) * valid7
    
    col8 = my_row - 8
    valid8 = cutlass.Float32(col8 >= 0)
    a_val8 = cutlass.Float32(sA_in[row_off + my_row, col_off + col8]) * valid8
    rA[8] = a_val8
    inv_d8_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 7)
    inv_d8_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 6)
    inv_d8_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 5)
    inv_d8_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 4)
    inv_d8_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 3)
    inv_d8_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 2)
    inv_d8_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 1)
    sum8 = rA[7] * inv_d8_k1 + rA[6] * inv_d8_k2 + rA[5] * inv_d8_k3 + rA[4] * inv_d8_k4 + rA[3] * inv_d8_k5 + rA[2] * inv_d8_k6 + rA[1] * inv_d8_k7
    rInv[8] = (-a_val8 - sum8) * valid8
    
    col9 = my_row - 9
    valid9 = cutlass.Float32(col9 >= 0)
    a_val9 = cutlass.Float32(sA_in[row_off + my_row, col_off + col9]) * valid9
    rA[9] = a_val9
    inv_d9_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 8)
    inv_d9_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 7)
    inv_d9_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 6)
    inv_d9_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 5)
    inv_d9_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 4)
    inv_d9_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 3)
    inv_d9_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 2)
    inv_d9_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 1)
    sum9 = rA[8] * inv_d9_k1 + rA[7] * inv_d9_k2 + rA[6] * inv_d9_k3 + rA[5] * inv_d9_k4 + rA[4] * inv_d9_k5 + rA[3] * inv_d9_k6 + rA[2] * inv_d9_k7 + rA[1] * inv_d9_k8
    rInv[9] = (-a_val9 - sum9) * valid9
    
    col10 = my_row - 10
    valid10 = cutlass.Float32(col10 >= 0)
    a_val10 = cutlass.Float32(sA_in[row_off + my_row, col_off + col10]) * valid10
    rA[10] = a_val10
    inv_d10_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 9)
    inv_d10_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 8)
    inv_d10_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 7)
    inv_d10_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 6)
    inv_d10_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 5)
    inv_d10_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 4)
    inv_d10_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 3)
    inv_d10_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 2)
    inv_d10_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 1)
    sum10 = rA[9] * inv_d10_k1 + rA[8] * inv_d10_k2 + rA[7] * inv_d10_k3 + rA[6] * inv_d10_k4 + rA[5] * inv_d10_k5 + rA[4] * inv_d10_k6 + rA[3] * inv_d10_k7 + rA[2] * inv_d10_k8 + rA[1] * inv_d10_k9
    rInv[10] = (-a_val10 - sum10) * valid10
    
    col11 = my_row - 11
    valid11 = cutlass.Float32(col11 >= 0)
    a_val11 = cutlass.Float32(sA_in[row_off + my_row, col_off + col11]) * valid11
    rA[11] = a_val11
    inv_d11_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 10)
    inv_d11_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 9)
    inv_d11_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 8)
    inv_d11_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 7)
    inv_d11_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 6)
    inv_d11_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 5)
    inv_d11_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 4)
    inv_d11_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 3)
    inv_d11_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 2)
    inv_d11_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 1)
    sum11 = rA[10] * inv_d11_k1 + rA[9] * inv_d11_k2 + rA[8] * inv_d11_k3 + rA[7] * inv_d11_k4 + rA[6] * inv_d11_k5 + rA[5] * inv_d11_k6 + rA[4] * inv_d11_k7 + rA[3] * inv_d11_k8 + rA[2] * inv_d11_k9 + rA[1] * inv_d11_k10
    rInv[11] = (-a_val11 - sum11) * valid11
    
    col12 = my_row - 12
    valid12 = cutlass.Float32(col12 >= 0)
    a_val12 = cutlass.Float32(sA_in[row_off + my_row, col_off + col12]) * valid12
    rA[12] = a_val12
    inv_d12_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 11)
    inv_d12_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 10)
    inv_d12_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 9)
    inv_d12_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 8)
    inv_d12_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 7)
    inv_d12_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 6)
    inv_d12_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 5)
    inv_d12_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 4)
    inv_d12_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 3)
    inv_d12_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 2)
    inv_d12_k11 = cute.arch.shuffle_sync(rInv[11], halfwarp_base + my_row - 1)
    sum12 = rA[11] * inv_d12_k1 + rA[10] * inv_d12_k2 + rA[9] * inv_d12_k3 + rA[8] * inv_d12_k4 + rA[7] * inv_d12_k5 + rA[6] * inv_d12_k6 + rA[5] * inv_d12_k7 + rA[4] * inv_d12_k8 + rA[3] * inv_d12_k9 + rA[2] * inv_d12_k10 + rA[1] * inv_d12_k11
    rInv[12] = (-a_val12 - sum12) * valid12
    
    col13 = my_row - 13
    valid13 = cutlass.Float32(col13 >= 0)
    a_val13 = cutlass.Float32(sA_in[row_off + my_row, col_off + col13]) * valid13
    rA[13] = a_val13
    inv_d13_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 12)
    inv_d13_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 11)
    inv_d13_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 10)
    inv_d13_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 9)
    inv_d13_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 8)
    inv_d13_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 7)
    inv_d13_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 6)
    inv_d13_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 5)
    inv_d13_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 4)
    inv_d13_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 3)
    inv_d13_k11 = cute.arch.shuffle_sync(rInv[11], halfwarp_base + my_row - 2)
    inv_d13_k12 = cute.arch.shuffle_sync(rInv[12], halfwarp_base + my_row - 1)
    sum13 = rA[12] * inv_d13_k1 + rA[11] * inv_d13_k2 + rA[10] * inv_d13_k3 + rA[9] * inv_d13_k4 + rA[8] * inv_d13_k5 + rA[7] * inv_d13_k6 + rA[6] * inv_d13_k7 + rA[5] * inv_d13_k8 + rA[4] * inv_d13_k9 + rA[3] * inv_d13_k10 + rA[2] * inv_d13_k11 + rA[1] * inv_d13_k12
    rInv[13] = (-a_val13 - sum13) * valid13
    
    col14 = my_row - 14
    valid14 = cutlass.Float32(col14 >= 0)
    a_val14 = cutlass.Float32(sA_in[row_off + my_row, col_off + col14]) * valid14
    rA[14] = a_val14
    inv_d14_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 13)
    inv_d14_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 12)
    inv_d14_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 11)
    inv_d14_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 10)
    inv_d14_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 9)
    inv_d14_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 8)
    inv_d14_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 7)
    inv_d14_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 6)
    inv_d14_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 5)
    inv_d14_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 4)
    inv_d14_k11 = cute.arch.shuffle_sync(rInv[11], halfwarp_base + my_row - 3)
    inv_d14_k12 = cute.arch.shuffle_sync(rInv[12], halfwarp_base + my_row - 2)
    inv_d14_k13 = cute.arch.shuffle_sync(rInv[13], halfwarp_base + my_row - 1)
    sum14 = rA[13] * inv_d14_k1 + rA[12] * inv_d14_k2 + rA[11] * inv_d14_k3 + rA[10] * inv_d14_k4 + rA[9] * inv_d14_k5 + rA[8] * inv_d14_k6 + rA[7] * inv_d14_k7 + rA[6] * inv_d14_k8 + rA[5] * inv_d14_k9 + rA[4] * inv_d14_k10 + rA[3] * inv_d14_k11 + rA[2] * inv_d14_k12 + rA[1] * inv_d14_k13
    rInv[14] = (-a_val14 - sum14) * valid14
    
    col15 = my_row - 15
    valid15 = cutlass.Float32(col15 >= 0)
    a_val15 = cutlass.Float32(sA_in[row_off + my_row, col_off + col15]) * valid15
    rA[15] = a_val15
    inv_d15_k1 = cute.arch.shuffle_sync(rInv[1], halfwarp_base + my_row - 14)
    inv_d15_k2 = cute.arch.shuffle_sync(rInv[2], halfwarp_base + my_row - 13)
    inv_d15_k3 = cute.arch.shuffle_sync(rInv[3], halfwarp_base + my_row - 12)
    inv_d15_k4 = cute.arch.shuffle_sync(rInv[4], halfwarp_base + my_row - 11)
    inv_d15_k5 = cute.arch.shuffle_sync(rInv[5], halfwarp_base + my_row - 10)
    inv_d15_k6 = cute.arch.shuffle_sync(rInv[6], halfwarp_base + my_row - 9)
    inv_d15_k7 = cute.arch.shuffle_sync(rInv[7], halfwarp_base + my_row - 8)
    inv_d15_k8 = cute.arch.shuffle_sync(rInv[8], halfwarp_base + my_row - 7)
    inv_d15_k9 = cute.arch.shuffle_sync(rInv[9], halfwarp_base + my_row - 6)
    inv_d15_k10 = cute.arch.shuffle_sync(rInv[10], halfwarp_base + my_row - 5)
    inv_d15_k11 = cute.arch.shuffle_sync(rInv[11], halfwarp_base + my_row - 4)
    inv_d15_k12 = cute.arch.shuffle_sync(rInv[12], halfwarp_base + my_row - 3)
    inv_d15_k13 = cute.arch.shuffle_sync(rInv[13], halfwarp_base + my_row - 2)
    inv_d15_k14 = cute.arch.shuffle_sync(rInv[14], halfwarp_base + my_row - 1)
    sum15 = rA[14] * inv_d15_k1 + rA[13] * inv_d15_k2 + rA[12] * inv_d15_k3 + rA[11] * inv_d15_k4 + rA[10] * inv_d15_k5 + rA[9] * inv_d15_k6 + rA[8] * inv_d15_k7 + rA[7] * inv_d15_k8 + rA[6] * inv_d15_k9 + rA[5] * inv_d15_k10 + rA[4] * inv_d15_k11 + rA[3] * inv_d15_k12 + rA[2] * inv_d15_k13 + rA[1] * inv_d15_k14
    rInv[15] = (-a_val15 - sum15) * valid15
    
    rInv[0] = cutlass.Float32(1.0)
    
    # Write to FP32 output
    sA_out[row_off + my_row, col_off + my_row] = rInv[0]
    sA_out[row_off + my_row, col_off + (my_row + 15) % 16] = rInv[1] * cutlass.Float32(my_row >= 1)
    sA_out[row_off + my_row, col_off + (my_row + 14) % 16] = rInv[2] * cutlass.Float32(my_row >= 2)
    sA_out[row_off + my_row, col_off + (my_row + 13) % 16] = rInv[3] * cutlass.Float32(my_row >= 3)
    sA_out[row_off + my_row, col_off + (my_row + 12) % 16] = rInv[4] * cutlass.Float32(my_row >= 4)
    sA_out[row_off + my_row, col_off + (my_row + 11) % 16] = rInv[5] * cutlass.Float32(my_row >= 5)
    sA_out[row_off + my_row, col_off + (my_row + 10) % 16] = rInv[6] * cutlass.Float32(my_row >= 6)
    sA_out[row_off + my_row, col_off + (my_row + 9) % 16] = rInv[7] * cutlass.Float32(my_row >= 7)
    sA_out[row_off + my_row, col_off + (my_row + 8) % 16] = rInv[8] * cutlass.Float32(my_row >= 8)
    sA_out[row_off + my_row, col_off + (my_row + 7) % 16] = rInv[9] * cutlass.Float32(my_row >= 9)
    sA_out[row_off + my_row, col_off + (my_row + 6) % 16] = rInv[10] * cutlass.Float32(my_row >= 10)
    sA_out[row_off + my_row, col_off + (my_row + 5) % 16] = rInv[11] * cutlass.Float32(my_row >= 11)
    sA_out[row_off + my_row, col_off + (my_row + 4) % 16] = rInv[12] * cutlass.Float32(my_row >= 12)
    sA_out[row_off + my_row, col_off + (my_row + 3) % 16] = rInv[13] * cutlass.Float32(my_row >= 13)
    sA_out[row_off + my_row, col_off + (my_row + 2) % 16] = rInv[14] * cutlass.Float32(my_row >= 14)
    sA_out[row_off + my_row, col_off + (my_row + 1) % 16] = rInv[15] * cutlass.Float32(my_row >= 15)


# 全局配置
BC = 16          # sub-chunk size (rows)
BK = 64          # key dimension (columns)
NUM_STAGES = 2   # double buffer
NUM_THREADS = 128
NUM_WARPS = 4

# 峰值带宽 (GB/s) - 根据不同 GPU 平台修改
# B200: 8000, H100 SXM: 3350, H100 PCIe: 2000, A100 SXM: 2039, A100 PCIe: 1555
PEAK_BW_GBS = 7672  # B200


@cute.kernel
def kda_Akk_kernel(
    tma_atom_g: cute.CopyAtom,
    tma_tensor_g: cute.Tensor,   # 5D view: [BC, BK, B*NT, H, num_k_tiles] float32
    tma_atom_q: cute.CopyAtom,
    tma_tensor_q: cute.Tensor,   # 5D view: [BC, BK, B*NT, H, num_k_tiles] fp16
    tma_atom_k: cute.CopyAtom,
    tma_tensor_k: cute.Tensor,   # 5D view: [BC, BK, B*NT, H, num_k_tiles] fp16
    g_smem_layout,               # 3D ComposedLayout with swizzle: (BC, BK, NUM_STAGES)
    qk_smem_layout,              # 3D ComposedLayout with swizzle: (BC, BK, NUM_STAGES * 2)
    beta_tensor: cute.Tensor,    # (B, T, H) fp16
    Akk_tensor: cute.Tensor,     # (B, T, H, BC) output
    Aqk_tensor: cute.Tensor,     # (B, T, H, BT) output
    accum_smem_layout: cute.Layout, # (BC, BC * 2, 2) for Aqk and Akk
    beta_smem_layout: cute.Layout, # (BT,)
    tiled_mma: cute.TiledMma,    # MMA configuration for ldmatrix
    tiled_copy_Q: cute.TiledCopy, # ldmatrix for Q (16x8)
    tiled_copy_K: cute.TiledCopy, # ldmatrix for K (8x8)
    tiled_copy_G: cute.TiledCopy, # s2r for G (16x8), MMA C layout: (8,4) threads, 2 vals each
    BT: cutlass.Constexpr[int],
    num_k_tiles: cutlass.Constexpr[int],
    seq_len: int,                # sequence length
    scale: cutlass.Float32,      # scale factor for Aqk
):
    """
    KDA Akk kernel using TMA for g/q/k loading.
    - Each block handles one (batch, chunk, head)
    - The block owns a [BT, K] chunk, tiled by (BC, BK)
      BT=64, BC=16 => 4 tiles along T; K=128, BK=64 => 2 tiles along K; total 8 tiles.
    - G: TMA load without swizzle -> direct indexing for element-wise
    - Q/K: TMA load with swizzle -> ldmatrix for MMA register loading
    """
    
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // 32
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_b, i_t, i_h = cute.arch.block_idx()  # batch, chunk, head
    
    # =============== Allocate shared memory ===============
    smem = cutlass.utils.SmemAllocator()
    
    # g: (BC, BK, NUM_STAGES) with swizzle - indexing handles swizzle automatically
    sG = smem.allocate_tensor(cutlass.Float32, g_smem_layout.outer, 128, swizzle=g_smem_layout.inner)
    
    # qk: (BC, BK, NUM_STAGES * 2) with swizzle, interleaved [q0][k0][q1][k1]
    # Use ldmatrix for loading to registers
    sQK = smem.allocate_tensor(cutlass.Float16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner)
    
    sAccum = smem.allocate_tensor(cutlass.Float32, accum_smem_layout, 128)
    sBeta = smem.allocate_tensor(cutlass.Float16, beta_smem_layout, 128)
    
    # Allocate mbarriers for TMA synchronization (one per stage)
    mbar_ptr = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    
    # TMA tile size in bytes
    g_tile_bytes = BC * BK * 4   # float32
    qk_tile_bytes = BC * BK * 2  # fp16
    total_tile_bytes = g_tile_bytes + qk_tile_bytes * 2  # g + q + k
    
    # Initialize mbarriers using pointer offset
    if tidx == 0:
        for stage in range(NUM_STAGES):
            cute.arch.mbarrier_init(mbar_ptr + stage, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    
    # Chunk start and tile base along T (each T tile is BC rows)
    t_start = i_t * BT
    t_tile_base = i_t * (BT // BC)  # base tile index in (T/BC) dimension
    
    # Compute global tile index offset: i_b * (T/BC) + t_tile_base
    num_t_tiles_per_batch = seq_len // BC
    g_tile_base = i_b * num_t_tiles_per_batch + t_tile_base
    
    # Load beta for this chunk: sBeta[0:BT] <- beta[t_start : t_start+BT]
    gBeta_batch = beta_tensor[(i_b, None, i_h)]   # (T,)
    if tidx < BT:
        sBeta[tidx] = gBeta_batch[t_start + tidx]
    cute.arch.barrier()
    
    # Iterate all tiles in the chunk:
    #   iter -> (t_sub, k_tile)
    #   t_sub in [0, BT/BC), k_tile in [0, num_k_tiles)
    num_t_tiles = BT // BC
    total_tiles = num_t_tiles * num_k_tiles

    # =============== Prefetch first stages using TMA ===============
    # IMPORTANT: Only warp 0 executes TMA copies to avoid multiple TMA requests
    prefetch_count = cutlass.min(NUM_STAGES - 1, total_tiles)
    if warp_idx == 0:
        for it in range(prefetch_count):
            stage = it % NUM_STAGES
            t_sub = it // num_k_tiles
            k_tile = it - t_sub * num_k_tiles
            
            # Global tile index in the 5D TMA tensor
            t_tile_idx = g_tile_base + t_sub
            
            # Expect bytes for this stage's mbarrier (using pointer offset)
            if tidx == 0:
                cute.arch.mbarrier_expect_tx(mbar_ptr + stage, total_tile_bytes)
            
            # TMA load g: global -> sG[(None, None, stage)]
            gG_src = cute.local_tile(tma_tensor_g, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx, i_h, k_tile))
            gG_src_2d = gG_src[(None, None, 0, 0, 0)]
            sG_stage = sG[(None, None, stage)]
            tma_sG, tma_gG = cpasync.tma_partition(
                tma_atom_g, 0, cute.make_layout(1),
                cute.group_modes(sG_stage, 0, 2),
                cute.group_modes(gG_src_2d, 0, 2),
            )
            cute.copy(tma_atom_g, tma_gG, tma_sG, tma_bar_ptr=mbar_ptr + stage)
            
            # TMA load q: global -> sQK[(None, None, stage * 2)]
            gQ_src = cute.local_tile(tma_tensor_q, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx, i_h, k_tile))
            gQ_src_2d = gQ_src[(None, None, 0, 0, 0)]
            sQ_stage = sQK[(None, None, stage * 2)]
            tma_sQ, tma_gQ = cpasync.tma_partition(
                tma_atom_q, 0, cute.make_layout(1),
                cute.group_modes(sQ_stage, 0, 2),
                cute.group_modes(gQ_src_2d, 0, 2),
            )
            cute.copy(tma_atom_q, tma_gQ, tma_sQ, tma_bar_ptr=mbar_ptr + stage)
            
            # TMA load k: global -> sQK[(None, None, stage * 2 + 1)]
            gK_src = cute.local_tile(tma_tensor_k, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx, i_h, k_tile))
            gK_src_2d = gK_src[(None, None, 0, 0, 0)]
            sK_stage = sQK[(None, None, stage * 2 + 1)]
            tma_sK, tma_gK = cpasync.tma_partition(
                tma_atom_k, 0, cute.make_layout(1),
                cute.group_modes(sK_stage, 0, 2),
                cute.group_modes(gK_src_2d, 0, 2),
            )
            cute.copy(tma_atom_k, tma_gK, tma_sK, tma_bar_ptr=mbar_ptr + stage)

    # =============== Main loop ===============
    # Accumulators defined outside loop - persist across k_tiles within same t_sub
    # Only need to write to sAccum once at k_tile == num_k_tiles - 1 for reduction
    aqk_c00_0 = cutlass.Float32(0.0)
    aqk_c00_1 = cutlass.Float32(0.0)
    aqk_c00_2 = cutlass.Float32(0.0)
    aqk_c00_3 = cutlass.Float32(0.0)
    aqk_c01_0 = cutlass.Float32(0.0)
    aqk_c01_1 = cutlass.Float32(0.0)
    aqk_c01_2 = cutlass.Float32(0.0)
    aqk_c01_3 = cutlass.Float32(0.0)
    
    akk_c00_0 = cutlass.Float32(0.0)
    akk_c00_1 = cutlass.Float32(0.0)
    akk_c00_2 = cutlass.Float32(0.0)
    akk_c00_3 = cutlass.Float32(0.0)
    akk_c01_0 = cutlass.Float32(0.0)
    akk_c01_1 = cutlass.Float32(0.0)
    akk_c01_2 = cutlass.Float32(0.0)
    akk_c01_3 = cutlass.Float32(0.0)
    
    for it in range(total_tiles):
        stage = it % NUM_STAGES
        phase = it // NUM_STAGES % 2

        # Arrive at current stage mbarrier (thread 0 only)
        if tidx == 0:
            cute.arch.mbarrier_arrive(mbar_ptr + stage)
        
        # Wait for current stage TMA data to be ready using mbarrier
        cute.arch.mbarrier_wait(mbar_ptr + stage, phase)

        # Issue next TMA loads (overlapped with compute)
        # IMPORTANT: Only warp 0 executes TMA copies
        next_it = it + prefetch_count
        if warp_idx == 0 and next_it < total_tiles:
            next_stage = next_it % NUM_STAGES
            t_sub_n = next_it // num_k_tiles
            k_tile_n = next_it - t_sub_n * num_k_tiles
            t_tile_idx_n = g_tile_base + t_sub_n
            
            # Expect bytes for next stage's mbarrier (using pointer offset)
            if tidx == 0:
                cute.arch.mbarrier_expect_tx(mbar_ptr + next_stage, total_tile_bytes)
            
            # TMA load next g
            gG_src_n = cute.local_tile(tma_tensor_g, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx_n, i_h, k_tile_n))
            gG_src_2d_n = gG_src_n[(None, None, 0, 0, 0)]
            sG_next = sG[(None, None, next_stage)]
            tma_sG_n, tma_gG_n = cpasync.tma_partition(
                tma_atom_g, 0, cute.make_layout(1),
                cute.group_modes(sG_next, 0, 2),
                cute.group_modes(gG_src_2d_n, 0, 2),
            )
            cute.copy(tma_atom_g, tma_gG_n, tma_sG_n, tma_bar_ptr=mbar_ptr + next_stage)
            
            # TMA load next q
            gQ_src_n = cute.local_tile(tma_tensor_q, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx_n, i_h, k_tile_n))
            gQ_src_2d_n = gQ_src_n[(None, None, 0, 0, 0)]
            sQ_next = sQK[(None, None, next_stage * 2)]
            tma_sQ_n, tma_gQ_n = cpasync.tma_partition(
                tma_atom_q, 0, cute.make_layout(1),
                cute.group_modes(sQ_next, 0, 2),
                cute.group_modes(gQ_src_2d_n, 0, 2),
            )
            cute.copy(tma_atom_q, tma_gQ_n, tma_sQ_n, tma_bar_ptr=mbar_ptr + next_stage)
            
            # TMA load next k
            gK_src_n = cute.local_tile(tma_tensor_k, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx_n, i_h, k_tile_n))
            gK_src_2d_n = gK_src_n[(None, None, 0, 0, 0)]
            sK_next = sQK[(None, None, next_stage * 2 + 1)]
            tma_sK_n, tma_gK_n = cpasync.tma_partition(
                tma_atom_k, 0, cute.make_layout(1),
                cute.group_modes(sK_next, 0, 2),
                cute.group_modes(gK_src_2d_n, 0, 2),
            )
            cute.copy(tma_atom_k, tma_gK_n, tma_sK_n, tma_bar_ptr=mbar_ptr + next_stage)

        # =============== Fused Element-wise + MMA using ldmatrix ===============
        # Strategy: Use ldmatrix to load Q/K from swizzled SMEM, load G directly (no swizzle),
        # compute element-wise in registers, then do MMA.
        #
        # MMA m16n8k8 layout:
        #   - group_id = lane_id // 4 (0-7)
        #   - tid_in_group = lane_id % 4 (0-3)
        #   - A[16,8]: a0=A[group_id, tid*2], a1=A[group_id+8, tid*2], a2=A[group_id, tid*2+1], a3=A[group_id+8, tid*2+1]
        #   - B[8,8]:  b0=B[group_id, tid*2], b1=B[group_id, tid*2+1]
        #
        # ldmatrix register mapping (from test_ldmatrix_layout.py):
        #   - tCrA[0] -> a0, tCrA[2] -> a1, tCrA[1] -> a2, tCrA[3] -> a3
        #   - tCrB[0] -> b0, tCrB[1] -> b1
        #
        # Each warp processes 2 k sub-tiles: warp0 -> k=0,1; warp1 -> k=2,3; etc.
        
        t_sub = it // num_k_tiles
        k_tile = it - t_sub * num_k_tiles
        t_abs_base = t_start + t_sub * BC
        
        # gn_row for normalization
        gn_row = cutlass.min(BC // 2, cutlass.max(0, seq_len - t_abs_base - 1))
        
        # Thread/warp mapping for MMA
        warp_id = tidx // 32
        lane_id = tidx % 32
        group_id = lane_id // 4      # 0-7, determines rows (group_id and group_id+8)
        tid_in_group = lane_id % 4   # 0-3, determines cols within 8-col tile (tid*2 and tid*2+1)
        k_start = warp_id * 2        # warp 0 -> k=0,1; warp 1 -> k=2,3; etc.
        
        # Get thread slices for ldmatrix and G copy
        thr_mma = tiled_mma.get_slice(lane_id)
        thr_copy_Q = tiled_copy_Q.get_slice(lane_id)
        thr_copy_K = tiled_copy_K.get_slice(lane_id)
        thr_copy_G = tiled_copy_G.get_slice(lane_id)
        
        # ========== Reset accumulators at k_tile == 0 (new t_sub) ==========
        if k_tile == 0:
            aqk_c00_0 = cutlass.Float32(0.0)
            aqk_c00_1 = cutlass.Float32(0.0)
            aqk_c00_2 = cutlass.Float32(0.0)
            aqk_c00_3 = cutlass.Float32(0.0)
            aqk_c01_0 = cutlass.Float32(0.0)
            aqk_c01_1 = cutlass.Float32(0.0)
            aqk_c01_2 = cutlass.Float32(0.0)
            aqk_c01_3 = cutlass.Float32(0.0)
            
            akk_c00_0 = cutlass.Float32(0.0)
            akk_c00_1 = cutlass.Float32(0.0)
            akk_c00_2 = cutlass.Float32(0.0)
            akk_c00_3 = cutlass.Float32(0.0)
            akk_c01_0 = cutlass.Float32(0.0)
            akk_c01_1 = cutlass.Float32(0.0)
            akk_c01_2 = cutlass.Float32(0.0)
            akk_c01_3 = cutlass.Float32(0.0)
        
        # Get 2D SMEM views for current stage
        sQ_stage = sQK[(None, None, stage * 2)]       # Q: (BC, BK)
        sK_stage = sQK[(None, None, stage * 2 + 1)]   # K: (BC, BK)
        sG_stage = sG[(None, None, stage)]            # G: (BC, BK)
        
        # ========== Fused element-wise + MMA loop using ldmatrix ==========
        for k_iter in cutlass.range_constexpr(2):
            k = k_start + k_iter
            k_offset = k * 8  # column offset for this k sub-tile
            col = tid_in_group * 2  # 0,2,4,6 within each 8-col tile
            
            # Row indices for MMA layout
            r0 = group_id       # rows 0-7
            r1 = group_id + 8   # rows 8-15
            
            # ===== Load Q tile (16x8) via ldmatrix =====
            sQ_tile = cute.local_tile(sQ_stage, tiler=(16, 8), coord=(0, k))
            tCsQ = thr_mma.partition_A(sQ_tile)
            tCrQ = tiled_mma.make_fragment_A(tCsQ)
            tCsQ_view = thr_copy_Q.partition_S(sQ_tile)
            tCrQ_view = thr_copy_Q.retile(tCrQ)
            cute.copy(tiled_copy_Q, tCsQ_view, tCrQ_view)
            
            # ===== Load K tile 0 (rows 0-7, 8x8) via ldmatrix =====
            sK_tile_n0 = cute.local_tile(sK_stage, tiler=(8, 8), coord=(0, k))
            tCsK_n0 = thr_mma.partition_B(sK_tile_n0)
            tCrK_n0 = tiled_mma.make_fragment_B(tCsK_n0)
            tCsK_n0_view = thr_copy_K.partition_S(sK_tile_n0)
            tCrK_n0_view = thr_copy_K.retile(tCrK_n0)
            cute.copy(tiled_copy_K, tCsK_n0_view, tCrK_n0_view)
            
            # ===== Load K tile 1 (rows 8-15, 8x8) via ldmatrix =====
            sK_tile_n1 = cute.local_tile(sK_stage, tiler=(8, 8), coord=(1, k))
            tCsK_n1 = thr_mma.partition_B(sK_tile_n1)
            tCrK_n1 = tiled_mma.make_fragment_B(tCsK_n1)
            tCsK_n1_view = thr_copy_K.partition_S(sK_tile_n1)
            tCrK_n1_view = thr_copy_K.retile(tCrK_n1)
            cute.copy(tiled_copy_K, tCsK_n1_view, tCrK_n1_view)
            
            # ===== Load G tile (16x8) via tiled copy: (8,4) threads, 2 vals each =====
            gn_0 = sG_stage[(gn_row, k_offset + col)]
            gn_1 = sG_stage[(gn_row, k_offset + col + 1)]

            sG_tile = cute.local_tile(sG_stage, tiler=(16, 8), coord=(0, k))
            tCsG = thr_mma.partition_C(sG_tile)
            tCrG = tiled_mma.make_fragment_C(tCsG)
            cute.copy(tiled_copy_G, thr_copy_G.partition_S(sG_tile), thr_copy_G.retile(tCrG))

            # ===== Compute exp2 factors =====
            # tCrG layout (MMA C/D m16n8k8):
            #   [0] = G[group_id, tid*2]
            #   [1] = G[group_id, tid*2+1]
            #   [2] = G[group_id+8, tid*2]
            #   [3] = G[group_id+8, tid*2+1]
            b_gm_r0_0 = tCrG[0] - gn_0
            b_gm_r0_1 = tCrG[1] - gn_1
            b_gm_r1_0 = tCrG[2] - gn_0
            b_gm_r1_1 = tCrG[3] - gn_1
            
            gq_r0_0 = cute.math.exp2(b_gm_r0_0)
            gq_r0_1 = cute.math.exp2(b_gm_r0_1)
            gq_r1_0 = cute.math.exp2(b_gm_r1_0)
            gq_r1_1 = cute.math.exp2(b_gm_r1_1)
            gk_r0_0 = cute.math.exp2(-b_gm_r0_0)
            gk_r0_1 = cute.math.exp2(-b_gm_r0_1)
            gk_r1_0 = cute.math.exp2(-b_gm_r1_0)
            gk_r1_1 = cute.math.exp2(-b_gm_r1_1)
            
            # ===== Extract Q values from ldmatrix registers =====
            # ldmatrix layout: tCrQ[0]->q[r0,col], tCrQ[2]->q[r1,col], tCrQ[1]->q[r0,col+1], tCrQ[3]->q[r1,col+1]
            q_r0_0 = tCrQ[0].to(cutlass.Float32)
            q_r1_0 = tCrQ[2].to(cutlass.Float32)
            q_r0_1 = tCrQ[1].to(cutlass.Float32)
            q_r1_1 = tCrQ[3].to(cutlass.Float32)
            
            # ===== Extract K values from ldmatrix registers =====
            # K tile 0 (rows 0-7): tCrK_n0[0]->k[r0,col], tCrK_n0[1]->k[r0,col+1]
            k_r0_0 = tCrK_n0[0].to(cutlass.Float32)
            k_r0_1 = tCrK_n0[1].to(cutlass.Float32)
            # K tile 1 (rows 8-15): tCrK_n1[0]->k[r1,col], tCrK_n1[1]->k[r1,col+1]
            k_r1_0 = tCrK_n1[0].to(cutlass.Float32)
            k_r1_1 = tCrK_n1[1].to(cutlass.Float32)
            
            # ===== Element-wise multiply for A matrices =====
            # A for Aqk (q * gq): MMA expects a0, a1, a2, a3 order
            a_aqk_0 = q_r0_0 * gq_r0_0  # a0: q[r0,col] * gq
            a_aqk_1 = q_r1_0 * gq_r1_0  # a1: q[r1,col] * gq
            a_aqk_2 = q_r0_1 * gq_r0_1  # a2: q[r0,col+1] * gq
            a_aqk_3 = q_r1_1 * gq_r1_1  # a3: q[r1,col+1] * gq
            
            # A for Akk (k * gq): MMA expects a0, a1, a2, a3 order
            a_akk_0 = k_r0_0 * gq_r0_0  # a0: k[r0,col] * gq
            a_akk_1 = k_r1_0 * gq_r1_0  # a1: k[r1,col] * gq
            a_akk_2 = k_r0_1 * gq_r0_1  # a2: k[r0,col+1] * gq
            a_akk_3 = k_r1_1 * gq_r1_1  # a3: k[r1,col+1] * gq
            
            # ===== Element-wise multiply for B matrices (k * gk) =====
            # B_tile_0 (rows 0-7): b0, b1
            b0_0 = k_r0_0 * gk_r0_0  # b0: k[r0,col] * gk
            b1_0 = k_r0_1 * gk_r0_1  # b1: k[r0,col+1] * gk
            # B_tile_1 (rows 8-15): b0, b1
            b0_1 = k_r1_0 * gk_r1_0  # b0: k[r1,col] * gk
            b1_1 = k_r1_1 * gk_r1_1  # b1: k[r1,col+1] * gk
            
            # ===== MMA directly with computed values =====
            # Aqk MMA: A[16,8] = q*gq, B[8,8] = k*gk (2 tiles for n=16)
            aqk_c00_0, aqk_c00_1, aqk_c00_2, aqk_c00_3 = mma_tf32_m16n8k8(a_aqk_0, a_aqk_1, a_aqk_2, a_aqk_3, b0_0, b1_0, aqk_c00_0, aqk_c00_1, aqk_c00_2, aqk_c00_3)
            aqk_c01_0, aqk_c01_1, aqk_c01_2, aqk_c01_3 = mma_tf32_m16n8k8(a_aqk_0, a_aqk_1, a_aqk_2, a_aqk_3, b0_1, b1_1, aqk_c01_0, aqk_c01_1, aqk_c01_2, aqk_c01_3)
            
            # Akk MMA: A[16,8] = k*gq, B[8,8] = k*gk (reuse B tiles!)
            akk_c00_0, akk_c00_1, akk_c00_2, akk_c00_3 = mma_tf32_m16n8k8(a_akk_0, a_akk_1, a_akk_2, a_akk_3, b0_0, b1_0, akk_c00_0, akk_c00_1, akk_c00_2, akk_c00_3)
            akk_c01_0, akk_c01_1, akk_c01_2, akk_c01_3 = mma_tf32_m16n8k8(a_akk_0, a_akk_1, a_akk_2, a_akk_3, b0_1, b1_1, akk_c01_0, akk_c01_1, akk_c01_2, akk_c01_3)
        
        # 6. Tree Reduction: only after processing all k_tiles for this t_sub
        # Uses half SMEM: only need (BC, BC*2, 2) instead of (BC, BC*4, 2)
        # 
        # Tree reduction strategy:
        #   Phase 1: Warps 0,1 write to sAccum
        #   Phase 2: Warps 2,3 load from sAccum, add to their registers
        #   Phase 3: Warp 2 writes (warp0+warp2) to sAccum
        #   Phase 4: Warp 3 loads, adds to get final (warp0+warp1+warp2+warp3)
        #   Phase 5: Warp 3 writes final to sAccum
        #   Phase 6: All warps read and write to global
        
        if k_tile == num_k_tiles - 1:
            t_abs_base = t_start + t_sub * BC
            
            # ========== Phase 1: Warps 0,1 write to sAccum ==========
            if warp_id < 2:
                col_base = warp_id * BC  # warp 0: 0, warp 1: 16
                sAccum[(group_id, col_base + tid_in_group * 2, 0)] = aqk_c00_0
                sAccum[(group_id, col_base + tid_in_group * 2 + 1, 0)] = aqk_c00_1
                sAccum[(group_id + 8, col_base + tid_in_group * 2, 0)] = aqk_c00_2
                sAccum[(group_id + 8, col_base + tid_in_group * 2 + 1, 0)] = aqk_c00_3
                sAccum[(group_id, col_base + 8 + tid_in_group * 2, 0)] = aqk_c01_0
                sAccum[(group_id, col_base + 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_1
                sAccum[(group_id + 8, col_base + 8 + tid_in_group * 2, 0)] = aqk_c01_2
                sAccum[(group_id + 8, col_base + 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_3
                
                sAccum[(group_id, col_base + tid_in_group * 2, 1)] = akk_c00_0
                sAccum[(group_id, col_base + tid_in_group * 2 + 1, 1)] = akk_c00_1
                sAccum[(group_id + 8, col_base + tid_in_group * 2, 1)] = akk_c00_2
                sAccum[(group_id + 8, col_base + tid_in_group * 2 + 1, 1)] = akk_c00_3
                sAccum[(group_id, col_base + 8 + tid_in_group * 2, 1)] = akk_c01_0
                sAccum[(group_id, col_base + 8 + tid_in_group * 2 + 1, 1)] = akk_c01_1
                sAccum[(group_id + 8, col_base + 8 + tid_in_group * 2, 1)] = akk_c01_2
                sAccum[(group_id + 8, col_base + 8 + tid_in_group * 2 + 1, 1)] = akk_c01_3
            
            cute.arch.barrier()
            
            # ========== Phase 2: Warps 2,3 load and add to registers ==========
            # Warp 2 loads warp 0's data (cols 0-15), warp 3 loads warp 1's data (cols 16-31)
            if warp_id >= 2:
                partner_col_base = (warp_id - 2) * BC  # warp 2: 0, warp 3: 16
                aqk_c00_0 = aqk_c00_0 + sAccum[(group_id, partner_col_base + tid_in_group * 2, 0)]
                aqk_c00_1 = aqk_c00_1 + sAccum[(group_id, partner_col_base + tid_in_group * 2 + 1, 0)]
                aqk_c00_2 = aqk_c00_2 + sAccum[(group_id + 8, partner_col_base + tid_in_group * 2, 0)]
                aqk_c00_3 = aqk_c00_3 + sAccum[(group_id + 8, partner_col_base + tid_in_group * 2 + 1, 0)]
                aqk_c01_0 = aqk_c01_0 + sAccum[(group_id, partner_col_base + 8 + tid_in_group * 2, 0)]
                aqk_c01_1 = aqk_c01_1 + sAccum[(group_id, partner_col_base + 8 + tid_in_group * 2 + 1, 0)]
                aqk_c01_2 = aqk_c01_2 + sAccum[(group_id + 8, partner_col_base + 8 + tid_in_group * 2, 0)]
                aqk_c01_3 = aqk_c01_3 + sAccum[(group_id + 8, partner_col_base + 8 + tid_in_group * 2 + 1, 0)]
                
                akk_c00_0 = akk_c00_0 + sAccum[(group_id, partner_col_base + tid_in_group * 2, 1)]
                akk_c00_1 = akk_c00_1 + sAccum[(group_id, partner_col_base + tid_in_group * 2 + 1, 1)]
                akk_c00_2 = akk_c00_2 + sAccum[(group_id + 8, partner_col_base + tid_in_group * 2, 1)]
                akk_c00_3 = akk_c00_3 + sAccum[(group_id + 8, partner_col_base + tid_in_group * 2 + 1, 1)]
                akk_c01_0 = akk_c01_0 + sAccum[(group_id, partner_col_base + 8 + tid_in_group * 2, 1)]
                akk_c01_1 = akk_c01_1 + sAccum[(group_id, partner_col_base + 8 + tid_in_group * 2 + 1, 1)]
                akk_c01_2 = akk_c01_2 + sAccum[(group_id + 8, partner_col_base + 8 + tid_in_group * 2, 1)]
                akk_c01_3 = akk_c01_3 + sAccum[(group_id + 8, partner_col_base + 8 + tid_in_group * 2 + 1, 1)]
            
            cute.arch.barrier()
            
            # ========== Phase 3: Warp 2 writes (warp0+warp2) to sAccum cols 0-15 ==========
            if warp_id == 2:
                sAccum[(group_id, tid_in_group * 2, 0)] = aqk_c00_0
                sAccum[(group_id, tid_in_group * 2 + 1, 0)] = aqk_c00_1
                sAccum[(group_id + 8, tid_in_group * 2, 0)] = aqk_c00_2
                sAccum[(group_id + 8, tid_in_group * 2 + 1, 0)] = aqk_c00_3
                sAccum[(group_id, 8 + tid_in_group * 2, 0)] = aqk_c01_0
                sAccum[(group_id, 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_1
                sAccum[(group_id + 8, 8 + tid_in_group * 2, 0)] = aqk_c01_2
                sAccum[(group_id + 8, 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_3
                
                sAccum[(group_id, tid_in_group * 2, 1)] = akk_c00_0
                sAccum[(group_id, tid_in_group * 2 + 1, 1)] = akk_c00_1
                sAccum[(group_id + 8, tid_in_group * 2, 1)] = akk_c00_2
                sAccum[(group_id + 8, tid_in_group * 2 + 1, 1)] = akk_c00_3
                sAccum[(group_id, 8 + tid_in_group * 2, 1)] = akk_c01_0
                sAccum[(group_id, 8 + tid_in_group * 2 + 1, 1)] = akk_c01_1
                sAccum[(group_id + 8, 8 + tid_in_group * 2, 1)] = akk_c01_2
                sAccum[(group_id + 8, 8 + tid_in_group * 2 + 1, 1)] = akk_c01_3
            
            cute.arch.barrier()
            
            # ========== Phase 4: Warp 3 loads (warp0+warp2) and adds to get final ==========
            if warp_id == 3:
                aqk_c00_0 = aqk_c00_0 + sAccum[(group_id, tid_in_group * 2, 0)]
                aqk_c00_1 = aqk_c00_1 + sAccum[(group_id, tid_in_group * 2 + 1, 0)]
                aqk_c00_2 = aqk_c00_2 + sAccum[(group_id + 8, tid_in_group * 2, 0)]
                aqk_c00_3 = aqk_c00_3 + sAccum[(group_id + 8, tid_in_group * 2 + 1, 0)]
                aqk_c01_0 = aqk_c01_0 + sAccum[(group_id, 8 + tid_in_group * 2, 0)]
                aqk_c01_1 = aqk_c01_1 + sAccum[(group_id, 8 + tid_in_group * 2 + 1, 0)]
                aqk_c01_2 = aqk_c01_2 + sAccum[(group_id + 8, 8 + tid_in_group * 2, 0)]
                aqk_c01_3 = aqk_c01_3 + sAccum[(group_id + 8, 8 + tid_in_group * 2 + 1, 0)]
                
                akk_c00_0 = akk_c00_0 + sAccum[(group_id, tid_in_group * 2, 1)]
                akk_c00_1 = akk_c00_1 + sAccum[(group_id, tid_in_group * 2 + 1, 1)]
                akk_c00_2 = akk_c00_2 + sAccum[(group_id + 8, tid_in_group * 2, 1)]
                akk_c00_3 = akk_c00_3 + sAccum[(group_id + 8, tid_in_group * 2 + 1, 1)]
                akk_c01_0 = akk_c01_0 + sAccum[(group_id, 8 + tid_in_group * 2, 1)]
                akk_c01_1 = akk_c01_1 + sAccum[(group_id, 8 + tid_in_group * 2 + 1, 1)]
                akk_c01_2 = akk_c01_2 + sAccum[(group_id + 8, 8 + tid_in_group * 2, 1)]
                akk_c01_3 = akk_c01_3 + sAccum[(group_id + 8, 8 + tid_in_group * 2 + 1, 1)]
            
            cute.arch.barrier()
            
            # ========== Phase 5: Warp 3 writes Aqk to slot 0 + prepared Akk to slot 1 ==========
            if warp_id == 3:
                # Aqk: raw values to slot 0
                sAccum[(group_id, tid_in_group * 2, 0)] = aqk_c00_0
                sAccum[(group_id, tid_in_group * 2 + 1, 0)] = aqk_c00_1
                sAccum[(group_id + 8, tid_in_group * 2, 0)] = aqk_c00_2
                sAccum[(group_id + 8, tid_in_group * 2 + 1, 0)] = aqk_c00_3
                sAccum[(group_id, 8 + tid_in_group * 2, 0)] = aqk_c01_0
                sAccum[(group_id, 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_1
                sAccum[(group_id + 8, 8 + tid_in_group * 2, 0)] = aqk_c01_2
                sAccum[(group_id + 8, 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_3
                
                # Akk: prepare unit lower triangular (diag=1, lower=val*beta, upper=0)
                # Branchless: prepared = val * beta * float(row > col) + float(row == col)
                r0 = group_id              # rows 0-7
                r1 = group_id + 8          # rows 8-15
                c0 = tid_in_group * 2      # cols 0,2,4,6
                c1 = tid_in_group * 2 + 1  # cols 1,3,5,7
                c2 = 8 + tid_in_group * 2  # cols 8,10,12,14
                c3 = 8 + tid_in_group * 2 + 1  # cols 9,11,13,15
                beta_r0 = cutlass.Float32(sBeta[(t_sub * BC + r0,)])
                beta_r1 = cutlass.Float32(sBeta[(t_sub * BC + r1,)])
                
                sAccum[(r0, c0, 1)] = akk_c00_0 * beta_r0 * cutlass.Float32(r0 > c0) + cutlass.Float32(r0 == c0)
                sAccum[(r0, c1, 1)] = akk_c00_1 * beta_r0 * cutlass.Float32(r0 > c1) + cutlass.Float32(r0 == c1)
                sAccum[(r1, c0, 1)] = akk_c00_2 * beta_r1 * cutlass.Float32(r1 > c0) + cutlass.Float32(r1 == c0)
                sAccum[(r1, c1, 1)] = akk_c00_3 * beta_r1 * cutlass.Float32(r1 > c1) + cutlass.Float32(r1 == c1)
                sAccum[(r0, c2, 1)] = akk_c01_0 * beta_r0 * cutlass.Float32(r0 > c2) + cutlass.Float32(r0 == c2)
                sAccum[(r0, c3, 1)] = akk_c01_1 * beta_r0 * cutlass.Float32(r0 > c3) + cutlass.Float32(r0 == c3)
                sAccum[(r1, c2, 1)] = akk_c01_2 * beta_r1 * cutlass.Float32(r1 > c2) + cutlass.Float32(r1 == c2)
                sAccum[(r1, c3, 1)] = akk_c01_3 * beta_r1 * cutlass.Float32(r1 > c3) + cutlass.Float32(r1 == c3)
            
            cute.arch.barrier()
            
            # ========== Phase 6a: Write Aqk from slot 0 to global ==========
            for local_row in cutlass.range_constexpr(4):
                row = warp_id * 4 + local_row
                if lane_id < 16:
                    col = lane_id
                    final_aqk = sAccum[(row, col, 0)]
                    val_aqk_out = cutlass.Float32(0.0)
                    if row >= col:
                        val_aqk_out = final_aqk * scale
                    Aqk_tensor[(i_b, t_abs_base + row, i_h, t_sub * BC + col)] = cutlass.Float16(val_aqk_out)
            
            cute.arch.barrier()
            
            # ========== Phase 6b: Warp 0 inverts 16x16 Akk (slot 1 -> slot 0) ==========
            if warp_id == 0:
                _invert_16x16_halfwarp_fp32(sAccum[(None, None, 1)], sAccum[(None, None, 0)], 0, lane_id)
            
            cute.arch.barrier()
            
            # ========== Phase 6c: Write inverted Akk from slot 0 to global ==========
            for local_row in cutlass.range_constexpr(4):
                row = warp_id * 4 + local_row
                if lane_id < 16:
                    col = lane_id
                    Akk_tensor[(i_b, t_abs_base + row, i_h, col)] = sAccum[(row, col, 0)]

        cute.arch.barrier()


@cute.jit
def run_kda_Akk(
    g_tensor: cute.Tensor,
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    beta_tensor: cute.Tensor,
    Akk_tensor: cute.Tensor,
    Aqk_tensor: cute.Tensor,
    scale: float,
    stream: cuda.CUstream
):
    B, seq_len, H, K = g_tensor.layout.shape
    BT = 64  # chunk size
    
    # Number of tiles in K dimension
    num_k_tiles = cute.ceil_div(K, BK)
    NT = cute.ceil_div(seq_len, BT)  # number of chunks
    
    # =============== Create TMA for g/q/k ===============
    # Original shape: (B, T, H, K), stride: (T*H*K, H*K, K, 1)
    # 5D view: [BC, BK, B*(T/BC), H, K/BK] = [16, 64, B*seq_len/16, H, 2]
    # This allows TMA to load a (BC, BK) tile per operation
    
    num_t_tiles_per_batch = seq_len // BC  # T/BC per batch
    num_t_tiles_total = B * num_t_tiles_per_batch  # total T tiles
    
    # Stride calculation for 5D view:
    # dim 0 (BC rows): stride = H*K (move one row in T dimension)
    # dim 1 (BK cols): stride = 1 (move one col in K dimension)
    # dim 2 (T tiles): stride = BC*H*K (move one BC-sized chunk in T)
    # dim 3 (H): stride = K
    # dim 4 (K tiles): stride = BK (select first or second half of K)
    s_bc = H * K
    s_bk = 1
    s_t_tile = BC * H * K
    s_h = K
    s_k_tile = BK
    
    gqk_view_layout = cute.make_layout(
        (BC, BK, num_t_tiles_total, H, num_k_tiles),
        stride=(s_bc, s_bk, s_t_tile, s_h, s_k_tile)
    )
    
    # Create view tensors for g, q and k
    g_view = cute.make_tensor(g_tensor.iterator, gqk_view_layout)
    q_view = cute.make_tensor(q_tensor.iterator, gqk_view_layout)
    k_view = cute.make_tensor(k_tensor.iterator, gqk_view_layout)
    
    # =============== SMEM layouts ===============
    # G: K-major layout with custom swizzle S<2,5,2> for Float32
    # For Float32 (32-bit), num_contiguous_bits = 1024 -> num_contiguous_elems = 1024 // 32 = 32
    # K-major: (8, 32), stride=(32, 1) - 8 rows, 32 cols, column-major within tile
    sw_g = cute.make_swizzle(2, 5, 2)  # S<2,5,2> like MN_SW128_32B but for K-major
    outer_g = cute.make_layout((8, 32), stride=(32, 1))  # K-major layout
    smem_atom_g = cute.make_composed_layout(sw_g, 0, outer_g)
    g_smem_layout_2d = cute.tile_to_shape(smem_atom_g, (BC, BK), order=(0, 1))
    g_smem_layout = cute.tile_to_shape(smem_atom_g, (BC, BK, NUM_STAGES), order=(0, 1, 2))

    print(f"g_smem_layout: {g_smem_layout}")
    print(f"g_smem_atom: {smem_atom_g}")
    print(f"g_smem_atom_2d: {g_smem_layout_2d}")
    
    # Q/K: Swizzled layout for TMA + ldmatrix
    smem_atom_qk = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float16)
    qk_smem_layout_2d = cute.tile_to_shape(smem_atom_qk, (BC, BK), order=(0, 1))
    
    # qk: (BC, BK, NUM_STAGES * 2) fp16, interleaved [q0][k0][q1][k1]
    qk_smem_layout = cute.tile_to_shape(smem_atom_qk, (BC, BK, NUM_STAGES * 2), order=(0, 1, 2))
    
    # TMA load atoms for g, q and k
    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
    
    # G: TMA with swizzle
    tma_atom_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
        tma_load_op, g_view, g_smem_layout_2d, cute.product_each(g_smem_layout_2d.shape), num_multicast=1
    )
    # Q/K: TMA with swizzle
    tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
        tma_load_op, q_view, qk_smem_layout_2d, cute.product_each(qk_smem_layout_2d.shape), num_multicast=1
    )
    tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
        tma_load_op, k_view, qk_smem_layout_2d, cute.product_each(qk_smem_layout_2d.shape), num_multicast=1
    )
    
    # =============== MMA and ldmatrix configuration ===============
    # MMA m16n8k8 for TF32
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 8))
    tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 8))
    
    # LdMatrix for Q (16x8 tile, 2 loads)
    atom_copy_Q = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.Float16)
    tiled_copy_Q = cute.make_tiled_copy_A(atom_copy_Q, tiled_mma)
    
    # LdMatrix for K (8x8 tile, 1 load)
    atom_copy_K = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 1), cutlass.Float16)
    tiled_copy_K = cute.make_tiled_copy_B(atom_copy_K, tiled_mma)
    
    # TiledCopy for G: SMEM->Register matching MMA C/D layout
    # Thread layout: (8,4), each copy instruction loads 2 Float32 (64 bits)
    # Covers a (16,8) tile in 2 rounds per thread
    copy_atom_G_s2r = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=64  # 2 x Float32 = 64 bits
    )
    tiled_copy_G_s2r = cute.make_tiled_copy_C(copy_atom_G_s2r, tiled_mma)
    
    # =============== SMEM layouts ===============
    # g/q/k use swizzle layouts defined above (g_smem_layout_swizzle, qk_smem_layout_swizzle)
    # Only need to define accum and beta layouts here
    
    # accum: (BC, BC * 2, 2) for Aqk (slot 0) and Akk (slot 1)
    # With tree reduction, only need space for 2 warps
    # Add 8-col padding to avoid bank conflict
    ACCUM_PAD = 8
    accum_smem_layout = cute.make_layout(
        (BC, BC * 2, 2),
        stride=(BC * 2 + ACCUM_PAD, 1, BC * (BC * 2 + ACCUM_PAD))
    )

    # beta: (BT,) fp16
    beta_smem_layout = cute.make_layout(
        (BT,),
        stride=(1,)
    )
    
    # SMEM size calculation
    # g: 16 * 64 * NUM_STAGES * 4 bytes (with swizzle)
    # qk (fp16): 16 * 64 * (NUM_STAGES * 2) * 2 bytes for q and k interleaved (with swizzle)
    # accum: (BC, BC*2+8, 2) = 16 * 40 * 2 * 4 bytes (8-col padding)
    # beta: BT * 2 bytes
    # mbarrier: 8 bytes per barrier * NUM_STAGES
    smem_bytes = (4 * BC * BK * NUM_STAGES +                     # sG (float32, swizzled)
                  2 * BC * BK * NUM_STAGES * 2 +                 # sQK (fp16, swizzled, interleaved q/k)
                  4 * BC * (BC * 2 + ACCUM_PAD) * 2 +            # sAccum (8-col padding)
                  2 * BT +                                        # sBeta
                  8 * NUM_STAGES +                                # mbarriers
                  256)                                            # alignment
    
    print(f"KDA Akk: B={B}, T={seq_len}, H={H}, K={K}")
    print(f"  Tiles: NT={NT}, num_k_tiles={num_k_tiles}")
    print(f"  SMEM: {smem_bytes} bytes\n")
    
    kda_Akk_kernel(
        tma_atom_g,
        tma_tensor_g,
        tma_atom_q,
        tma_tensor_q,
        tma_atom_k,
        tma_tensor_k,
        g_smem_layout,       # 3D: (BC, BK, NUM_STAGES) - with swizzle
        qk_smem_layout,      # 3D: (BC, BK, NUM_STAGES * 2) - with swizzle
        beta_tensor,
        Akk_tensor,
        Aqk_tensor,
        accum_smem_layout,
        beta_smem_layout,
        tiled_mma,           # MMA configuration
        tiled_copy_Q,        # ldmatrix for Q
        tiled_copy_K,        # ldmatrix for K
        tiled_copy_G_s2r,    # s2r for G matching MMA C layout
        BT,
        num_k_tiles,
        seq_len,
        scale,
    ).launch(
        grid=(B, NT, H),
        # grid=(1, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


if __name__ == "__main__":
    print("KDA Akk TMA Test")
    print("=" * 50)
    
    # Test parameters
    # NOTE: For debug writeback, Akk/Aqk are (B, T, H, BC) float32, keep sizes small.
    B, seq_len, H, K = 1, 8192, 96, 128  # batch, seq_len, heads, head_dim
    BT = 64  # chunk size
    warmup_iters = 5
    test_iters = 100
    use_profiler = True
    
    # Create test tensors
    g = torch.randn(B, seq_len, H, K, dtype=torch.float32, device='cuda')
    q = torch.randn(B, seq_len, H, K, dtype=torch.float16, device='cuda')
    k = torch.randn(B, seq_len, H, K, dtype=torch.float16, device='cuda')
    beta = torch.randn(B, seq_len, H, dtype=torch.float16, device='cuda')
    # L2 norm for k
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    q = q.to(torch.float16)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    k = k.to(torch.float16)
    
    # Output tensors (debug writeback buffers)
    # - Akk: (B, seq_len, H, BC) fp32
    # - Aqk: (B, seq_len, H, BT) fp16
    Akk = torch.zeros(B, seq_len, H, BC, dtype=torch.float32, device='cuda')
    Aqk = torch.zeros(B, seq_len, H, BT, dtype=torch.float16, device='cuda')
    
    # Create cute tensors
    g_tensor = from_dlpack(g, assumed_align=16)
    g_tensor.element_type = cutlass.Float32
    
    q_tensor = from_dlpack(q, assumed_align=16)
    q_tensor.element_type = cutlass.Float16
    
    k_tensor = from_dlpack(k, assumed_align=16)
    k_tensor.element_type = cutlass.Float16

    beta_tensor = from_dlpack(beta, assumed_align=16)
    beta_tensor.element_type = cutlass.Float16
    
    Akk_tensor = from_dlpack(Akk, assumed_align=16)
    Akk_tensor.element_type = cutlass.Float32
    
    Aqk_tensor = from_dlpack(Aqk, assumed_align=16)
    Aqk_tensor.element_type = cutlass.Float16
    
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    # scale = 1 / sqrt(K)
    scale = 1.0 / (K ** 0.5)
    
    print("Compiling kernel...")
    compiled = cute.compile(run_kda_Akk, g_tensor, q_tensor, k_tensor, beta_tensor, Akk_tensor, Aqk_tensor, scale, stream)
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup_iters):
        compiled(g_tensor, q_tensor, k_tensor, beta_tensor, Akk_tensor, Aqk_tensor, scale, stream)
    torch.cuda.synchronize()
    
    print("\n✓ Kernel executed successfully!")

    # =============== Correctness check against Triton reference ===============
    print("\n" + "=" * 50)
    print("Correctness Check (vs Triton reference kernel)")
    print("=" * 50)
    
    # Allocate reference outputs
    Aqk_ref = torch.zeros(B, seq_len, H, BT, device='cuda', dtype=torch.float16)
    Akk_ref = torch.zeros(B, seq_len, H, BC, device='cuda', dtype=torch.float32)
    
    # Run Triton reference kernel
    NT = triton.cdiv(seq_len, BT)
    NC = triton.cdiv(BT, BC)
    BK_triton = triton.next_power_of_2(K)
    
    grid = (NT, NC, B * H)
    chunk_kda_fwd_kernel_intra_sub_chunk[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        Aqk=Aqk_ref,
        Akk=Akk_ref,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        T=seq_len,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK_triton,
        IS_VARLEN=False,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    torch.cuda.synchronize()
    
    # Compare results
    # Note: CuTe outputs - Aqk: (B, seq_len, H, BT), Akk: (B, seq_len, H, BC)
    # Triton outputs   - Aqk_ref: (B, seq_len, H, BT), Akk_ref: (B, seq_len, H, BC)
    
    # Aqk comparison (fp16)
    max_diff_aqk = (Aqk.float() - Aqk_ref.float()).abs().max().item()
    mean_diff_aqk = (Aqk.float() - Aqk_ref.float()).abs().mean().item()
    
    # Akk comparison (fp32)
    max_diff_akk = (Akk - Akk_ref).abs().max().item()
    mean_diff_akk = (Akk - Akk_ref).abs().mean().item()
    
    print(f"\nAqk (fp16):")
    print(f"  max |CuTe - Triton| = {max_diff_aqk:.6e}")
    print(f"  mean|CuTe - Triton| = {mean_diff_aqk:.6e}")
    
    print(f"\nAkk (fp32):")
    print(f"  max |CuTe - Triton| = {max_diff_akk:.6e}")
    print(f"  mean|CuTe - Triton| = {mean_diff_akk:.6e}")
    
    # Print sample for inspection (b=0, h=0, first sub-chunk)
    # print("\n--- Sample: Aqk[0, 0:8, 0, 0:8] ---")
    # print("CuTe:")
    # print(Aqk[0, 0:8, 0, 0:8].detach().cpu())
    # print("Triton:")
    # print(Aqk_ref[0, 0:8, 0, 0:8].detach().cpu())
    
    # print("\n--- Sample: Akk[0, 0:8, 0, 0:8] ---")
    # print("CuTe:")
    # print(Akk[0, 0:8, 0, 0:8].detach().cpu())
    # print("Triton:")
    # print(Akk_ref[0, 0:8, 0, 0:8].detach().cpu())
    
    # Pass/Fail check
    # TF32 MMA has ~1e-3 relative error, so use relaxed thresholds
    aqk_threshold = 1e-3  # fp16 output
    akk_threshold = 1e-3  # fp32 output
    
    if max_diff_aqk > aqk_threshold or max_diff_akk > akk_threshold:
        print(f"\n✗ FAIL: Aqk diff > {aqk_threshold} or Akk diff > {akk_threshold}")
    else:
        print(f"\n✓ PASS: Results match within tolerance")
    
    # Benchmark
    print(f"\nBenchmarking: {test_iters} iterations...")

    # L2 cache eviction buffer (match benchmark_all.py idea)
    dummy_buffer = torch.empty(int(80 * 1024 * 1024 / 4), dtype=torch.float32, device='cuda')

    # =============== CUDA Event Timing ===============
    print(f"\nCUDA Event Timing: {test_iters} iterations...")
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    cuda_event_times_ms = []
    for i in range(test_iters):
        # Evict L2 cache
        _ = dummy_buffer.sum()
        torch.cuda.synchronize()
        
        # Record start
        start_event.record()
        
        # Run kernel
        compiled(g_tensor, q_tensor, k_tensor, beta_tensor, Akk_tensor, Aqk_tensor, scale, stream)
        
        # Record end
        end_event.record()
        
        # Synchronize and get elapsed time
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        cuda_event_times_ms.append(elapsed_ms)
    
    cuda_event_times = torch.tensor(cuda_event_times_ms, dtype=torch.float64)
    cuda_event_mean_ms = cuda_event_times.mean().item()
    cuda_event_min_ms = cuda_event_times.min().item()
    cuda_event_std_ms = cuda_event_times.std().item()
    
    print(f"  ✓ CUDA Event Mean: {cuda_event_mean_ms:.4f} ms")
    print(f"  ✓ CUDA Event Min:  {cuda_event_min_ms:.4f} ms")
    print(f"  ✓ CUDA Event Std:  {cuda_event_std_ms:.4f} ms")

    # =============== torch.profiler Timing ===============
    print(f"\nProfiling with torch.profiler: {test_iters} iterations...")
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    )

    with profiler:
        for i in range(test_iters):
            _ = dummy_buffer.sum()
            torch.cuda.synchronize()
            with torch.profiler.record_function(f"kda_Akk_iter{i}"):
                compiled(g_tensor, q_tensor, k_tensor, beta_tensor, Akk_tensor, Aqk_tensor, scale, stream)
            torch.cuda.synchronize()

    trace_dir = "profiler_traces"
    os.makedirs(trace_dir, exist_ok=True)
    trace_file = os.path.join(trace_dir, "kda_Akk.json")
    profiler.export_chrome_trace(trace_file)
    print(f"  ✓ Profiler trace saved to {trace_file}")

    profiler_times_us = []
    try:
        with open(trace_file, "r") as f:
            trace_data = json.load(f)
        for event in trace_data.get("traceEvents", []):
            # only kernel events
            if event.get("cat") != "kernel":
                continue
            name = event.get("name", "")
            dur = event.get("dur", 0)  # microseconds
            # Only extract CUTLASS kernels, ignore dummy_buffer reductions
            if dur > 0 and "kernel_cutlass" in name:
                profiler_times_us.append(dur)
        if profiler_times_us:
            print(f"  ✓ Parsed {len(profiler_times_us)} CUTLASS kernel timings from profiler trace")
        else:
            print("  ⚠ No CUTLASS kernel timings found in profiler trace")
    except Exception as e:
        print(f"  ✗ Failed to parse profiler trace: {e}")
        profiler_times_us = []
    
    # Data size (bytes moved by this debug kernel)
    read_bytes = B * seq_len * H * K * (4 + 2 + 2) + (B * seq_len * H * 2)
    write_bytes = (B * seq_len * H * BC * 4) + (B * seq_len * H * BC * 2)
    data_bytes = read_bytes + write_bytes
    data_mb = data_bytes / 1024 / 1024
    data_gib = data_bytes / 1000 / 1000 / 1000
    peak_bw = PEAK_BW_GBS
    
    # CuTe metrics
    cute_bw_cuda = data_gib / (cuda_event_mean_ms / 1000.0)
    cute_peak_cuda = cute_bw_cuda / peak_bw * 100.0
    
    cute_profiler_mean_ms = 0.0
    cute_bw_profiler = 0.0
    cute_peak_profiler = 0.0
    if len(profiler_times_us) > 0:
        cute_profiler_mean_ms = torch.tensor(profiler_times_us, dtype=torch.float64).mean().item() / 1000.0
        cute_bw_profiler = data_gib / (cute_profiler_mean_ms / 1000.0)
        cute_peak_profiler = cute_bw_profiler / peak_bw * 100.0
    
    # =============== Triton Benchmark ===============
    print(f"\nTriton Benchmark ({test_iters} iters)...")
    
    # Warmup
    for _ in range(10):
        chunk_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q, k=k, g=g, beta=beta, Aqk=Aqk_ref, Akk=Akk_ref, scale=scale,
            cu_seqlens=None, chunk_indices=None,
            T=seq_len, H=H, K=K, BT=BT, BC=BC, BK=BK_triton,
            IS_VARLEN=False, USE_GATHER=IS_GATHER_SUPPORTED,
        )
    torch.cuda.synchronize()
    
    # CUDA Event Timing for Triton
    triton_cuda_event_times_ms = []
    for i in range(test_iters):
        _ = dummy_buffer.sum()
        torch.cuda.synchronize()
        start_event.record()
        chunk_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q, k=k, g=g, beta=beta, Aqk=Aqk_ref, Akk=Akk_ref, scale=scale,
            cu_seqlens=None, chunk_indices=None,
            T=seq_len, H=H, K=K, BT=BT, BC=BC, BK=BK_triton,
            IS_VARLEN=False, USE_GATHER=IS_GATHER_SUPPORTED,
        )
        end_event.record()
        torch.cuda.synchronize()
        triton_cuda_event_times_ms.append(start_event.elapsed_time(end_event))
    
    triton_cuda_event_times = torch.tensor(triton_cuda_event_times_ms, dtype=torch.float64)
    triton_cuda_event_mean_ms = triton_cuda_event_times.mean().item()
    
    # torch.profiler Timing for Triton
    profiler_triton = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, with_stack=False,
    )
    with profiler_triton:
        for i in range(test_iters):
            _ = dummy_buffer.sum()
            torch.cuda.synchronize()
            chunk_kda_fwd_kernel_intra_sub_chunk[grid](
                q=q, k=k, g=g, beta=beta, Aqk=Aqk_ref, Akk=Akk_ref, scale=scale,
                cu_seqlens=None, chunk_indices=None,
                T=seq_len, H=H, K=K, BT=BT, BC=BC, BK=BK_triton,
                IS_VARLEN=False, USE_GATHER=IS_GATHER_SUPPORTED,
            )
            torch.cuda.synchronize()
    
    trace_file_triton = os.path.join(trace_dir, "kda_Akk_triton.json")
    profiler_triton.export_chrome_trace(trace_file_triton)
    
    triton_times_us = []
    try:
        with open(trace_file_triton, "r") as f:
            trace_data = json.load(f)
        for event in trace_data.get("traceEvents", []):
            if event.get("cat") == "kernel":
                name = event.get("name", "")
                dur = event.get("dur", 0)
                if dur > 0 and "chunk_kda" in name.lower():
                    triton_times_us.append(dur)
    except:
        pass
    
    # Triton metrics
    triton_bw_cuda = data_gib / (triton_cuda_event_mean_ms / 1000.0)
    triton_peak_cuda = triton_bw_cuda / peak_bw * 100.0
    
    triton_profiler_mean_ms = 0.0
    triton_bw_profiler = 0.0
    triton_peak_profiler = 0.0
    if len(triton_times_us) > 0:
        triton_profiler_mean_ms = torch.tensor(triton_times_us, dtype=torch.float64).mean().item() / 1000.0
        triton_bw_profiler = data_gib / (triton_profiler_mean_ms / 1000.0)
        triton_peak_profiler = triton_bw_profiler / peak_bw * 100.0
    
    # =============== Summary Table ===============
    print("TMA LDMatrix KDA Akk")
    print("\n" + "=" * 80)
    print(f"Performance Summary (B={B}, T={seq_len}, H={H}, K={K}, Data={data_mb:.1f}MB, Peak={peak_bw}GB/s)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'CuTe':<18} | {'Triton':<18} | {'Speedup':<10}")
    print("-" * 80)
    
    # CUDA Event row
    speedup_cuda = triton_cuda_event_mean_ms / cuda_event_mean_ms
    print(f"{'CUDA Event (ms)':<20} | {cuda_event_mean_ms:<18.4f} | {triton_cuda_event_mean_ms:<18.4f} | {speedup_cuda:<10.2f}x")
    print(f"{'  → BW (GB/s)':<20} | {cute_bw_cuda:<18.1f} | {triton_bw_cuda:<18.1f} |")
    print(f"{'  → Peak%':<20} | {cute_peak_cuda:<18.2f} | {triton_peak_cuda:<18.2f} |")
    
    # Profiler row
    if cute_profiler_mean_ms > 0 and triton_profiler_mean_ms > 0:
        speedup_profiler = triton_profiler_mean_ms / cute_profiler_mean_ms
        print("-" * 80)
        print(f"{'Profiler (ms)':<20} | {cute_profiler_mean_ms:<18.4f} | {triton_profiler_mean_ms:<18.4f} | {speedup_profiler:<10.2f}x")
        print(f"{'  → BW (GB/s)':<20} | {cute_bw_profiler:<18.1f} | {triton_bw_profiler:<18.1f} |")
        print(f"{'  → Peak%':<20} | {cute_peak_profiler:<18.2f} | {triton_peak_profiler:<18.2f} |")
    
    print("=" * 80)
    if speedup_cuda > 1:
        print(f"✓ CuTe is {speedup_cuda:.2f}x FASTER than Triton (CUDA Event)")
    else:
        print(f"✗ Triton is {1/speedup_cuda:.2f}x FASTER than CuTe (CUDA Event)")
    print("=" * 80)
    print("DONE")
    
    del compiled