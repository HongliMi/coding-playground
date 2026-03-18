# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Lower Triangular Matrix Inversion using CuTe DSL with TMA - FlashInfer Style.

Optimizations:
1. Double buffering: separate input (sA_in) and output (sA_out) SMEM
2. No swizzle - direct access to 64x64 SMEM with stride=64
3. Transpose view for B operand (zero-cost, no data movement)
4. ldmatrix for efficient SMEM->Register loading
"""

import math
import cutlass
import cutlass.cute as cute
from cutlass import utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


# ===========================================================================
# Kernel Constants
# ===========================================================================
THREADS_PER_CTA = 128  # 4 warps
BLOCK_SIZE = 64
SUB_BLOCK_SIZE = 16


# ===========================================================================
# Device Kernel: Lower Triangular Matrix Inversion (64x64) - FlashInfer Style
# ===========================================================================
@cute.kernel
def solve_tril_kernel_v2(
    tma_load_atom: cute.CopyAtom,
    tma_store_atom: cute.CopyAtom,
    tma_load_tensor: cute.Tensor,
    tma_store_tensor: cute.Tensor,
):
    """
    64x64 Lower Triangular Matrix Inversion Kernel.
    
    Memory Layout:
    - sA_in: Input buffer (64x64 FP16, row-major, stride=64)
    - sA_out: Output buffer (64x64 FP16, row-major, stride=64)
    - sT: Temporary buffer for each warp (16x16 FP16)
    
    Uses FlashInfer-style transpose view for B operand.
    """
    # Thread indices
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32
    
    # Block indices
    batch_idx, _, _ = cute.arch.block_idx()
    
    # Allocate shared memory
    smem = cutlass.utils.SmemAllocator()
    
    # 64x64 row-major layout (stride=64, no swizzle!)
    smem_layout_fp16 = cute.make_layout((1, BLOCK_SIZE, BLOCK_SIZE), stride=(0, BLOCK_SIZE, 1))
    sA_in = smem.allocate_tensor(cutlass.Float16, smem_layout_fp16, 128)
    sA_out = smem.allocate_tensor(cutlass.Float16, smem_layout_fp16, 128)
    
    # Temporary buffers for intermediate results (one per warp, 16x16)
    sT_layout = cute.make_layout((16, 16), stride=(16, 1))
    sT_0 = smem.allocate_tensor(cutlass.Float16, sT_layout, 16)
    sT_1 = smem.allocate_tensor(cutlass.Float16, sT_layout, 16)
    sT_2 = smem.allocate_tensor(cutlass.Float16, sT_layout, 16)
    
    # Allocate mbarrier for TMA
    mbar_ptr = smem.allocate_array(cutlass.Int64, 1)
    
    tile_bytes = BLOCK_SIZE * BLOCK_SIZE * 2
    
    # Initialize mbarrier
    if tidx == 0:
        cute.arch.mbarrier_init(mbar_ptr, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    
    # Get global tiles
    gA = cute.local_tile(tma_load_tensor, (1, BLOCK_SIZE, BLOCK_SIZE), (batch_idx, 0, None))
    gAi = cute.local_tile(tma_store_tensor, (1, BLOCK_SIZE, BLOCK_SIZE), (batch_idx, 0, None))
    
    gA_tile = gA[(0, None, None, 0)]
    gAi_tile = gAi[(0, None, None, 0)]
    sA_in_tile = sA_in[(0, None, None)]
    sA_out_tile = sA_out[(0, None, None)]
    
    # =========================================================================
    # TMA Load: float16 Global -> sA_in
    # =========================================================================
    if warp_idx == 0:
        if tidx == 0:
            cute.arch.mbarrier_expect_tx(mbar_ptr, tile_bytes)
        
        tma_sDst, tma_gSrc = cpasync.tma_partition(
            tma_load_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sA_in_tile, 0, 2),
            cute.group_modes(gA_tile, 0, 2),
        )
        cute.copy(tma_load_atom, tma_gSrc, tma_sDst, tma_bar_ptr=mbar_ptr)
    
    if warp_idx == 0 and tidx == 0:
        cute.arch.mbarrier_arrive(mbar_ptr)
    cute.arch.mbarrier_wait(mbar_ptr, 0)
    
    # =========================================================================
    # Setup MMA (shared by all phases)
    # =========================================================================
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(
        mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 16)
    )
    
    atom_copy_s2r_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Float16
    )
    atom_copy_s2r_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 2), cutlass.Float16
    )
    tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
    tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
    
    # =========================================================================
    # STAGE 1: Invert Diagonal 16x16 Blocks (4 half-warps in parallel)
    # =========================================================================
    halfwarp_idx = (warp_idx * 2 + (lane_id // 16)) % 4
    diag_offset = halfwarp_idx * SUB_BLOCK_SIZE
    
    _invert_16x16_halfwarp_fp16(sA_in_tile, sA_out_tile, diag_offset, lane_id)
    cute.arch.sync_threads()
    
    # =========================================================================
    # STAGE 2a: First-level off-diagonal blocks (3 warps in parallel)
    # Ai21 = -(Ai22 @ A21) @ Ai11
    # Ai32 = -(Ai33 @ A32) @ Ai22  
    # Ai43 = -(Ai44 @ A43) @ Ai33
    # =========================================================================
    if warp_idx == 0:
        _phase2a_chain_mma_flashinfer(
            sA_in_tile, sA_out_tile, sT_0,
            1, 1,   # Tril1 = Ai22 at block(1,1) from sA_out
            1, 0,   # B = A21 at block(1,0) from sA_in
            0, 0,   # Tril2 = Ai11 at block(0,0) from sA_out
            1, 0,   # output at block(1,0) to sA_out
            lane_id, tiled_mma, tiled_copy_s2r_A, tiled_copy_s2r_B,
        )
    if warp_idx == 1:
        _phase2a_chain_mma_flashinfer(
            sA_in_tile, sA_out_tile, sT_1,
            2, 2,   # Ai33
            2, 1,   # A32
            1, 1,   # Ai22
            2, 1,   # output Ai32
            lane_id, tiled_mma, tiled_copy_s2r_A, tiled_copy_s2r_B,
        )
    if warp_idx == 2:
        _phase2a_chain_mma_flashinfer(
            sA_in_tile, sA_out_tile, sT_2,
            3, 3,   # Ai44
            3, 2,   # A43
            2, 2,   # Ai33
            3, 2,   # output Ai43
            lane_id, tiled_mma, tiled_copy_s2r_A, tiled_copy_s2r_B,
        )
    
    cute.arch.sync_threads()
    
    # =========================================================================
    # STAGE 2b: Second-level off-diagonal blocks (2 warps)
    # Ai31 = -Ai33 @ (A31 @ Ai11 + A32 @ Ai21)
    # Ai42 = -Ai44 @ (A42 @ Ai22 + A43 @ Ai32)
    # =========================================================================
    if warp_idx == 0:
        _phase2b_sum_mma_flashinfer(
            sA_in_tile, sA_out_tile, sT_0,
            2, 0,   # B1 = A31 from sA_in
            0, 0,   # C1 = Ai11 from sA_out
            2, 1,   # B2 = A32 from sA_in
            1, 0,   # C2 = Ai21 from sA_out
            2, 2,   # Ai = Ai33 from sA_out
            2, 0,   # output Ai31
            lane_id, tiled_mma, tiled_copy_s2r_A, tiled_copy_s2r_B,
        )
    if warp_idx == 1:
        _phase2b_sum_mma_flashinfer(
            sA_in_tile, sA_out_tile, sT_1,
            3, 1,   # B1 = A42 from sA_in
            1, 1,   # C1 = Ai22 from sA_out
            3, 2,   # B2 = A43 from sA_in
            2, 1,   # C2 = Ai32 from sA_out
            3, 3,   # Ai = Ai44 from sA_out
            3, 1,   # output Ai42
            lane_id, tiled_mma, tiled_copy_s2r_A, tiled_copy_s2r_B,
        )
    
    cute.arch.sync_threads()
    
    # =========================================================================
    # STAGE 2c: Third-level off-diagonal block (1 warp)
    # Ai41 = -Ai44 @ (A41 @ Ai11 + A42 @ Ai21 + A43 @ Ai31)
    # =========================================================================
    if warp_idx == 0:
        _phase2c_sum3_mma_flashinfer(
            sA_in_tile, sA_out_tile, sT_0,
            3, 0, 0, 0,   # B1=A41, C1=Ai11
            3, 1, 1, 0,   # B2=A42, C2=Ai21
            3, 2, 2, 0,   # B3=A43, C3=Ai31
            3, 3,         # Ai=Ai44
            3, 0,         # output Ai41
            lane_id, tiled_mma, tiled_copy_s2r_A, tiled_copy_s2r_B,
        )
    
    cute.arch.sync_threads()
    
    # =========================================================================
    # TMA Store: sA_out -> Global
    # =========================================================================
    tma_sSrc, tma_gDst = cpasync.tma_partition(
        tma_store_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sA_out_tile, 0, 2),
        cute.group_modes(gAi_tile, 0, 2),
    )
    cute.copy(tma_store_atom, tma_sSrc, tma_gDst)


# ===========================================================================
# Helper: 16x16 diagonal block inversion (FP16 output)
# ===========================================================================
@dsl_user_op
def _invert_16x16_halfwarp_fp16(
    sA_in: cute.Tensor,   # (64, 64) fp16 - input
    sA_out: cute.Tensor,  # (64, 64) fp16 - output
    diag_offset,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    """Invert a 16x16 lower triangular block, write FP16 result directly."""
    my_row = lane_id % 16
    halfwarp_base = (lane_id // 16) * 16
    
    row_off = diag_offset
    col_off = diag_offset
    
    # Registers for computation (FP32 for precision)
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
    
    # d=3 to d=15
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
    
    # Write directly to FP16 output
    sA_out[row_off + my_row, col_off + my_row] = cutlass.Float16(rInv[0])
    sA_out[row_off + my_row, col_off + (my_row + 15) % 16] = cutlass.Float16(rInv[1] * cutlass.Float32(my_row >= 1))
    sA_out[row_off + my_row, col_off + (my_row + 14) % 16] = cutlass.Float16(rInv[2] * cutlass.Float32(my_row >= 2))
    sA_out[row_off + my_row, col_off + (my_row + 13) % 16] = cutlass.Float16(rInv[3] * cutlass.Float32(my_row >= 3))
    sA_out[row_off + my_row, col_off + (my_row + 12) % 16] = cutlass.Float16(rInv[4] * cutlass.Float32(my_row >= 4))
    sA_out[row_off + my_row, col_off + (my_row + 11) % 16] = cutlass.Float16(rInv[5] * cutlass.Float32(my_row >= 5))
    sA_out[row_off + my_row, col_off + (my_row + 10) % 16] = cutlass.Float16(rInv[6] * cutlass.Float32(my_row >= 6))
    sA_out[row_off + my_row, col_off + (my_row + 9) % 16] = cutlass.Float16(rInv[7] * cutlass.Float32(my_row >= 7))
    sA_out[row_off + my_row, col_off + (my_row + 8) % 16] = cutlass.Float16(rInv[8] * cutlass.Float32(my_row >= 8))
    sA_out[row_off + my_row, col_off + (my_row + 7) % 16] = cutlass.Float16(rInv[9] * cutlass.Float32(my_row >= 9))
    sA_out[row_off + my_row, col_off + (my_row + 6) % 16] = cutlass.Float16(rInv[10] * cutlass.Float32(my_row >= 10))
    sA_out[row_off + my_row, col_off + (my_row + 5) % 16] = cutlass.Float16(rInv[11] * cutlass.Float32(my_row >= 11))
    sA_out[row_off + my_row, col_off + (my_row + 4) % 16] = cutlass.Float16(rInv[12] * cutlass.Float32(my_row >= 12))
    sA_out[row_off + my_row, col_off + (my_row + 3) % 16] = cutlass.Float16(rInv[13] * cutlass.Float32(my_row >= 13))
    sA_out[row_off + my_row, col_off + (my_row + 2) % 16] = cutlass.Float16(rInv[14] * cutlass.Float32(my_row >= 14))
    sA_out[row_off + my_row, col_off + (my_row + 1) % 16] = cutlass.Float16(rInv[15] * cutlass.Float32(my_row >= 15))


# ===========================================================================
# Helper: Phase 2a Chain MMA - FlashInfer Style
# ===========================================================================
@dsl_user_op
def _phase2a_chain_mma_flashinfer(
    sA_in: cute.Tensor,       # (64, 64) input fp16 smem
    sA_out: cute.Tensor,      # (64, 64) output fp16 smem
    sT: cute.Tensor,          # (16, 16) temp buffer
    tril1_br, tril1_bc,       # Tril1 block position (from sA_out)
    b_br, b_bc,               # B block position (from sA_in)
    tril2_br, tril2_bc,       # Tril2 block position (from sA_out)
    out_br, out_bc,           # output block position (to sA_out)
    lane_id,
    tiled_mma: cute.TiledMma,
    tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy,
    *,
    loc=None,
    ip=None,
):
    """
    Compute: Result = -(Tril1 @ B) @ Tril2
    
    FlashInfer style: direct access to 64x64 SMEM with transpose view for B operand.
    """
    # Get 16x16 sub-blocks using local_tile
    sA_Tril1 = cute.local_tile(sA_out, tiler=(16, 16), coord=(tril1_br, tril1_bc))
    sA_B = cute.local_tile(sA_in, tiler=(16, 16), coord=(b_br, b_bc))
    sA_Tril2 = cute.local_tile(sA_out, tiler=(16, 16), coord=(tril2_br, tril2_bc))
    
    # Create transposed views for B operands (zero-cost!)
    # Original layout: (16, 16) with stride (64, 1) - row-major
    # Transposed view: (16, 16) with stride (1, 64) - col-major
    sA_B_T_layout = cute.make_layout((16, 16), stride=(1, 64))
    sA_B_T = cute.make_tensor(sA_B.iterator, sA_B_T_layout)
    
    sA_Tril2_T_layout = cute.make_layout((16, 16), stride=(1, 64))
    sA_Tril2_T = cute.make_tensor(sA_Tril2.iterator, sA_Tril2_T_layout)
    
    # MMA setup
    thr_mma = tiled_mma.get_slice(lane_id)
    thr_copy_A = tiled_copy_s2r_A.get_slice(lane_id)
    thr_copy_B = tiled_copy_s2r_B.get_slice(lane_id)
    
    # =========================================================================
    # First MMA: T = Tril1 @ B
    # A = Tril1 (row-major from sA_out)
    # B = B (transposed view from sA_in)
    # =========================================================================
    # Load Tril1 as A operand
    tCsA = thr_mma.partition_A(sA_Tril1)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_Tril1), thr_copy_A.retile(tCrA))
    
    # Accumulators for T
    sC_part_layout = cute.make_layout((16, 8), stride=(8, 1))
    sCp = cute.make_tensor(sT.iterator, sC_part_layout)
    tCsCp = thr_mma.partition_C(sCp)
    tCrT0 = tiled_mma.make_fragment_C(tCsCp)
    tCrT0.fill(0.0)
    tCrT1 = tiled_mma.make_fragment_C(tCsCp)
    tCrT1.fill(0.0)
    
    # Load B tile 0 (N=0..7) using transposed view
    sB_T_tile0 = cute.local_tile(sA_B_T, tiler=(8, 16), coord=(0, 0))
    tCsB0 = thr_mma.partition_B(sB_T_tile0)
    tCrB0 = tiled_mma.make_fragment_B(tCsB0)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sB_T_tile0), thr_copy_B.retile(tCrB0))
    cute.gemm(tiled_mma, tCrT0, tCrA, tCrB0, tCrT0)
    
    # Load B tile 1 (N=8..15) using transposed view
    sB_T_tile1 = cute.local_tile(sA_B_T, tiler=(8, 16), coord=(1, 0))
    tCsB1 = thr_mma.partition_B(sB_T_tile1)
    tCrB1 = tiled_mma.make_fragment_B(tCsB1)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sB_T_tile1), thr_copy_B.retile(tCrB1))
    cute.gemm(tiled_mma, tCrT1, tCrA, tCrB1, tCrT1)
    
    # Store T to temp buffer
    _store_acc_to_smem_fp16(tCrT0, sT, 0, lane_id)
    _store_acc_to_smem_fp16(tCrT1, sT, 8, lane_id)
    
    cute.arch.sync_warp()
    
    # =========================================================================
    # Second MMA: Result = T @ Tril2
    # A = T (from sT, row-major 16x16 with stride=16)
    # B = Tril2 (transposed view from sA_out)
    # =========================================================================
    # Load T as A operand
    tCsT = thr_mma.partition_A(sT)
    tCrT_A = tiled_mma.make_fragment_A(tCsT)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sT), thr_copy_A.retile(tCrT_A))
    
    # Accumulators for output
    tCrOut0 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut0.fill(0.0)
    tCrOut1 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut1.fill(0.0)
    
    # Load Tril2 tile 0 (N=0..7) using transposed view
    sTril2_T_tile0 = cute.local_tile(sA_Tril2_T, tiler=(8, 16), coord=(0, 0))
    tCsC0 = thr_mma.partition_B(sTril2_T_tile0)
    tCrC0 = tiled_mma.make_fragment_B(tCsC0)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sTril2_T_tile0), thr_copy_B.retile(tCrC0))
    cute.gemm(tiled_mma, tCrOut0, tCrT_A, tCrC0, tCrOut0)
    
    # Load Tril2 tile 1 (N=8..15) using transposed view
    sTril2_T_tile1 = cute.local_tile(sA_Tril2_T, tiler=(8, 16), coord=(1, 0))
    tCsC1 = thr_mma.partition_B(sTril2_T_tile1)
    tCrC1 = tiled_mma.make_fragment_B(tCsC1)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sTril2_T_tile1), thr_copy_B.retile(tCrC1))
    cute.gemm(tiled_mma, tCrOut1, tCrT_A, tCrC1, tCrOut1)
    
    # Store negated result to sA_out
    out_row = out_br * 16
    out_col = out_bc * 16
    _store_neg_acc_to_smem_fp16(tCrOut0, sA_out, out_row, out_col, 0, lane_id)
    _store_neg_acc_to_smem_fp16(tCrOut1, sA_out, out_row, out_col, 8, lane_id)


# ===========================================================================
# Helper: Phase 2b Sum MMA - FlashInfer Style
# ===========================================================================
@dsl_user_op
def _phase2b_sum_mma_flashinfer(
    sA_in: cute.Tensor,
    sA_out: cute.Tensor,
    sT: cute.Tensor,
    b1_br, b1_bc,             # B1 from sA_in
    c1_br, c1_bc,             # C1 from sA_out
    b2_br, b2_bc,             # B2 from sA_in
    c2_br, c2_bc,             # C2 from sA_out
    ai_br, ai_bc,             # Ai from sA_out
    out_br, out_bc,
    lane_id,
    tiled_mma: cute.TiledMma,
    tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy,
    *,
    loc=None,
    ip=None,
):
    """Compute: Result = -Ai @ (B1 @ C1 + B2 @ C2)"""
    # Get sub-blocks
    sA_B1 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b1_br, b1_bc))
    sA_C1 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c1_br, c1_bc))
    sA_B2 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b2_br, b2_bc))
    sA_C2 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c2_br, c2_bc))
    sA_Ai = cute.local_tile(sA_out, tiler=(16, 16), coord=(ai_br, ai_bc))
    
    # Create transposed views for B operands
    sA_C1_T_layout = cute.make_layout((16, 16), stride=(1, 64))
    sA_C1_T = cute.make_tensor(sA_C1.iterator, sA_C1_T_layout)
    sA_C2_T_layout = cute.make_layout((16, 16), stride=(1, 64))
    sA_C2_T = cute.make_tensor(sA_C2.iterator, sA_C2_T_layout)
    
    thr_mma = tiled_mma.get_slice(lane_id)
    thr_copy_A = tiled_copy_s2r_A.get_slice(lane_id)
    thr_copy_B = tiled_copy_s2r_B.get_slice(lane_id)
    
    sC_part_layout = cute.make_layout((16, 8), stride=(8, 1))
    sCp = cute.make_tensor(sT.iterator, sC_part_layout)
    tCsCp = thr_mma.partition_C(sCp)
    tCrT0 = tiled_mma.make_fragment_C(tCsCp)
    tCrT0.fill(0.0)
    tCrT1 = tiled_mma.make_fragment_C(tCsCp)
    tCrT1.fill(0.0)
    
    # T = B1 @ C1
    tCsA = thr_mma.partition_A(sA_B1)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B1), thr_copy_A.retile(tCrA))
    
    sC1_T_tile0 = cute.local_tile(sA_C1_T, tiler=(8, 16), coord=(0, 0))
    tCsB0 = thr_mma.partition_B(sC1_T_tile0)
    tCrB0 = tiled_mma.make_fragment_B(tCsB0)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC1_T_tile0), thr_copy_B.retile(tCrB0))
    cute.gemm(tiled_mma, tCrT0, tCrA, tCrB0, tCrT0)
    
    sC1_T_tile1 = cute.local_tile(sA_C1_T, tiler=(8, 16), coord=(1, 0))
    tCsB1 = thr_mma.partition_B(sC1_T_tile1)
    tCrB1 = tiled_mma.make_fragment_B(tCsB1)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC1_T_tile1), thr_copy_B.retile(tCrB1))
    cute.gemm(tiled_mma, tCrT1, tCrA, tCrB1, tCrT1)
    
    # T += B2 @ C2
    tCsA2 = thr_mma.partition_A(sA_B2)
    tCrA2 = tiled_mma.make_fragment_A(tCsA2)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B2), thr_copy_A.retile(tCrA2))
    
    sC2_T_tile0 = cute.local_tile(sA_C2_T, tiler=(8, 16), coord=(0, 0))
    tCsB0_2 = thr_mma.partition_B(sC2_T_tile0)
    tCrB0_2 = tiled_mma.make_fragment_B(tCsB0_2)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC2_T_tile0), thr_copy_B.retile(tCrB0_2))
    cute.gemm(tiled_mma, tCrT0, tCrA2, tCrB0_2, tCrT0)
    
    sC2_T_tile1 = cute.local_tile(sA_C2_T, tiler=(8, 16), coord=(1, 0))
    tCsB1_2 = thr_mma.partition_B(sC2_T_tile1)
    tCrB1_2 = tiled_mma.make_fragment_B(tCsB1_2)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC2_T_tile1), thr_copy_B.retile(tCrB1_2))
    cute.gemm(tiled_mma, tCrT1, tCrA2, tCrB1_2, tCrT1)
    
    # Store T
    _store_acc_to_smem_fp16(tCrT0, sT, 0, lane_id)
    _store_acc_to_smem_fp16(tCrT1, sT, 8, lane_id)
    
    cute.arch.sync_warp()
    
    # Result = -Ai @ T
    # Create transposed view for T (sT has stride=16)
    sT_T_layout = cute.make_layout((16, 16), stride=(1, 16))
    sT_T = cute.make_tensor(sT.iterator, sT_T_layout)
    
    tCsAi = thr_mma.partition_A(sA_Ai)
    tCrAi = tiled_mma.make_fragment_A(tCsAi)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_Ai), thr_copy_A.retile(tCrAi))
    
    tCrOut0 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut0.fill(0.0)
    tCrOut1 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut1.fill(0.0)
    
    sT_T_tile0 = cute.local_tile(sT_T, tiler=(8, 16), coord=(0, 0))
    tCsC0 = thr_mma.partition_B(sT_T_tile0)
    tCrC0 = tiled_mma.make_fragment_B(tCsC0)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sT_T_tile0), thr_copy_B.retile(tCrC0))
    cute.gemm(tiled_mma, tCrOut0, tCrAi, tCrC0, tCrOut0)
    
    sT_T_tile1 = cute.local_tile(sT_T, tiler=(8, 16), coord=(1, 0))
    tCsC1 = thr_mma.partition_B(sT_T_tile1)
    tCrC1 = tiled_mma.make_fragment_B(tCsC1)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sT_T_tile1), thr_copy_B.retile(tCrC1))
    cute.gemm(tiled_mma, tCrOut1, tCrAi, tCrC1, tCrOut1)
    
    out_row = out_br * 16
    out_col = out_bc * 16
    _store_neg_acc_to_smem_fp16(tCrOut0, sA_out, out_row, out_col, 0, lane_id)
    _store_neg_acc_to_smem_fp16(tCrOut1, sA_out, out_row, out_col, 8, lane_id)


# ===========================================================================
# Helper: Phase 2c Sum3 MMA - FlashInfer Style
# ===========================================================================
@dsl_user_op
def _phase2c_sum3_mma_flashinfer(
    sA_in: cute.Tensor,
    sA_out: cute.Tensor,
    sT: cute.Tensor,
    b1_br, b1_bc, c1_br, c1_bc,
    b2_br, b2_bc, c2_br, c2_bc,
    b3_br, b3_bc, c3_br, c3_bc,
    ai_br, ai_bc,
    out_br, out_bc,
    lane_id,
    tiled_mma: cute.TiledMma,
    tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy,
    *,
    loc=None,
    ip=None,
):
    """Compute: Result = -Ai @ (B1 @ C1 + B2 @ C2 + B3 @ C3)"""
    # Get sub-blocks
    sA_B1 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b1_br, b1_bc))
    sA_C1 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c1_br, c1_bc))
    sA_B2 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b2_br, b2_bc))
    sA_C2 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c2_br, c2_bc))
    sA_B3 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b3_br, b3_bc))
    sA_C3 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c3_br, c3_bc))
    sA_Ai = cute.local_tile(sA_out, tiler=(16, 16), coord=(ai_br, ai_bc))
    
    # Create transposed views
    sA_C1_T = cute.make_tensor(sA_C1.iterator, cute.make_layout((16, 16), stride=(1, 64)))
    sA_C2_T = cute.make_tensor(sA_C2.iterator, cute.make_layout((16, 16), stride=(1, 64)))
    sA_C3_T = cute.make_tensor(sA_C3.iterator, cute.make_layout((16, 16), stride=(1, 64)))
    
    thr_mma = tiled_mma.get_slice(lane_id)
    thr_copy_A = tiled_copy_s2r_A.get_slice(lane_id)
    thr_copy_B = tiled_copy_s2r_B.get_slice(lane_id)
    
    sC_part_layout = cute.make_layout((16, 8), stride=(8, 1))
    sCp = cute.make_tensor(sT.iterator, sC_part_layout)
    tCsCp = thr_mma.partition_C(sCp)
    tCrT0 = tiled_mma.make_fragment_C(tCsCp)
    tCrT0.fill(0.0)
    tCrT1 = tiled_mma.make_fragment_C(tCsCp)
    tCrT1.fill(0.0)
    
    # T = B1 @ C1
    tCsA = thr_mma.partition_A(sA_B1)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B1), thr_copy_A.retile(tCrA))
    
    sC1_T_tile0 = cute.local_tile(sA_C1_T, tiler=(8, 16), coord=(0, 0))
    tCsB0 = thr_mma.partition_B(sC1_T_tile0)
    tCrB0 = tiled_mma.make_fragment_B(tCsB0)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC1_T_tile0), thr_copy_B.retile(tCrB0))
    cute.gemm(tiled_mma, tCrT0, tCrA, tCrB0, tCrT0)
    
    sC1_T_tile1 = cute.local_tile(sA_C1_T, tiler=(8, 16), coord=(1, 0))
    tCsB1 = thr_mma.partition_B(sC1_T_tile1)
    tCrB1 = tiled_mma.make_fragment_B(tCsB1)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC1_T_tile1), thr_copy_B.retile(tCrB1))
    cute.gemm(tiled_mma, tCrT1, tCrA, tCrB1, tCrT1)
    
    # T += B2 @ C2
    tCsA2 = thr_mma.partition_A(sA_B2)
    tCrA2 = tiled_mma.make_fragment_A(tCsA2)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B2), thr_copy_A.retile(tCrA2))
    
    sC2_T_tile0 = cute.local_tile(sA_C2_T, tiler=(8, 16), coord=(0, 0))
    tCsB0_2 = thr_mma.partition_B(sC2_T_tile0)
    tCrB0_2 = tiled_mma.make_fragment_B(tCsB0_2)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC2_T_tile0), thr_copy_B.retile(tCrB0_2))
    cute.gemm(tiled_mma, tCrT0, tCrA2, tCrB0_2, tCrT0)
    
    sC2_T_tile1 = cute.local_tile(sA_C2_T, tiler=(8, 16), coord=(1, 0))
    tCsB1_2 = thr_mma.partition_B(sC2_T_tile1)
    tCrB1_2 = tiled_mma.make_fragment_B(tCsB1_2)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC2_T_tile1), thr_copy_B.retile(tCrB1_2))
    cute.gemm(tiled_mma, tCrT1, tCrA2, tCrB1_2, tCrT1)
    
    # T += B3 @ C3
    tCsA3 = thr_mma.partition_A(sA_B3)
    tCrA3 = tiled_mma.make_fragment_A(tCsA3)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B3), thr_copy_A.retile(tCrA3))
    
    sC3_T_tile0 = cute.local_tile(sA_C3_T, tiler=(8, 16), coord=(0, 0))
    tCsB0_3 = thr_mma.partition_B(sC3_T_tile0)
    tCrB0_3 = tiled_mma.make_fragment_B(tCsB0_3)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC3_T_tile0), thr_copy_B.retile(tCrB0_3))
    cute.gemm(tiled_mma, tCrT0, tCrA3, tCrB0_3, tCrT0)
    
    sC3_T_tile1 = cute.local_tile(sA_C3_T, tiler=(8, 16), coord=(1, 0))
    tCsB1_3 = thr_mma.partition_B(sC3_T_tile1)
    tCrB1_3 = tiled_mma.make_fragment_B(tCsB1_3)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC3_T_tile1), thr_copy_B.retile(tCrB1_3))
    cute.gemm(tiled_mma, tCrT1, tCrA3, tCrB1_3, tCrT1)
    
    # Store T
    _store_acc_to_smem_fp16(tCrT0, sT, 0, lane_id)
    _store_acc_to_smem_fp16(tCrT1, sT, 8, lane_id)
    
    cute.arch.sync_warp()
    
    # Result = -Ai @ T
    sT_T_layout = cute.make_layout((16, 16), stride=(1, 16))
    sT_T = cute.make_tensor(sT.iterator, sT_T_layout)
    
    tCsAi = thr_mma.partition_A(sA_Ai)
    tCrAi = tiled_mma.make_fragment_A(tCsAi)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_Ai), thr_copy_A.retile(tCrAi))
    
    tCrOut0 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut0.fill(0.0)
    tCrOut1 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut1.fill(0.0)
    
    sT_T_tile0 = cute.local_tile(sT_T, tiler=(8, 16), coord=(0, 0))
    tCsC0 = thr_mma.partition_B(sT_T_tile0)
    tCrC0 = tiled_mma.make_fragment_B(tCsC0)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sT_T_tile0), thr_copy_B.retile(tCrC0))
    cute.gemm(tiled_mma, tCrOut0, tCrAi, tCrC0, tCrOut0)
    
    sT_T_tile1 = cute.local_tile(sT_T, tiler=(8, 16), coord=(1, 0))
    tCsC1 = thr_mma.partition_B(sT_T_tile1)
    tCrC1 = tiled_mma.make_fragment_B(tCsC1)
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sT_T_tile1), thr_copy_B.retile(tCrC1))
    cute.gemm(tiled_mma, tCrOut1, tCrAi, tCrC1, tCrOut1)
    
    out_row = out_br * 16
    out_col = out_bc * 16
    _store_neg_acc_to_smem_fp16(tCrOut0, sA_out, out_row, out_col, 0, lane_id)
    _store_neg_acc_to_smem_fp16(tCrOut1, sA_out, out_row, out_col, 8, lane_id)


# ===========================================================================
# Helper functions for storing accumulator results
# ===========================================================================
@dsl_user_op
def _store_acc_to_smem_fp16(acc: cute.Tensor, sOut: cute.Tensor, col_offset, lane_id, *, loc=None, ip=None):
    """Store FP32 accumulator to FP16 SMEM (row-major 16x16)."""
    group_id = lane_id // 4
    thread_in_group = lane_id % 4
    row0 = group_id
    row1 = group_id + 8
    col_base = thread_in_group * 2
    sOut[row0, col_offset + col_base] = cutlass.Float16(acc[0])
    sOut[row0, col_offset + col_base + 1] = cutlass.Float16(acc[1])
    sOut[row1, col_offset + col_base] = cutlass.Float16(acc[2])
    sOut[row1, col_offset + col_base + 1] = cutlass.Float16(acc[3])


@dsl_user_op
def _store_neg_acc_to_smem_fp16(acc: cute.Tensor, sOut: cute.Tensor, base_row, base_col, col_offset, lane_id, *, loc=None, ip=None):
    """Store negated FP32 accumulator to FP16 SMEM (64x64)."""
    group_id = lane_id // 4
    thread_in_group = lane_id % 4
    row0 = group_id
    row1 = group_id + 8
    col_base = thread_in_group * 2
    sOut[base_row + row0, base_col + col_offset + col_base] = cutlass.Float16(-acc[0])
    sOut[base_row + row0, base_col + col_offset + col_base + 1] = cutlass.Float16(-acc[1])
    sOut[base_row + row1, base_col + col_offset + col_base] = cutlass.Float16(-acc[2])
    sOut[base_row + row1, base_col + col_offset + col_base + 1] = cutlass.Float16(-acc[3])


# ===========================================================================
# Host JIT Function
# ===========================================================================
@cute.jit
def solve_tril_host_v2(
    A: cute.Tensor,
    Ai: cute.Tensor,
    batch_size: cutlass.Constexpr[int],
):
    smem_layout = cute.make_layout((1, BLOCK_SIZE, BLOCK_SIZE), stride=(0, BLOCK_SIZE, 1))
    cta_tiler = cute.product_each(smem_layout.shape)
    
    tma_load_atom, tma_load_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        A,
        smem_layout,
        cta_tiler,
    )
    
    tma_store_atom, tma_store_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        Ai,
        smem_layout,
        cta_tiler,
    )
    
    grid = (batch_size, 1, 1)
    
    solve_tril_kernel_v2(
        tma_load_atom,
        tma_store_atom,
        tma_load_tensor,
        tma_store_tensor,
    ).launch(
        grid=grid,
        block=(THREADS_PER_CTA, 1, 1),
    )


# ===========================================================================
# Test Function
# ===========================================================================
def test_solve_tril_v2():
    import torch
    import time
    
    BATCH_SIZE = 96 * 128  # H=96, num_chunks=8192/64=128
    WARMUP_ITERS = 5
    BENCH_ITERS = 100
    
    print("=" * 60)
    print("Benchmarking V2 - FlashInfer Style (No Swizzle, Transpose View)")
    print("=" * 60)
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Matrix size: 64x64")
    print(f"  Data type: float16")
    
    cutlass.cuda.initialize_cuda_context()
    
    torch.manual_seed(42)
    A = torch.randn(BATCH_SIZE, 64, 64, device="cuda", dtype=torch.float16)
    A = A.tril(-1)
    
    Ai = torch.zeros_like(A)
    
    A_cute = from_dlpack(A, assumed_align=16)
    Ai_cute = from_dlpack(Ai, assumed_align=16)
    
    print("\nCompiling kernel...")
    compiled_kernel = cute.compile(
        solve_tril_host_v2,
        A_cute,
        Ai_cute,
        BATCH_SIZE,
    )
    torch.cuda.synchronize()
    print("Compilation done.")
    
    print(f"\nWarmup ({WARMUP_ITERS} iterations)...")
    for _ in range(WARMUP_ITERS):
        compiled_kernel(A_cute, Ai_cute)
    torch.cuda.synchronize()
    
    print(f"\nBenchmarking ({BENCH_ITERS} iterations)...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(BENCH_ITERS):
        compiled_kernel(A_cute, Ai_cute)
    end_event.record()
    
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    mean_time = total_time_ms / BENCH_ITERS
    
    data_size_mb = BATCH_SIZE * 64 * 64 * 2 / (1024 * 1024)
    total_data_mb = data_size_mb * 2
    bandwidth_gbps = total_data_mb / (mean_time / 1000) / 1024
    
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"  Total time:    {total_time_ms:.3f} ms ({BENCH_ITERS} iterations)")
    print(f"  Time per iter: {mean_time:.3f} ms")
    print(f"  Bandwidth:     {bandwidth_gbps:.1f} GB/s")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_solve_tril_v2()

