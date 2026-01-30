# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
KDA Akk/Aqk Kernel - cp.async Pipeline
Load g (float32), q (fp16), k (fp16) with async copy
Tile size: 16x64 (BC x BK), double buffer
"""

import torch
import triton
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda
import json
import os
import sys

# Add path for Triton reference kernel
sys.path.insert(0, "/home/menyu/workspace/hongli/origin_space/flash-linear-attention")
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
    tid_in_group = lane_id % 4
    
    a0 = cutlass.Float32(sA[group_id, tid_in_group])
    a1 = cutlass.Float32(sA[group_id + 8, tid_in_group])
    a2 = cutlass.Float32(sA[group_id, tid_in_group + 4])
    a3 = cutlass.Float32(sA[group_id + 8, tid_in_group + 4])
    
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
    tid_in_group = lane_id % 4
    
    # Load with transposed indices (swap row/col)
    b0 = cutlass.Float32(sB[tid_in_group, group_id])
    b1 = cutlass.Float32(sB[tid_in_group + 4, group_id])
    
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


# 全局配置
BC = 16          # sub-chunk size (rows)
BK = 64          # key dimension (columns)
NUM_STAGES = 2   # double buffer
NUM_THREADS = 128
NUM_WARPS = 4


@cute.kernel
def kda_Akk_kernel(
    tiled_copy_g: cute.TiledCopy,
    tiled_copy_qk: cute.TiledCopy,
    g_tensor: cute.Tensor,       # (B, T, H, K) float32
    q_tensor: cute.Tensor,       # (B, T, H, K) fp16
    k_tensor: cute.Tensor,       # (B, T, H, K) fp16
    beta_tensor: cute.Tensor,    # (B, T, H) fp16
    Akk_tensor: cute.Tensor,     # (B, T, H, BC) output (debug: write back loaded g[:, :BC])
    Aqk_tensor: cute.Tensor,     # (B, T, H, BT) output (debug: write back loaded q[:, :BC] as fp16 into [:BC])
    g_smem_layout: cute.Layout,  # (BC, BK, NUM_STAGES)
    qk_smem_layout: cute.Layout, # (BC, BK, NUM_STAGES * 2) for q and k
    qk_smem_layout_f32: cute.Layout, # (BC, BK, NUM_STAGES) float32 view for r_qgq
    accum_smem_layout: cute.Layout, # (BC, BC * NUM_WARPS, 2) for Aqk and Akk
    beta_smem_layout: cute.Layout, # (BT,)
    BT: cutlass.Constexpr[int],
    num_k_tiles: cutlass.Constexpr[int],
    seq_len: int,                # sequence length (renamed from T to avoid shadowing cutlass.T)
    scale: cutlass.Float32,      # scale factor for Aqk
):
    """
    Debug kernel:
    - Each block handles one (batch, chunk, head)
    - The block owns a [BT, K] chunk, tiled by (BC, BK)
      BT=64, BC=16 => 4 tiles along T; K=128, BK=64 => 2 tiles along K; total 8 tiles.
    - We use cp.async pipeline to load g/q/k into smem, and also stage beta (fp16, length=BT) into smem.
      Then write back ONLY the first BC columns
      for correctness validation:
        - Akk: g[..., :BC] -> (B, T, H, BC) fp32
        - Aqk: q[..., :BC] -> (B, T, H, BT) fp16  (only [:BC] is written)
    """
    
    tidx, _, _ = cute.arch.thread_idx()
    i_b, i_t, i_h = cute.arch.block_idx()  # batch, chunk, head
    
    # Allocate shared memory
    smem = cutlass.utils.SmemAllocator()
    sG = smem.allocate_tensor(cutlass.Float32, g_smem_layout, 128)
    
    # Allocate q/k smem with dual view (fp16 for loading, fp32 for storing r_qgq)
    # q+k: (BC, BK, NUM_STAGES * 2) fp16 = (16, 64, 4) * 2 bytes = 8KB
    # Reused as: (BC, BK, NUM_STAGES) fp32 = (16, 64, 2) * 4 bytes = 8KB
    QK_SMEM_BYTES = BC * BK * NUM_STAGES * 2 * 2  # 8KB
    ptr_sQK_raw = smem.allocate(QK_SMEM_BYTES, 128)
    
    # fp16 view (for loading q/k from global memory)
    ptr_sQK_fp16 = cute.recast_ptr(ptr_sQK_raw, dtype=cutlass.Float16)
    sQK = cute.make_tensor(ptr_sQK_fp16, qk_smem_layout)
    
    # fp32 view (for storing r_qgq, reusing q+k space)
    ptr_sQK_f32 = cute.recast_ptr(ptr_sQK_raw, dtype=cutlass.Float32)
    sQK_f32 = cute.make_tensor(ptr_sQK_f32, qk_smem_layout_f32)

    sAccum = smem.allocate_tensor(cutlass.Float32, accum_smem_layout, 128)
    sBeta = smem.allocate_tensor(cutlass.Float16, beta_smem_layout, 128)
    
    # Chunk start and tile base along T (each T tile is BC rows)
    t_start = i_t * BT
    t_tile_base = i_t * (BT // BC)  # base tile index in (T/BC) dimension
    
    # 获取当前 batch 和 head 的数据 slice
    # g_tensor shape: (B, T, H, K) -> select (T, K) for current batch and head
    gG_batch = g_tensor[(i_b, None, i_h, None)]   # (T, K)
    gQ_batch = q_tensor[(i_b, None, i_h, None)]   # (T, K)
    gK_batch = k_tensor[(i_b, None, i_h, None)]   # (T, K)
    gBeta_batch = beta_tensor[(i_b, None, i_h)]   # (T,)

    # Stage beta for this chunk: sBeta[0:BT] <- beta[t_start : t_start+BT]
    if tidx < BT:
        sBeta[tidx] = gBeta_batch[t_start + tidx]
    cute.arch.barrier()
    
    # 对 T 和 K 维度分 tiles: (BC, BK, T/BC, K/BK) - 4D tensor
    gG = cute.local_tile(gG_batch, (BC, BK), (None, None))
    gQ = cute.local_tile(gQ_batch, (BC, BK), (None, None))
    gK = cute.local_tile(gK_batch, (BC, BK), (None, None))
    
    # Get thread-level copy partitions
    thr_copy_g = tiled_copy_g.get_slice(tidx)
    thr_copy_qk = tiled_copy_qk.get_slice(tidx)
    
    # Iterate all tiles in the chunk:
    #   iter -> (t_sub, k_tile)
    #   t_sub in [0, BT/BC), k_tile in [0, num_k_tiles)
    num_t_tiles = BT // BC
    total_tiles = num_t_tiles * num_k_tiles

    # =============== Prefetch first stages ===============
    prefetch_count = cutlass.min(NUM_STAGES - 1, total_tiles)
    for it in range(prefetch_count):
        stage = it % NUM_STAGES
        t_sub = it // num_k_tiles
        k_tile = it - t_sub * num_k_tiles

        t_tile_idx = t_tile_base + t_sub

        # Load g tile
        gG_tile = gG[(None, None, t_tile_idx, k_tile)]
        sG_stage = sG[(None, None, stage)]
        thr_gG = thr_copy_g.partition_S(gG_tile)
        thr_sG = thr_copy_g.partition_D(sG_stage)
        cute.copy(tiled_copy_g, thr_gG, thr_sG)

        # Load q tile (interleaved: stage*2)
        gQ_tile = gQ[(None, None, t_tile_idx, k_tile)]
        sQ_stage = sQK[(None, None, stage * 2)]
        thr_gQ = thr_copy_qk.partition_S(gQ_tile)
        thr_sQ = thr_copy_qk.partition_D(sQ_stage)
        cute.copy(tiled_copy_qk, thr_gQ, thr_sQ)

        # Load k tile (interleaved: stage*2 + 1)
        gK_tile = gK[(None, None, t_tile_idx, k_tile)]
        sK_stage = sQK[(None, None, stage * 2 + 1)]
        thr_gK = thr_copy_qk.partition_S(gK_tile)
        thr_sK = thr_copy_qk.partition_D(sK_stage)
        cute.copy(tiled_copy_qk, thr_gK, thr_sK)

        cute.arch.cp_async_commit_group()

    # =============== Main loop ===============
    for it in range(total_tiles):
        stage = it % NUM_STAGES

        # Wait for current stage data to be ready
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
        cute.arch.barrier()

        # Issue next async loads (overlapped with writeback)
        next_it = it + prefetch_count
        if next_it < total_tiles:
            next_stage = next_it % NUM_STAGES
            t_sub_n = next_it // num_k_tiles
            k_tile_n = next_it - t_sub_n * num_k_tiles
            t_tile_idx_n = t_tile_base + t_sub_n

            # Load next g tile
            gG_next = gG[(None, None, t_tile_idx_n, k_tile_n)]
            sG_next = sG[(None, None, next_stage)]
            thr_gG = thr_copy_g.partition_S(gG_next)
            thr_sG = thr_copy_g.partition_D(sG_next)
            cute.copy(tiled_copy_g, thr_gG, thr_sG)

            # Load next q tile (interleaved: next_stage*2)
            gQ_next = gQ[(None, None, t_tile_idx_n, k_tile_n)]
            sQ_next = sQK[(None, None, next_stage * 2)]
            thr_gQ = thr_copy_qk.partition_S(gQ_next)
            thr_sQ = thr_copy_qk.partition_D(sQ_next)
            cute.copy(tiled_copy_qk, thr_gQ, thr_sQ)

            # Load next k tile (interleaved: next_stage*2 + 1)
            gK_next = gK[(None, None, t_tile_idx_n, k_tile_n)]
            sK_next = sQK[(None, None, next_stage * 2 + 1)]
            thr_gK = thr_copy_qk.partition_S(gK_next)
            thr_sK = thr_copy_qk.partition_D(sK_next)
            cute.copy(tiled_copy_qk, thr_gK, thr_sK)

            cute.arch.cp_async_commit_group()

        # =============== Compute (current stage) ===============
        # 1. compute b_gm (ref: chunk_intra.py)
        # Layout: 4 warps × 4 rows/warp × 2 cols/thread
        #   - BC=16 rows, BK=64 cols, 128 threads (4 warps)
        #   - Each warp handles 4 rows, each thread handles 2 consecutive cols
        #   - Each thread: 4 rows × 2 cols = 8 elements in registers
        t_sub = it // num_k_tiles
        k_tile = it - t_sub * num_k_tiles
        t_abs_base = t_start + t_sub * BC

        # gn_row = min(BC//2, max(0, seq_len - t_abs_base - 1))
        gn_row = cutlass.min(BC // 2, cutlass.max(0, seq_len - t_abs_base - 1))

        # Thread mapping
        warp_id = tidx // 32
        lane_id = tidx % 32
        row_base = warp_id * 4      # each warp owns 4 rows
        col_base = lane_id * 2      # each thread owns 2 consecutive cols

        # Pre-allocate register tensors: (4 rows, 2 cols) per thread
        # r_qgq = q * exp2(b_gm)  -> for Aqk: dot(q*gq, kgk^T)
        r_qgq = cute.make_rmem_tensor(
            cute.make_layout((4, 2), stride=(2, 1)),
            cutlass.Float32
        )
        # r_kgq = k * exp2(b_gm)  -> for Akk: dot(k*gq, kgk^T)
        r_kgq = cute.make_rmem_tensor(
            cute.make_layout((4, 2), stride=(2, 1)),
            cutlass.Float32
        )
        # r_kgk = k * exp2(-b_gm) -> B matrix for both Aqk and Akk
        r_kgk = cute.make_rmem_tensor(
            cute.make_layout((4, 2), stride=(2, 1)),
            cutlass.Float32
        )

        # Load gn values (same row for all, broadcast across columns)
        gn_val_0 = sG[(gn_row, col_base, stage)]
        gn_val_1 = sG[(gn_row, col_base + 1, stage)]

        # 2. compute all fused element-wise products in one pass
        for ri in range(4):
            r = row_base + ri
            if t_abs_base + r < seq_len:
                g_val_0 = sG[(r, col_base, stage)]
                g_val_1 = sG[(r, col_base + 1, stage)]
                # Load q values from sQK (interleaved: stage*2)
                q_val_0 = sQK[(r, col_base, stage * 2)]
                q_val_1 = sQK[(r, col_base + 1, stage * 2)]
                # Load k values from sQK (interleaved: stage*2 + 1)
                k_val_0 = sQK[(r, col_base, stage * 2 + 1)]
                k_val_1 = sQK[(r, col_base + 1, stage * 2 + 1)]
                
                b_gm_0 = g_val_0 - gn_val_0
                b_gm_1 = g_val_1 - gn_val_1
                gq_0 = cute.math.exp2(b_gm_0)
                gq_1 = cute.math.exp2(b_gm_1)
                gk_0 = cute.math.exp2(-b_gm_0)
                gk_1 = cute.math.exp2(-b_gm_1)
                
                # Fused: q * gq (for Aqk)
                r_qgq[(ri, 0)] = q_val_0 * gq_0
                r_qgq[(ri, 1)] = q_val_1 * gq_1
                # Fused: k * gq (for Akk)
                r_kgq[(ri, 0)] = k_val_0 * gq_0
                r_kgq[(ri, 1)] = k_val_1 * gq_1
                # Fused: k * gk (B matrix, will be transposed in GEMM)
                r_kgk[(ri, 0)] = k_val_0 * gk_0
                r_kgk[(ri, 1)] = k_val_1 * gk_1
            else:
                # Out of bounds: set to 0 (same as Triton's tl.where(m_c, ..., 0.))
                r_qgq[(ri, 0)] = cutlass.Float32(0.0)
                r_qgq[(ri, 1)] = cutlass.Float32(0.0)
                r_kgq[(ri, 0)] = cutlass.Float32(0.0)
                r_kgq[(ri, 1)] = cutlass.Float32(0.0)
                r_kgk[(ri, 0)] = cutlass.Float32(0.0)
                r_kgk[(ri, 1)] = cutlass.Float32(0.0)
        
        # 3. barrier to ensure all threads finished reading q/k from sQK before overwriting
        cute.arch.barrier()
        
        # 4. write r_kgk and r_qgq to SMEM
        # r_kgk -> sG (reuse g's current stage)
        # r_qgq -> sQK_f32 (reuse q+k position as float32)
        for ri in range(4):
            r = row_base + ri
            sG[(r, col_base, stage)] = r_kgk[(ri, 0)]
            sG[(r, col_base + 1, stage)] = r_kgk[(ri, 1)]
            sQK_f32[(r, col_base, stage)] = r_qgq[(ri, 0)]
            sQK_f32[(r, col_base + 1, stage)] = r_qgq[(ri, 1)]
        
        cute.arch.barrier()
        
        # 5. MMA: Compute both Aqk and Akk
        #   Aqk = r_qgq @ r_kgk^T  -> sAccum[:, :, 0]
        #   Akk = r_kgq @ r_kgk^T  -> sAccum[:, :, 1]
        # 
        # Data in SMEM:
        #   A matrix (r_qgq): sQK_f32[:, :, stage] - 16x64 fp32
        #   B matrix (r_kgk): sG[:, :, stage] - 16x64 fp32
        #   A matrix (r_kgq): still in registers, need to write to sQK_f32 after Aqk MMA
        #
        # Strategy: Sequential MMA
        #   1. First compute Aqk using r_qgq (already in sQK_f32)
        #   2. Then write r_kgq to sQK_f32 and compute Akk
        #   B matrix (r_kgk) is shared for both MMAs
        
        t_sub = it // num_k_tiles
        k_tile = it - t_sub * num_k_tiles
        
        lane_id_mma = tidx % 32
        k_start = warp_id * 2  # warp 0 -> k=0,1; warp 1 -> k=2,3; etc.
        group_id = lane_id_mma // 4
        tid_in_group = lane_id_mma % 4
        warp_col_base = warp_id * BC  # warp_id * 16
        
        # ========== Aqk MMA (A = r_qgq, already in sQK_f32) ==========
        # Initialize accumulators first (CuTe DSL requires init before control flow)
        aqk_c00_0 = cutlass.Float32(0.0)
        aqk_c00_1 = cutlass.Float32(0.0)
        aqk_c00_2 = cutlass.Float32(0.0)
        aqk_c00_3 = cutlass.Float32(0.0)
        aqk_c01_0 = cutlass.Float32(0.0)
        aqk_c01_1 = cutlass.Float32(0.0)
        aqk_c01_2 = cutlass.Float32(0.0)
        aqk_c01_3 = cutlass.Float32(0.0)
        
        # k_tile > 0: Load previous partial sums from sAccum[:, :, 0]
        if k_tile != 0:
            aqk_c00_0 = sAccum[(group_id, warp_col_base + tid_in_group * 2, 0)]
            aqk_c00_1 = sAccum[(group_id, warp_col_base + tid_in_group * 2 + 1, 0)]
            aqk_c00_2 = sAccum[(group_id + 8, warp_col_base + tid_in_group * 2, 0)]
            aqk_c00_3 = sAccum[(group_id + 8, warp_col_base + tid_in_group * 2 + 1, 0)]
            aqk_c01_0 = sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2, 0)]
            aqk_c01_1 = sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2 + 1, 0)]
            aqk_c01_2 = sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2, 0)]
            aqk_c01_3 = sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2 + 1, 0)]
        
        # Aqk MMA loop
        for k_iter in cutlass.range_constexpr(2):
            k = k_start + k_iter
            
            # Load A tile from sQK_f32 (r_qgq)
            sA_tile = cute.local_tile(sQK_f32[(None, None, stage)], tiler=(16, 8), coord=(0, k))
            a0, a1, a2, a3 = load_A_tf32(sA_tile, lane_id_mma)
            
            # Load B tiles from sG (r_kgk) - shared for both Aqk and Akk
            sB_tile_0 = cute.local_tile(sG[(None, None, stage)], tiler=(8, 8), coord=(0, k))
            b0_0, b1_0 = load_B_tf32_from_rowmajor(sB_tile_0, lane_id_mma)
            
            aqk_c00_0, aqk_c00_1, aqk_c00_2, aqk_c00_3 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0_0, b1_0, aqk_c00_0, aqk_c00_1, aqk_c00_2, aqk_c00_3)
            
            sB_tile_1 = cute.local_tile(sG[(None, None, stage)], tiler=(8, 8), coord=(1, k))
            b0_1, b1_1 = load_B_tf32_from_rowmajor(sB_tile_1, lane_id_mma)
            
            aqk_c01_0, aqk_c01_1, aqk_c01_2, aqk_c01_3 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0_1, b1_1, aqk_c01_0, aqk_c01_1, aqk_c01_2, aqk_c01_3)
        
        # Store Aqk partial result to sAccum[:, :, 0]
        sAccum[(group_id, warp_col_base + tid_in_group * 2, 0)] = aqk_c00_0
        sAccum[(group_id, warp_col_base + tid_in_group * 2 + 1, 0)] = aqk_c00_1
        sAccum[(group_id + 8, warp_col_base + tid_in_group * 2, 0)] = aqk_c00_2
        sAccum[(group_id + 8, warp_col_base + tid_in_group * 2 + 1, 0)] = aqk_c00_3
        sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2, 0)] = aqk_c01_0
        sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_1
        sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2, 0)] = aqk_c01_2
        sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_3
        
        # ========== Write r_kgq to sQK_f32 (overwrite r_qgq) ==========
        cute.arch.barrier()
        for ri in range(4):
            r = row_base + ri
            sQK_f32[(r, col_base, stage)] = r_kgq[(ri, 0)]
            sQK_f32[(r, col_base + 1, stage)] = r_kgq[(ri, 1)]
        cute.arch.barrier()
        
        # ========== Akk MMA (A = r_kgq, now in sQK_f32) ==========
        # Initialize accumulators first (CuTe DSL requires init before control flow)
        akk_c00_0 = cutlass.Float32(0.0)
        akk_c00_1 = cutlass.Float32(0.0)
        akk_c00_2 = cutlass.Float32(0.0)
        akk_c00_3 = cutlass.Float32(0.0)
        akk_c01_0 = cutlass.Float32(0.0)
        akk_c01_1 = cutlass.Float32(0.0)
        akk_c01_2 = cutlass.Float32(0.0)
        akk_c01_3 = cutlass.Float32(0.0)
        
        # k_tile > 0: Load previous partial sums from sAccum[:, :, 1]
        if k_tile != 0:
            akk_c00_0 = sAccum[(group_id, warp_col_base + tid_in_group * 2, 1)]
            akk_c00_1 = sAccum[(group_id, warp_col_base + tid_in_group * 2 + 1, 1)]
            akk_c00_2 = sAccum[(group_id + 8, warp_col_base + tid_in_group * 2, 1)]
            akk_c00_3 = sAccum[(group_id + 8, warp_col_base + tid_in_group * 2 + 1, 1)]
            akk_c01_0 = sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2, 1)]
            akk_c01_1 = sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2 + 1, 1)]
            akk_c01_2 = sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2, 1)]
            akk_c01_3 = sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2 + 1, 1)]
        
        # Akk MMA loop
        for k_iter in cutlass.range_constexpr(2):
            k = k_start + k_iter
            
            # Load A tile from sQK_f32 (r_kgq)
            sA_tile = cute.local_tile(sQK_f32[(None, None, stage)], tiler=(16, 8), coord=(0, k))
            a0, a1, a2, a3 = load_A_tf32(sA_tile, lane_id_mma)
            
            # Load B tiles from sG (r_kgk) - same as before
            sB_tile_0 = cute.local_tile(sG[(None, None, stage)], tiler=(8, 8), coord=(0, k))
            b0_0, b1_0 = load_B_tf32_from_rowmajor(sB_tile_0, lane_id_mma)
            
            akk_c00_0, akk_c00_1, akk_c00_2, akk_c00_3 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0_0, b1_0, akk_c00_0, akk_c00_1, akk_c00_2, akk_c00_3)
            
            sB_tile_1 = cute.local_tile(sG[(None, None, stage)], tiler=(8, 8), coord=(1, k))
            b0_1, b1_1 = load_B_tf32_from_rowmajor(sB_tile_1, lane_id_mma)
            
            akk_c01_0, akk_c01_1, akk_c01_2, akk_c01_3 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0_1, b1_1, akk_c01_0, akk_c01_1, akk_c01_2, akk_c01_3)
        
        # Store Akk partial result to sAccum[:, :, 1]
        sAccum[(group_id, warp_col_base + tid_in_group * 2, 1)] = akk_c00_0
        sAccum[(group_id, warp_col_base + tid_in_group * 2 + 1, 1)] = akk_c00_1
        sAccum[(group_id + 8, warp_col_base + tid_in_group * 2, 1)] = akk_c00_2
        sAccum[(group_id + 8, warp_col_base + tid_in_group * 2 + 1, 1)] = akk_c00_3
        sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2, 1)] = akk_c01_0
        sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2 + 1, 1)] = akk_c01_1
        sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2, 1)] = akk_c01_2
        sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2 + 1, 1)] = akk_c01_3
        
        cute.arch.barrier()
        
        # 6. Reduction: only after processing all k_tiles for this t_sub
        if k_tile == num_k_tiles - 1:
            # Sum all 4 warps' partial results for both Aqk and Akk
            # 128 threads, 16x16=256 elements, so 2 elements per thread
            elem_idx = tidx * 2
            if elem_idx < BC * BC:
                row0 = elem_idx // BC
                col0 = elem_idx % BC
                row1 = (elem_idx + 1) // BC
                col1 = (elem_idx + 1) % BC
                
                # Aqk reduction with scale, apply mask (row >= col)
                sum_aqk_0 = sAccum[(row0, col0, 0)] + sAccum[(row0, col0 + 16, 0)] + sAccum[(row0, col0 + 32, 0)] + sAccum[(row0, col0 + 48, 0)]
                sum_aqk_1 = sAccum[(row1, col1, 0)] + sAccum[(row1, col1 + 16, 0)] + sAccum[(row1, col1 + 32, 0)] + sAccum[(row1, col1 + 48, 0)]
                # mask: row >= col (lower triangular with diagonal), else 0
                if row0 >= col0:
                    sAccum[(row0, col0, 0)] = sum_aqk_0 * scale
                else:
                    sAccum[(row0, col0, 0)] = cutlass.Float32(0.0)
                if row1 >= col1:
                    sAccum[(row1, col1, 0)] = sum_aqk_1 * scale
                else:
                    sAccum[(row1, col1, 0)] = cutlass.Float32(0.0)
                
                # Akk reduction with beta, apply mask (row > col)
                # beta index: t_sub * BC + row (within BT range)
                beta_idx_0 = t_sub * BC + row0
                beta_idx_1 = t_sub * BC + row1
                beta_0 = cutlass.Float32(sBeta[(beta_idx_0,)])
                beta_1 = cutlass.Float32(sBeta[(beta_idx_1,)])
                
                sum_akk_0 = sAccum[(row0, col0, 1)] + sAccum[(row0, col0 + 16, 1)] + sAccum[(row0, col0 + 32, 1)] + sAccum[(row0, col0 + 48, 1)]
                sum_akk_1 = sAccum[(row1, col1, 1)] + sAccum[(row1, col1 + 16, 1)] + sAccum[(row1, col1 + 32, 1)] + sAccum[(row1, col1 + 48, 1)]
                # mask: row > col (strictly lower triangular), else 0
                if row0 > col0:
                    sAccum[(row0, col0, 1)] = sum_akk_0 * beta_0
                else:
                    sAccum[(row0, col0, 1)] = cutlass.Float32(0.0)
                if row1 > col1:
                    sAccum[(row1, col1, 1)] = sum_akk_1 * beta_1
                else:
                    sAccum[(row1, col1, 1)] = cutlass.Float32(0.0)
            
            cute.arch.barrier()
            
            # 7. Apply mask and write back to global memory
            # Aqk: mask m_Aqk = (row >= col), i.e., lower triangular including diagonal
            # Akk: mask m_Akk = (row > col), i.e., strictly lower triangular
            #
            # Output layout:
            #   Aqk_tensor: (B, T, H, BT) -> write to [i_b, t_abs_base + row, i_h, t_sub * BC + col]
            #   Akk_tensor: (B, T, H, BC) -> write to [i_b, t_abs_base + row, i_h, col]
            t_abs_base = t_start + t_sub * BC
            
            # 128 threads, 256 elements, 2 elements per thread
            linear = tidx
            while linear < (BC * BC):
                row = linear // BC
                col = linear % BC
                
                # Read from reduced sAccum[:, 0:16, :] (scale/beta and mask already applied)
                val_aqk = sAccum[(row, col, 0)]  # fp32, masked with 0 for upper triangle
                val_akk = sAccum[(row, col, 1)]  # fp32, masked with 0 for upper triangle + diagonal
                
                # Direct write back (mask already applied in reduction)
                Aqk_tensor[(i_b, t_abs_base + row, i_h, t_sub * BC + col)] = cutlass.Float16(val_aqk)
                Akk_tensor[(i_b, t_abs_base + row, i_h, col)] = val_akk
                
                linear = linear + NUM_THREADS

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
    
    # =============== Create tiled copy for g (float32) ===============
    # 128 bits = 4 floats per copy
    # Thread layout: (8, 16) -> 128 threads
    # Val layout: (1, 4) -> 4 floats per thread
    # Coverage per iteration: 8 rows x 64 cols, need 2 iterations for 16 rows
    copy_atom_g = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128
    )
    thread_layout_g = cute.make_layout(
        (8, 16),       # 8 rows x 16 threads per row
        stride=(16, 1)
    )
    val_layout_g = cute.make_layout((1, 4))  # 4 floats per thread
    tiled_copy_g = cute.make_tiled_copy_tv(copy_atom_g, thread_layout_g, val_layout_g)
    
    # =============== Create tiled copy for q/k (fp16) ===============
    # 128 bits = 8 halfs per copy
    # Thread layout: (16, 8) -> 128 threads
    # Val layout: (1, 8) -> 8 halfs per thread
    # Coverage: 16 rows x 64 cols in one pass
    copy_atom_qk = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float16,
        num_bits_per_copy=128
    )
    thread_layout_qk = cute.make_layout(
        (16, 8),       # 16 rows x 8 threads per row
        stride=(8, 1)
    )
    val_layout_qk = cute.make_layout((1, 8))  # 8 halfs per thread
    tiled_copy_qk = cute.make_tiled_copy_tv(copy_atom_qk, thread_layout_qk, val_layout_qk)
    
    # =============== SMEM layouts ===============
    # g: (BC, BK, NUM_STAGES) = (16, 64, 2)
    g_smem_layout = cute.make_layout(
        (BC, BK, NUM_STAGES),
        stride=(BK, 1, BC * BK)
    )
    # q and k share smem with interleaved stages: (BC, BK, NUM_STAGES * 2)
    # Interleaved layout: [q0][k0][q1][k1] so q and k for same stage are adjacent
    # This allows reusing q+k area (2 consecutive float16 stages) as one float32 stage
    qk_smem_layout = cute.make_layout(
        (BC, BK, NUM_STAGES * 2),
        stride=(BK, 1, BC * BK)
    )
    
    # Float32 view for reusing q+k smem space (for r_qgq)
    # Each fp32 stage spans 2 fp16 stages (q+k), so stride on 3rd dim is doubled
    qk_smem_layout_f32 = cute.make_layout(
        (BC, BK, NUM_STAGES),
        stride=(BK, 1, BC * BK * 2)
    )
    
    # accum: (BC, BC * NUM_WARPS, 2) for Aqk (slot 0) and Akk (slot 1)
    accum_smem_layout = cute.make_layout(
        (BC, BC * NUM_WARPS, 2),
        stride=(BC * NUM_WARPS * 2, 2, 1)
    )

    # beta: (BT,) fp16
    beta_smem_layout = cute.make_layout(
        (BT,),
        stride=(1,)
    )
    
    # SMEM size calculation
    # g: 16 * 64 * 2 * 4 bytes = 8192 bytes
    # qk: 16 * 64 * 4 * 2 bytes = 8192 bytes
    # accum: 16 * 64 * 2 * 4 bytes = 8192 bytes (Aqk + Akk)
    # beta: BT * 2 bytes
    smem_bytes = (4 * BC * BK * NUM_STAGES +       # sG
                  2 * BC * BK * NUM_STAGES * 2 +   # sQK
                  4 * BC * BC * NUM_WARPS * 2 +    # sAccum (2 slots: Aqk, Akk)
                  2 * BT +                          # sBeta
                  128)                              # alignment
    
    print(f"KDA Akk: B={B}, T={seq_len}, H={H}, K={K}")
    print(f"  Tiles: NT={NT}, num_k_tiles={num_k_tiles}")
    print(f"  SMEM: {smem_bytes} bytes\n")
    
    kda_Akk_kernel(
        tiled_copy_g,
        tiled_copy_qk,
        g_tensor,
        q_tensor,
        k_tensor,
        beta_tensor,
        Akk_tensor,
        Aqk_tensor,
        g_smem_layout,
        qk_smem_layout,
        qk_smem_layout_f32,
        accum_smem_layout,
        beta_smem_layout,
        BT,
        num_k_tiles,
        seq_len,
        scale,
    ).launch(
        grid=(B, NT, H),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


if __name__ == "__main__":
    print("KDA Akk cp.async Test")
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
    
    # Output tensors (debug writeback buffers)
    # - Akk: (B, seq_len, H, BC) fp32
    # - Aqk: (B, seq_len, H, BT) fp16
    Akk = torch.empty(B, seq_len, H, BC, dtype=torch.float32, device='cuda')
    Aqk = torch.empty(B, seq_len, H, BT, dtype=torch.float16, device='cuda')
    
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
    Aqk_ref = torch.empty(B, seq_len, H, BT, device='cuda', dtype=torch.float16)
    Akk_ref = torch.empty(B, seq_len, H, BC, device='cuda', dtype=torch.float32)
    
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
    print("\n--- Sample: Aqk[0, 0:8, 0, 0:8] ---")
    print("CuTe:")
    print(Aqk[0, 0:8, 0, 0:8].detach().cpu())
    print("Triton:")
    print(Aqk_ref[0, 0:8, 0, 0:8].detach().cpu())
    
    print("\n--- Sample: Akk[0, 0:8, 0, 0:8] ---")
    print("CuTe:")
    print(Akk[0, 0:8, 0, 0:8].detach().cpu())
    print("Triton:")
    print(Akk_ref[0, 0:8, 0, 0:8].detach().cpu())
    
    # Pass/Fail check
    # TF32 MMA has ~1e-3 relative error, so use relaxed thresholds
    aqk_threshold = 1e-2  # fp16 output
    akk_threshold = 1e-3  # fp32 output
    
    if max_diff_aqk > aqk_threshold or max_diff_akk > akk_threshold:
        print(f"\n✗ FAIL: Aqk diff > {aqk_threshold} or Akk diff > {akk_threshold}")
    else:
        print(f"\n✓ PASS: Results match within tolerance")
    
    # Benchmark (torch.profiler only)
    print(f"\nBenchmarking: {test_iters} iterations...")

    # L2 cache eviction buffer (match benchmark_all.py idea)
    dummy_buffer = torch.empty(int(80 * 1024 * 1024 / 4), dtype=torch.float32, device='cuda')

    if not use_profiler:
        raise SystemExit("use_profiler=False is not supported (CUDA Events timing removed).")

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
    # - Read:  g(fp32) + q(fp16) + k(fp16)  -> full K
    #          beta(fp16)                   -> BT per token (here modeled as full seq_len)
    # - Write: Akk(fp32) -> only first BC columns
    #          Aqk(fp16) -> only first BC columns (even though Aqk is allocated with BT columns)
    read_bytes = B * seq_len * H * K * (4 + 2 + 2) + (B * seq_len * H * 2)
    write_bytes = (B * seq_len * H * BC * 4) + (B * seq_len * H * BC * 2)
    data_bytes = read_bytes + write_bytes
    data_mb = data_bytes / 1024 / 1024
    # Use decimal GB to match "peak bandwidth = 8192" convention
    data_gib = data_bytes / 1000 / 1000 / 1000
    
    print("\n" + "=" * 50)
    print(f"✓ CuTe Results:")
    print(f"  Data: {data_mb:.2f} MB")
    cute_mean_ms = None
    if len(profiler_times_us) > 0:
        prof = torch.tensor(profiler_times_us, dtype=torch.float64)
        prof_ms = prof / 1000.0
        cute_mean_ms = prof_ms.mean().item()
        min_ms = prof_ms.min().item()
        bw_gibs = data_gib / (cute_mean_ms / 1000.0)
        peak_bw = 4814
        peak_pct = bw_gibs / peak_bw * 100.0
        print(f"  Bytes (read/write/total): {read_bytes} / {write_bytes} / {data_bytes}")
        print(f"  Profiler Mean: {cute_mean_ms:.4f} ms (kernel_cutlass only)")
        print(f"  Profiler Min:  {min_ms:.4f} ms (kernel_cutlass only)")
        print(f"  BW (profiler mean): {bw_gibs:.1f} GB/s")
        print(f"  Peak% (peak={peak_bw:.0f}): {peak_pct:.2f}%")
        if len(profiler_times_us) != test_iters:
            print(f"  Note: kernel_count={len(profiler_times_us)} != test_iters={test_iters} (mean is over kernel events).")
    else:
        print("  ✗ No kernel timings parsed; bandwidth unavailable.")
    print("=" * 50)
    
    # =============== Triton Benchmark ===============
    print(f"\n" + "=" * 50)
    print(f"Triton Benchmark: {test_iters} iterations...")
    print("=" * 50)
    
    # Warmup
    for _ in range(10):
        chunk_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q, k=k, g=g, beta=beta, Aqk=Aqk_ref, Akk=Akk_ref, scale=scale,
            cu_seqlens=None, chunk_indices=None,
            T=seq_len, H=H, K=K, BT=BT, BC=BC, BK=BK_triton,
            IS_VARLEN=False, USE_GATHER=IS_GATHER_SUPPORTED,
        )
    torch.cuda.synchronize()
    
    # Profile Triton
    profiler_triton = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    )
    
    with profiler_triton:
        for i in range(test_iters):
            _ = dummy_buffer.sum()
            torch.cuda.synchronize()
            with torch.profiler.record_function(f"triton_iter{i}"):
                chunk_kda_fwd_kernel_intra_sub_chunk[grid](
                    q=q, k=k, g=g, beta=beta, Aqk=Aqk_ref, Akk=Akk_ref, scale=scale,
                    cu_seqlens=None, chunk_indices=None,
                    T=seq_len, H=H, K=K, BT=BT, BC=BC, BK=BK_triton,
                    IS_VARLEN=False, USE_GATHER=IS_GATHER_SUPPORTED,
                )
            torch.cuda.synchronize()
    
    trace_file_triton = os.path.join(trace_dir, "kda_Akk_triton.json")
    profiler_triton.export_chrome_trace(trace_file_triton)
    print(f"  ✓ Triton profiler trace saved to {trace_file_triton}")
    
    triton_times_us = []
    try:
        with open(trace_file_triton, "r") as f:
            trace_data = json.load(f)
        for event in trace_data.get("traceEvents", []):
            if event.get("cat") != "kernel":
                continue
            name = event.get("name", "")
            dur = event.get("dur", 0)
            # Extract Triton kernels (usually named chunk_kda_fwd_kernel_intra_sub_chunk or similar)
            if dur > 0 and "chunk_kda" in name.lower():
                triton_times_us.append(dur)
        if triton_times_us:
            print(f"  ✓ Parsed {len(triton_times_us)} Triton kernel timings from profiler trace")
        else:
            print("  ⚠ No Triton kernel timings found in profiler trace")
    except Exception as e:
        print(f"  ✗ Failed to parse Triton profiler trace: {e}")
        triton_times_us = []
    
    print("\n" + "=" * 50)
    print(f"✓ Triton Results:")
    triton_mean_ms = None
    if len(triton_times_us) > 0:
        prof_triton = torch.tensor(triton_times_us, dtype=torch.float64)
        prof_triton_ms = prof_triton / 1000.0
        triton_mean_ms = prof_triton_ms.mean().item()
        triton_min_ms = prof_triton_ms.min().item()
        triton_bw_gibs = data_gib / (triton_mean_ms / 1000.0)
        triton_peak_pct = triton_bw_gibs / peak_bw * 100.0
        print(f"  Profiler Mean: {triton_mean_ms:.4f} ms")
        print(f"  Profiler Min:  {triton_min_ms:.4f} ms")
        print(f"  BW (profiler mean): {triton_bw_gibs:.1f} GB/s")
        print(f"  Peak% (peak={peak_bw:.0f}): {triton_peak_pct:.2f}%")
    else:
        print("  ✗ No kernel timings parsed; bandwidth unavailable.")
    print("=" * 50)
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("Performance Comparison:")
    print("=" * 50)
    if cute_mean_ms and triton_mean_ms:
        speedup = triton_mean_ms / cute_mean_ms
        print(f"  CuTe Mean:   {cute_mean_ms:.4f} ms")
        print(f"  Triton Mean: {triton_mean_ms:.4f} ms")
        print(f"  Speedup (Triton/CuTe): {speedup:.2f}x")
        if speedup > 1:
            print(f"  → CuTe is {speedup:.2f}x FASTER than Triton")
        else:
            print(f"  → Triton is {1/speedup:.2f}x FASTER than CuTe")
    print("=" * 50)
    print("DONE")
    
    del compiled
