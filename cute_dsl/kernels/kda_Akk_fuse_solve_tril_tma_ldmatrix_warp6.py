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
    """Invert a 16x16 unit lower triangular block, write FP32 result.
    
    Register-optimized: eliminates rA register array by re-reading values
    from shared memory (sA_in). Sums are computed incrementally so only
    one shuffle result + one SMEM read are live at a time (instead of up
    to 14 shuffle results simultaneously at d=15).
    
    Register savings vs original manual unrolling:
      - rA[16] eliminated:  ~16 FP32 registers saved
      - Named shuffle temps eliminated: ~14 FP32 registers saved at peak
      - Total: ~30 fewer live registers at peak (d=15)
    """
    my_row = lane_id % 16
    halfwarp_base = (lane_id // 16) * 16
    
    row_off = diag_offset
    col_off = diag_offset
    
    # Only rInv register array needed (rA eliminated - values re-read from sA_in)
    rInv = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    
    rInv[0] = cutlass.Float32(1.0)
    for x in range(1, 16):
        rInv[x] = cutlass.Float32(0.0)
    
    # Anti-diagonal sweep d=1..15
    # For each d, compute: rInv[d] = (-A[my_row, my_row-d] - sum) * valid
    # where sum = sum_{j=1}^{d-1} A[my_row, my_row-(d-j)] * shuffle(rInv[j], my_row-d+j)
    for d in range(1, 16):
        col_d = my_row - d
        valid = cutlass.Float32(col_d >= 0)
        # Read A[my_row, my_row-d] directly from shared memory
        a_val = cutlass.Float32(sA_in[row_off + my_row, col_off + col_d]) * valid
        
        # Incremental sum to minimize live registers:
        # each iteration only needs acc + one shuffle result + one SMEM read
        acc = cutlass.Float32(0.0)
        for j in range(1, d):
            # Re-read A[my_row, my_row-(d-j)] from SMEM (replaces rA[d-j])
            a_re = cutlass.Float32(sA_in[row_off + my_row, col_off + my_row - (d - j)])
            # Get Inv[my_row-d+j, my_row-d] via shuffle from thread (my_row-d+j)
            inv_shfl = cute.arch.shuffle_sync(rInv[j], halfwarp_base + my_row - d + j)
            acc = acc + a_re * inv_shfl
        
        rInv[d] = (-a_val - acc) * valid
    
    rInv[0] = cutlass.Float32(1.0)
    
    # Write inverse to FP32 output
    sA_out[row_off + my_row, col_off + my_row] = rInv[0]
    for d in range(1, 16):
        sA_out[row_off + my_row, col_off + (my_row + 16 - d) % 16] = rInv[d] * cutlass.Float32(my_row >= d)


# ===========================================================================
# Named barrier for compute warps (0-3, 128 threads)
# ===========================================================================
@dsl_user_op
def compute_warp_barrier(*, loc=None, ip=None):
    """Named barrier sync for compute warps (0-3, 128 threads).
    Uses bar.sync with barrier_id=2. membar.cta ensures shared memory
    writes are visible across warps (bar.sync alone only guarantees
    memory fence when thread_count == CTA size).
    """
    llvm.inline_asm(
        T.i32(),
        [],
        "membar.cta; bar.sync 2, 128; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


# 全局配置
BC = 16          # sub-chunk size (rows)
BK = 64          # key dimension (columns)
NUM_STAGES = 2   # double buffer
NUM_WARPS = 6    # warps 0-3: MMA compute, warp 4: writeback + inversion, warp 5: TMA producer
NUM_ACC_PIPE_STAGES = 2  # double buffer for accumulator pipeline (warp specialization)

# MMA C layout offsets for m16n16 output (two n-tiles of m16n8k8)
# Index order: [c00_0, c00_1, c00_2, c00_3, c01_0, c01_1, c01_2, c01_3]
_C_ROW = [0, 0, 8, 8, 0, 0, 8, 8]   # added to group_id
_C_COL = [0, 1, 0, 1, 8, 9, 8, 9]   # added to col_base + tid_in_group*2

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
    accum_smem_layout: cute.Layout, # (BC, BC * 2, NUM_ACC_PIPE_STAGES * 2) double-buffered Aqk/Akk
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
    - 6 warps: warps 0-3 do MMA compute + tree reduction,
      warp 4 is dedicated to writeback (Aqk/Akk) + 16x16 inversion,
      warp 5 is dedicated TMA producer
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
    
    # Allocate mbarriers for accumulator pipeline (warp specialization)
    # acc_full: producer (warp 3, 32 threads) signals data ready -> consumer (warp 4) waits
    # acc_empty: consumer (warp 4, 1 thread) signals buffer free -> producer (warps 0-3) waits
    acc_full_mbar_ptr = smem.allocate_array(cutlass.Int64, NUM_ACC_PIPE_STAGES)
    acc_empty_mbar_ptr = smem.allocate_array(cutlass.Int64, NUM_ACC_PIPE_STAGES)
    
    # tma_consumed: compute warps (0-3, 128 threads) signal stage consumed -> TMA warp (5) waits
    tma_consumed_mbar_ptr = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    
    # TMA tile size in bytes
    g_tile_bytes = BC * BK * 4   # float32
    qk_tile_bytes = BC * BK * 2  # fp16
    total_tile_bytes = g_tile_bytes + qk_tile_bytes * 2  # g + q + k
    
    # Initialize mbarriers using pointer offset
    if tidx == 0:
        for stage in range(NUM_STAGES):
            cute.arch.mbarrier_init(mbar_ptr + stage, 1)
        for stage in range(NUM_ACC_PIPE_STAGES):
            cute.arch.mbarrier_init(acc_full_mbar_ptr + stage, 32)   # 32 threads in warp 3 arrive
            cute.arch.mbarrier_init(acc_empty_mbar_ptr + stage, 1)   # 1 thread in warp 4 arrives
        for stage in range(NUM_STAGES):
            cute.arch.mbarrier_init(tma_consumed_mbar_ptr + stage, 128)  # 128 threads in warps 0-3 arrive
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    
    # Pre-arrive acc_empty mbarriers so buffers start as "empty" (available)
    # This flips phase from 0->1, so first mbarrier_wait(phase=0) succeeds immediately
    if tidx == 0:
        for stage in range(NUM_ACC_PIPE_STAGES):
            cute.arch.mbarrier_arrive(acc_empty_mbar_ptr + stage)
    
    # Pre-arrive tma_consumed mbarriers so stages start as "consumed" (available for TMA)
    # All 128 threads of warps 0-3 arrive, flipping phase from 0->1
    if warp_idx < 4:
        for stage in range(NUM_STAGES):
            cute.arch.mbarrier_arrive(tma_consumed_mbar_ptr + stage)
    
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

    # =============== Warp-specialized main loop ===============
    # Warps 0-3: MMA compute + tree reduction (producer)
    # Warp 4:    Dedicated writeback + inversion (consumer)
    # Warp 5:    Dedicated TMA producer
    # Double-buffered sAccum with mbarrier pipeline for overlap.
    # Double-buffered sG/sQK with tma_consumed mbarrier for stage reuse.
    
    if warp_idx < 4:
        # ======= Compute warps (0-3): MMA + Tree Reduction =======
        # Accumulators: rmem_tensor[8] for aqk and akk (m16n16 = 2 n-tiles of m16n8k8)
        aqk = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
        akk = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
        for _i in cutlass.range_constexpr(8, unroll=True):
            aqk[_i] = cutlass.Float32(0.0)
            akk[_i] = cutlass.Float32(0.0)
        
        for it in range(total_tiles):
            stage = it % NUM_STAGES
            phase = it // NUM_STAGES % 2

            # Wait for current stage TMA data to be ready (warp 5 handles TMA)
            cute.arch.mbarrier_wait(mbar_ptr + stage, phase)

            # =============== Fused Element-wise + MMA using ldmatrix ===============
            t_sub = it // num_k_tiles
            k_tile = it - t_sub * num_k_tiles
            t_abs_base = t_start + t_sub * BC
            
            # Thread/warp mapping (needed by both MMA and tree reduction)
            warp_id = tidx // 32
            lane_id = tidx % 32
            group_id = lane_id // 4      # 0-7, determines rows (group_id and group_id+8)
            tid_in_group = lane_id % 4   # 0-3, determines cols within 8-col tile (tid*2 and tid*2+1)
            
            # ===== MMA computation (warps 0-3) =====
            # gn_row for normalization
            gn_row = cutlass.min(BC // 2, cutlass.max(0, seq_len - t_abs_base - 1))
            k_start = warp_id * 2        # warp 0 -> k=0,1; warp 1 -> k=2,3; etc.
            
            # Get thread slices for ldmatrix and G copy
            thr_mma = tiled_mma.get_slice(lane_id)
            thr_copy_Q = tiled_copy_Q.get_slice(lane_id)
            thr_copy_K = tiled_copy_K.get_slice(lane_id)
            thr_copy_G = tiled_copy_G.get_slice(lane_id)
            
            # ========== Reset accumulators at k_tile == 0 (new t_sub) ==========
            if k_tile == 0:
                for _i in cutlass.range_constexpr(8, unroll=True):
                    aqk[_i] = cutlass.Float32(0.0)
                    akk[_i] = cutlass.Float32(0.0)
            
            # Get 2D SMEM views for current stage
            sQ_stage = sQK[(None, None, stage * 2)]       # Q: (BC, BK)
            sK_stage = sQK[(None, None, stage * 2 + 1)]   # K: (BC, BK)
            sG_stage = sG[(None, None, stage)]            # G: (BC, BK)
            
            # ========== Fused element-wise + MMA loop using ldmatrix ==========
            for k_iter in cutlass.range_constexpr(2, unroll=True):
                k = k_start + k_iter
                k_offset = k * 8  # column offset for this k sub-tile
                col = tid_in_group * 2  # 0,2,4,6 within each 8-col tile
                
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

                # After last k_iter's SMEM reads, signal warp 5 that stage is consumed
                # All 128 threads arrive; warp 5 can then reuse this stage buffer
                # This lets TMA overlap with the remaining register-only computation (exp2 + MMA)
                if k_iter == 1:
                    cute.arch.mbarrier_arrive(tma_consumed_mbar_ptr + stage)

                # ===== Compute exp2 factors =====
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
                q_r0_0 = tCrQ[0].to(cutlass.Float32)
                q_r1_0 = tCrQ[2].to(cutlass.Float32)
                q_r0_1 = tCrQ[1].to(cutlass.Float32)
                q_r1_1 = tCrQ[3].to(cutlass.Float32)
                
                # ===== Extract K values from ldmatrix registers =====
                k_r0_0 = tCrK_n0[0].to(cutlass.Float32)
                k_r0_1 = tCrK_n0[1].to(cutlass.Float32)
                k_r1_0 = tCrK_n1[0].to(cutlass.Float32)
                k_r1_1 = tCrK_n1[1].to(cutlass.Float32)
                
                # ===== Element-wise multiply for A matrices =====
                a_aqk_0 = q_r0_0 * gq_r0_0
                a_aqk_1 = q_r1_0 * gq_r1_0
                a_aqk_2 = q_r0_1 * gq_r0_1
                a_aqk_3 = q_r1_1 * gq_r1_1
                
                a_akk_0 = k_r0_0 * gq_r0_0
                a_akk_1 = k_r1_0 * gq_r1_0
                a_akk_2 = k_r0_1 * gq_r0_1
                a_akk_3 = k_r1_1 * gq_r1_1
                
                # ===== Element-wise multiply for B matrices (k * gk) =====
                b0_0 = k_r0_0 * gk_r0_0
                b1_0 = k_r0_1 * gk_r0_1
                b0_1 = k_r1_0 * gk_r1_0
                b1_1 = k_r1_1 * gk_r1_1
                
                # ===== MMA directly with computed values =====
                aqk[0], aqk[1], aqk[2], aqk[3] = mma_tf32_m16n8k8(a_aqk_0, a_aqk_1, a_aqk_2, a_aqk_3, b0_0, b1_0, aqk[0], aqk[1], aqk[2], aqk[3])
                aqk[4], aqk[5], aqk[6], aqk[7] = mma_tf32_m16n8k8(a_aqk_0, a_aqk_1, a_aqk_2, a_aqk_3, b0_1, b1_1, aqk[4], aqk[5], aqk[6], aqk[7])
                akk[0], akk[1], akk[2], akk[3] = mma_tf32_m16n8k8(a_akk_0, a_akk_1, a_akk_2, a_akk_3, b0_0, b1_0, akk[0], akk[1], akk[2], akk[3])
                akk[4], akk[5], akk[6], akk[7] = mma_tf32_m16n8k8(a_akk_0, a_akk_1, a_akk_2, a_akk_3, b0_1, b1_1, akk[4], akk[5], akk[6], akk[7])
            
            # =============== Tree Reduction + Pipeline Signal ===============
            # Only after processing all k_tiles for this t_sub.
            # Uses double-buffered sAccum: buf = t_sub % NUM_ACC_PIPE_STAGES
            # Warps 0-3 sync via compute_warp_barrier (bar.sync 2, 128).
            # After Phase 5, warp 3 signals acc_full -> warp 4 can consume.
            # Before Phase 1, warps 0-3 wait acc_empty -> buffer is free.
            
            if k_tile == num_k_tiles - 1:
                t_abs_base = t_start + t_sub * BC
                
                # Double-buffer index for sAccum pipeline
                acc_buf = t_sub % NUM_ACC_PIPE_STAGES
                acc_pipe_phase = (t_sub // NUM_ACC_PIPE_STAGES) % 2
                aqk_slot = acc_buf * 2        # Aqk slot in sAccum dim-2
                akk_slot = acc_buf * 2 + 1    # Akk slot in sAccum dim-2
                
                # Wait for buffer to be free (warp 4 done with this buffer)
                cute.arch.mbarrier_wait(acc_empty_mbar_ptr + acc_buf, acc_pipe_phase)
                
                # ========== Phase 1: Warps 0,1 write to sAccum ==========
                if warp_id < 2:
                    col_base = warp_id * BC  # warp 0: 0, warp 1: 16
                    for _i in cutlass.range_constexpr(8, unroll=True):
                        sAccum[(group_id + _C_ROW[_i], col_base + tid_in_group * 2 + _C_COL[_i], aqk_slot)] = aqk[_i]
                        sAccum[(group_id + _C_ROW[_i], col_base + tid_in_group * 2 + _C_COL[_i], akk_slot)] = akk[_i]
                
                compute_warp_barrier()
                
                # ========== Phase 2: Warps 2,3 load and add to registers ==========
                if warp_id >= 2:
                    if warp_id < 4:
                        partner_col_base = (warp_id - 2) * BC
                        for _i in cutlass.range_constexpr(8, unroll=True):
                            aqk[_i] = aqk[_i] + sAccum[(group_id + _C_ROW[_i], partner_col_base + tid_in_group * 2 + _C_COL[_i], aqk_slot)]
                            akk[_i] = akk[_i] + sAccum[(group_id + _C_ROW[_i], partner_col_base + tid_in_group * 2 + _C_COL[_i], akk_slot)]
                
                compute_warp_barrier()
                
                # ========== Phase 3: Warp 2 writes (warp0+warp2) to sAccum cols 0-15 ==========
                if warp_id == 2:
                    for _i in cutlass.range_constexpr(8, unroll=True):
                        sAccum[(group_id + _C_ROW[_i], tid_in_group * 2 + _C_COL[_i], aqk_slot)] = aqk[_i]
                        sAccum[(group_id + _C_ROW[_i], tid_in_group * 2 + _C_COL[_i], akk_slot)] = akk[_i]
                
                compute_warp_barrier()
                
                # ========== Phase 4: Warp 3 loads (warp0+warp2) and adds to get final ==========
                if warp_id == 3:
                    for _i in cutlass.range_constexpr(8, unroll=True):
                        aqk[_i] = aqk[_i] + sAccum[(group_id + _C_ROW[_i], tid_in_group * 2 + _C_COL[_i], aqk_slot)]
                        akk[_i] = akk[_i] + sAccum[(group_id + _C_ROW[_i], tid_in_group * 2 + _C_COL[_i], akk_slot)]
                
                compute_warp_barrier()
                
                # ========== Phase 5: Warp 3 writes Aqk + prepared Akk to sAccum ==========
                if warp_id == 3:
                    # Aqk: raw values to aqk_slot
                    for _i in cutlass.range_constexpr(8, unroll=True):
                        sAccum[(group_id + _C_ROW[_i], tid_in_group * 2 + _C_COL[_i], aqk_slot)] = aqk[_i]
                    
                    # Akk: prepare unit lower triangular (diag=1, lower=val*beta, upper=0)
                    beta_r0 = cutlass.Float32(sBeta[(t_sub * BC + group_id,)])
                    beta_r1 = cutlass.Float32(sBeta[(t_sub * BC + group_id + 8,)])
                    _betas = [beta_r0, beta_r0, beta_r1, beta_r1, beta_r0, beta_r0, beta_r1, beta_r1]
                    for _i in cutlass.range_constexpr(8, unroll=True):
                        row = group_id + _C_ROW[_i]
                        col = tid_in_group * 2 + _C_COL[_i]
                        sAccum[(row, col, akk_slot)] = akk[_i] * _betas[_i] * cutlass.Float32(row > col) + cutlass.Float32(row == col)
                    
                    # Signal warp 4: data is ready in this buffer
                    # mbarrier_arrive includes release fence for this thread's writes
                    cute.arch.mbarrier_arrive(acc_full_mbar_ptr + acc_buf)
    
    elif warp_idx == 4:
        # ======= Warp 4: Dedicated writeback + inversion (pipelined) =======
        # Warp 4 has its own independent loop over t_sub values.
        # It waits for acc_full (data ready), does writeback + inversion,
        # then signals acc_empty (buffer free) so compute warps can reuse.
        lane_id_w4 = tidx % 32
        
        for t_sub_w4 in range(num_t_tiles):
            acc_buf = t_sub_w4 % NUM_ACC_PIPE_STAGES
            acc_pipe_phase = (t_sub_w4 // NUM_ACC_PIPE_STAGES) % 2
            t_abs_base = t_start + t_sub_w4 * BC
            aqk_slot = acc_buf * 2
            akk_slot = acc_buf * 2 + 1
            
            # Wait for compute warps to produce data in this buffer
            cute.arch.mbarrier_wait(acc_full_mbar_ptr + acc_buf, acc_pipe_phase)
            
            # ========== Phase 6a: Write Aqk from sAccum to global ==========
            for local_row in cutlass.range_constexpr(BC, unroll=True):
                row = local_row
                if lane_id_w4 < BC:
                    col = lane_id_w4
                    final_aqk = sAccum[(row, col, aqk_slot)]
                    val_aqk_out = cutlass.Float32(0.0)
                    if row >= col:
                        val_aqk_out = final_aqk * scale
                    Aqk_tensor[(i_b, t_abs_base + row, i_h, t_sub_w4 * BC + col)] = cutlass.Float16(val_aqk_out)
            
            # ========== Phase 6b: Invert 16x16 Akk (akk_slot -> aqk_slot) ==========
            _invert_16x16_halfwarp_fp32(sAccum[(None, None, akk_slot)], sAccum[(None, None, aqk_slot)], 0, lane_id_w4)
            
            # ========== Phase 6c: Write inverted Akk from sAccum to global ==========
            for local_row in cutlass.range_constexpr(BC, unroll=True):
                row = local_row
                if lane_id_w4 < BC:
                    col = lane_id_w4
                    Akk_tensor[(i_b, t_abs_base + row, i_h, col)] = sAccum[(row, col, aqk_slot)]
            
            # Signal compute warps: this buffer is now free to reuse
            if lane_id_w4 == 0:
                cute.arch.mbarrier_arrive(acc_empty_mbar_ptr + acc_buf)
    
    else:
        # ======= Warp 5: Dedicated TMA producer =======
        # Warp 5 runs its own loop over all tiles.
        # It waits for tma_consumed (stage free), then issues TMA + signals data ready.
        # This fully decouples TMA loading from compute warps 0-3.
        lane_id_w5 = tidx % 32
        
        for it in range(total_tiles):
            stage = it % NUM_STAGES
            tma_consumed_phase = (it // NUM_STAGES) % 2
            t_sub = it // num_k_tiles
            k_tile = it - t_sub * num_k_tiles
            
            # Global tile index in the 5D TMA tensor
            t_tile_idx = g_tile_base + t_sub
            
            # Wait for compute warps to finish reading this stage buffer
            cute.arch.mbarrier_wait(tma_consumed_mbar_ptr + stage, tma_consumed_phase)
            
            # Expect bytes for this stage's mbarrier
            if lane_id_w5 == 0:
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
            
            # Signal compute warps: TMA issued, data will be ready
            # mbarrier_arrive completes the manual arrive (count=1)
            if lane_id_w5 == 0:
                cute.arch.mbarrier_arrive(mbar_ptr + stage)


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
    
    # accum: (BC, BC * 2, NUM_ACC_PIPE_STAGES * 2)
    # Double-buffered: buf 0 has slots (0,1) = (Aqk, Akk), buf 1 has slots (2,3) = (Aqk, Akk)
    # Add 8-col padding to avoid bank conflict
    ACCUM_PAD = 8
    num_accum_slots = NUM_ACC_PIPE_STAGES * 2  # 4 slots total
    accum_smem_layout = cute.make_layout(
        (BC, BC * 2, num_accum_slots),
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
    # accum: (BC, BC*2+8, num_accum_slots) * 4 bytes (8-col padding, double buffered)
    # beta: BT * 2 bytes
    # mbarrier: 8 bytes per barrier * (NUM_STAGES + NUM_STAGES + 2 * NUM_ACC_PIPE_STAGES)
    #   TMA data ready: NUM_STAGES, tma_consumed: NUM_STAGES, acc pipeline: 2 * NUM_ACC_PIPE_STAGES
    smem_bytes = (4 * BC * BK * NUM_STAGES +                     # sG (float32, swizzled)
                  2 * BC * BK * NUM_STAGES * 2 +                 # sQK (fp16, swizzled, interleaved q/k)
                  4 * BC * (BC * 2 + ACCUM_PAD) * num_accum_slots + # sAccum (double-buffered)
                  2 * BT +                                        # sBeta
                  8 * (2 * NUM_STAGES + 2 * NUM_ACC_PIPE_STAGES) + # mbarriers (TMA + tma_consumed + acc pipeline)
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
        block=[NUM_WARPS * 32, 1, 1],
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