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
    tiled_copy_g: cute.TiledCopy,       # cp.async for G
    tma_atom_q: cute.CopyAtom,          # TMA for Q
    tma_tensor_q: cute.Tensor,          # 5D view for Q TMA
    tma_atom_k: cute.CopyAtom,          # TMA for K
    tma_tensor_k: cute.Tensor,          # 5D view for K TMA
    g_tensor: cute.Tensor,              # (B, T, H, K) float32
    beta_tensor: cute.Tensor,           # (B, T, H) fp16
    Akk_tensor: cute.Tensor,            # (B, T, H, BC) output
    Aqk_tensor: cute.Tensor,            # (B, T, H, BT) output
    g_smem_layout: cute.Layout,         # (BC, BK, NUM_STAGES) with padding, no swizzle
    qk_smem_layout,                     # ComposedLayout with swizzle: (BC, BK, NUM_STAGES * 2)
    accum_smem_layout: cute.Layout,     # (BC, BC * 2, 2) for Aqk and Akk
    beta_smem_layout: cute.Layout,      # (BT,)
    tiled_mma: cute.TiledMma,           # MMA configuration for ldmatrix
    tiled_copy_Q: cute.TiledCopy,       # ldmatrix for Q (16x8)
    tiled_copy_K: cute.TiledCopy,       # ldmatrix for K (8x8)
    BT: cutlass.Constexpr[int],
    num_k_tiles: cutlass.Constexpr[int],
    seq_len: int,                       # sequence length
    scale: cutlass.Float32,             # scale factor for Aqk
):
    """
    KDA Akk kernel using cp.async for G loading, TMA for Q/K loading with ldmatrix for MMA.
    - Each block handles one (batch, chunk, head)
    - The block owns a [BT, K] chunk, tiled by (BC, BK)
      BT=64, BC=16 => 4 tiles along T; K=128, BK=64 => 2 tiles along K; total 8 tiles.
    - G: cp.async load with 8-col padding -> direct indexing for element-wise
    - Q/K: TMA load with swizzle -> ldmatrix for MMA register loading
    """
    
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // 32
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_b, i_t, i_h = cute.arch.block_idx()  # batch, chunk, head
    
    # =============== Allocate shared memory ===============
    smem = cutlass.utils.SmemAllocator()
    
    # g: (BC, BK, NUM_STAGES) with 8-col padding - no swizzle
    sG = smem.allocate_tensor(cutlass.Float32, g_smem_layout, 128)
    
    # qk: (BC, BK, NUM_STAGES * 2) with swizzle, interleaved [q0][k0][q1][k1]
    # Use TMA for loading, ldmatrix for SMEM->register
    sQK = smem.allocate_tensor(cutlass.Float16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner)
    
    sAccum = smem.allocate_tensor(cutlass.Float32, accum_smem_layout, 128)
    sBeta = smem.allocate_tensor(cutlass.Float16, beta_smem_layout, 128)
    
    # Allocate mbarriers for TMA synchronization (one per stage)
    mbar_ptr = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    
    # TMA tile size in bytes (only Q+K use TMA)
    qk_tile_bytes = BC * BK * 2  # fp16
    total_tma_bytes = qk_tile_bytes * 2  # q + k
    
    # Initialize mbarriers
    if tidx == 0:
        for stage in range(NUM_STAGES):
            cute.arch.mbarrier_init(mbar_ptr + stage, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    
    # Chunk start and tile base along T (each T tile is BC rows)
    t_start = i_t * BT
    t_tile_base = i_t * (BT // BC)  # base tile index in (T/BC) dimension
    
    # Compute global tile index offset for TMA: i_b * (T/BC) + t_tile_base
    num_t_tiles_per_batch = seq_len // BC
    g_tile_base = i_b * num_t_tiles_per_batch + t_tile_base
    
    # 获取当前 batch 和 head 的数据 slice
    gG_batch = g_tensor[(i_b, None, i_h, None)]   # (T, K) for G (cpasync)
    gBeta_batch = beta_tensor[(i_b, None, i_h)]   # (T,)

    # Stage beta for this chunk: sBeta[0:BT] <- beta[t_start : t_start+BT]
    if tidx < BT:
        sBeta[tidx] = gBeta_batch[t_start + tidx]
    cute.arch.barrier()
    
    # G: 对 T 和 K 维度分 tiles: (BC, BK, T/BC, K/BK) - 4D tensor
    gG = cute.local_tile(gG_batch, (BC, BK), (None, None))
    
    # Get thread-level copy partitions for G (cpasync)
    thr_copy_g = tiled_copy_g.get_slice(tidx)
    
    # Iterate all tiles in the chunk:
    #   iter -> (t_sub, k_tile)
    #   t_sub in [0, BT/BC), k_tile in [0, num_k_tiles)
    num_t_tiles = BT // BC
    total_tiles = num_t_tiles * num_k_tiles

    # Get thread slices for ldmatrix (moved outside loop for efficiency)
    lane_id = tidx % 32
    thr_mma = tiled_mma.get_slice(lane_id)
    thr_copy_Q_ld = tiled_copy_Q.get_slice(lane_id)
    thr_copy_K_ld = tiled_copy_K.get_slice(lane_id)

    # =============== Prefetch first stages ===============
    # G: cp.async, Q/K: TMA (warp 0 only issues TMA)
    prefetch_count = cutlass.min(NUM_STAGES - 1, total_tiles)
    for it in range(prefetch_count):
        stage = it % NUM_STAGES
        t_sub = it // num_k_tiles
        k_tile = it - t_sub * num_k_tiles
        t_tile_idx = t_tile_base + t_sub
        t_tile_idx_tma = g_tile_base + t_sub  # for TMA 5D view

        # Load G tile via cp.async (all threads)
        gG_tile = gG[(None, None, t_tile_idx, k_tile)]
        sG_stage = sG[(None, None, stage)]
        thr_gG = thr_copy_g.partition_S(gG_tile)
        thr_sG = thr_copy_g.partition_D(sG_stage)
        cute.copy(tiled_copy_g, thr_gG, thr_sG)
        cute.arch.cp_async_commit_group()

        # Load Q/K tiles via TMA (warp 0 only)
        if warp_idx == 0:
            # Expect bytes for this stage's mbarrier
            if tidx == 0:
                cute.arch.mbarrier_expect_tx(mbar_ptr + stage, total_tma_bytes)
            
            # TMA load Q: global -> sQK[(None, None, stage * 2)]
            gQ_src = cute.local_tile(tma_tensor_q, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx_tma, i_h, k_tile))
            gQ_src_2d = gQ_src[(None, None, 0, 0, 0)]
            sQ_stage = sQK[(None, None, stage * 2)]
            tma_sQ, tma_gQ = cpasync.tma_partition(
                tma_atom_q, 0, cute.make_layout(1),
                cute.group_modes(sQ_stage, 0, 2),
                cute.group_modes(gQ_src_2d, 0, 2),
            )
            cute.copy(tma_atom_q, tma_gQ, tma_sQ, tma_bar_ptr=mbar_ptr + stage)
            
            # TMA load K: global -> sQK[(None, None, stage * 2 + 1)]
            gK_src = cute.local_tile(tma_tensor_k, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx_tma, i_h, k_tile))
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

        # Wait for G (cp.async)
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
        
        # Arrive at mbarrier for TMA
        if tidx == 0:
            cute.arch.mbarrier_arrive(mbar_ptr + stage)
        
        # Wait for Q/K (TMA via mbarrier)
        cute.arch.mbarrier_wait(mbar_ptr + stage, phase)

        # Issue next loads (overlapped with compute)
        next_it = it + prefetch_count
        if next_it < total_tiles:
            next_stage = next_it % NUM_STAGES
            t_sub_n = next_it // num_k_tiles
            k_tile_n = next_it - t_sub_n * num_k_tiles
            t_tile_idx_n = t_tile_base + t_sub_n
            t_tile_idx_tma_n = g_tile_base + t_sub_n  # for TMA

            # Load next G tile via cp.async (all threads)
            gG_next = gG[(None, None, t_tile_idx_n, k_tile_n)]
            sG_next = sG[(None, None, next_stage)]
            thr_gG = thr_copy_g.partition_S(gG_next)
            thr_sG = thr_copy_g.partition_D(sG_next)
            cute.copy(tiled_copy_g, thr_gG, thr_sG)
            cute.arch.cp_async_commit_group()

            # Load next Q/K tiles via TMA (warp 0 only)
            if warp_idx == 0:
                # Expect bytes for next stage's mbarrier
                if tidx == 0:
                    cute.arch.mbarrier_expect_tx(mbar_ptr + next_stage, total_tma_bytes)
                
                # TMA load next Q
                gQ_src_n = cute.local_tile(tma_tensor_q, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx_tma_n, i_h, k_tile_n))
                gQ_src_2d_n = gQ_src_n[(None, None, 0, 0, 0)]
                sQ_next = sQK[(None, None, next_stage * 2)]
                tma_sQ_n, tma_gQ_n = cpasync.tma_partition(
                    tma_atom_q, 0, cute.make_layout(1),
                    cute.group_modes(sQ_next, 0, 2),
                    cute.group_modes(gQ_src_2d_n, 0, 2),
                )
                cute.copy(tma_atom_q, tma_gQ_n, tma_sQ_n, tma_bar_ptr=mbar_ptr + next_stage)
                
                # TMA load next K
                gK_src_n = cute.local_tile(tma_tensor_k, (BC, BK, 1, 1, 1), (0, 0, t_tile_idx_tma_n, i_h, k_tile_n))
                gK_src_2d_n = gK_src_n[(None, None, 0, 0, 0)]
                sK_next = sQK[(None, None, next_stage * 2 + 1)]
                tma_sK_n, tma_gK_n = cpasync.tma_partition(
                    tma_atom_k, 0, cute.make_layout(1),
                    cute.group_modes(sK_next, 0, 2),
                    cute.group_modes(gK_src_2d_n, 0, 2),
                )
                cute.copy(tma_atom_k, tma_gK_n, tma_sK_n, tma_bar_ptr=mbar_ptr + next_stage)

        # =============== Fused Element-wise + MMA using ldmatrix ===============
        # Strategy: Use ldmatrix to load Q/K from SMEM (with padding), load G directly,
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
        group_id = lane_id // 4      # 0-7, determines rows (group_id and group_id+8)
        tid_in_group = lane_id % 4   # 0-3, determines cols within 8-col tile (tid*2 and tid*2+1)
        k_start = warp_id * 2        # warp 0 -> k=0,1; warp 1 -> k=2,3; etc.
        
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
            tCsQ_view = thr_copy_Q_ld.partition_S(sQ_tile)
            tCrQ_view = thr_copy_Q_ld.retile(tCrQ)
            cute.copy(tiled_copy_Q, tCsQ_view, tCrQ_view)
            
            # ===== Load K tile 0 (rows 0-7, 8x8) via ldmatrix =====
            sK_tile_n0 = cute.local_tile(sK_stage, tiler=(8, 8), coord=(0, k))
            tCsK_n0 = thr_mma.partition_B(sK_tile_n0)
            tCrK_n0 = tiled_mma.make_fragment_B(tCsK_n0)
            tCsK_n0_view = thr_copy_K_ld.partition_S(sK_tile_n0)
            tCrK_n0_view = thr_copy_K_ld.retile(tCrK_n0)
            cute.copy(tiled_copy_K, tCsK_n0_view, tCrK_n0_view)
            
            # ===== Load K tile 1 (rows 8-15, 8x8) via ldmatrix =====
            sK_tile_n1 = cute.local_tile(sK_stage, tiler=(8, 8), coord=(1, k))
            tCsK_n1 = thr_mma.partition_B(sK_tile_n1)
            tCrK_n1 = tiled_mma.make_fragment_B(tCsK_n1)
            tCsK_n1_view = thr_copy_K_ld.partition_S(sK_tile_n1)
            tCrK_n1_view = thr_copy_K_ld.retile(tCrK_n1)
            cute.copy(tiled_copy_K, tCsK_n1_view, tCrK_n1_view)
            
            # ===== Load G values directly (no swizzle, with padding) =====
            gn_0 = sG_stage[(gn_row, k_offset + col)]
            gn_1 = sG_stage[(gn_row, k_offset + col + 1)]
            g_r0_0 = sG_stage[(r0, k_offset + col)]
            g_r0_1 = sG_stage[(r0, k_offset + col + 1)]
            g_r1_0 = sG_stage[(r1, k_offset + col)]
            g_r1_1 = sG_stage[(r1, k_offset + col + 1)]
            
            # ===== Compute exp2 factors =====
            b_gm_r0_0 = g_r0_0 - gn_0
            b_gm_r0_1 = g_r0_1 - gn_1
            b_gm_r1_0 = g_r1_0 - gn_0
            b_gm_r1_1 = g_r1_1 - gn_1
            
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
            
            # ========== Phase 5: Warp 3 writes final result to sAccum cols 0-15 ==========
            if warp_id == 3:
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
            
            # ========== Phase 6: All warps read final and write to global ==========
            # Each warp handles 4 rows (4 warps * 4 rows = 16 = BC)
            # Final result is now in sAccum[0:16, 0:16, :]
            for local_row in cutlass.range_constexpr(4):
                row = warp_id * 4 + local_row
                
                # Only 16 threads per warp write (lane 0-15)
                if lane_id < 16:
                    col = lane_id
                    
                    # Read final reduced values from sAccum
                    final_aqk = sAccum[(row, col, 0)]
                    final_akk = sAccum[(row, col, 1)]
                    
                    # Initialize outputs before control flow (CuTe DSL requirement)
                    val_aqk_out = cutlass.Float32(0.0)
                    val_akk_out = cutlass.Float32(0.0)
                    
                    # Aqk: mask row >= col (lower triangular with diagonal)
                    if row >= col:
                        val_aqk_out = final_aqk * scale
                    
                    # Akk: mask row > col (strictly lower triangular)
                    beta_idx = t_sub * BC + row
                    beta_val = cutlass.Float32(sBeta[(beta_idx,)])
                    if row > col:
                        val_akk_out = final_akk * beta_val
                    
                    # Write directly to global memory
                    Aqk_tensor[(i_b, t_abs_base + row, i_h, t_sub * BC + col)] = cutlass.Float16(val_aqk_out)
                    Akk_tensor[(i_b, t_abs_base + row, i_h, col)] = val_akk_out

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
    
    # =============== Create TMA for Q/K ===============
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
    
    qk_view_layout = cute.make_layout(
        (BC, BK, num_t_tiles_total, H, num_k_tiles),
        stride=(s_bc, s_bk, s_t_tile, s_h, s_k_tile)
    )
    
    # Create view tensors for q and k
    q_view = cute.make_tensor(q_tensor.iterator, qk_view_layout)
    k_view = cute.make_tensor(k_tensor.iterator, qk_view_layout)
    
    # =============== SMEM layouts ===============
    # G: 8-col padding, no swizzle (for element-wise access)
    G_ROW_STRIDE = BK + 8  # 72
    g_smem_layout = cute.make_layout(
        (BC, BK, NUM_STAGES),
        stride=(G_ROW_STRIDE, 1, BC * G_ROW_STRIDE)
    )
    
    # Q/K: Swizzled layout for TMA + ldmatrix
    smem_atom_qk = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float16)
    qk_smem_layout_2d = cute.tile_to_shape(smem_atom_qk, (BC, BK), order=(0, 1))
    qk_smem_layout = cute.tile_to_shape(smem_atom_qk, (BC, BK, NUM_STAGES * 2), order=(0, 1, 2))
    
    # TMA load atoms for q and k
    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
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
    
    # =============== Accum and beta SMEM layouts ===============
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
    # g: 16 * 72 * NUM_STAGES * 4 bytes (with 8-col padding, no swizzle)
    # qk (fp16): 16 * 64 * (NUM_STAGES * 2) * 2 bytes (with swizzle)
    # accum: (BC, BC*2+8, 2) = 16 * 40 * 2 * 4 bytes (8-col padding)
    # beta: BT * 2 bytes
    # mbarrier: 8 bytes per barrier * NUM_STAGES
    smem_bytes = (4 * BC * G_ROW_STRIDE * NUM_STAGES +           # sG (float32, with padding)
                  2 * BC * BK * NUM_STAGES * 2 +                 # sQK (fp16, swizzled, interleaved q/k)
                  4 * BC * (BC * 2 + ACCUM_PAD) * 2 +            # sAccum (8-col padding)
                  2 * BT +                                        # sBeta
                  8 * NUM_STAGES +                                # mbarriers
                  256)                                            # alignment
    
    print(f"KDA Akk (G:cpasync + QK:TMA + ldmatrix): B={B}, T={seq_len}, H={H}, K={K}")
    print(f"  Tiles: NT={NT}, num_k_tiles={num_k_tiles}")
    print(f"  SMEM: {smem_bytes} bytes\n")
    
    kda_Akk_kernel(
        tiled_copy_g,        # cp.async for G
        tma_atom_q,          # TMA for Q
        tma_tensor_q,
        tma_atom_k,          # TMA for K
        tma_tensor_k,
        g_tensor,
        beta_tensor,
        Akk_tensor,
        Aqk_tensor,
        g_smem_layout,       # 3D: (BC, BK, NUM_STAGES) - with padding, no swizzle
        qk_smem_layout,      # 3D ComposedLayout: (BC, BK, NUM_STAGES * 2) - with swizzle
        accum_smem_layout,
        beta_smem_layout,
        tiled_mma,           # MMA configuration
        tiled_copy_Q,        # ldmatrix for Q
        tiled_copy_K,        # ldmatrix for K
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
    print("KDA Akk (G:cpasync + QK:TMA + ldmatrix) Test")
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
    
    # Pass/Fail check
    # TF32 MMA has ~1e-3 relative error, so use relaxed thresholds
    aqk_threshold = 1e-2  # fp16 output
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
    trace_file = os.path.join(trace_dir, "kda_Akk_g_cpasync_qk_tma_ldmatrix.json")
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
    print("G:cpasync + QK:TMA + LDMatrix KDA Akk")
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
