import os
# 指定 Triton 缓存目录到 scratch 空间（避免 home 目录空间不足）
os.environ["TRITON_CACHE_DIR"] = "/home/scratch.peiyuanz_gpu/mhl/.triton_cache"

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
    qk_smem_layout_f32: cute.Layout, # (BC, BK, 2) float32 for r_qgq (slot 0) and r_kgq (slot 1)
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
    
    # Allocate q/k smem (fp16 for loading q/k from global memory)
    # q+k: (BC, BK, NUM_STAGES * 2) fp16 = (16, 64, 4) * 2 bytes = 8KB
    sQK = smem.allocate_tensor(cutlass.Float16, qk_smem_layout, 128)
    
    # Allocate separate fp32 smem for r_qgq (cannot reuse sQK due to incompatible strides)
    # (BC, BK, NUM_STAGES) fp32 = (16, 64, 2) * 4 bytes = 8KB
    sQK_f32 = smem.allocate_tensor(cutlass.Float32, qk_smem_layout_f32, 128)

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
        
        # 4. write r_kgk, r_qgq, and r_kgq to SMEM
        # r_kgk -> sG[:, :, stage]
        # r_qgq -> sQK_f32[:, :, 0]
        # r_kgq -> sQK_f32[:, :, 1]
        # This allows fused MMA loop: load B once, compute both Aqk and Akk!
        for ri in range(4):
            r = row_base + ri
            sG[(r, col_base, stage)] = r_kgk[(ri, 0)]
            sG[(r, col_base + 1, stage)] = r_kgk[(ri, 1)]
            sQK_f32[(r, col_base, 0)] = r_qgq[(ri, 0)]
            sQK_f32[(r, col_base + 1, 0)] = r_qgq[(ri, 1)]
            sQK_f32[(r, col_base, 1)] = r_kgq[(ri, 0)]
            sQK_f32[(r, col_base + 1, 1)] = r_kgq[(ri, 1)]
        
        cute.arch.barrier()
        
        # 5. Fused MMA: Compute both Aqk and Akk in single loop
        #   Aqk = r_qgq @ r_kgk^T  -> sAccum[:, :, 0]
        #   Akk = r_kgq @ r_kgk^T  -> sAccum[:, :, 1]
        # 
        # Data in SMEM:
        #   A matrix (r_qgq): sQK_f32[:, :, 0] - 16x64 fp32
        #   A matrix (r_kgq): sQK_f32[:, :, 1] - 16x64 fp32
        #   B matrix (r_kgk): sG[:, :, stage] - 16x64 fp32
        #
        # Strategy: Fused MMA - load B once, compute both Aqk and Akk
        # This halves the sG memory reads!
        
        
        t_sub = it // num_k_tiles # sub_idx
        k_tile = it - t_sub * num_k_tiles # k_tile_idx
        
        lane_id_mma = tidx % 32 # lane_id
        k_start = warp_id * 2  # warp 0 -> k=0,1; warp 1 -> k=2,3; etc.
        group_id = lane_id_mma // 4 # group_id
        tid_in_group = lane_id_mma % 4 # tid_in_group
        warp_col_base = warp_id * BC  # warp_id * 16
        
        # ========== Reset accumulators at k_tile == 0 (new t_sub) ==========
        # Accumulators persist in registers across k_tiles - no sAccum load/store needed!
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
        # k_tile > 0: accumulators already have partial sums from previous k_tile in registers!
        
        # ========== Fused MMA loop: load B once, compute both Aqk and Akk ==========
        for k_iter in cutlass.range_constexpr(2):
            k = k_start + k_iter
            
            # Load B tiles from sG (r_kgk) - ONCE for both Aqk and Akk!
            sB_tile_0 = cute.local_tile(sG[(None, None, stage)], tiler=(8, 8), coord=(0, k))
            b0_0, b1_0 = load_B_tf32_from_rowmajor(sB_tile_0, lane_id_mma)
            
            sB_tile_1 = cute.local_tile(sG[(None, None, stage)], tiler=(8, 8), coord=(1, k))
            b0_1, b1_1 = load_B_tf32_from_rowmajor(sB_tile_1, lane_id_mma)
            
            # Load A tile for Aqk (r_qgq from slot 0)
            sA_tile_aqk = cute.local_tile(sQK_f32[(None, None, 0)], tiler=(16, 8), coord=(0, k))
            a0_aqk, a1_aqk, a2_aqk, a3_aqk = load_A_tf32(sA_tile_aqk, lane_id_mma)
            
            # Aqk MMA
            aqk_c00_0, aqk_c00_1, aqk_c00_2, aqk_c00_3 = mma_tf32_m16n8k8(a0_aqk, a1_aqk, a2_aqk, a3_aqk, b0_0, b1_0, aqk_c00_0, aqk_c00_1, aqk_c00_2, aqk_c00_3)
            aqk_c01_0, aqk_c01_1, aqk_c01_2, aqk_c01_3 = mma_tf32_m16n8k8(a0_aqk, a1_aqk, a2_aqk, a3_aqk, b0_1, b1_1, aqk_c01_0, aqk_c01_1, aqk_c01_2, aqk_c01_3)
            
            # Load A tile for Akk (r_kgq from slot 1)
            sA_tile_akk = cute.local_tile(sQK_f32[(None, None, 1)], tiler=(16, 8), coord=(0, k))
            a0_akk, a1_akk, a2_akk, a3_akk = load_A_tf32(sA_tile_akk, lane_id_mma)
            
            # Akk MMA (reuse b0_0, b1_0, b0_1, b1_1!)
            akk_c00_0, akk_c00_1, akk_c00_2, akk_c00_3 = mma_tf32_m16n8k8(a0_akk, a1_akk, a2_akk, a3_akk, b0_0, b1_0, akk_c00_0, akk_c00_1, akk_c00_2, akk_c00_3)
            akk_c01_0, akk_c01_1, akk_c01_2, akk_c01_3 = mma_tf32_m16n8k8(a0_akk, a1_akk, a2_akk, a3_akk, b0_1, b1_1, akk_c01_0, akk_c01_1, akk_c01_2, akk_c01_3)
        
        # 6. Reduction: only after processing all k_tiles for this t_sub
        # Accumulators stay in registers until final k_tile - then store to sAccum for cross-warp reduction
        if k_tile == num_k_tiles - 1:
            # Store final accumulated results to sAccum (only once per t_sub!)
            sAccum[(group_id, warp_col_base + tid_in_group * 2, 0)] = aqk_c00_0
            sAccum[(group_id, warp_col_base + tid_in_group * 2 + 1, 0)] = aqk_c00_1
            sAccum[(group_id + 8, warp_col_base + tid_in_group * 2, 0)] = aqk_c00_2
            sAccum[(group_id + 8, warp_col_base + tid_in_group * 2 + 1, 0)] = aqk_c00_3
            sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2, 0)] = aqk_c01_0
            sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_1
            sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2, 0)] = aqk_c01_2
            sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2 + 1, 0)] = aqk_c01_3
            
            sAccum[(group_id, warp_col_base + tid_in_group * 2, 1)] = akk_c00_0
            sAccum[(group_id, warp_col_base + tid_in_group * 2 + 1, 1)] = akk_c00_1
            sAccum[(group_id + 8, warp_col_base + tid_in_group * 2, 1)] = akk_c00_2
            sAccum[(group_id + 8, warp_col_base + tid_in_group * 2 + 1, 1)] = akk_c00_3
            sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2, 1)] = akk_c01_0
            sAccum[(group_id, warp_col_base + 8 + tid_in_group * 2 + 1, 1)] = akk_c01_1
            sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2, 1)] = akk_c01_2
            sAccum[(group_id + 8, warp_col_base + 8 + tid_in_group * 2 + 1, 1)] = akk_c01_3
            
            cute.arch.barrier()
            # Each warp handles 4 rows (4 warps * 4 rows = 16 = BC)
            # Use shuffle to reduce across 4 warps' partial sums
            #
            # sAccum layout: (BC=16, BC*NUM_WARPS=64, 2)
            #   col 0..15 = warp0, col 16..31 = warp1, col 32..47 = warp2, col 48..63 = warp3
            # Goal: final[row, col] = sum(sAccum[row, col + warp*16]) for warp in 0..3
            
            t_abs_base = t_start + t_sub * BC
            
            for local_row in cutlass.range_constexpr(4):
                row = warp_id * 4 + local_row
                
                # 32 threads read 64 elements (2 per thread, stride 32)
                # Thread i reads col i and col i+32 (consecutive access, no bank conflict)
                col_a = lane_id        # 0..31
                col_b = lane_id + 32   # 32..63
                
                # Read Aqk and Akk partial sums
                val_a_aqk = sAccum[(row, col_a, 0)]
                val_b_aqk = sAccum[(row, col_b, 0)]
                val_a_akk = sAccum[(row, col_a, 1)]
                val_b_akk = sAccum[(row, col_b, 1)]
                
                # Step 1: Add elements 32 apart
                # Thread 0..15 now has: warp0[i] + warp2[i]
                # Thread 16..31 now has: warp1[i-16] + warp3[i-16]
                sum_aqk = val_a_aqk + val_b_aqk
                sum_akk = val_a_akk + val_b_akk
                
                # Step 2: shfl_xor(16) exchanges between thread i and thread i^16
                # Thread 0 <-> Thread 16, Thread 1 <-> Thread 17, etc.
                partner_aqk = cute.arch.shuffle_sync_bfly(sum_aqk, offset=16, mask=-1, mask_and_clamp=31)
                partner_akk = cute.arch.shuffle_sync_bfly(sum_akk, offset=16, mask=-1, mask_and_clamp=31)
                
                # Step 3: Final reduction (all 4 warps summed)
                final_aqk = sum_aqk + partner_aqk
                final_akk = sum_akk + partner_akk
                
                # Step 4: First 16 threads apply mask, scale/beta, and write back to global
                if lane_id < 16:
                    col = lane_id
                    
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
    # Add 8 columns padding to reduce bank conflict: row_stride = 64 + 8 = 72
    G_ROW_STRIDE = BK + 8  # 72
    g_smem_layout = cute.make_layout(
        (BC, BK, NUM_STAGES),
        stride=(G_ROW_STRIDE, 1, BC * G_ROW_STRIDE)
    )
    # q and k share smem with interleaved stages: (BC, BK, NUM_STAGES * 2)
    # Interleaved layout: [q0][k0][q1][k1] so q and k for same stage are adjacent
    # This allows reusing q+k area (2 consecutive float16 stages) as one float32 stage
    qk_smem_layout = cute.make_layout(
        (BC, BK, NUM_STAGES * 2),
        stride=(BK, 1, BC * BK)
    )
    
    # Float32 smem for r_qgq and r_kgq (both stored, allows fused MMA loop)
    # (BC, BK, 2) = (16, 64, 2) - slot 0 for r_qgq, slot 1 for r_kgq
    # Add 8 columns padding to reduce bank conflict: row_stride = 64 + 8 = 72, 72 % 32 = 8
    QK_F32_ROW_STRIDE = BK + 8  # 72
    qk_smem_layout_f32 = cute.make_layout(
        (BC, BK, 2),
        stride=(QK_F32_ROW_STRIDE, 1, BC * QK_F32_ROW_STRIDE)
    )
    
    # accum: (BC, BC * NUM_WARPS, 2) for Aqk (slot 0) and Akk (slot 1)
    # Row-major with slot as outermost dimension (two separate stages)
    # Add 8 floats padding per row to reduce bank conflict
    # row_stride = 64 + 8 = 72, 72 % 32 = 8 → reduces to 2-way conflict
    ACCUM_ROW_STRIDE = BC * NUM_WARPS + 8  # 72 (col_count + padding)
    accum_smem_layout = cute.make_layout(
        (BC, BC * NUM_WARPS, 2),
        stride=(ACCUM_ROW_STRIDE, 1, BC * ACCUM_ROW_STRIDE)
    )

    # beta: (BT,) fp16
    beta_smem_layout = cute.make_layout(
        (BT,),
        stride=(1,)
    )
    
    # SMEM size calculation
    # g: 16 * 72 * 2 * 4 bytes = 9216 bytes (with 8-col padding)
    # qk (fp16): 16 * 64 * 4 * 2 bytes = 8192 bytes
    # qk_f32: 16 * 72 * 2 * 4 bytes = 9216 bytes (2 slots: r_qgq + r_kgq, with 8-col padding)
    # accum: with padding, two separate stages
    #        max_offset = (BC-1)*72 + (64-1)*1 + 1*1152 + 1 = 2296 floats
    # beta: BT * 2 bytes
    accum_max_offset = (BC - 1) * ACCUM_ROW_STRIDE + (BC * NUM_WARPS - 1) + BC * ACCUM_ROW_STRIDE + 1
    smem_bytes = (4 * BC * G_ROW_STRIDE * NUM_STAGES +  # sG (with 8-col padding)
                  2 * BC * BK * NUM_STAGES * 2 +        # sQK (fp16)
                  4 * BC * QK_F32_ROW_STRIDE * 2 +      # sQK_f32 (fp32, 2 slots, with 8-col padding)
                  4 * accum_max_offset +                # sAccum (with padding)
                  2 * BT +                               # sBeta
                  128)                                   # alignment
    
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