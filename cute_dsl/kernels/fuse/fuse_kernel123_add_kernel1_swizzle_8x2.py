"""
Fused K1+K2+K3 Kernel for KDA.

Fuses gate activation + cumsum + scaling (K1), intra sub-chunk Aqk/Akk (K2),
and inter sub-chunk solve + merged inverse (K3) into a single kernel.

Grid: (NT/4, H, B)
Block: 928 threads (29 warps), warp-specialized with mbarrier pipeline:
  Warp 0:      TMA producer (independent loop, double-buffered stages)
  Warps 1-16:  K1 compute (8×2: 8 row groups × 2 col groups, vec2, 128B/warp store)
  Warps 17-20: Store A_qk, A_kk (consumer of TMA, no K1 dependency)
  Warps 21-28: K2 MMA compute (TODO)
Mbarriers: tma_mbars[2], store_stage_mbars[2] (count=640, warps 1-20)
SMEM: ~160KB (q+k+g × [64,128] bf16 × 2 stages + g_cumsum [64,128] fp32 × 2 stages)

Inputs:
  g       [B,T,H,K]   bf16  raw gate
  k       [B,T,H,K]   bf16
  q       [B,T,H,K]   bf16
  A_log   [H]          fp32  per-head log decay
  beta    [B,T,H]      bf16  used for Akk unit lower triangular
  scale                fp32  1/sqrt(K)

Outputs (g_cumsum stays in SMEM, not written to GMEM):
  k_scaled   [B,T,H,K]   bf16
  q_scaled   [B,T,H,K]   bf16
  kg         [B,T,H,K]   bf16
  gk_last_exp[B,NT,H,K]  fp32
  A_qk       [B,T,H,BT]  bf16  full merged (diagonal + off-diagonal)
  A_kk       [B,T,H,BT]  bf16  full inverted lower triangular
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute import KeepPTX, KeepCUBIN
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import torch
import time

B200_PEAK_BW_GBS = 7672  # GB/s

BT = 64
BC = 16
K_DIM = 128
K_PAD = 8
K_STRIDE = K_DIM + K_PAD  # 136, padded row stride to avoid bank conflicts
CHUNKS_PER_BLOCK = 4

NUM_K1_WARPS = 16
NUM_STORE_WARPS = 4
NUM_MMA_WARPS = 10
NUM_WARPS = 1 + NUM_K1_WARPS + NUM_STORE_WARPS + NUM_MMA_WARPS  # 29
THREADS = NUM_WARPS * 32  # 928

K1_ROW_GROUPS = 8
K1_COL_GROUPS = 2
ROWS_PER_K1_WARP = BT // K1_ROW_GROUPS       # 8
K1_COLS_PER_WARP = K_DIM // K1_COL_GROUPS     # 64
ROWS_PER_STORE_WARP = BT // NUM_STORE_WARPS   # 16

VEC = K1_COLS_PER_WARP // 32  # 2 (vec2 bf16 = 4B per thread, 128B per warp)
K_VEC = K_DIM // VEC          # 64
NUM_STAGES = 2
PARTIAL_COLS = K_DIM + 4      # 132, padded 4 cols to avoid bank conflicts in prefix sum
PARTIAL_COLS_PER_WARP = K_DIM // NUM_K1_WARPS  # 8 cols per warp in prefix phase

LOG2E = 1.4426950408889634   # log2(e) = 1/ln(2)
LN2 = 0.6931471805599453     # ln(2)
RCP_LN2 = LOG2E              # alias for cumsum_scale


@dsl_user_op
def store_internal_barrier(*, loc=None, ip=None):
    """Named barrier for store warps (17-20, 128 threads). barrier_id=1."""
    llvm.inline_asm(
        T.i32(), [],
        "membar.cta; bar.sync 1, 128; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def k1_internal_barrier(*, loc=None, ip=None):
    """Named barrier for K1 warps (1-16, 512 threads). barrier_id=2.
    Ensures SMEM writes (partial_last) are visible before cross-warp reads."""
    llvm.inline_asm(
        T.i32(), [],
        "membar.cta; bar.sync 2, 512; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


SHFL_W8_CLAMP = 0x1800  # shfl width=8: segmask=0x18 in bits[12:8], maxLaneId=0 in bits[4:0]


@cute.kernel
def fused_kernel123(
    tma_atom_Q: cute.CopyAtom, tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom, tma_tensor_K: cute.Tensor,
    tma_atom_G: cute.CopyAtom, tma_tensor_G: cute.Tensor,
    mA_log: cute.Tensor,          # [H] fp32, per-head log decay
    mBeta: cute.Tensor,           # [B,T,H] bf16
    scale: cutlass.Float32,       # 1/sqrt(K)
    mKscaled: cute.Tensor,        # [B,T,H,K_VEC,VEC] bf16 output
    mKg: cute.Tensor,             # [B,T,H,K_VEC,VEC] bf16 output
    mQscaled: cute.Tensor,        # [B,T,H,K_VEC,VEC] bf16 output
    mGkLast: cute.Tensor,         # [B,NT,H,K_VEC,VEC] fp32 output
    mAqk: cute.Tensor,            # [B,T,H,BT] bf16 output
    mAkk: cute.Tensor,            # [B,T,H,BT] bf16 output
    tiled_copy_qk,                # TiledCopy for reading from swizzled Q/K SMEM
    qk_smem_layout,               # (BT, K_DIM, NUM_STAGES) swizzled
    g_smem_layout,                # (BT, K_DIM, NUM_STAGES) row-major
    g_cumsum_layout,
    num_chunks: int,
):
    i_cg, i_h, i_b = cute.arch.block_idx()
    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32

    chunk_base = i_cg * CHUNKS_PER_BLOCK

    # =====================================================================
    # SMEM allocation
    # =====================================================================
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.BFloat16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner)
    sK = smem.allocate_tensor(cutlass.BFloat16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner)
    sG = smem.allocate_tensor(cutlass.BFloat16, g_smem_layout, 128)
    sGcum = smem.allocate_tensor(cutlass.Float32, g_cumsum_layout, 128)
    # Partial last: [8 row_groups, 132 cols] fp32
    # 128 data cols + 4 padding cols for bank conflict avoidance in prefix sum
    # stride=(132,1): row-major, padding ensures stride%32=4 → no bank conflict
    partial_last_layout = cute.make_layout(
        (K1_ROW_GROUPS, PARTIAL_COLS),
        stride=(PARTIAL_COLS, 1))
    sPartialLast = smem.allocate_tensor(cutlass.Float32, partial_last_layout, 128)

    # Mbarrier protocol:
    #   tma_mbars[2]:         Warp 0 → Warps 1-20 (consumers wait for TMA data)
    #   store_stage_mbars[2]: Warps 1-20 (640 threads arrive) → Warp 0 (stage reuse)
    tma_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    store_stage_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)

    bytes_per_stage = BT * K_DIM * 2 * 3  # Q + K + G, all bf16

    if tidx == 0:
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_init(tma_mbars + s, 1)
            cute.arch.mbarrier_init(store_stage_mbars + s, (NUM_K1_WARPS + NUM_STORE_WARPS) * 32)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # Pre-arrive: warps 1-20 (K1 + store) flip phase
    if warp_idx >= 1 and warp_idx < 1 + NUM_K1_WARPS + NUM_STORE_WARPS:
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_arrive(store_stage_mbars + s)

    # =================================================================
    # Warp 0: TMA Producer
    # =================================================================
    if warp_idx == 0:
        for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
            s = chunk_iter % NUM_STAGES
            phase = chunk_iter // NUM_STAGES % 2
            chunk_idx = chunk_base + chunk_iter
            i_bnt = i_b * num_chunks + chunk_idx

            cute.arch.mbarrier_wait(store_stage_mbars + s, phase)

            if lane_id == 0:
                cute.arch.mbarrier_expect_tx(tma_mbars + s, bytes_per_stage)

            sQ_s = sQ[(None, None, s)]
            gQ = cute.local_tile(tma_tensor_Q, (BT, K_DIM, 1, 1), (0, 0, i_bnt, i_h))
            ts, tg = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1),
                cute.group_modes(sQ_s, 0, 2), cute.group_modes(gQ[(None, None, 0, 0)], 0, 2))
            cute.copy(tma_atom_Q, tg, ts, tma_bar_ptr=tma_mbars + s)

            sK_s = sK[(None, None, s)]
            gK = cute.local_tile(tma_tensor_K, (BT, K_DIM, 1, 1), (0, 0, i_bnt, i_h))
            ts, tg = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1),
                cute.group_modes(sK_s, 0, 2), cute.group_modes(gK[(None, None, 0, 0)], 0, 2))
            cute.copy(tma_atom_K, tg, ts, tma_bar_ptr=tma_mbars + s)

            sG_s = sG[(None, None, s)]
            gG = cute.local_tile(tma_tensor_G, (BT, K_DIM, 1, 1), (0, 0, i_bnt, i_h))
            ts, tg = cpasync.tma_partition(tma_atom_G, 0, cute.make_layout(1),
                cute.group_modes(sG_s, 0, 2), cute.group_modes(gG[(None, None, 0, 0)], 0, 2))
            cute.copy(tma_atom_G, tg, ts, tma_bar_ptr=tma_mbars + s)

            if lane_id == 0:
                cute.arch.mbarrier_arrive(tma_mbars + s)

    # =================================================================
    # Warps 1-16: K1 gate activation + cumsum + scaling (8×2 layout)
    #
    #           cols 0-63      cols 64-127
    # rows 0-7     warp 0        warp 8
    # rows 8-15    warp 1        warp 9
    # rows 16-23   warp 2        warp 10
    # rows 24-31   warp 3        warp 11
    # rows 32-39   warp 4        warp 12
    # rows 40-47   warp 5        warp 13
    # rows 48-55   warp 6        warp 14
    # rows 56-63   warp 7        warp 15
    #
    # Each warp: 8 rows × 64 cols, 32 threads × vec2
    # GMEM store: 32 threads × 4B (bf16×2) = 128 bytes = 1 sector
    # Cross-warp prefix: 8 row-warps per col group (independent)
    # =================================================================
    if warp_idx >= 1 and warp_idx < 1 + NUM_K1_WARPS:
        k1_warp = warp_idx - 1
        warp_row_group = k1_warp % K1_ROW_GROUPS   # 0..7
        warp_col_group = k1_warp // K1_ROW_GROUPS   # 0 or 1
        k1_row_start = warp_row_group * ROWS_PER_K1_WARP
        col_base = warp_col_group * K1_COLS_PER_WARP + lane_id * VEC
        col_vec_idx = warp_col_group * (K1_COLS_PER_WARP // VEC) + lane_id  # 0..63

        exp_A = cute.exp(mA_log[i_h], fastmath=True)
        cumsum_scale = cutlass.Float32(RCP_LN2)

        # Register arrays
        rAcc = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
        rPrefix = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
        rGkLast = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
        rKsOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
        rQsOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
        rKgOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
        rGkOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)

        for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
            s = chunk_iter % NUM_STAGES
            phase = chunk_iter // NUM_STAGES % 2
            chunk_idx = chunk_base + chunk_iter
            chunk_start = chunk_idx * BT

            cute.arch.mbarrier_wait(tma_mbars + s, phase)

            csG = sG[(None, None, s)]
            csGcum = sGcum[(None, None, s)]
            csQ = sQ[(None, None, s)]
            csK = sK[(None, None, s)]

            # ---- Pass 1: activate G, store in registers, local prefix sum ----
            rGact = cute.make_rmem_tensor(cute.make_layout((ROWS_PER_K1_WARP, VEC)), cutlass.Float32)
            for vi in cutlass.range_constexpr(VEC):
                rAcc[vi] = cutlass.Float32(0.0)

            for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                row = k1_row_start + ri
                for vi in cutlass.range_constexpr(VEC):
                    c = col_base + vi
                    g_val = csG[row, c].to(cutlass.Float32)
                    softplus_g = cute.log2(cutlass.Float32(1.0) + cute.exp2(g_val * LOG2E, fastmath=True), fastmath=True) * LN2
                    g_activated = -exp_A * softplus_g
                    rGact[ri, vi] = g_activated
                    rAcc[vi] = rAcc[vi] + g_activated

            # ---- Write partial_last to SMEM: 8×128 (padded to 8×132) ----
            for vi in cutlass.range_constexpr(VEC):
                sPartialLast[warp_row_group, col_base + vi] = rAcc[vi]

            # ---- Barrier: sync warps 1-16 (512 threads) ----
            k1_internal_barrier()

            # ---- Shuffle-based prefix sum across 8 row groups ----
            # Each of 16 warps handles 8 columns, 2 iterations of 4 cols × 8 rows
            # Thread mapping (column-major): row = lane_id % 8, col_in_group = lane_id // 8
            prefix_col_start = k1_warp * PARTIAL_COLS_PER_WARP
            row_in_prefix = lane_id % K1_ROW_GROUPS
            col_in_group = lane_id // K1_ROW_GROUPS  # 0..3

            for j in cutlass.range_constexpr(PARTIAL_COLS_PER_WARP // 4):
                col = prefix_col_start + j * 4 + col_in_group
                val = cutlass.Float32(sPartialLast[row_in_prefix, col])

                tmp = cute.arch.shuffle_sync_up(val, 1, mask=-1, mask_and_clamp=SHFL_W8_CLAMP)
                if row_in_prefix >= 1:
                    val = val + tmp
                tmp = cute.arch.shuffle_sync_up(val, 2, mask=-1, mask_and_clamp=SHFL_W8_CLAMP)
                if row_in_prefix >= 2:
                    val = val + tmp
                tmp = cute.arch.shuffle_sync_up(val, 4, mask=-1, mask_and_clamp=SHFL_W8_CLAMP)
                if row_in_prefix >= 4:
                    val = val + tmp

                sPartialLast[row_in_prefix, col] = val

            # ---- Barrier: prefix sum done, read results ----
            k1_internal_barrier()

            # rGkLast = total sum (inclusive prefix at last row)
            for vi in cutlass.range_constexpr(VEC): 
                rGkLast[vi] = sPartialLast[K1_ROW_GROUPS - 1, col_base + vi]

            # rPrefix = exclusive prefix (inclusive prefix at row_group - 1)
            for vi in cutlass.range_constexpr(VEC):
                rPrefix[vi] = cutlass.Float32(0.0)
            if warp_row_group > 0:
                for vi in cutlass.range_constexpr(VEC):
                    rPrefix[vi] = sPartialLast[warp_row_group - 1, col_base + vi]

            # ---- Pass 2: re-scan with offset, compute & store row-by-row ----
            for vi in cutlass.range_constexpr(VEC):
                rAcc[vi] = rPrefix[vi]

            thr_copy_qk = tiled_copy_qk.get_slice(lane_id)

            for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                row = k1_row_start + ri
                t = chunk_start + row

                # Load K from swizzled SMEM via tiled_copy
                sK_tile = cute.local_tile(csK, tiler=(1, K1_COLS_PER_WARP), coord=(row, warp_col_group))
                tCsK = thr_copy_qk.partition_S(sK_tile)
                tCrK = cute.make_fragment_like(tCsK)
                cute.copy(tiled_copy_qk, tCsK, thr_copy_qk.retile(tCrK))

                # Load Q from swizzled SMEM via tiled_copy
                sQ_tile = cute.local_tile(csQ, tiler=(1, K1_COLS_PER_WARP), coord=(row, warp_col_group))
                tCsQ = thr_copy_qk.partition_S(sQ_tile)
                tCrQ = cute.make_fragment_like(tCsQ)
                cute.copy(tiled_copy_qk, tCsQ, thr_copy_qk.retile(tCrQ))

                for vi in cutlass.range_constexpr(VEC):
                    c = col_base + vi
                    rAcc[vi] = rAcc[vi] + rGact[ri, vi]

                    cs = rAcc[vi] * cumsum_scale

                    k_val = tCrK[vi].to(cutlass.Float32)
                    q_val = tCrQ[vi].to(cutlass.Float32)

                    exp2_cs = cute.exp2(cs, fastmath=True)
                    gk_last_cs = rGkLast[vi] * cumsum_scale
                    exp2_kg = cute.exp2(gk_last_cs - cs, fastmath=True)

                    csGcum[row, c] = cs #mhl-debug

                    rKsOut[vi] = (k_val * exp2_cs).to(cutlass.BFloat16)
                    rQsOut[vi] = (q_val * exp2_cs * scale).to(cutlass.BFloat16)
                    rKgOut[vi] = (k_val * exp2_kg).to(cutlass.BFloat16)

                cute.autovec_copy(rKsOut, mKscaled[i_b, t, i_h, col_vec_idx, None])
                cute.autovec_copy(rQsOut, mQscaled[i_b, t, i_h, col_vec_idx, None])
                cute.autovec_copy(rKgOut, mKg[i_b, t, i_h, col_vec_idx, None])

            # gk_last: only one row-warp per col group stores
            if warp_row_group == 0:
                for vi in cutlass.range_constexpr(VEC):
                    rGkOut[vi] = cute.exp2(rGkLast[vi] * cumsum_scale, fastmath=True)
                cute.autovec_copy(rGkOut, mGkLast[i_b, chunk_idx, i_h, col_vec_idx, None])

            cute.arch.mbarrier_arrive(store_stage_mbars + s)

    # =================================================================
    # Warps 17-20: Store A_qk, A_kk
    # =================================================================
    if warp_idx >= 1 + NUM_K1_WARPS and warp_idx < 1 + NUM_K1_WARPS + NUM_STORE_WARPS:
        store_warp = warp_idx - (1 + NUM_K1_WARPS)
        st_row_start = store_warp * ROWS_PER_STORE_WARP

        for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
            s = chunk_iter % NUM_STAGES
            phase = chunk_iter // NUM_STAGES % 2
            chunk_idx = chunk_base + chunk_iter
            chunk_start = chunk_idx * BT

            cute.arch.mbarrier_wait(tma_mbars + s, phase)

            csQ = sQ[(None, None, s)]
            csK = sK[(None, None, s)]

            thr_copy_qk_st = tiled_copy_qk.get_slice(lane_id)

            for ri in cutlass.range_constexpr(ROWS_PER_STORE_WARP):
                row = st_row_start + ri
                t = chunk_start + row

                # Load Q from swizzled SMEM via tiled_copy (first BT=64 cols)
                sQ_tile = cute.local_tile(csQ, tiler=(1, BT), coord=(row, 0))
                tCsQ_st = thr_copy_qk_st.partition_S(sQ_tile)
                tCrQ_st = cute.make_fragment_like(tCsQ_st)
                cute.copy(tiled_copy_qk, tCsQ_st, thr_copy_qk_st.retile(tCrQ_st))

                # Load K from swizzled SMEM via tiled_copy (first BT=64 cols)
                sK_tile = cute.local_tile(csK, tiler=(1, BT), coord=(row, 0))
                tCsK_st = thr_copy_qk_st.partition_S(sK_tile)
                tCrK_st = cute.make_fragment_like(tCsK_st)
                cute.copy(tiled_copy_qk, tCsK_st, thr_copy_qk_st.retile(tCrK_st))

                # Write to GMEM: thread lane_id gets cols [2*lane_id, 2*lane_id+1]
                for vi in cutlass.range_constexpr(BT // 32):
                    c = lane_id * 2 + vi
                    mAqk[i_b, t, i_h, c] = tCrQ_st[vi]
                    mAkk[i_b, t, i_h, c] = tCrK_st[vi]

            cute.arch.mbarrier_arrive(store_stage_mbars + s)

    # =================================================================
    # Warps 21-28: K2 MMA Compute (idle for now)
    # =================================================================
    if warp_idx >= 1 + NUM_K1_WARPS + NUM_STORE_WARPS:
        _dummy = cutlass.Int32(0)


# =========================================================================
# Host function
# =========================================================================
def make_host_function(B, NT, H):
    _B, _NT, _H = B, NT, H
    _T = _NT * BT
    _BNT = _B * _NT

    s_row = _H * K_DIM
    s_col = 1
    s_bnt = BT * _H * K_DIM
    s_h = K_DIM

    @cute.jit
    def host_fn(mQ, mK, mG, mA_log, mBeta, scale,
                mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk):
        view_layout = cute.make_layout(
            (BT, K_DIM, _BNT, _H),
            stride=(s_row, s_col, s_bnt, s_h),
        )
        mQ_view = cute.make_tensor(mQ.iterator, view_layout)
        mK_view = cute.make_tensor(mK.iterator, view_layout)
        mG_view = cute.make_tensor(mG.iterator, view_layout)

        # Q/K SMEM: tcgen05 K_SW128 swizzle
        smem_atom_qk = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16)
        qk_smem_2d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM), order=(0, 1))
        qk_smem_3d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM, NUM_STAGES), order=(0, 1, 2))

        # G SMEM: row-major, no swizzle
        g_smem_2d = cute.make_layout((BT, K_DIM), stride=(K_DIM, 1))
        g_smem_3d = cute.make_layout((BT, K_DIM, NUM_STAGES), stride=(K_DIM, 1, BT * K_DIM))

        tma_op = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_op, mQ_view, qk_smem_2d,
            cute.product_each(qk_smem_2d.shape), num_multicast=1)
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_op, mK_view, qk_smem_2d,
            cute.product_each(qk_smem_2d.shape), num_multicast=1)
        tma_atom_G, tma_tensor_G = cpasync.make_tiled_tma_atom(
            tma_op, mG_view, g_smem_2d,
            cute.product_each(g_smem_2d.shape), num_multicast=1)

        g_cumsum_layout = cute.make_layout(
            (BT, K_DIM, NUM_STAGES), stride=(K_STRIDE, 1, BT * K_STRIDE))

        # Tiled copy for reading Q/K from swizzled SMEM (32 threads × vec2 bf16)
        copy_atom_qk_s2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=32  # 2 x bf16 = 32 bits
        )
        tiled_copy_qk_s2r = cute.make_tiled_copy_tv(
            copy_atom_qk_s2r,
            thr_layout=cute.make_layout((1, 32)),  # 32 threads along cols
            val_layout=cute.make_layout((1, 2))    # 2 values per thread
        )

        # Vec2 views for vectorized stores (warps 1-16, 8×2 layout)
        out_v2_layout = cute.make_layout(
            (_B, _T, _H, K_VEC, VEC),
            stride=(_T * _H * K_DIM, _H * K_DIM, K_DIM, VEC, 1),
        )
        mKscaled_v2 = cute.make_tensor(mKscaled.iterator, out_v2_layout)
        mQscaled_v2 = cute.make_tensor(mQscaled.iterator, out_v2_layout)
        mKg_v2 = cute.make_tensor(mKg.iterator, out_v2_layout)

        gklast_v2_layout = cute.make_layout(
            (_B, _NT, _H, K_VEC, VEC),
            stride=(_NT * _H * K_DIM, _H * K_DIM, K_DIM, VEC, 1),
        )
        mGkLast_v2 = cute.make_tensor(mGkLast.iterator, gklast_v2_layout)

        # Q/K swizzled + G row-major + g_cumsum fp32 padded + sPartialLast + mbarriers
        smem_size = (BT * K_DIM * 2 * 2 * NUM_STAGES        # Q + K swizzled bf16
                     + BT * K_DIM * 2 * NUM_STAGES           # G row-major bf16
                     + BT * K_STRIDE * 4 * NUM_STAGES        # g_cumsum fp32 padded
                     + K1_ROW_GROUPS * PARTIAL_COLS * 4       # sPartialLast fp32
                     + 256)                                   # mbarriers + alignment

        fused_kernel123(
            tma_atom_Q, tma_tensor_Q,
            tma_atom_K, tma_tensor_K,
            tma_atom_G, tma_tensor_G,
            mA_log, mBeta, scale,
            mKscaled_v2, mKg_v2, mQscaled_v2, mGkLast_v2, mAqk, mAkk,
            tiled_copy_qk_s2r,
            qk_smem_3d, g_smem_3d, g_cumsum_layout, _NT,
        ).launch(
            grid=(_NT // CHUNKS_PER_BLOCK, _H, _B),
            # grid=(1, 1, 1),
            block=(THREADS, 1, 1),
            smem=smem_size,
        )

    return host_fn


# =========================================================================
# Test
# =========================================================================
def test():
    cutlass.cuda.initialize_cuda_context()

    B, H, K = 1, 96, 128
    NT = 128
    T = NT * BT

    print("=" * 60)
    print("Fused K1+K2+K3 Kernel")
    print("=" * 60)
    print(f"B={B}, T={T}, H={H}, K={K}, NT={NT}, BT={BT}, BC={BC}")
    print(f"Grid: ({NT // CHUNKS_PER_BLOCK}, {H}, {B}), Block: {THREADS} ({NUM_WARPS} warps)")
    print()

    torch.manual_seed(42)
    scale = 1.0 / (K ** 0.5)

    # --- Inputs ---
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)

    # --- Outputs (g_cumsum stays in SMEM, not written to GMEM) ---
    k_scaled = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    kg = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    q_scaled = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    gk_last = torch.empty(B, NT, H, K, device="cuda", dtype=torch.float32)
    A_qk = torch.empty(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    A_kk = torch.empty(B, T, H, BT, device="cuda", dtype=torch.bfloat16)

    host_fn = make_host_function(B, NT, H)

    dl = from_dlpack

    def _ct(t, etype):
        r = dl(t, assumed_align=16)
        r.element_type = etype
        return r

    mQ = _ct(q, cutlass.BFloat16)
    mK = _ct(k, cutlass.BFloat16)
    mG = _ct(g, cutlass.BFloat16)
    mA_log = _ct(A_log, cutlass.Float32)
    mBeta = _ct(beta, cutlass.BFloat16)
    mKscaled = _ct(k_scaled, cutlass.BFloat16)
    mKg = _ct(kg, cutlass.BFloat16)
    mQscaled = _ct(q_scaled, cutlass.BFloat16)
    mGkLast = _ct(gk_last, cutlass.Float32)
    mAqk = _ct(A_qk, cutlass.BFloat16)
    mAkk = _ct(A_kk, cutlass.BFloat16)

    print("Compiling...")
    compiled = cute.compile[KeepPTX, KeepCUBIN](
        host_fn,
        mQ, mK, mG, mA_log, mBeta, scale,
        mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk,
    )

    print("Running...")
    compiled(mQ, mK, mG, mA_log, mBeta, scale,
             mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk)
    torch.cuda.synchronize()
    print("Kernel executed successfully!")

    # TODO: correctness checks against reference K1+K2+K3

    num_warmup, num_iters = 20, 100

    for _ in range(num_warmup):
        compiled(mQ, mK, mG, mA_log, mBeta, scale,
                 mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        compiled(mQ, mK, mG, mA_log, mBeta, scale,
                 mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) / num_iters * 1e6

    read_bytes = B * T * H * K * 2 * 3 + H * 4 + B * T * H * 2
    write_bytes = (B * T * H * K * 2 * 3         # k_scaled, kg, q_scaled bf16
                   + B * NT * H * K * 4           # gk_last_exp fp32
                   + B * T * H * BT * 2 * 2)     # A_qk + A_kk bf16
    total_bytes = read_bytes + write_bytes
    bw = total_bytes / (elapsed_us * 1e-6) / 1e9

    print(f"\n[Benchmark] {num_iters} iterations")
    print(f"  Kernel time: {elapsed_us:.2f} us")
    print(f"  Read:  {read_bytes / 1e6:.1f} MB")
    print(f"  Write: {write_bytes / 1e6:.1f} MB")
    print(f"  Total: {total_bytes / 1e6:.1f} MB")
    print(f"  Bandwidth: {bw:.1f} GB/s")
    print(f"  Utilization: {bw / B200_PEAK_BW_GBS * 100:.1f}% of B200 peak ({B200_PEAK_BW_GBS} GB/s)")
    print("=" * 60)


if __name__ == "__main__":
    test()
