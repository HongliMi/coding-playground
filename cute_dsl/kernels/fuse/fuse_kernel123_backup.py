"""
Fused K1+K2+K3 Kernel for KDA.

Fuses gate activation + cumsum + scaling (K1), intra sub-chunk Aqk/Akk (K2),
and inter sub-chunk solve + merged inverse (K3) into a single kernel.

Grid: (NT/4, H, B)
Block: 992 threads (31 warps), warp-specialized with mbarrier pipeline:
  Warp 0:      TMA producer (independent loop, double-buffered stages)
  Warps 1-16:  K1 compute (8×2: 8 row groups × 2 col groups, vec2, 128B/warp store)
  Warps 17-20: Store/Inversion warps (consume sAqk/sAkk from MMA, do inversion + GMEM write)
  Warps 21-30: K2 MMA compute (10 warps, 1 tile per warp, full K=128 iteration)
Mbarriers: tma_mbars[2], store_stage_mbars[2], k1_done_mbars[2], mma_done_mbars[2]
SMEM: ~160KB (q+k+g × [64,128] bf16 × 2 stages + g_cumsum [64,128] fp32 × 2 stages
      + sAqk [16,168,2,2] fp32 + sAkk [16,168,2,2] fp32)

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
from cutlass._mlir import ir
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
NUM_WARPS = 1 + NUM_K1_WARPS + NUM_STORE_WARPS + NUM_MMA_WARPS  # 31
THREADS = NUM_WARPS * 32  # 992

# MMA tile constants
NUM_SUB_CHUNKS = BT // BC  # 4
NUM_TILES = NUM_SUB_CHUNKS * (NUM_SUB_CHUNKS + 1) // 2  # 10 lower-tri tiles
MMA_K_TILE = 8             # m16n8k8 k-dimension
NUM_MMA_K_TILES = K_DIM // MMA_K_TILE  # 16 iterations over K=128
AQK_TILE_COLS = NUM_TILES * BC  # 160
AQK_TILE_PAD = 8           # padding to avoid bank conflicts
AQK_TILE_STRIDE = AQK_TILE_COLS + AQK_TILE_PAD  # 168

# 10 MMA warps = 10 lower-tri tiles, 1 tile per warp (no K-half split)
# Each warp independently processes its full tile over all 16 k_blocks

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

# Compile-time tile (i_q, i_k) lookup for 10 lower-triangular tiles
_TILE_IQ = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
_TILE_IK = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

LOG2E = 1.4426950408889634   # log2(e) = 1/ln(2)
LN2 = 0.6931471805599453     # ln(2)
RCP_LN2 = LOG2E              # alias for cumsum_scale

# Per-warp register budget via setmaxnreg (PTX SM90+)
# Total must fit in 65536 regs: 1×32×32 + 16×32×72 + 4×32×32 + 10×32×72 = 65024
NUM_REGS_LOW = 32    # TMA producer (warp 0) and Store warps (17-20)
NUM_REGS_K1 = 72     # K1 gate/cumsum/scaling warps (1-16)
NUM_REGS_MMA = 72    # MMA compute warps (21-30)


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


@dsl_user_op
def read_clock64(*, loc=None, ip=None):
    """Read GPU cycle counter (%clock64)"""
    result = llvm.inline_asm(
        T.i64(), [],
        "mov.u64 $0, %clock64;",
        "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return cutlass.Int64(result)


NUM_PROFILE_SLOTS = 8
SHFL_W8_CLAMP = 0x1800  # shfl width=8: segmask=0x18 in bits[12:8], max LaneId=0 in bits[4:0]


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
    tiled_copy_qk,                # TiledCopy for reading from swizzled Q/K SMEM (K1 warps)
    tiled_mma_k2,                 # TiledMma for K2 MMA warps (m16n8k8)
    tiled_copy_mma_A,             # TiledCopy A-operand from swizzled SMEM (16x8)
    tiled_copy_mma_B,             # TiledCopy B-operand from swizzled SMEM (8x8)
    qk_smem_layout,               # (BT, K_DIM, NUM_STAGES) swizzled
    g_smem_layout,                # (BT, K_DIM, NUM_STAGES) row-major
    g_cumsum_layout,
    mProfile: cute.Tensor,         # [8] int64, cycle profiling
    num_chunks: int,
):
    i_cg, i_h, i_b = cute.arch.block_idx()
    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32

    is_first_block = (i_cg == 0) & (i_h == 0) & (i_b == 0)

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

    # sAqk/sAkk: [16, 168, 2] bf16 — 10 tiles × 16 cols = 160 + 8 pad, 2 stages
    # Each warp writes its tile directly, no K-half reduction needed
    aqk_tile_layout = cute.make_layout(
        (BC, AQK_TILE_STRIDE, NUM_STAGES),
        stride=(AQK_TILE_STRIDE, 1, BC * AQK_TILE_STRIDE))
    sAqk = smem.allocate_tensor(cutlass.BFloat16, aqk_tile_layout, 128)
    sAkk = smem.allocate_tensor(cutlass.BFloat16, aqk_tile_layout, 128)

    # Mbarrier protocol:
    #   tma_mbars[2]:         Warp 0 → K1+MMA warps (consumers wait for TMA data)
    #   store_stage_mbars[2]: K1+MMA warps arrive → Warp 0 (stage reuse)
    #   k1_done_mbars[2]:     K1 warps → MMA warps (g_cumsum ready)
    #   mma_done_mbars[2]:    MMA warps → Store warps (sAqk/sAkk ready)
    tma_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    store_stage_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    k1_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    mma_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    store_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)

    bytes_per_stage = BT * K_DIM * 2 * 3  # Q + K + G, all bf16

    if tidx == 0:
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_init(tma_mbars + s, 1)
            # Only MMA (10 warps) read from TMA SMEM; K1 finishes before MMA
            cute.arch.mbarrier_init(store_stage_mbars + s, NUM_MMA_WARPS * 32)
            # K1 (16 warps) signal g_cumsum ready
            cute.arch.mbarrier_init(k1_done_mbars + s, NUM_K1_WARPS * 32)
            # MMA (10 warps) signal sAqk/sAkk ready
            cute.arch.mbarrier_init(mma_done_mbars + s, NUM_MMA_WARPS * 32)
            # Store (4 warps) signal sAqk/sAkk stage free for MMA reuse
            cute.arch.mbarrier_init(store_done_mbars + s, NUM_STORE_WARPS * 32)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # Pre-arrive: only needed when consumer waits BEFORE producer arrives
    # in the first iteration. Pre-arrive count must EXACTLY equal init count.
    #
    # store_stage_mbars: TMA waits before MMA arrives → need pre-arrive
    #   init=MMA(10)*32=320, pre-arrive by MMA(10 warps)=320 ✓
    # store_done_mbars: MMA waits before Store arrives → need pre-arrive
    #   init=Store(4)*32=128, pre-arrive by first 4 MMA warps=128 ✓
    # k1_done_mbars: MMA waits AFTER K1 arrives → no pre-arrive needed
    # mma_done_mbars: Store waits AFTER MMA arrives → no pre-arrive needed
    if warp_idx >= 1 + NUM_K1_WARPS + NUM_STORE_WARPS and warp_idx < 1 + NUM_K1_WARPS + NUM_STORE_WARPS + NUM_MMA_WARPS:
        mma_warp_tmp = warp_idx - (1 + NUM_K1_WARPS + NUM_STORE_WARPS)
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_arrive(store_stage_mbars + s)  # all 10 MMA warps
            if mma_warp_tmp < NUM_STORE_WARPS:  # first 4 MMA warps only (128 threads = init count)
                cute.arch.mbarrier_arrive(store_done_mbars + s)

    # =================================================================
    # Warp 0: TMA Producer
    # =================================================================
    if warp_idx == 0:
        t_tma_start = read_clock64()
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

        t_tma_end = read_clock64()
        if is_first_block and lane_id == 0:
            mProfile[0] = t_tma_start
            mProfile[1] = t_tma_end

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
        t_k1_start = read_clock64()
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

            # Signal g_cumsum ready for MMA warps
            cute.arch.mbarrier_arrive(k1_done_mbars + s)

        t_k1_end = read_clock64()
        if is_first_block and warp_idx == 1 and lane_id == 0:
            mProfile[2] = t_k1_start
            mProfile[3] = t_k1_end

    # =================================================================
    # Warps 17-20: Store/Inversion warps
    # Wait for MMA warps to fill sAqk/sAkk, then write to GMEM.
    # TODO: add block inversion for A_kk
    # =================================================================
    if warp_idx >= 1 + NUM_K1_WARPS and warp_idx < 1 + NUM_K1_WARPS + NUM_STORE_WARPS:
        t_store_start = read_clock64()
        store_warp = warp_idx - (1 + NUM_K1_WARPS)

        for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
            s = chunk_iter % NUM_STAGES
            phase = chunk_iter // NUM_STAGES % 2
            chunk_idx = chunk_base + chunk_iter
            chunk_start = chunk_idx * BT

            # Wait for MMA warps to finish writing sAqk/sAkk for this stage
            cute.arch.mbarrier_wait(mma_done_mbars + s, phase)

            csAqk = sAqk[(None, None, s)]
            csAkk = sAkk[(None, None, s)]

            # 4 store warps × 32 threads = 128 threads
            # 10 tiles, each 16×16. Each warp handles 4 rows per tile (16/4=4).
            # K-half reduction already done in sAqk/sAkk by MMA warps.
            for tile_idx in cutlass.range_constexpr(NUM_TILES):
                i_q = _TILE_IQ[tile_idx]
                i_k = _TILE_IK[tile_idx]
                is_diag = _TILE_IQ[tile_idx] == _TILE_IK[tile_idx]
                tile_col_base = tile_idx * BC
                gmem_row_base = chunk_start + i_q * BC
                gmem_col_base = i_k * BC

                for ri in cutlass.range_constexpr(BC // NUM_STORE_WARPS):  # 4 rows per warp
                    local_row = store_warp * (BC // NUM_STORE_WARPS) + ri
                    if lane_id < BC:
                        local_col = lane_id
                        aqk_val = csAqk[local_row, tile_col_base + local_col]
                        akk_val = csAkk[local_row, tile_col_base + local_col]

                        if is_diag and local_row < local_col:
                            aqk_val = cutlass.BFloat16(0.0)
                            akk_val = cutlass.BFloat16(0.0)

                        mAqk[i_b, gmem_row_base + local_row, i_h, gmem_col_base + local_col] = aqk_val
                        mAkk[i_b, gmem_row_base + local_row, i_h, gmem_col_base + local_col] = akk_val

            # Signal sAqk/sAkk stage is free for MMA to reuse
            cute.arch.mbarrier_arrive(store_done_mbars + s)

        t_store_end = read_clock64()
        if is_first_block and warp_idx == 1 + NUM_K1_WARPS and lane_id == 0:
            mProfile[4] = t_store_start
            mProfile[5] = t_store_end

    # =================================================================
    # Warps 21-30: K2+K3 MMA Compute (10 warps = 5 pairs × 2 K-halves)
    #
    # 10 MMA warps: each pair of 2 warps cooperates on the same tile,
    # splitting K=128 into two 64-col halves. Each pair processes 2 tiles.
    #   my_pair   = mma_warp // 2  (0..4)
    #   my_k_half = mma_warp % 2   (0 or 1)
    #
    # Pair-to-tile mapping (10 lower-tri tiles, 2 per pair):
    #   pair 0: tile0=(0,0), tile1=(1,0)
    #   pair 1: tile2=(1,1), tile3=(2,0)
    #   pair 2: tile4=(2,1), tile5=(2,2)
    #   pair 3: tile6=(3,0), tile7=(3,1)
    #   pair 4: tile8=(3,2), tile9=(3,3)
    #
    # For tile (i_q, i_k):
    #   A_qk[i_q,i_k] = scale * Q_gated[i_q] @ K_gated[i_k]^T
    #   A_kk[i_q,i_k] = beta  * Kq_gated[i_q] @ K_gated[i_k]^T
    # =================================================================
    if warp_idx >= 1 + NUM_K1_WARPS + NUM_STORE_WARPS:
        t_mma_start = read_clock64()
        mma_warp = warp_idx - (1 + NUM_K1_WARPS + NUM_STORE_WARPS)  # 0..9

        # Decode (i_q, i_k) from mma_warp (fixed per warp, computed once)
        # tile layout: 0|(0,0) 1|(1,0) 2|(1,1) 3|(2,0) 4|(2,1) 5|(2,2) 6|(3,0) 7|(3,1) 8|(3,2) 9|(3,3)
        my_i_q = cutlass.Int32(0)
        my_i_k = cutlass.Int32(0)
        if mma_warp < 1:
            my_i_q = cutlass.Int32(0)
            my_i_k = mma_warp
        elif mma_warp < 3:
            my_i_q = cutlass.Int32(1)
            my_i_k = mma_warp - 1
        elif mma_warp < 6:
            my_i_q = cutlass.Int32(2)
            my_i_k = mma_warp - 3
        else:
            my_i_q = cutlass.Int32(3)
            my_i_k = mma_warp - 6

        q_row_base = my_i_q * BC
        k_row_base = my_i_k * BC
        tile_col_base = mma_warp * BC

        # MMA thread mapping (m16n8k8):
        group_id = lane_id // 4
        tid_in_group = lane_id % 4
        row0, row1 = group_id, group_id + 8
        col0, col1 = tid_in_group * 2, tid_in_group * 2 + 1
        col2, col3 = 8 + tid_in_group * 2, 8 + tid_in_group * 2 + 1

        # Setup tiled_copy/mma partitions (once, reused across chunks)
        thr_mma = tiled_mma_k2.get_slice(lane_id)
        thr_copy_A = tiled_copy_mma_A.get_slice(lane_id)
        thr_copy_B = tiled_copy_mma_B.get_slice(lane_id)

        for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
            s = chunk_iter % NUM_STAGES
            phase = chunk_iter // NUM_STAGES % 2
            chunk_idx = chunk_base + chunk_iter
            chunk_start = chunk_idx * BT

            # Wait for K1 to finish g_cumsum
            cute.arch.mbarrier_wait(k1_done_mbars + s, phase)
            # Wait for Store to free sAqk/sAkk stage
            cute.arch.mbarrier_wait(store_done_mbars + s, phase)

            csQ = sQ[(None, None, s)]
            csK = sK[(None, None, s)]
            csGcum = sGcum[(None, None, s)]

            _z = cutlass.Float32(0.0)

            # Wait for TMA data
            cute.arch.mbarrier_wait(tma_mbars + s, phase)

            csAqk = sAqk[(None, None, s)]
            csAkk = sAkk[(None, None, s)]

            # Per-tile beta
            beta_row0 = mBeta[i_b, chunk_start + q_row_base + row0, i_h].to(cutlass.Float32)
            beta_row1 = mBeta[i_b, chunk_start + q_row_base + row1, i_h].to(cutlass.Float32)

            # Accumulators (16 fp32 regs)
            acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = _z, _z, _z, _z
            acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = _z, _z, _z, _z
            acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = _z, _z, _z, _z
            acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = _z, _z, _z, _z

            # Iterate over all 16 k_blocks (K=128, no K-half split)
            for k_block in cutlass.range_constexpr(NUM_MMA_K_TILES):
                # Load A: Q[i_q] and Kq[i_q]
                sQ_tile = cute.local_tile(csQ, tiler=(16, 8), coord=(my_i_q, k_block))
                tCrQ = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sQ_tile))
                cute.copy(tiled_copy_mma_A, thr_copy_A.partition_S(sQ_tile), thr_copy_A.retile(tCrQ))

                sKq_tile = cute.local_tile(csK, tiler=(16, 8), coord=(my_i_q, k_block))
                tCrKq = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sKq_tile))
                cute.copy(tiled_copy_mma_A, thr_copy_A.partition_S(sKq_tile), thr_copy_A.retile(tCrKq))

                # Gate A with g_cumsum
                g_col0 = k_block * MMA_K_TILE + tid_in_group * 2
                g_col1 = g_col0 + 1
                g_norm_0 = csGcum[q_row_base, g_col0]
                g_norm_1 = csGcum[q_row_base, g_col1]
                gate_q_0 = cute.exp2(csGcum[q_row_base + row0, g_col0] - g_norm_0, fastmath=True)
                gate_q_1 = cute.exp2(csGcum[q_row_base + row0, g_col1] - g_norm_1, fastmath=True)
                gate_q_2 = cute.exp2(csGcum[q_row_base + row1, g_col0] - g_norm_0, fastmath=True)
                gate_q_3 = cute.exp2(csGcum[q_row_base + row1, g_col1] - g_norm_1, fastmath=True)

                qa0 = tCrQ[0].to(cutlass.Float32) * gate_q_0
                qa1 = tCrQ[2].to(cutlass.Float32) * gate_q_2
                qa2 = tCrQ[1].to(cutlass.Float32) * gate_q_1
                qa3 = tCrQ[3].to(cutlass.Float32) * gate_q_3
                ka0 = tCrKq[0].to(cutlass.Float32) * gate_q_0
                ka1 = tCrKq[2].to(cutlass.Float32) * gate_q_2
                ka2 = tCrKq[1].to(cutlass.Float32) * gate_q_1
                ka3 = tCrKq[3].to(cutlass.Float32) * gate_q_3

                # Load B: K[i_k] — two n-halves
                sK_tile_n0 = cute.local_tile(csK, tiler=(8, 8), coord=(my_i_k * 2, k_block))
                tCrK_n0 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n0))
                cute.copy(tiled_copy_mma_B, thr_copy_B.partition_S(sK_tile_n0), thr_copy_B.retile(tCrK_n0))

                sK_tile_n1 = cute.local_tile(csK, tiler=(8, 8), coord=(my_i_k * 2 + 1, k_block))
                tCrK_n1 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n1))
                cute.copy(tiled_copy_mma_B, thr_copy_B.partition_S(sK_tile_n1), thr_copy_B.retile(tCrK_n1))

                # Gate B with g_cumsum
                gk_n0_0 = cute.exp2(g_norm_0 - csGcum[k_row_base + group_id, g_col0], fastmath=True)
                gk_n0_1 = cute.exp2(g_norm_1 - csGcum[k_row_base + group_id, g_col1], fastmath=True)
                k_n0_b0 = tCrK_n0[0].to(cutlass.Float32) * gk_n0_0
                k_n0_b1 = tCrK_n0[1].to(cutlass.Float32) * gk_n0_1

                gk_n1_0 = cute.exp2(g_norm_0 - csGcum[k_row_base + group_id + 8, g_col0], fastmath=True)
                gk_n1_1 = cute.exp2(g_norm_1 - csGcum[k_row_base + group_id + 8, g_col1], fastmath=True)
                k_n1_b0 = tCrK_n1[0].to(cutlass.Float32) * gk_n1_0
                k_n1_b1 = tCrK_n1[1].to(cutlass.Float32) * gk_n1_1

                # 4 MMA calls: Aqk×2 n-halves + Akk×2 n-halves
                acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = mma_tf32_m16n8k8(
                    qa0, qa1, qa2, qa3, k_n0_b0, k_n0_b1,
                    acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3)
                acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = mma_tf32_m16n8k8(
                    qa0, qa1, qa2, qa3, k_n1_b0, k_n1_b1,
                    acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3)
                acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = mma_tf32_m16n8k8(
                    ka0, ka1, ka2, ka3, k_n0_b0, k_n0_b1,
                    acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3)
                acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = mma_tf32_m16n8k8(
                    ka0, ka1, ka2, ka3, k_n1_b0, k_n1_b1,
                    acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3)

            # Write results directly to sAqk/sAkk as bf16 (no reduction needed)
            csAqk[row0, tile_col_base + col0] = (acc_aqk_n0_0 * scale).to(cutlass.BFloat16)
            csAqk[row0, tile_col_base + col1] = (acc_aqk_n0_1 * scale).to(cutlass.BFloat16)
            csAqk[row1, tile_col_base + col0] = (acc_aqk_n0_2 * scale).to(cutlass.BFloat16)
            csAqk[row1, tile_col_base + col1] = (acc_aqk_n0_3 * scale).to(cutlass.BFloat16)
            csAqk[row0, tile_col_base + col2] = (acc_aqk_n1_0 * scale).to(cutlass.BFloat16)
            csAqk[row0, tile_col_base + col3] = (acc_aqk_n1_1 * scale).to(cutlass.BFloat16)
            csAqk[row1, tile_col_base + col2] = (acc_aqk_n1_2 * scale).to(cutlass.BFloat16)
            csAqk[row1, tile_col_base + col3] = (acc_aqk_n1_3 * scale).to(cutlass.BFloat16)

            csAkk[row0, tile_col_base + col0] = (acc_akk_n0_0 * beta_row0).to(cutlass.BFloat16)
            csAkk[row0, tile_col_base + col1] = (acc_akk_n0_1 * beta_row0).to(cutlass.BFloat16)
            csAkk[row1, tile_col_base + col0] = (acc_akk_n0_2 * beta_row1).to(cutlass.BFloat16)
            csAkk[row1, tile_col_base + col1] = (acc_akk_n0_3 * beta_row1).to(cutlass.BFloat16)
            csAkk[row0, tile_col_base + col2] = (acc_akk_n1_0 * beta_row0).to(cutlass.BFloat16)
            csAkk[row0, tile_col_base + col3] = (acc_akk_n1_1 * beta_row0).to(cutlass.BFloat16)
            csAkk[row1, tile_col_base + col2] = (acc_akk_n1_2 * beta_row1).to(cutlass.BFloat16)
            csAkk[row1, tile_col_base + col3] = (acc_akk_n1_3 * beta_row1).to(cutlass.BFloat16)

            # Signal MMA done for this stage, and stage done for TMA reuse
            cute.arch.mbarrier_arrive(mma_done_mbars + s)
            cute.arch.mbarrier_arrive(store_stage_mbars + s)

        t_mma_end = read_clock64()
        if is_first_block and warp_idx == 1 + NUM_K1_WARPS + NUM_STORE_WARPS and lane_id == 0:
            mProfile[6] = t_mma_start
            mProfile[7] = t_mma_end


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
                mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk, mProfile):
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

        # MMA tiled_mma and copy atoms for K2 MMA warps (m16n8k8 TF32)
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 8))
        tiled_mma_k2 = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 8))

        tiled_copy_mma_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16),
            tiled_mma_k2)
        tiled_copy_mma_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 1), cutlass.BFloat16),
            tiled_mma_k2)

        # Q/K swizzled + G row-major + g_cumsum fp32 padded + sPartialLast
        # + sAqk/sAkk fp32 + mbarriers
        smem_size = (BT * K_DIM * 2 * 2 * NUM_STAGES        # Q + K swizzled bf16
                     + BT * K_DIM * 2 * NUM_STAGES           # G row-major bf16
                     + BT * K_STRIDE * 4 * NUM_STAGES        # g_cumsum fp32 padded
                     + K1_ROW_GROUPS * PARTIAL_COLS * 4       # sPartialLast fp32
                     + BC * AQK_TILE_STRIDE * 2 * NUM_STAGES * 2  # sAqk + sAkk bf16
                     + 512)                                   # mbarriers + alignment

        fused_kernel123(
            tma_atom_Q, tma_tensor_Q,
            tma_atom_K, tma_tensor_K,
            tma_atom_G, tma_tensor_G,
            mA_log, mBeta, scale,
            mKscaled_v2, mKg_v2, mQscaled_v2, mGkLast_v2, mAqk, mAkk,
            tiled_copy_qk_s2r,
            tiled_mma_k2, tiled_copy_mma_A, tiled_copy_mma_B,
            qk_smem_3d, g_smem_3d, g_cumsum_layout, mProfile, _NT,
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
    smem_est = (BT * K_DIM * 2 * 2 * NUM_STAGES
                + BT * K_DIM * 2 * NUM_STAGES
                + BT * K_STRIDE * 4 * NUM_STAGES
                + K1_ROW_GROUPS * PARTIAL_COLS * 4
                + BC * AQK_TILE_STRIDE * 2 * NUM_STAGES * 2
                + 512)
    print(f"Grid: ({NT // CHUNKS_PER_BLOCK}, {H}, {B}), Block: {THREADS} ({NUM_WARPS} warps)")
    print(f"SMEM: {smem_est} bytes ({smem_est / 1024:.1f} KB)")
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
    profile = torch.zeros(NUM_PROFILE_SLOTS, device="cuda", dtype=torch.int64)

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
    mProfile = _ct(profile, cutlass.Int64)

    print("Compiling...")
    compiled = cute.compile[KeepPTX, KeepCUBIN](
        host_fn,
        mQ, mK, mG, mA_log, mBeta, scale,
        mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk, mProfile,
    )

    print("Running...")
    compiled(mQ, mK, mG, mA_log, mBeta, scale,
             mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk, mProfile)
    torch.cuda.synchronize()
    print("Kernel executed successfully!")

    # --- Cycle profile (block 0, representative thread per warp group) ---
    p = profile.cpu().numpy()
    tma_cycles = p[1] - p[0]
    k1_cycles = p[3] - p[2]
    store_cycles = p[5] - p[4]
    mma_cycles = p[7] - p[6]
    wall_start = min(p[0], p[2], p[4], p[6])
    wall_end = max(p[1], p[3], p[5], p[7])
    wall_cycles = wall_end - wall_start

    print("\n[Cycle Profile] (block 0, lane 0 of each warp group)")
    print(f"  TMA   (warp  0):       {tma_cycles:12d} cycles  [{p[0]:16d} → {p[1]:16d}]")
    print(f"  K1    (warps 1-16):    {k1_cycles:12d} cycles  [{p[2]:16d} → {p[3]:16d}]")
    print(f"  Store (warps 17-20):   {store_cycles:12d} cycles  [{p[4]:16d} → {p[5]:16d}]")
    print(f"  MMA   (warps 21-30):   {mma_cycles:12d} cycles  [{p[6]:16d} → {p[7]:16d}]")
    print(f"  Wall (min start→max end): {wall_cycles:10d} cycles")

    num_warmup, num_iters = 20, 100

    for _ in range(num_warmup):
        compiled(mQ, mK, mG, mA_log, mBeta, scale,
                 mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk, mProfile)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        compiled(mQ, mK, mG, mA_log, mBeta, scale,
                 mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk, mProfile)
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
