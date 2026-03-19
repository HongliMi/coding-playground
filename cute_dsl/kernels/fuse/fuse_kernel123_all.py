"""
Fused K1+K2+K3 Kernel for KDA.

Fuses gate activation + cumsum + scaling (K1), intra sub-chunk Aqk/Akk (K2),
and inter sub-chunk solve + merged inverse (K3) into a single kernel.

Grid: (NT/4, H, B)
Block: 1024 threads (32 warps), warp-specialized with setmaxnreg (all groups 4-aligned):
  Warps 0-15:  TMA+K1 fused (8×2, vec2, prefetch pipeline) – 4 WGs, 72 regs
  Warps 16-27: K2 MMA compute (10 active + 2 idle for WG alignment) – 3 WGs, 56 regs
  Warps 28-31: Store/Inversion warps – 1 WG, 24 regs

Pipeline (prefetch overlap):
  Warps 0-15:  prefetch chunk 0→stage 0 (warp 0), then loop:
                  TMA next chunk (warp 0), wait cur chunk, K1 compute, arrive(k1_done)
  Warps 16-27: wait(k1_done) + wait(store_done) + wait(tma), MMA, arrive(mma_done + stage_reuse)
  Warps 28-31: wait(mma_done), store sAqk/sAkk→GMEM, arrive(store_done)

Mbarriers:
  tma_mbars[2]:          count=1, warp 0 lane 0 → K1+MMA wait for TMA data
  stage_reuse_mbars[2]:  count=384, MMA(12 warps) → warp 0 waits before TMA reuse
  k1_done_mbars[2]:      count=512, K1(16 warps) → MMA waits for g_cumsum ready
  mma_done_mbars[2]:     count=384, MMA(12 warps) → Store waits for sAqk/sAkk ready
  store_done_mbars[2]:   count=128, Store(4 warps) → MMA waits for sAqk/sAkk stage free

SMEM: ~215KB (q+k+g × [64,128] bf16 × 2 stages + g_cumsum [64,136] fp32 × 2 stages
      + sPartialLast [8,132] fp32 + sAqk [16,168,2] bf16
      + sAkk [64,72,2] fp32 block-transposed upper-tri layout)

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
  A_kk       [B,T,H,BT]  fp32  block-transposed upper-tri (input to akk_inv)
"""

import sys
sys.path.insert(0, '/home/scratch.peiyuanz_gpu/mhl/Personal_workspace/scripts/kda_optimized')
sys.path.insert(0, '/home/scratch.peiyuanz_gpu/mhl/Personal_workspace/flash-linear-attention')

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
import subprocess
import os
import glob
import cuda.bindings.driver as cuda_drv

from act_cumsum_scale_fused_cute import act_cumsum_scale_fused_v2_vec as ref_k1
from intra_parellel_cute import run_kda_Akk as ref_k2
from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_inter_solve_fused as ref_k3
from Akk_inverse_lower_triangle import akk_inv_host

B200_PEAK_BW_GBS = 7672  # GB/s

BT = 64
BC = 16
K_DIM = 128
K_PAD = 8
K_STRIDE = K_DIM + K_PAD  # 136, padded row stride to avoid bank conflicts
CHUNKS_PER_BLOCK = 4

NUM_K1_TMA_WARPS = 16    # Warps 0-15:  fused TMA + K1 (4 warpgroups, 8×2)
NUM_MMA_WARPS = 12        # Warps 16-27: MMA (3 warpgroups, 10 active + 2 idle)
NUM_MMA_ACTIVE = 10       # only 10 warps do actual MMA work
NUM_STORE_WARPS = 4       # Warps 28-31: Store/Inversion (1 warpgroup)
NUM_WARPS = NUM_K1_TMA_WARPS + NUM_MMA_WARPS + NUM_STORE_WARPS  # 32
THREADS = NUM_WARPS * 32  # 1024

NUM_SUB_CHUNKS = BT // BC  # 4
NUM_TILES = NUM_SUB_CHUNKS * (NUM_SUB_CHUNKS + 1) // 2  # 10 lower-tri tiles
MMA_K_TILE = 8
NUM_MMA_K_TILES = K_DIM // MMA_K_TILE  # 16
AQK_TILE_COLS = NUM_TILES * BC  # 160
AQK_TILE_PAD = 8
AQK_TILE_STRIDE = AQK_TILE_COLS + AQK_TILE_PAD  # 168

AKK_PAD = 8
AKK_STRIDE = BT + AKK_PAD  # 72

K1_ROW_GROUPS = 8
K1_COL_GROUPS = 2
ROWS_PER_K1_WARP = BT // K1_ROW_GROUPS       # 8
K1_COLS_PER_WARP = K_DIM // K1_COL_GROUPS     # 64
ROWS_PER_STORE_WARP = BT // NUM_STORE_WARPS   # 16

VEC = K1_COLS_PER_WARP // 32  # 2
K_VEC = K_DIM // VEC          # 64
NUM_STAGES = 2
PARTIAL_COLS = K_DIM + 4      # 132
PARTIAL_COLS_PER_WARP = K_DIM // NUM_K1_TMA_WARPS  # 8

_TILE_IQ = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
_TILE_IK = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
RCP_LN2 = LOG2E

# Per-warpgroup register budget via setmaxnreg (requires 4-warp aligned groups)
# WG0-3 (warps 0-15):  K1+TMA  → alloc(72)
# WG4-6 (warps 16-27): MMA     → dealloc(56)
# WG7   (warps 28-31): Store   → dealloc(24)
# Total: (4×72 + 3×56 + 1×24) × 128 threads/WG = 60160 ≤ 65536
# Compiler uses REG:64; any setmaxnreg ≥ 64 is sufficient
NUM_REGS_K1 = 72
NUM_REGS_MMA = 56
NUM_REGS_STORE = 24


@dsl_user_op
def k1_internal_barrier(*, loc=None, ip=None):
    """Named barrier for K1+TMA warps (0-15, 512 threads). barrier_id=2."""
    llvm.inline_asm(
        T.i32(), [],
        "membar.cta; bar.sync 2, 512; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def mma_tf32_m16n8k8(
    a0, a1, a2, a3,
    b0, b1,
    c0, c1, c2, c3,
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


SHFL_W8_CLAMP = 0x1800


@cute.kernel
def fused_kernel123(
    tma_atom_Q: cute.CopyAtom, tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom, tma_tensor_K: cute.Tensor,
    tma_atom_G: cute.CopyAtom, tma_tensor_G: cute.Tensor,
    mA_log: cute.Tensor,
    mBeta: cute.Tensor,
    scale: cutlass.Float32,
    mKscaled: cute.Tensor,
    mKg: cute.Tensor,
    mQscaled: cute.Tensor,
    mGkLast: cute.Tensor,
    mAqk: cute.Tensor,
    mAkk: cute.Tensor,
    tiled_copy_qk_k1,
    tiled_mma_k2,
    tiled_copy_mma_A,
    tiled_copy_mma_B,
    tiled_copy_Gcum_norm,
    tiled_copy_Gcum_gate,
    qk_smem_layout,
    g_smem_layout,
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
    partial_last_layout = cute.make_layout(
        (K1_ROW_GROUPS, PARTIAL_COLS),
        stride=(PARTIAL_COLS, 1))
    sPartialLast = smem.allocate_tensor(cutlass.Float32, partial_last_layout, 128)

    aqk_tile_layout = cute.make_layout(
        (BC, AQK_TILE_STRIDE, NUM_STAGES),
        stride=(AQK_TILE_STRIDE, 1, BC * AQK_TILE_STRIDE))
    sAqk = smem.allocate_tensor(cutlass.BFloat16, aqk_tile_layout, 128)

    akk_tile_layout = cute.make_layout(
        (BT, AKK_STRIDE, NUM_STAGES),
        stride=(AKK_STRIDE, 1, BT * AKK_STRIDE))
    sAkk = smem.allocate_tensor(cutlass.Float32, akk_tile_layout, 128)

    # =====================================================================
    # Mbarrier allocation & init
    # =====================================================================
    tma_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    stage_reuse_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    k1_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    mma_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    store_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)

    bytes_per_stage = BT * K_DIM * 2 * 3

    if tidx == 0:
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_init(tma_mbars + s, 1)
            cute.arch.mbarrier_init(stage_reuse_mbars + s, NUM_MMA_WARPS * 32)
            cute.arch.mbarrier_init(k1_done_mbars + s, NUM_K1_TMA_WARPS * 32)
            cute.arch.mbarrier_init(mma_done_mbars + s, NUM_MMA_WARPS * 32)
            cute.arch.mbarrier_init(store_done_mbars + s, NUM_STORE_WARPS * 32)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # =====================================================================
    # setmaxnreg: redistribute registers across warpgroups
    # =====================================================================
    if warp_idx < NUM_K1_TMA_WARPS:
        cute.arch.warpgroup_reg_alloc(NUM_REGS_K1)
    if warp_idx >= NUM_K1_TMA_WARPS and warp_idx < NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
        cute.arch.warpgroup_reg_dealloc(NUM_REGS_MMA)
    if warp_idx >= NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
        cute.arch.warpgroup_reg_dealloc(NUM_REGS_STORE)

    # =====================================================================
    # Pre-arrive (MMA warps only)
    # stage_reuse_mbars: warp 0 waits before MMA arrives → pre-arrive all 12 MMA warps
    # store_done_mbars:  MMA waits before Store arrives → pre-arrive first 4 MMA warps
    # =====================================================================
    if warp_idx >= NUM_K1_TMA_WARPS and warp_idx < NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
        mma_warp_tmp = warp_idx - NUM_K1_TMA_WARPS
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_arrive(stage_reuse_mbars + s)
            if mma_warp_tmp < NUM_STORE_WARPS:
                cute.arch.mbarrier_arrive(store_done_mbars + s)

    # =================================================================
    # Warps 0-15: Fused TMA + K1 (8×2 layout, vec2, with TMA prefetch)
    #
    #              cols 0-63       cols 64-127
    # rows 0-7      warp 0 (TMA)    warp 8
    # rows 8-15     warp 1          warp 9
    # ...
    # rows 56-63    warp 7          warp 15
    # =================================================================
    if warp_idx < NUM_K1_TMA_WARPS:
        k1_warp = warp_idx
        warp_row_group = k1_warp % K1_ROW_GROUPS
        warp_col_group = k1_warp // K1_ROW_GROUPS
        k1_row_start = warp_row_group * ROWS_PER_K1_WARP
        col_base = warp_col_group * K1_COLS_PER_WARP + lane_id * VEC
        col_vec_idx = warp_col_group * (K1_COLS_PER_WARP // VEC) + lane_id

        exp_A = cute.exp(mA_log[i_h], fastmath=True)
        cumsum_scale = cutlass.Float32(RCP_LN2)

        rAcc = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
        rPrefix = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
        rGkLast = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
        rKsOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
        rQsOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
        rKgOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
        rGkOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)

        # ---- Step 1: TMA prefetch chunk 0 → stage 0 (warp 0 only) ----
        if warp_idx == 0:
            prefetch_i_bnt = i_b * num_chunks + chunk_base
            cute.arch.mbarrier_wait(stage_reuse_mbars, 0)

            if lane_id == 0:
                cute.arch.mbarrier_expect_tx(tma_mbars, bytes_per_stage)

            sQ_pf = sQ[(None, None, 0)]
            gQ_pf = cute.local_tile(tma_tensor_Q, (BT, K_DIM, 1, 1), (0, 0, prefetch_i_bnt, i_h))
            ts_pf, tg_pf = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1),
                cute.group_modes(sQ_pf, 0, 2), cute.group_modes(gQ_pf[(None, None, 0, 0)], 0, 2))
            cute.copy(tma_atom_Q, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)

            sK_pf = sK[(None, None, 0)]
            gK_pf = cute.local_tile(tma_tensor_K, (BT, K_DIM, 1, 1), (0, 0, prefetch_i_bnt, i_h))
            ts_pf, tg_pf = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1),
                cute.group_modes(sK_pf, 0, 2), cute.group_modes(gK_pf[(None, None, 0, 0)], 0, 2))
            cute.copy(tma_atom_K, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)

            sG_pf = sG[(None, None, 0)]
            gG_pf = cute.local_tile(tma_tensor_G, (BT, K_DIM, 1, 1), (0, 0, prefetch_i_bnt, i_h))
            ts_pf, tg_pf = cpasync.tma_partition(tma_atom_G, 0, cute.make_layout(1),
                cute.group_modes(sG_pf, 0, 2), cute.group_modes(gG_pf[(None, None, 0, 0)], 0, 2))
            cute.copy(tma_atom_G, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)

            if lane_id == 0:
                cute.arch.mbarrier_arrive(tma_mbars)

        # ---- Step 2: main processing loop ----
        for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
            cur_stage = chunk_iter % NUM_STAGES
            cur_phase = chunk_iter // NUM_STAGES % 2
            chunk_idx = chunk_base + chunk_iter
            chunk_start = chunk_idx * BT

            # -- TMA for NEXT chunk (warp 0 only, skip last iteration) --
            if chunk_iter < CHUNKS_PER_BLOCK - 1:
                if warp_idx == 0:
                    next_stage = (chunk_iter + 1) % NUM_STAGES
                    next_phase = (chunk_iter + 1) // NUM_STAGES % 2
                    next_i_bnt = i_b * num_chunks + chunk_base + chunk_iter + 1

                    cute.arch.mbarrier_wait(stage_reuse_mbars + next_stage, next_phase)

                    if lane_id == 0:
                        cute.arch.mbarrier_expect_tx(tma_mbars + next_stage, bytes_per_stage)

                    sQ_ns = sQ[(None, None, next_stage)]
                    gQ_ns = cute.local_tile(tma_tensor_Q, (BT, K_DIM, 1, 1), (0, 0, next_i_bnt, i_h))
                    ts_ns, tg_ns = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1),
                        cute.group_modes(sQ_ns, 0, 2), cute.group_modes(gQ_ns[(None, None, 0, 0)], 0, 2))
                    cute.copy(tma_atom_Q, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)

                    sK_ns = sK[(None, None, next_stage)]
                    gK_ns = cute.local_tile(tma_tensor_K, (BT, K_DIM, 1, 1), (0, 0, next_i_bnt, i_h))
                    ts_ns, tg_ns = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1),
                        cute.group_modes(sK_ns, 0, 2), cute.group_modes(gK_ns[(None, None, 0, 0)], 0, 2))
                    cute.copy(tma_atom_K, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)

                    sG_ns = sG[(None, None, next_stage)]
                    gG_ns = cute.local_tile(tma_tensor_G, (BT, K_DIM, 1, 1), (0, 0, next_i_bnt, i_h))
                    ts_ns, tg_ns = cpasync.tma_partition(tma_atom_G, 0, cute.make_layout(1),
                        cute.group_modes(sG_ns, 0, 2), cute.group_modes(gG_ns[(None, None, 0, 0)], 0, 2))
                    cute.copy(tma_atom_G, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)

                    if lane_id == 0:
                        cute.arch.mbarrier_arrive(tma_mbars + next_stage)

            # -- Wait for current chunk TMA data --
            cute.arch.mbarrier_wait(tma_mbars + cur_stage, cur_phase)

            csG = sG[(None, None, cur_stage)]
            csGcum = sGcum[(None, None, cur_stage)]
            csQ = sQ[(None, None, cur_stage)]
            csK = sK[(None, None, cur_stage)]

            # ---- Pass 1: activate G, local prefix sum ----
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

            for vi in cutlass.range_constexpr(VEC):
                sPartialLast[warp_row_group, col_base + vi] = rAcc[vi]

            k1_internal_barrier()

            # ---- Shuffle-based prefix sum across 8 row groups ----
            prefix_col_start = k1_warp * PARTIAL_COLS_PER_WARP
            row_in_prefix = lane_id % K1_ROW_GROUPS
            col_in_group = lane_id // K1_ROW_GROUPS

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

            k1_internal_barrier()

            for vi in cutlass.range_constexpr(VEC):
                rGkLast[vi] = sPartialLast[K1_ROW_GROUPS - 1, col_base + vi]

            for vi in cutlass.range_constexpr(VEC):
                rPrefix[vi] = cutlass.Float32(0.0)
            if warp_row_group > 0:
                for vi in cutlass.range_constexpr(VEC):
                    rPrefix[vi] = sPartialLast[warp_row_group - 1, col_base + vi]

            # ---- Pass 2: re-scan with offset, compute & store ----
            for vi in cutlass.range_constexpr(VEC):
                rAcc[vi] = rPrefix[vi]

            thr_copy_k1 = tiled_copy_qk_k1.get_slice(lane_id)

            for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                row = k1_row_start + ri
                t = chunk_start + row

                sK_tile = cute.local_tile(csK, tiler=(1, K1_COLS_PER_WARP), coord=(row, warp_col_group))
                tCsK = thr_copy_k1.partition_S(sK_tile)
                tCrK = cute.make_fragment_like(tCsK)
                cute.copy(tiled_copy_qk_k1, tCsK, thr_copy_k1.retile(tCrK))

                sQ_tile = cute.local_tile(csQ, tiler=(1, K1_COLS_PER_WARP), coord=(row, warp_col_group))
                tCsQ = thr_copy_k1.partition_S(sQ_tile)
                tCrQ = cute.make_fragment_like(tCsQ)
                cute.copy(tiled_copy_qk_k1, tCsQ, thr_copy_k1.retile(tCrQ))

                for vi in cutlass.range_constexpr(VEC):
                    c = col_base + vi
                    rAcc[vi] = rAcc[vi] + rGact[ri, vi]

                    cs = rAcc[vi] * cumsum_scale

                    k_val = tCrK[vi].to(cutlass.Float32)
                    q_val = tCrQ[vi].to(cutlass.Float32)

                    exp2_cs = cute.exp2(cs, fastmath=True)
                    gk_last_cs = rGkLast[vi] * cumsum_scale
                    exp2_kg = cute.exp2(gk_last_cs - cs, fastmath=True)

                    csGcum[row, c] = cs

                    rKsOut[vi] = (k_val * exp2_cs).to(cutlass.BFloat16)
                    rQsOut[vi] = (q_val * exp2_cs * scale).to(cutlass.BFloat16)
                    rKgOut[vi] = (k_val * exp2_kg).to(cutlass.BFloat16)

                cute.autovec_copy(rKsOut, mKscaled[i_b, t, i_h, col_vec_idx, None])
                cute.autovec_copy(rQsOut, mQscaled[i_b, t, i_h, col_vec_idx, None])
                cute.autovec_copy(rKgOut, mKg[i_b, t, i_h, col_vec_idx, None])

            if warp_row_group == 0:
                for vi in cutlass.range_constexpr(VEC):
                    rGkOut[vi] = cute.exp2(rGkLast[vi] * cumsum_scale, fastmath=True)
                cute.autovec_copy(rGkOut, mGkLast[i_b, chunk_idx, i_h, col_vec_idx, None])

            cute.arch.mbarrier_arrive(k1_done_mbars + cur_stage)

    # =================================================================
    # Warps 16-27: K2 MMA Compute (10 active + 2 idle for WG alignment)
    # =================================================================
    if warp_idx >= NUM_K1_TMA_WARPS and warp_idx < NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
        mma_warp = warp_idx - NUM_K1_TMA_WARPS  # 0..11

        # Decode (i_q, i_k) — only for active warps (0..9)
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
        elif mma_warp < NUM_MMA_ACTIVE:
            my_i_q = cutlass.Int32(3)
            my_i_k = mma_warp - 6

        q_row_base = my_i_q * BC
        k_row_base = my_i_k * BC
        tile_col_base = mma_warp * BC

        akk_row_base = k_row_base   # block-transposed: Akk(i_q,i_k) → sAkk row i_k
        akk_col_base = q_row_base   # block-transposed: Akk(i_q,i_k) → sAkk col i_q

        norm_row = q_row_base
        if my_i_q == my_i_k:
            norm_row = q_row_base + cutlass.Int32(BC // 2)

        group_id = lane_id // 4
        tid_in_group = lane_id % 4
        row0, row1 = group_id, group_id + 8
        col0, col1 = tid_in_group * 2, tid_in_group * 2 + 1
        col2, col3 = 8 + tid_in_group * 2, 8 + tid_in_group * 2 + 1

        thr_mma = tiled_mma_k2.get_slice(lane_id)
        thr_copy_A = tiled_copy_mma_A.get_slice(lane_id)
        thr_copy_B = tiled_copy_mma_B.get_slice(lane_id)
        thr_copy_Gn = tiled_copy_Gcum_norm.get_slice(tid_in_group)
        thr_copy_Ggate = tiled_copy_Gcum_gate.get_slice(lane_id)

        for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
            s = chunk_iter % NUM_STAGES
            phase = chunk_iter // NUM_STAGES % 2
            chunk_idx = chunk_base + chunk_iter
            chunk_start = chunk_idx * BT

            cute.arch.mbarrier_wait(k1_done_mbars + s, phase)
            cute.arch.mbarrier_wait(store_done_mbars + s, phase)
            # cute.arch.mbarrier_wait(tma_mbars + s, phase)

            if mma_warp < NUM_MMA_ACTIVE:
                csQ = sQ[(None, None, s)]
                csK = sK[(None, None, s)]
                csGcum = sGcum[(None, None, s)]
                csAqk = sAqk[(None, None, s)]
                csAkk = sAkk[(None, None, s)]

                _z = cutlass.Float32(0.0)

                beta_row0 = mBeta[i_b, chunk_start + q_row_base + row0, i_h].to(cutlass.Float32)
                beta_row1 = mBeta[i_b, chunk_start + q_row_base + row1, i_h].to(cutlass.Float32)

                acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = _z, _z, _z, _z
                acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = _z, _z, _z, _z
                acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = _z, _z, _z, _z
                acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = _z, _z, _z, _z

                for k_block in cutlass.range_constexpr(NUM_MMA_K_TILES):
                    sQ_tile = cute.local_tile(csQ, tiler=(16, 8), coord=(my_i_q, k_block))
                    tCrQ = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sQ_tile))
                    cute.copy(tiled_copy_mma_A, thr_copy_A.partition_S(sQ_tile), thr_copy_A.retile(tCrQ))

                    sKq_tile = cute.local_tile(csK, tiler=(16, 8), coord=(my_i_q, k_block))
                    tCrKq = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sKq_tile))
                    cute.copy(tiled_copy_mma_A, thr_copy_A.partition_S(sKq_tile), thr_copy_A.retile(tCrKq))

                    sGn_tile = cute.local_tile(csGcum, tiler=(1, 8), coord=(norm_row, k_block))
                    tCsGn = thr_copy_Gn.partition_S(sGn_tile)
                    tCrGn = cute.make_fragment_like(tCsGn, cutlass.Float32)
                    cute.copy(tiled_copy_Gcum_norm, tCsGn, thr_copy_Gn.retile(tCrGn))
                    g_norm_0 = tCrGn[0]
                    g_norm_1 = tCrGn[1]

                    sGq_tile = cute.local_tile(csGcum, tiler=(16, 8), coord=(my_i_q, k_block))
                    tCrGq = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGq_tile))
                    cute.copy(tiled_copy_Gcum_gate, thr_copy_Ggate.partition_S(sGq_tile), thr_copy_Ggate.retile(tCrGq))
                    gate_q_0 = cute.exp2(tCrGq[0] - g_norm_0, fastmath=True)
                    gate_q_1 = cute.exp2(tCrGq[1] - g_norm_1, fastmath=True)
                    gate_q_2 = cute.exp2(tCrGq[2] - g_norm_0, fastmath=True)
                    gate_q_3 = cute.exp2(tCrGq[3] - g_norm_1, fastmath=True)

                    qa0 = tCrQ[0].to(cutlass.Float32) * gate_q_0
                    qa1 = tCrQ[2].to(cutlass.Float32) * gate_q_2
                    qa2 = tCrQ[1].to(cutlass.Float32) * gate_q_1
                    qa3 = tCrQ[3].to(cutlass.Float32) * gate_q_3
                    ka0 = tCrKq[0].to(cutlass.Float32) * gate_q_0
                    ka1 = tCrKq[2].to(cutlass.Float32) * gate_q_2
                    ka2 = tCrKq[1].to(cutlass.Float32) * gate_q_1
                    ka3 = tCrKq[3].to(cutlass.Float32) * gate_q_3

                    sK_tile_n0 = cute.local_tile(csK, tiler=(8, 8), coord=(my_i_k * 2, k_block))
                    tCrK_n0 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n0))
                    cute.copy(tiled_copy_mma_B, thr_copy_B.partition_S(sK_tile_n0), thr_copy_B.retile(tCrK_n0))

                    sK_tile_n1 = cute.local_tile(csK, tiler=(8, 8), coord=(my_i_k * 2 + 1, k_block))
                    tCrK_n1 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n1))
                    cute.copy(tiled_copy_mma_B, thr_copy_B.partition_S(sK_tile_n1), thr_copy_B.retile(tCrK_n1))

                    sGk_tile = cute.local_tile(csGcum, tiler=(16, 8), coord=(my_i_k, k_block))
                    tCrGk = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGk_tile))
                    cute.copy(tiled_copy_Gcum_gate, thr_copy_Ggate.partition_S(sGk_tile), thr_copy_Ggate.retile(tCrGk))
                    gk_n0_0 = cute.exp2(g_norm_0 - tCrGk[0], fastmath=True)
                    gk_n0_1 = cute.exp2(g_norm_1 - tCrGk[1], fastmath=True)
                    k_n0_b0 = tCrK_n0[0].to(cutlass.Float32) * gk_n0_0
                    k_n0_b1 = tCrK_n0[1].to(cutlass.Float32) * gk_n0_1

                    gk_n1_0 = cute.exp2(g_norm_0 - tCrGk[2], fastmath=True)
                    gk_n1_1 = cute.exp2(g_norm_1 - tCrGk[3], fastmath=True)
                    k_n1_b0 = tCrK_n1[0].to(cutlass.Float32) * gk_n1_0
                    k_n1_b1 = tCrK_n1[1].to(cutlass.Float32) * gk_n1_1

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

                csAqk[row0, tile_col_base + col0] = (acc_aqk_n0_0 * scale).to(cutlass.BFloat16)
                csAqk[row0, tile_col_base + col1] = (acc_aqk_n0_1 * scale).to(cutlass.BFloat16)
                csAqk[row1, tile_col_base + col0] = (acc_aqk_n0_2 * scale).to(cutlass.BFloat16)
                csAqk[row1, tile_col_base + col1] = (acc_aqk_n0_3 * scale).to(cutlass.BFloat16)
                csAqk[row0, tile_col_base + col2] = (acc_aqk_n1_0 * scale).to(cutlass.BFloat16)
                csAqk[row0, tile_col_base + col3] = (acc_aqk_n1_1 * scale).to(cutlass.BFloat16)
                csAqk[row1, tile_col_base + col2] = (acc_aqk_n1_2 * scale).to(cutlass.BFloat16)
                csAqk[row1, tile_col_base + col3] = (acc_aqk_n1_3 * scale).to(cutlass.BFloat16)

                csAkk[akk_row_base + row0, akk_col_base + col0] = acc_akk_n0_0 * beta_row0
                csAkk[akk_row_base + row0, akk_col_base + col1] = acc_akk_n0_1 * beta_row0
                csAkk[akk_row_base + row1, akk_col_base + col0] = acc_akk_n0_2 * beta_row1
                csAkk[akk_row_base + row1, akk_col_base + col1] = acc_akk_n0_3 * beta_row1
                csAkk[akk_row_base + row0, akk_col_base + col2] = acc_akk_n1_0 * beta_row0
                csAkk[akk_row_base + row0, akk_col_base + col3] = acc_akk_n1_1 * beta_row0
                csAkk[akk_row_base + row1, akk_col_base + col2] = acc_akk_n1_2 * beta_row1
                csAkk[akk_row_base + row1, akk_col_base + col3] = acc_akk_n1_3 * beta_row1

            # All 12 MMA warps (including idle) participate in barriers
            cute.arch.mbarrier_arrive(mma_done_mbars + s)
            cute.arch.mbarrier_arrive(stage_reuse_mbars + s)

    # =================================================================
    # Warps 28-31: Store/Inversion warps
    # =================================================================
    if warp_idx >= NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
        store_warp = warp_idx - (NUM_K1_TMA_WARPS + NUM_MMA_WARPS)

        for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
            s = chunk_iter % NUM_STAGES
            phase = chunk_iter // NUM_STAGES % 2
            chunk_idx = chunk_base + chunk_iter
            chunk_start = chunk_idx * BT

            cute.arch.mbarrier_wait(mma_done_mbars + s, phase)

            csAqk = sAqk[(None, None, s)]
            csAkk = sAkk[(None, None, s)]

            for tile_idx in cutlass.range_constexpr(NUM_TILES):
                i_q = _TILE_IQ[tile_idx]
                i_k = _TILE_IK[tile_idx]
                is_diag = _TILE_IQ[tile_idx] == _TILE_IK[tile_idx]
                aqk_col_base = tile_idx * BC
                akk_smem_rb = _TILE_IK[tile_idx] * BC
                akk_smem_cb = _TILE_IQ[tile_idx] * BC
                gmem_aqk_row = chunk_start + i_q * BC
                gmem_aqk_col = i_k * BC

                for ri in cutlass.range_constexpr(BC // NUM_STORE_WARPS):
                    local_row = store_warp * (BC // NUM_STORE_WARPS) + ri
                    if lane_id < BC:
                        local_col = lane_id
                        aqk_val = csAqk[local_row, aqk_col_base + local_col]
                        akk_val_f32 = cutlass.Float32(csAkk[akk_smem_rb + local_row, akk_smem_cb + local_col])

                        if is_diag and local_row < local_col:
                            aqk_val = cutlass.BFloat16(0.0)
                        if is_diag and local_row <= local_col:
                            akk_val_f32 = cutlass.Float32(0.0)

                        mAqk[i_b, gmem_aqk_row + local_row, i_h, gmem_aqk_col + local_col] = aqk_val
                        mAkk[i_b, chunk_start + i_k * BC + local_row, i_h, i_q * BC + local_col] = akk_val_f32

            cute.arch.mbarrier_arrive(store_done_mbars + s)


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

        smem_atom_qk = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16)
        qk_smem_2d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM), order=(0, 1))
        qk_smem_3d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM, NUM_STAGES), order=(0, 1, 2))

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

        copy_atom_qk_k1 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=32
        )
        tiled_copy_qk_k1 = cute.make_tiled_copy_tv(
            copy_atom_qk_k1,
            thr_layout=cute.make_layout((1, 32)),
            val_layout=cute.make_layout((1, 2))
        )

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

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 8))
        tiled_mma_k2 = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 8))

        tiled_copy_mma_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16),
            tiled_mma_k2)
        tiled_copy_mma_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 1), cutlass.BFloat16),
            tiled_mma_k2)

        copy_atom_Gcum = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=64
        )
        tiled_copy_Gcum_norm = cute.make_tiled_copy_tv(
            copy_atom_Gcum,
            thr_layout=cute.make_layout((1, 4)),
            val_layout=cute.make_layout((1, 2))
        )
        tiled_copy_Gcum_gate = cute.make_tiled_copy_C(copy_atom_Gcum, tiled_mma_k2)

        smem_size = (BT * K_DIM * 2 * 2 * NUM_STAGES
                     + BT * K_DIM * 2 * NUM_STAGES
                     + BT * K_STRIDE * 4 * NUM_STAGES
                     + K1_ROW_GROUPS * PARTIAL_COLS * 4
                     + BC * AQK_TILE_STRIDE * 2 * NUM_STAGES
                     + BT * AKK_STRIDE * 4 * NUM_STAGES
                     + 512)

        fused_kernel123(
            tma_atom_Q, tma_tensor_Q,
            tma_atom_K, tma_tensor_K,
            tma_atom_G, tma_tensor_G,
            mA_log, mBeta, scale,
            mKscaled_v2, mKg_v2, mQscaled_v2, mGkLast_v2, mAqk, mAkk,
            tiled_copy_qk_k1,
            tiled_mma_k2, tiled_copy_mma_A, tiled_copy_mma_B,
            tiled_copy_Gcum_norm, tiled_copy_Gcum_gate,
            qk_smem_3d, g_smem_3d, g_cumsum_layout, _NT,
        ).launch(
            grid=(_NT // CHUNKS_PER_BLOCK, _H, _B),
            block=(THREADS, 1, 1),
            smem=smem_size,
        )

    return host_fn


# =========================================================================
# Test Helpers
# =========================================================================
_k2_ref_cache = {}
_k2_ref_dummy = {}


def _ct_ref(t, etype):
    r = from_dlpack(t, assumed_align=16)
    r.element_type = etype
    return r


def _get_k2_ref_dummy(device):
    dev_idx = device.index if device.index is not None else 0
    if dev_idx not in _k2_ref_dummy:
        _k2_ref_dummy[dev_idx] = (
            _ct_ref(torch.empty(2, dtype=torch.int64, device=device), cutlass.Int64),
            _ct_ref(torch.empty(1, 2, dtype=torch.int64, device=device), cutlass.Int64),
        )
    return _k2_ref_dummy[dev_idx]


def _compare(name, ref, fused):
    """Compare two tensors, masking out NaN/inf from both sides."""
    r, f = ref.float(), fused.float()
    valid = torch.isfinite(r) & torch.isfinite(f)
    n_valid = valid.sum().item()
    r_nan = r.isnan().sum().item()
    f_nan = f.isnan().sum().item()

    if n_valid == 0:
        print(f"  {name:<15s}  no valid elements  nan: ref={r_nan} fused={f_nan}")
        return

    rv, fv = r[valid], f[valid]
    diff = (rv - fv).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    n = diff.numel()
    n_ge_1e2 = (diff >= 1e-2).sum().item()
    n_ge_1e3 = (diff >= 1e-3).sum().item()
    pct_1e2 = (n - n_ge_1e2) / n * 100
    pct_1e3 = (n - n_ge_1e3) / n * 100

    nan_info = ""
    if r_nan > 0 or f_nan > 0:
        nan_info = f"  nan: ref={r_nan} fused={f_nan}"
    print(f"  {name:<15s}  max={max_diff:.6f}  mean={mean_diff:.6f}")
    print(f"         |diff|<1e-2: {pct_1e2:.3f}% ({n_ge_1e2} outliers),  "
          f"|diff|<1e-3: {pct_1e3:.3f}% ({n_ge_1e3} outliers){nan_info}")


def _compile_fused(B, NT, H, q, k, g, A_log, beta, scale,
                   k_scaled, kg, q_scaled, gk_last, A_qk, A_kk,
                   keep_artifacts=False):
    """Compile fused K1+K2 kernel, return (compiled_fn, cute_args)."""
    host_fn = make_host_function(B, NT, H)
    dl = from_dlpack

    def _ct(t, etype):
        r = dl(t, assumed_align=16)
        r.element_type = etype
        return r

    ct_args = (
        _ct(q, cutlass.BFloat16), _ct(k, cutlass.BFloat16),
        _ct(g, cutlass.BFloat16), _ct(A_log, cutlass.Float32),
        _ct(beta, cutlass.BFloat16), scale,
        _ct(k_scaled, cutlass.BFloat16), _ct(kg, cutlass.BFloat16),
        _ct(q_scaled, cutlass.BFloat16), _ct(gk_last, cutlass.Float32),
        _ct(A_qk, cutlass.BFloat16), _ct(A_kk, cutlass.Float32),
    )

    if keep_artifacts:
        compiled = cute.compile[KeepPTX, KeepCUBIN](host_fn, *ct_args)
    else:
        compiled = cute.compile(host_fn, *ct_args)
    return compiled, ct_args


def _sass_analysis():
    """Dump SASS and resource usage from the latest cubin."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cubin_files = glob.glob(os.path.join(script_dir, "*.cubin"))
    if not cubin_files:
        cubin_files = glob.glob("*.cubin")
    if not cubin_files:
        print("  No cubin file found")
        return

    cubin_file = max(cubin_files, key=os.path.getmtime)
    result = subprocess.run(["cuobjdump", "-sass", cubin_file],
                            capture_output=True, text=True)
    if result.returncode == 0:
        sass_file = os.path.join(script_dir, "fuse_kernel123_8x2_ws.sass")
        with open(sass_file, 'w') as f:
            f.write(result.stdout)
        stl = result.stdout.count("STL")
        ldl = result.stdout.count("LDL")
        print(f"  SASS: STL(spill store)={stl}, LDL(spill load)={ldl}")

    result = subprocess.run(["cuobjdump", "-res-usage", cubin_file],
                            capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if any(kw in line.lower() for kw in ['reg', 'smem', 'stack']):
                print(f"  {line.strip()}")


# =========================================================================
# --verify: Correctness verification (fused K1+K2 vs separate K1 → K2 → K3)
# =========================================================================
def verify(B=1, H=96, K=128, NT=128):
    cutlass.cuda.initialize_cuda_context()
    T = NT * BT
    scale = 1.0 / (K ** 0.5)

    print("=" * 72)
    print("Correctness: fused K1+K2 vs reference K1 → K2 → K3")
    print("=" * 72)
    print(f"B={B}, T={T}, H={H}, K={K}, NT={NT}, BT={BT}, BC={BC}")
    print()

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device="cuda"), dim=-1).to(torch.bfloat16)
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)

    # --- Reference K1 → K2 → K3 ---
    print("[1/3] Running reference K1 → K2 → K3 ...")
    g_cumsum, k_scaled_ref, kg_ref, q_scaled_ref, gk_last_ref = ref_k1(
        g=g.contiguous(), k=k.contiguous(), q=q.contiguous(),
        A_log=A_log.float().contiguous(), cumsum_scale=RCP_LN2, attn_scale=scale,
    )

    A_qk_ref = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    A_kkd_ref = torch.zeros(B, T, H, BC, device="cuda", dtype=torch.float32)
    g_ct = _ct_ref(g_cumsum.contiguous(), cutlass.Float32)
    q_ct = _ct_ref(q.contiguous(), cutlass.BFloat16)
    k_ct = _ct_ref(k.contiguous(), cutlass.BFloat16)
    b_ct = _ct_ref(beta.contiguous(), cutlass.BFloat16)
    ak_ct = _ct_ref(A_kkd_ref, cutlass.Float32)
    aq_ct = _ct_ref(A_qk_ref, cutlass.BFloat16)
    du, di = _get_k2_ref_dummy(q.device)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    cache_key = (B, T, H, K, q.device.index or 0)
    if cache_key not in _k2_ref_cache:
        _k2_ref_cache[cache_key] = cute.compile(
            ref_k2, g_ct, q_ct, k_ct, b_ct, ak_ct, aq_ct,
            float(scale), stream, du, di, 0, cutlass.Int32(0))
    _k2_ref_cache[cache_key](
        g_ct, q_ct, k_ct, b_ct, ak_ct, aq_ct,
        float(scale), stream, du, di, 0)
    torch.cuda.synchronize()

    A_kk_ref = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    ref_k3[(NT, B * H)](
        q=q, k=k, g=g_cumsum, beta=beta,
        Aqk=A_qk_ref, Akkd=A_kkd_ref, Akk=A_kk_ref,
        scale=scale, cu_seqlens=None, chunk_indices=None,
        T=T, H=H, K=K, BT=BT, BC=BC, USE_SAFE_GATE=True)
    torch.cuda.synchronize()
    print("      Done.")

    # --- Fused K1+K2 ---
    print("[2/3] Compiling + running fused kernel ...")
    k_scaled_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    kg_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    q_scaled_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    gk_last_f = torch.empty(B, NT, H, K, device="cuda", dtype=torch.float32)
    A_qk_f = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    A_kk_f = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.float32)

    compiled, ct_args = _compile_fused(
        B, NT, H, q, k, g, A_log, beta, scale,
        k_scaled_f, kg_f, q_scaled_f, gk_last_f, A_qk_f, A_kk_f)
    compiled(*ct_args)
    torch.cuda.synchronize()
    print("      Done.")

    # --- Invert A_kk using akk_inv ---
    print("[3/4] Inverting A_kk via akk_inv ...")
    # A_kk_f is [B, T, H, BT] fp32, akk_inv outputs [B, T, H, BT] bf16
    A_kk_inv_bf16 = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    A_kk_in_ct = _ct_ref(A_kk_f, cutlass.Float32)
    A_kk_inv_ct = _ct_ref(A_kk_inv_bf16, cutlass.BFloat16)

    akk_compiled = cute.compile(akk_inv_host, A_kk_in_ct, A_kk_inv_ct, B, NT, H)
    akk_compiled(A_kk_in_ct, A_kk_inv_ct)
    torch.cuda.synchronize()
    print("      Done.")

    # --- Compare ---
    print()
    print("[4/4] Comparing outputs:")
    print("  --- K1 outputs (fused vs ref_k1) ---")
    _compare("k_scaled", k_scaled_ref, k_scaled_f)
    _compare("q_scaled", q_scaled_ref, q_scaled_f)
    _compare("kg", kg_ref, kg_f)
    _compare("gk_last_exp", gk_last_ref, gk_last_f)

    print("  --- K2+K3 outputs (fused+akk_inv vs ref_k1->k2->k3) ---")
    _compare("A_qk", A_qk_ref, A_qk_f)
    _compare("A_kk", A_kk_ref, A_kk_inv_bf16)

    print()
    print("=" * 72)


# =========================================================================
# --bench: Performance benchmark (fused K1+K2+K3 vs separate K1→K2→K3)
# =========================================================================
def bench(B=1, H=96, K=128, NT=128, num_warmup=20, num_iters=100):
    cutlass.cuda.initialize_cuda_context()
    T = NT * BT
    scale = 1.0 / (K ** 0.5)

    print("=" * 72)
    print("Benchmark: separate K1+K2+K3 vs fused K1+K2+K3")
    print("=" * 72)
    print(f"B={B}, T={T}, H={H}, K={K}, NT={NT}")
    print(f"Warmup={num_warmup}, Iters={num_iters}")
    print()

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device="cuda"), dim=-1).to(torch.bfloat16)
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)

    # --- Compile reference K2 ---
    print("Compiling reference K2...")
    A_qk_ref = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    A_kkd_ref = torch.zeros(B, T, H, BC, device="cuda", dtype=torch.float32)
    A_kk_ref = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.bfloat16)

    g_dummy = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)
    g_ct = _ct_ref(g_dummy, cutlass.Float32)
    q_ct = _ct_ref(q.contiguous(), cutlass.BFloat16)
    k_ct = _ct_ref(k.contiguous(), cutlass.BFloat16)
    b_ct = _ct_ref(beta.contiguous(), cutlass.BFloat16)
    ak_ct = _ct_ref(A_kkd_ref, cutlass.Float32)
    aq_ct = _ct_ref(A_qk_ref, cutlass.BFloat16)
    du, di = _get_k2_ref_dummy(q.device)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    k2_compiled = cute.compile(
        ref_k2, g_ct, q_ct, k_ct, b_ct, ak_ct, aq_ct,
        float(scale), stream, du, di, 0, cutlass.Int32(0))

    # --- Compile fused K1+K2 ---
    print("Compiling fused K1+K2...")
    k_scaled_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    kg_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    q_scaled_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    gk_last_f = torch.empty(B, NT, H, K, device="cuda", dtype=torch.float32)
    A_qk_f = torch.empty(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    A_kk_f = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.float32)

    compiled, ct_args = _compile_fused(
        B, NT, H, q, k, g, A_log, beta, scale,
        k_scaled_f, kg_f, q_scaled_f, gk_last_f, A_qk_f, A_kk_f,
        keep_artifacts=True)

    try:
        _sass_analysis()
    except Exception as e:
        print(f"  SASS analysis failed: {e}")

    # --- Compile akk_inv ---
    print("Compiling akk_inv...")
    A_kk_inv_out = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    akk_in_ct = _ct_ref(A_kk_f, cutlass.Float32)
    akk_out_ct = _ct_ref(A_kk_inv_out, cutlass.BFloat16)
    akk_compiled = cute.compile(akk_inv_host, akk_in_ct, akk_out_ct, B, NT, H)
    print()

    def run_ref_k12():
        g_cumsum, _, _, _, _ = ref_k1(
            g=g.contiguous(), k=k.contiguous(), q=q.contiguous(),
            A_log=A_log.float().contiguous(), cumsum_scale=RCP_LN2, attn_scale=scale)
        g_ct_live = _ct_ref(g_cumsum.contiguous(), cutlass.Float32)
        k2_compiled(g_ct_live, q_ct, k_ct, b_ct, ak_ct, aq_ct,
                    float(scale), stream, du, di, 0)

    def run_ref_k123():
        g_cumsum, _, _, _, _ = ref_k1(
            g=g.contiguous(), k=k.contiguous(), q=q.contiguous(),
            A_log=A_log.float().contiguous(), cumsum_scale=RCP_LN2, attn_scale=scale)
        g_ct_live = _ct_ref(g_cumsum.contiguous(), cutlass.Float32)
        k2_compiled(g_ct_live, q_ct, k_ct, b_ct, ak_ct, aq_ct,
                    float(scale), stream, du, di, 0)
        ref_k3[(NT, B * H)](
            q=q, k=k, g=g_cumsum, beta=beta,
            Aqk=A_qk_ref, Akkd=A_kkd_ref, Akk=A_kk_ref,
            scale=scale, cu_seqlens=None, chunk_indices=None,
            T=T, H=H, K=K, BT=BT, BC=BC, USE_SAFE_GATE=True)

    def run_fused():
        compiled(*ct_args)

    def run_fused_k123():
        compiled(*ct_args)
        akk_compiled(akk_in_ct, akk_out_ct)

    # --- Warmup ---
    print("Warming up...")
    for _ in range(num_warmup):
        run_ref_k12()
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        run_ref_k123()
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        run_fused()
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        run_fused_k123()
    torch.cuda.synchronize()

    # --- Timing (CUDA events) ---
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    for _ in range(num_iters):
        run_ref_k12()
    end_ev.record()
    torch.cuda.synchronize()
    ref_k12_us = start_ev.elapsed_time(end_ev) / num_iters * 1000

    start_ev.record()
    for _ in range(num_iters):
        run_ref_k123()
    end_ev.record()
    torch.cuda.synchronize()
    ref_k123_us = start_ev.elapsed_time(end_ev) / num_iters * 1000

    start_ev.record()
    for _ in range(num_iters):
        run_fused()
    end_ev.record()
    torch.cuda.synchronize()
    fused_us = start_ev.elapsed_time(end_ev) / num_iters * 1000

    start_ev.record()
    for _ in range(num_iters):
        run_fused_k123()
    end_ev.record()
    torch.cuda.synchronize()
    fused_k123_us = start_ev.elapsed_time(end_ev) / num_iters * 1000

    # --- Report ---
    print()
    print(f"{'Results':^72}")
    print("-" * 72)
    print(f"  Separate K1+K2:        {ref_k12_us:>8.1f} us")
    print(f"  Separate K1+K2+K3:     {ref_k123_us:>8.1f} us")
    print(f"  Fused K1+K2:           {fused_us:>8.1f} us")
    print(f"  Fused K1+K2+K3:       {fused_k123_us:>8.1f} us")
    print(f"  Speedup K12:           {ref_k12_us / fused_us:>8.2f}x")
    print(f"  Speedup K123:          {ref_k123_us / fused_k123_us:>8.2f}x")
    print("-" * 72)

    read_bytes = B * T * H * K * 2 * 3 + H * 4 + B * T * H * 2
    write_bytes_fused = (B * T * H * K * 2 * 3
                         + B * NT * H * K * 4
                         + B * T * H * BT * 2    # A_qk bf16
                         + B * T * H * BT * 4)   # A_kk fp32
    total_fused = read_bytes + write_bytes_fused
    bw_fused = total_fused / (fused_us * 1e-6) / 1e9

    # akk_inv reads [B,T,H,BT] fp32 + writes [B,T,H,BT] bf16
    inv_io = B * T * H * BT * (4 + 2)
    total_k123 = total_fused + inv_io
    bw_k123 = total_k123 / (fused_k123_us * 1e-6) / 1e9

    print(f"  Fused K12 IO:  {total_fused / 1e6:.1f} MB  BW: {bw_fused:.1f} GB/s "
          f"({bw_fused / B200_PEAK_BW_GBS * 100:.1f}%)")
    print(f"  Fused K123 IO: {total_k123 / 1e6:.1f} MB  BW: {bw_k123:.1f} GB/s "
          f"({bw_k123 / B200_PEAK_BW_GBS * 100:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fused K1+K2+K3 kernel test (8x2 warp-specialized)")
    parser.add_argument("--verify", action="store_true", help="Correctness verification")
    parser.add_argument("--bench", action="store_true", help="Performance benchmark")
    args = parser.parse_args()

    if not args.verify and not args.bench:
        args.verify = True
        args.bench = True

    if args.verify:
        verify()
        print()
    if args.bench:
        bench()

