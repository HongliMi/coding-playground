"""
Fused K1+K2+K3 Kernel for KDA.

Fuses gate activation + cumsum + scaling (K1), intra sub-chunk Aqk/Akk (K2),
and inter sub-chunk solve + merged inverse (K3) into a single kernel.

Grid: (NT/4, H, B)
Block: 1024 threads (32 warps), warp-specialized with setmaxnreg (all groups 4-aligned):
  Warps 0-15:  TMA+K1 fused (8×2, vec2, prefetch pipeline) – 4 WGs, 80 regs
  Warps 16-27: K2 MMA compute (10 active + 2 idle for WG alignment) – 3 WGs, 56 regs
  Warps 28-31: Store + Akk inversion warps – 1 WG, 56 regs

Pipeline (prefetch overlap):
  Warps 0-15:  prefetch chunk 0→stage 0 (warp 0), then loop:
                  TMA next chunk (warp 0), wait cur chunk, K1 compute, arrive(k1_done)
  Warps 16-27: wait(k1_done) + wait(store_done) + wait(tma), MMA, arrive(mma_done + stage_reuse)
  Warps 28-31: wait(mma_done), store Aqk→GMEM, invert Akk 64×64→store→GMEM, arrive(store_done)

Mbarriers:
  tma_mbars[2]:          count=1, warp 0 lane 0 → K1+MMA wait for TMA data
  stage_reuse_mbars[2]:  count=384, MMA(12 warps) → warp 0 waits before TMA reuse
  k1_done_mbars[2]:      count=512, K1(16 warps) → MMA waits for g_cumsum ready
  mma_done_mbars[2]:     count=384, MMA(12 warps) → Store waits for sAqk/sAkk ready
  store_done_mbars[2]:   count=128, Store(4 warps) → MMA waits for sAqk/sAkk stage free

SMEM: ~178KB (q+k+g × [64,128] bf16 × 2 stages + g_cumsum [64,128] fp32 × 2 stages
      + sAqk [16,168,2] bf16 + sAkk [16,168,2] bf16
      + sAkk_in [64,64] bf16 + sAkk_out [64,64] bf16 + 3× sT_inv [16,16] bf16)

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
  A_kk       [B,T,H,BT]  bf16  inverted unit lower triangular (solve_tril fused)
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
from fla.ops.utils.constant import RCP_LN2
from chunk_fwd import chunk_kda_fwd
from cutile_kernel import launch_kda_kernel

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
# WG7   (warps 28-31): Store+Inv → dealloc(56)  [diag inv rInv[16] + MMA chain fragments]
# Total: (16×72 + 12×56 + 4×56) × 32 = (1152+672+224) × 32 = 65536 ≤ 65536
NUM_REGS_K1 = 64
NUM_REGS_MMA = 56
NUM_REGS_STORE = 72


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
def store_internal_barrier(*, loc=None, ip=None):
    """Named barrier for Store warps (28-31, 128 threads). barrier_id=1."""
    llvm.inline_asm(
        T.i32(), [],
        "membar.cta; bar.sync 1, 128; mov.u32 $0, 0;",
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

INV_BLOCK = 16  # sub-block size for Akk inversion (4 × 16×16 = 64×64)

# =========================================================================
# Akk Inversion Helpers (bf16, adapted from solve_tril_cute_v2.py)
# =========================================================================

@dsl_user_op
def _store_acc_to_smem_bf16(acc: cute.Tensor, sOut: cute.Tensor, col_offset, lane_id, *, loc=None, ip=None):
    """Store FP32 accumulator to BF16 SMEM (row-major 16×16, stride=16)."""
    group_id = lane_id // 4
    thread_in_group = lane_id % 4
    row0 = group_id
    row1 = group_id + 8
    col_base = thread_in_group * 2
    sOut[row0, col_offset + col_base] = cutlass.BFloat16(acc[0])
    sOut[row0, col_offset + col_base + 1] = cutlass.BFloat16(acc[1])
    sOut[row1, col_offset + col_base] = cutlass.BFloat16(acc[2])
    sOut[row1, col_offset + col_base + 1] = cutlass.BFloat16(acc[3])


@dsl_user_op
def _store_neg_acc_to_smem_bf16(acc: cute.Tensor, sOut: cute.Tensor, base_row, base_col, col_offset, lane_id, *, loc=None, ip=None):
    """Store negated FP32 accumulator to BF16 SMEM (64×64, stride=64)."""
    group_id = lane_id // 4
    thread_in_group = lane_id % 4
    row0 = group_id
    row1 = group_id + 8
    col_base = thread_in_group * 2
    sOut[base_row + row0, base_col + col_offset + col_base] = cutlass.BFloat16(-acc[0])
    sOut[base_row + row0, base_col + col_offset + col_base + 1] = cutlass.BFloat16(-acc[1])
    sOut[base_row + row1, base_col + col_offset + col_base] = cutlass.BFloat16(-acc[2])
    sOut[base_row + row1, base_col + col_offset + col_base + 1] = cutlass.BFloat16(-acc[3])


@dsl_user_op
def _invert_16x16_halfwarp_bf16(
    sA_in: cute.Tensor, sA_out: cute.Tensor,
    diag_offset, lane_id,
    *, loc=None, ip=None,
):
    """Invert a 16×16 unit lower triangular block. BF16 I/O.

    Register-optimized: eliminates rA array by re-reading from SMEM.
    Incremental accumulation keeps only acc + 1 shuffle + 1 SMEM read live.
    Peak regs: rInv[16] + ~4 temps ≈ 20 fp32 (vs ~50 in unrolled version).
    """
    my_row = lane_id % 16
    halfwarp_base = (lane_id // 16) * 16
    row_off = diag_offset
    col_off = diag_offset

    rInv = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)

    rInv[0] = cutlass.Float32(1.0)
    for x in range(1, 16):
        rInv[x] = cutlass.Float32(0.0)

    for d in range(1, 16):
        col_d = my_row - d
        valid = cutlass.Float32(col_d >= 0)
        a_val = cutlass.Float32(sA_in[row_off + my_row, col_off + col_d]) * valid

        acc = cutlass.Float32(0.0)
        for j in range(1, d):
            a_re = cutlass.Float32(sA_in[row_off + my_row, col_off + my_row - (d - j)])
            inv_shfl = cute.arch.shuffle_sync(rInv[j], halfwarp_base + my_row - d + j)
            acc = acc + a_re * inv_shfl

        rInv[d] = (-a_val - acc) * valid

    rInv[0] = cutlass.Float32(1.0)

    sA_out[row_off + my_row, col_off + my_row] = cutlass.BFloat16(rInv[0])
    for d in range(1, 16):
        sA_out[row_off + my_row, col_off + (my_row + 16 - d) % 16] = cutlass.BFloat16(
            rInv[d] * cutlass.Float32(my_row >= d))


@dsl_user_op
def _phase2a_chain_mma_bf16(
    sA_in: cute.Tensor, sA_out: cute.Tensor, sT: cute.Tensor,
    tril1_br, tril1_bc, b_br, b_bc, tril2_br, tril2_bc, out_br, out_bc,
    lane_id,
    tiled_mma: cute.TiledMma,
    tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy,
    *, loc=None, ip=None,
):
    """Result = -(Tril1 @ B) @ Tril2.  BF16 I/O, 64×64 stride=64 SMEM."""
    sA_Tril1 = cute.local_tile(sA_out, tiler=(16, 16), coord=(tril1_br, tril1_bc))
    sA_B = cute.local_tile(sA_in, tiler=(16, 16), coord=(b_br, b_bc))
    sA_Tril2 = cute.local_tile(sA_out, tiler=(16, 16), coord=(tril2_br, tril2_bc))

    sA_B_T = cute.make_tensor(sA_B.iterator, cute.make_layout((16, 16), stride=(1, 64)))
    sA_Tril2_T = cute.make_tensor(sA_Tril2.iterator, cute.make_layout((16, 16), stride=(1, 64)))

    thr_mma = tiled_mma.get_slice(lane_id)
    thr_copy_A = tiled_copy_s2r_A.get_slice(lane_id)
    thr_copy_B = tiled_copy_s2r_B.get_slice(lane_id)

    tCsA = thr_mma.partition_A(sA_Tril1)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_Tril1), thr_copy_A.retile(tCrA))

    sC_part_layout = cute.make_layout((16, 8), stride=(8, 1))
    sCp = cute.make_tensor(sT.iterator, sC_part_layout)
    tCsCp = thr_mma.partition_C(sCp)
    tCrT0 = tiled_mma.make_fragment_C(tCsCp)
    tCrT0.fill(0.0)
    tCrT1 = tiled_mma.make_fragment_C(tCsCp)
    tCrT1.fill(0.0)

    sB_T_tile0 = cute.local_tile(sA_B_T, tiler=(8, 16), coord=(0, 0))
    tCrB0 = tiled_mma.make_fragment_B(thr_mma.partition_B(sB_T_tile0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sB_T_tile0), thr_copy_B.retile(tCrB0))
    cute.gemm(tiled_mma, tCrT0, tCrA, tCrB0, tCrT0)

    sB_T_tile1 = cute.local_tile(sA_B_T, tiler=(8, 16), coord=(1, 0))
    tCrB1 = tiled_mma.make_fragment_B(thr_mma.partition_B(sB_T_tile1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sB_T_tile1), thr_copy_B.retile(tCrB1))
    cute.gemm(tiled_mma, tCrT1, tCrA, tCrB1, tCrT1)

    _store_acc_to_smem_bf16(tCrT0, sT, 0, lane_id)
    _store_acc_to_smem_bf16(tCrT1, sT, 8, lane_id)
    cute.arch.sync_warp()

    tCsT = thr_mma.partition_A(sT)
    tCrT_A = tiled_mma.make_fragment_A(tCsT)
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sT), thr_copy_A.retile(tCrT_A))

    tCrOut0 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut0.fill(0.0)
    tCrOut1 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut1.fill(0.0)

    sTril2_T_tile0 = cute.local_tile(sA_Tril2_T, tiler=(8, 16), coord=(0, 0))
    tCrC0 = tiled_mma.make_fragment_B(thr_mma.partition_B(sTril2_T_tile0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sTril2_T_tile0), thr_copy_B.retile(tCrC0))
    cute.gemm(tiled_mma, tCrOut0, tCrT_A, tCrC0, tCrOut0)

    sTril2_T_tile1 = cute.local_tile(sA_Tril2_T, tiler=(8, 16), coord=(1, 0))
    tCrC1 = tiled_mma.make_fragment_B(thr_mma.partition_B(sTril2_T_tile1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sTril2_T_tile1), thr_copy_B.retile(tCrC1))
    cute.gemm(tiled_mma, tCrOut1, tCrT_A, tCrC1, tCrOut1)

    out_row = out_br * 16
    out_col = out_bc * 16
    _store_neg_acc_to_smem_bf16(tCrOut0, sA_out, out_row, out_col, 0, lane_id)
    _store_neg_acc_to_smem_bf16(tCrOut1, sA_out, out_row, out_col, 8, lane_id)


@dsl_user_op
def _phase2b_sum_mma_bf16(
    sA_in: cute.Tensor, sA_out: cute.Tensor, sT: cute.Tensor,
    b1_br, b1_bc, c1_br, c1_bc, b2_br, b2_bc, c2_br, c2_bc,
    ai_br, ai_bc, out_br, out_bc,
    lane_id,
    tiled_mma: cute.TiledMma,
    tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy,
    *, loc=None, ip=None,
):
    """Result = -Ai @ (B1 @ C1 + B2 @ C2).  BF16 I/O."""
    sA_B1 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b1_br, b1_bc))
    sA_C1 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c1_br, c1_bc))
    sA_B2 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b2_br, b2_bc))
    sA_C2 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c2_br, c2_bc))
    sA_Ai = cute.local_tile(sA_out, tiler=(16, 16), coord=(ai_br, ai_bc))

    sA_C1_T = cute.make_tensor(sA_C1.iterator, cute.make_layout((16, 16), stride=(1, 64)))
    sA_C2_T = cute.make_tensor(sA_C2.iterator, cute.make_layout((16, 16), stride=(1, 64)))

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

    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_B1))
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B1), thr_copy_A.retile(tCrA))
    sC1_T_t0 = cute.local_tile(sA_C1_T, tiler=(8, 16), coord=(0, 0))
    tCrB0 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC1_T_t0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC1_T_t0), thr_copy_B.retile(tCrB0))
    cute.gemm(tiled_mma, tCrT0, tCrA, tCrB0, tCrT0)
    sC1_T_t1 = cute.local_tile(sA_C1_T, tiler=(8, 16), coord=(1, 0))
    tCrB1 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC1_T_t1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC1_T_t1), thr_copy_B.retile(tCrB1))
    cute.gemm(tiled_mma, tCrT1, tCrA, tCrB1, tCrT1)

    tCrA2 = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_B2))
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B2), thr_copy_A.retile(tCrA2))
    sC2_T_t0 = cute.local_tile(sA_C2_T, tiler=(8, 16), coord=(0, 0))
    tCrB0_2 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC2_T_t0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC2_T_t0), thr_copy_B.retile(tCrB0_2))
    cute.gemm(tiled_mma, tCrT0, tCrA2, tCrB0_2, tCrT0)
    sC2_T_t1 = cute.local_tile(sA_C2_T, tiler=(8, 16), coord=(1, 0))
    tCrB1_2 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC2_T_t1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC2_T_t1), thr_copy_B.retile(tCrB1_2))
    cute.gemm(tiled_mma, tCrT1, tCrA2, tCrB1_2, tCrT1)

    _store_acc_to_smem_bf16(tCrT0, sT, 0, lane_id)
    _store_acc_to_smem_bf16(tCrT1, sT, 8, lane_id)
    cute.arch.sync_warp()

    sT_T = cute.make_tensor(sT.iterator, cute.make_layout((16, 16), stride=(1, 16)))
    tCrAi = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_Ai))
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_Ai), thr_copy_A.retile(tCrAi))

    tCrOut0 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut0.fill(0.0)
    tCrOut1 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut1.fill(0.0)

    sT_T_t0 = cute.local_tile(sT_T, tiler=(8, 16), coord=(0, 0))
    tCrC0 = tiled_mma.make_fragment_B(thr_mma.partition_B(sT_T_t0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sT_T_t0), thr_copy_B.retile(tCrC0))
    cute.gemm(tiled_mma, tCrOut0, tCrAi, tCrC0, tCrOut0)
    sT_T_t1 = cute.local_tile(sT_T, tiler=(8, 16), coord=(1, 0))
    tCrC1 = tiled_mma.make_fragment_B(thr_mma.partition_B(sT_T_t1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sT_T_t1), thr_copy_B.retile(tCrC1))
    cute.gemm(tiled_mma, tCrOut1, tCrAi, tCrC1, tCrOut1)

    out_row = out_br * 16
    out_col = out_bc * 16
    _store_neg_acc_to_smem_bf16(tCrOut0, sA_out, out_row, out_col, 0, lane_id)
    _store_neg_acc_to_smem_bf16(tCrOut1, sA_out, out_row, out_col, 8, lane_id)


@dsl_user_op
def _phase2c_sum3_mma_bf16(
    sA_in: cute.Tensor, sA_out: cute.Tensor, sT: cute.Tensor,
    b1_br, b1_bc, c1_br, c1_bc,
    b2_br, b2_bc, c2_br, c2_bc,
    b3_br, b3_bc, c3_br, c3_bc,
    ai_br, ai_bc, out_br, out_bc,
    lane_id,
    tiled_mma: cute.TiledMma,
    tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy,
    *, loc=None, ip=None,
):
    """Result = -Ai @ (B1@C1 + B2@C2 + B3@C3).  BF16 I/O."""
    sA_B1 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b1_br, b1_bc))
    sA_C1 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c1_br, c1_bc))
    sA_B2 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b2_br, b2_bc))
    sA_C2 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c2_br, c2_bc))
    sA_B3 = cute.local_tile(sA_in, tiler=(16, 16), coord=(b3_br, b3_bc))
    sA_C3 = cute.local_tile(sA_out, tiler=(16, 16), coord=(c3_br, c3_bc))
    sA_Ai = cute.local_tile(sA_out, tiler=(16, 16), coord=(ai_br, ai_bc))

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

    # T = B1@C1
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_B1))
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B1), thr_copy_A.retile(tCrA))
    sC1_T_t0 = cute.local_tile(sA_C1_T, tiler=(8, 16), coord=(0, 0))
    tCrB0 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC1_T_t0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC1_T_t0), thr_copy_B.retile(tCrB0))
    cute.gemm(tiled_mma, tCrT0, tCrA, tCrB0, tCrT0)
    sC1_T_t1 = cute.local_tile(sA_C1_T, tiler=(8, 16), coord=(1, 0))
    tCrB1 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC1_T_t1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC1_T_t1), thr_copy_B.retile(tCrB1))
    cute.gemm(tiled_mma, tCrT1, tCrA, tCrB1, tCrT1)

    # T += B2@C2
    tCrA2 = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_B2))
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B2), thr_copy_A.retile(tCrA2))
    sC2_T_t0 = cute.local_tile(sA_C2_T, tiler=(8, 16), coord=(0, 0))
    tCrB0_2 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC2_T_t0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC2_T_t0), thr_copy_B.retile(tCrB0_2))
    cute.gemm(tiled_mma, tCrT0, tCrA2, tCrB0_2, tCrT0)
    sC2_T_t1 = cute.local_tile(sA_C2_T, tiler=(8, 16), coord=(1, 0))
    tCrB1_2 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC2_T_t1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC2_T_t1), thr_copy_B.retile(tCrB1_2))
    cute.gemm(tiled_mma, tCrT1, tCrA2, tCrB1_2, tCrT1)

    # T += B3@C3
    tCrA3 = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_B3))
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_B3), thr_copy_A.retile(tCrA3))
    sC3_T_t0 = cute.local_tile(sA_C3_T, tiler=(8, 16), coord=(0, 0))
    tCrB0_3 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC3_T_t0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC3_T_t0), thr_copy_B.retile(tCrB0_3))
    cute.gemm(tiled_mma, tCrT0, tCrA3, tCrB0_3, tCrT0)
    sC3_T_t1 = cute.local_tile(sA_C3_T, tiler=(8, 16), coord=(1, 0))
    tCrB1_3 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC3_T_t1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sC3_T_t1), thr_copy_B.retile(tCrB1_3))
    cute.gemm(tiled_mma, tCrT1, tCrA3, tCrB1_3, tCrT1)

    _store_acc_to_smem_bf16(tCrT0, sT, 0, lane_id)
    _store_acc_to_smem_bf16(tCrT1, sT, 8, lane_id)
    cute.arch.sync_warp()

    # Result = -Ai @ T
    sT_T = cute.make_tensor(sT.iterator, cute.make_layout((16, 16), stride=(1, 16)))
    tCrAi = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_Ai))
    cute.copy(tiled_copy_s2r_A, thr_copy_A.partition_S(sA_Ai), thr_copy_A.retile(tCrAi))

    tCrOut0 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut0.fill(0.0)
    tCrOut1 = tiled_mma.make_fragment_C(tCsCp)
    tCrOut1.fill(0.0)

    sT_T_t0 = cute.local_tile(sT_T, tiler=(8, 16), coord=(0, 0))
    tCrC0 = tiled_mma.make_fragment_B(thr_mma.partition_B(sT_T_t0))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sT_T_t0), thr_copy_B.retile(tCrC0))
    cute.gemm(tiled_mma, tCrOut0, tCrAi, tCrC0, tCrOut0)
    sT_T_t1 = cute.local_tile(sT_T, tiler=(8, 16), coord=(1, 0))
    tCrC1 = tiled_mma.make_fragment_B(thr_mma.partition_B(sT_T_t1))
    cute.copy(tiled_copy_s2r_B, thr_copy_B.partition_S(sT_T_t1), thr_copy_B.retile(tCrC1))
    cute.gemm(tiled_mma, tCrOut1, tCrAi, tCrC1, tCrOut1)

    out_row = out_br * 16
    out_col = out_bc * 16
    _store_neg_acc_to_smem_bf16(tCrOut0, sA_out, out_row, out_col, 0, lane_id)
    _store_neg_acc_to_smem_bf16(tCrOut1, sA_out, out_row, out_col, 8, lane_id)


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
    tiled_mma_inv,
    tiled_copy_inv_A,
    tiled_copy_inv_B,
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
    sAkk = smem.allocate_tensor(cutlass.BFloat16, aqk_tile_layout, 128)

    akk_inv_layout = cute.make_layout((BT, BT), stride=(BT, 1))
    sAkk_in = smem.allocate_tensor(cutlass.BFloat16, akk_inv_layout, 128)
    sAkk_out = smem.allocate_tensor(cutlass.BFloat16, akk_inv_layout, 128)
    sT_inv_layout = cute.make_layout((INV_BLOCK, INV_BLOCK), stride=(INV_BLOCK, 1))
    sT_inv_0 = smem.allocate_tensor(cutlass.BFloat16, sT_inv_layout, 16)
    sT_inv_1 = smem.allocate_tensor(cutlass.BFloat16, sT_inv_layout, 16)
    sT_inv_2 = smem.allocate_tensor(cutlass.BFloat16, sT_inv_layout, 16)

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
        cute.arch.setmaxregister_increase(NUM_REGS_K1)
    if warp_idx >= NUM_K1_TMA_WARPS and warp_idx < NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
        cute.arch.setmaxregister_decrease(NUM_REGS_MMA)
    if warp_idx >= NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
        cute.arch.setmaxregister_increase(NUM_REGS_STORE)

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

        group_id = lane_id // 4
        tid_in_group = lane_id % 4
        row0, row1 = group_id, group_id + 8
        col0, col1 = tid_in_group * 2, tid_in_group * 2 + 1
        col2, col3 = 8 + tid_in_group * 2, 8 + tid_in_group * 2 + 1

        thr_mma = tiled_mma_k2.get_slice(lane_id)
        thr_copy_A = tiled_copy_mma_A.get_slice(lane_id)
        thr_copy_B = tiled_copy_mma_B.get_slice(lane_id)

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

                    sK_tile_n0 = cute.local_tile(csK, tiler=(8, 8), coord=(my_i_k * 2, k_block))
                    tCrK_n0 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n0))
                    cute.copy(tiled_copy_mma_B, thr_copy_B.partition_S(sK_tile_n0), thr_copy_B.retile(tCrK_n0))

                    sK_tile_n1 = cute.local_tile(csK, tiler=(8, 8), coord=(my_i_k * 2 + 1, k_block))
                    tCrK_n1 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n1))
                    cute.copy(tiled_copy_mma_B, thr_copy_B.partition_S(sK_tile_n1), thr_copy_B.retile(tCrK_n1))

                    gk_n0_0 = cute.exp2(g_norm_0 - csGcum[k_row_base + group_id, g_col0], fastmath=True)
                    gk_n0_1 = cute.exp2(g_norm_1 - csGcum[k_row_base + group_id, g_col1], fastmath=True)
                    k_n0_b0 = tCrK_n0[0].to(cutlass.Float32) * gk_n0_0
                    k_n0_b1 = tCrK_n0[1].to(cutlass.Float32) * gk_n0_1

                    gk_n1_0 = cute.exp2(g_norm_0 - csGcum[k_row_base + group_id + 8, g_col0], fastmath=True)
                    gk_n1_1 = cute.exp2(g_norm_1 - csGcum[k_row_base + group_id + 8, g_col1], fastmath=True)
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

                csAkk[row0, tile_col_base + col0] = (acc_akk_n0_0 * beta_row0).to(cutlass.BFloat16)
                csAkk[row0, tile_col_base + col1] = (acc_akk_n0_1 * beta_row0).to(cutlass.BFloat16)
                csAkk[row1, tile_col_base + col0] = (acc_akk_n0_2 * beta_row1).to(cutlass.BFloat16)
                csAkk[row1, tile_col_base + col1] = (acc_akk_n0_3 * beta_row1).to(cutlass.BFloat16)
                csAkk[row0, tile_col_base + col2] = (acc_akk_n1_0 * beta_row0).to(cutlass.BFloat16)
                csAkk[row0, tile_col_base + col3] = (acc_akk_n1_1 * beta_row0).to(cutlass.BFloat16)
                csAkk[row1, tile_col_base + col2] = (acc_akk_n1_2 * beta_row1).to(cutlass.BFloat16)
                csAkk[row1, tile_col_base + col3] = (acc_akk_n1_3 * beta_row1).to(cutlass.BFloat16)

            # All 12 MMA warps (including idle) participate in barriers
            cute.arch.mbarrier_arrive(mma_done_mbars + s)
            cute.arch.mbarrier_arrive(stage_reuse_mbars + s)

    # =================================================================
    # Warps 28-31: Store/Inversion warps
    #   1) Store Aqk → GMEM (with upper-tri zeroing for diag tiles)
    #   2) Rearrange Akk from tiled sAkk(16,168) → contiguous sAkk_in(64,64)
    #   3) Invert 64×64 unit lower-tri via 4 stages (solve_tril style)
    #   4) Store inverted sAkk_out → GMEM
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

            # --- Step 1: Store Aqk directly to GMEM ---
            for tile_idx in cutlass.range_constexpr(NUM_TILES):
                i_q = _TILE_IQ[tile_idx]
                i_k = _TILE_IK[tile_idx]
                is_diag = _TILE_IQ[tile_idx] == _TILE_IK[tile_idx]
                tile_col_base = tile_idx * BC
                gmem_row_base = chunk_start + i_q * BC
                gmem_col_base = i_k * BC

                for ri in cutlass.range_constexpr(BC // NUM_STORE_WARPS):
                    local_row = store_warp * (BC // NUM_STORE_WARPS) + ri
                    if lane_id < BC:
                        local_col = lane_id
                        aqk_val = csAqk[local_row, tile_col_base + local_col]
                        if is_diag and local_row < local_col:
                            aqk_val = cutlass.BFloat16(0.0)
                        mAqk[i_b, gmem_row_base + local_row, i_h, gmem_col_base + local_col] = aqk_val

            # --- Step 2: Rearrange Akk tiled → contiguous 64×64 ---
            # 4 warps × 32 lanes = 128 threads copy 10 tiles × 16×16 = 2560 elements
            # Each tile: 16 rows × 16 cols. 4 rows per warp per tile.
            for tile_idx in cutlass.range_constexpr(NUM_TILES):
                i_q = _TILE_IQ[tile_idx]
                i_k = _TILE_IK[tile_idx]
                is_diag = _TILE_IQ[tile_idx] == _TILE_IK[tile_idx]
                tile_col_base = tile_idx * BC
                dst_row_base = i_q * BC
                dst_col_base = i_k * BC

                for ri in cutlass.range_constexpr(BC // NUM_STORE_WARPS):
                    local_row = store_warp * (BC // NUM_STORE_WARPS) + ri
                    if lane_id < BC:
                        local_col = lane_id
                        val = csAkk[local_row, tile_col_base + local_col]
                        if is_diag and local_row < local_col:
                            val = cutlass.BFloat16(0.0)
                        sAkk_in[dst_row_base + local_row, dst_col_base + local_col] = val

            # Zero upper-triangle region: columns beyond the lower-tri blocks
            for ri in cutlass.range_constexpr(BT // NUM_STORE_WARPS):
                row = store_warp * (BT // NUM_STORE_WARPS) + ri
                block_row = row // BC
                for cj in cutlass.range_constexpr(BT // 32):
                    col = lane_id + cj * 32
                    block_col = col // BC
                    if block_col > block_row:
                        sAkk_in[row, col] = cutlass.BFloat16(0.0)

            store_internal_barrier()

            # --- Step 3: Invert 64×64 unit lower-tri (solve_tril stages) ---

            # Stage 1: Invert 4 diagonal 16×16 blocks (4 half-warps in parallel)
            halfwarp_idx = (store_warp * 2 + (lane_id // 16)) % 4
            diag_off = halfwarp_idx * INV_BLOCK
            _invert_16x16_halfwarp_bf16(sAkk_in, sAkk_out, diag_off, lane_id)

            store_internal_barrier()

            # Stage 2a: First-level off-diagonals (3 warps)
            # Ai21 = -(Ai22 @ A21) @ Ai11
            # Ai32 = -(Ai33 @ A32) @ Ai22
            # Ai43 = -(Ai44 @ A43) @ Ai33
            if store_warp == 0:
                _phase2a_chain_mma_bf16(
                    sAkk_in, sAkk_out, sT_inv_0,
                    1, 1, 1, 0, 0, 0, 1, 0,
                    lane_id, tiled_mma_inv, tiled_copy_inv_A, tiled_copy_inv_B)
            if store_warp == 1:
                _phase2a_chain_mma_bf16(
                    sAkk_in, sAkk_out, sT_inv_1,
                    2, 2, 2, 1, 1, 1, 2, 1,
                    lane_id, tiled_mma_inv, tiled_copy_inv_A, tiled_copy_inv_B)
            if store_warp == 2:
                _phase2a_chain_mma_bf16(
                    sAkk_in, sAkk_out, sT_inv_2,
                    3, 3, 3, 2, 2, 2, 3, 2,
                    lane_id, tiled_mma_inv, tiled_copy_inv_A, tiled_copy_inv_B)

            store_internal_barrier()

            # Stage 2b: Second-level off-diagonals (2 warps)
            # Ai31 = -Ai33 @ (A31 @ Ai11 + A32 @ Ai21)
            # Ai42 = -Ai44 @ (A42 @ Ai22 + A43 @ Ai32)
            if store_warp == 0:
                _phase2b_sum_mma_bf16(
                    sAkk_in, sAkk_out, sT_inv_0,
                    2, 0, 0, 0, 2, 1, 1, 0,
                    2, 2, 2, 0,
                    lane_id, tiled_mma_inv, tiled_copy_inv_A, tiled_copy_inv_B)
            if store_warp == 1:
                _phase2b_sum_mma_bf16(
                    sAkk_in, sAkk_out, sT_inv_1,
                    3, 1, 1, 1, 3, 2, 2, 1,
                    3, 3, 3, 1,
                    lane_id, tiled_mma_inv, tiled_copy_inv_A, tiled_copy_inv_B)

            store_internal_barrier()

            # Stage 2c: Third-level off-diagonal (1 warp)
            # Ai41 = -Ai44 @ (A41 @ Ai11 + A42 @ Ai21 + A43 @ Ai31)
            if store_warp == 0:
                _phase2c_sum3_mma_bf16(
                    sAkk_in, sAkk_out, sT_inv_0,
                    3, 0, 0, 0,
                    3, 1, 1, 0,
                    3, 2, 2, 0,
                    3, 3, 3, 0,
                    lane_id, tiled_mma_inv, tiled_copy_inv_A, tiled_copy_inv_B)

            store_internal_barrier()

            # --- Step 4: Store inverted Akk_out → GMEM ---
            for ri in cutlass.range_constexpr(BT // NUM_STORE_WARPS):
                row = store_warp * (BT // NUM_STORE_WARPS) + ri
                for cj in cutlass.range_constexpr(BT // 32):
                    col = lane_id + cj * 32
                    mAkk[i_b, chunk_start + row, i_h, col] = sAkk_out[row, col]

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

        # Inversion MMA: m16n8k16 for 16×16 block MMA in solve_tril stages
        mma_op_inv = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        tiled_mma_inv = cute.make_tiled_mma(
            mma_op_inv, cute.make_layout((1, 1, 1)),
            permutation_mnk=(16, 8, 16))
        tiled_copy_inv_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16),
            tiled_mma_inv)
        tiled_copy_inv_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 2), cutlass.BFloat16),
            tiled_mma_inv)

        smem_size = (BT * K_DIM * 2 * 2 * NUM_STAGES       # sQ + sK
                     + BT * K_DIM * 2 * NUM_STAGES           # sG
                     + BT * K_STRIDE * 4 * NUM_STAGES        # sGcum (fp32)
                     + K1_ROW_GROUPS * PARTIAL_COLS * 4       # sPartialLast
                     + BC * AQK_TILE_STRIDE * 2 * NUM_STAGES * 2  # sAqk + sAkk
                     + BT * BT * 2 * 2                        # sAkk_in + sAkk_out (64×64 bf16 × 2)
                     + INV_BLOCK * INV_BLOCK * 2 * 3          # sT_inv_0/1/2 (16×16 bf16 × 3)
                     + 512)

        fused_kernel123(
            tma_atom_Q, tma_tensor_Q,
            tma_atom_K, tma_tensor_K,
            tma_atom_G, tma_tensor_G,
            mA_log, mBeta, scale,
            mKscaled_v2, mKg_v2, mQscaled_v2, mGkLast_v2, mAqk, mAkk,
            tiled_copy_qk_k1,
            tiled_mma_k2, tiled_copy_mma_A, tiled_copy_mma_B,
            tiled_mma_inv, tiled_copy_inv_A, tiled_copy_inv_B,
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


def _compare(name, ref, fused, atol=0.5, rtol=0.1):
    """Compare two tensors, masking out NaN/inf from both sides."""
    r, f = ref.float(), fused.float()
    valid = torch.isfinite(r) & torch.isfinite(f)
    n_valid = valid.sum().item()
    n_total = r.numel()
    r_nan = r.isnan().sum().item()
    f_nan = f.isnan().sum().item()

    if n_valid == 0:
        print(f"  [FAIL] {name:<15s}  no valid elements  ref_nan={r_nan}  fused_nan={f_nan}")
        return False

    rv, fv = r[valid], f[valid]
    diff = (rv - fv).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(rv.unsqueeze(0), fv.unsqueeze(0)).item()

    ok = max_diff <= atol or torch.allclose(rv, fv, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    nan_info = ""
    if r_nan > 0 or f_nan > 0:
        nan_info = f"  nan: ref={r_nan} fused={f_nan}"
    print(f"  [{status}] {name:<15s}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
          f"cos_sim={cos:.8f}  valid={n_valid}/{n_total}{nan_info}")
    return ok


def _compile_fused(B, NT, H, q, k, g, A_log, beta, scale,
                   k_scaled, kg, q_scaled, gk_last, A_qk, A_kk,
                   keep_artifacts=False):
    """Compile fused K123 kernel, return (compiled_fn, cute_args)."""
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
        _ct(A_qk, cutlass.BFloat16), _ct(A_kk, cutlass.BFloat16),
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
        sass_file = os.path.join(script_dir, "fuse_kernel123_all.sass")
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
# --verify: Correctness verification (fused K123 vs separate K1→K2→K3)
# =========================================================================
def verify(B=1, H=96, K=128, NT=128):
    cutlass.cuda.initialize_cuda_context()
    T = NT * BT
    scale = 1.0 / (K ** 0.5)

    print("=" * 72)
    print("Correctness: fused K1+K2+K3 vs reference K1 → K2 → K3")
    print("=" * 72)
    print(f"B={B}, T={T}, H={H}, K={K}, NT={NT}, BT={BT}, BC={BC}")
    print()

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)

    # --- Reference K1 → K2 → K3 (no intermediate syncs) ---
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

    A_kk_ref = torch.zeros(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    ref_k3[(NT, B * H)](
        q=q, k=k, g=g_cumsum, beta=beta,
        Aqk=A_qk_ref, Akkd=A_kkd_ref, Akk=A_kk_ref,
        scale=scale, cu_seqlens=None, chunk_indices=None,
        T=T, H=H, K=K, BT=BT, BC=BC, USE_SAFE_GATE=True)
    torch.cuda.synchronize()
    print("      Done.")

    # --- Fused K1+K2+K3 ---
    print("[2/3] Compiling + running fused kernel ...")
    k_scaled_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    kg_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    q_scaled_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    gk_last_f = torch.empty(B, NT, H, K, device="cuda", dtype=torch.float32)
    A_qk_f = torch.empty(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    A_kk_f = torch.empty(B, T, H, BT, device="cuda", dtype=torch.bfloat16)

    compiled, ct_args = _compile_fused(
        B, NT, H, q, k, g, A_log, beta, scale,
        k_scaled_f, kg_f, q_scaled_f, gk_last_f, A_qk_f, A_kk_f)
    compiled(*ct_args)
    torch.cuda.synchronize()
    print("      Done.")

    # --- Compare ---
    print()
    print("[3/3] Comparing outputs:")
    print("  --- K1 outputs ---")
    all_ok = True
    all_ok &= _compare("k_scaled", k_scaled_ref, k_scaled_f, atol=0.01, rtol=0.01)
    all_ok &= _compare("q_scaled", q_scaled_ref, q_scaled_f, atol=0.01, rtol=0.01)
    all_ok &= _compare("kg", kg_ref, kg_f, atol=0.01, rtol=0.01)
    all_ok &= _compare("gk_last_exp", gk_last_ref, gk_last_f, atol=0.01, rtol=0.01)

    print("  --- K2+K3 outputs ---")
    all_ok &= _compare("A_qk", A_qk_ref, A_qk_f, atol=0.5, rtol=0.2)
    all_ok &= _compare("A_kk", A_kk_ref, A_kk_f, atol=1.0, rtol=0.3)

    print()
    if all_ok:
        print("ALL COMPARISONS PASSED")
    else:
        print("SOME COMPARISONS FAILED (check diffs above)")
    print("=" * 72)
    return all_ok


# =========================================================================
# --bench: Performance benchmark (K1+K2+K3 only, follows chunk_fwd.py)
# =========================================================================
def bench(B=1, H=96, K=128, NT=128, num_warmup=20, num_iters=100):
    cutlass.cuda.initialize_cuda_context()
    T = NT * BT
    scale = 1.0 / (K ** 0.5)

    print("=" * 72)
    print("Benchmark: separate K1+K2+K3 vs fused K123")
    print("=" * 72)
    print(f"B={B}, T={T}, H={H}, K={K}, NT={NT}")
    print(f"Warmup={num_warmup}, Iters={num_iters}")
    print()

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)

    # --- Compile reference K2 (CuTe, same as chunk_fwd.py _launch_k2_eqlen) ---
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

    # --- Compile fused K123 ---
    print("Compiling fused K123...")
    k_scaled_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    kg_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    q_scaled_f = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    gk_last_f = torch.empty(B, NT, H, K, device="cuda", dtype=torch.float32)
    A_qk_f = torch.empty(B, T, H, BT, device="cuda", dtype=torch.bfloat16)
    A_kk_f = torch.empty(B, T, H, BT, device="cuda", dtype=torch.bfloat16)

    compiled, ct_args = _compile_fused(
        B, NT, H, q, k, g, A_log, beta, scale,
        k_scaled_f, kg_f, q_scaled_f, gk_last_f, A_qk_f, A_kk_f,
        keep_artifacts=True)

    try:
        _sass_analysis()
    except Exception as e:
        print(f"  SASS analysis failed: {e}")
    print()

    # --- Reference K1→K2→K3 (follows chunk_fwd.py exactly, no K4) ---
    def run_ref():
        # K1: gate activation + cumsum + scaling (same as chunk_fwd line 179)
        g_cumsum, _, _, _, _ = ref_k1(
            g=g.contiguous(), k=k.contiguous(), q=q.contiguous(),
            A_log=A_log.float().contiguous(), cumsum_scale=RCP_LN2, attn_scale=scale)
        # K2: intra sub-chunk attention (same as chunk_fwd _launch_k2_eqlen)
        g_ct_live = _ct_ref(g_cumsum.contiguous(), cutlass.Float32)
        k2_compiled(g_ct_live, q_ct, k_ct, b_ct, ak_ct, aq_ct,
                    float(scale), stream, du, di, 0)
        # K3: inter-chunk solve (same as chunk_fwd line 206)
        ref_k3[(NT, B * H)](
            q=q, k=k, g=g_cumsum, beta=beta,
            Aqk=A_qk_ref, Akkd=A_kkd_ref, Akk=A_kk_ref,
            scale=scale, cu_seqlens=None, chunk_indices=None,
            T=T, H=H, K=K, BT=BT, BC=BC, USE_SAFE_GATE=True)

    def run_fused():
        compiled(*ct_args)

    # --- Warmup ---
    print("Warming up...")
    for _ in range(num_warmup):
        run_ref()
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        run_fused()
    torch.cuda.synchronize()

    # --- Timing (CUDA events) ---
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    for _ in range(num_iters):
        run_ref()
    end_ev.record()
    torch.cuda.synchronize()
    ref_us = start_ev.elapsed_time(end_ev) / num_iters * 1000

    start_ev.record()
    for _ in range(num_iters):
        run_fused()
    end_ev.record()
    torch.cuda.synchronize()
    fused_us = start_ev.elapsed_time(end_ev) / num_iters * 1000

    # --- Report ---
    print()
    print(f"{'Results':^72}")
    print("-" * 72)
    print(f"  Reference (K1+K2+K3):  {ref_us:>8.1f} us")
    print(f"  Fused K123:            {fused_us:>8.1f} us")
    print(f"  Speedup:               {ref_us / fused_us:>8.2f}x")
    print("-" * 72)

    read_bytes = B * T * H * K * 2 * 3 + H * 4 + B * T * H * 2
    write_bytes = (B * T * H * K * 2 * 3
                   + B * NT * H * K * 4
                   + B * T * H * BT * 2 * 2)
    total_bytes = read_bytes + write_bytes
    bw = total_bytes / (fused_us * 1e-6) / 1e9

    print(f"  K123 IO: {total_bytes / 1e6:.1f} MB  BW: {bw:.1f} GB/s "
          f"({bw / B200_PEAK_BW_GBS * 100:.1f}% of {B200_PEAK_BW_GBS} GB/s)")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fused K1+K2+K3 kernel test")
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

