"""
Akk 64×64 Lower Triangular Block Inversion — TF32 MMA, FP32 SMEM.

Non-swizzled padded SMEM layout (stride 72, 8-col padding) for bank-conflict
reduction.  Global→SMEM via cp.async (128-bit vectorised).  All subsequent
SMEM accesses are direct scalar indexing inside dsl_user_op functions so that
loads and MMA compile in the same unit for optimal instruction scheduling.

Architecture:
  - 4 warps (128 threads) per CTA, each CTA processes one 64×64 Akk matrix
  - sAkk [64,64] fp32 with stride 72 (8 padding cols per row)
  - sTemp [16, 24, 2] fp32: inter-warp communication buffer (C-layout only)
  - C→B layout conversion via warp shuffles (16 shfl + 8 selp, no SMEM)
  - TF32 MMA m16n8k8 for all 16×16 block matmuls, FP32 accumulators
  - Diagonal inversion via anti-diagonal sweep with shuffles (in-place on sAkk)

Block layout in sAkk (4×4 sub-blocks of 16×16):
  Upper tri = INPUT, Diagonal = in-place inversion, Lower tri = OUTPUT
         col 0-15    col 16-31   col 32-47   col 48-63
  row 0-15:  [Akk00→Ai00] [Akk10]      [Akk20]      [Akk30]
  row 16-31: [Ai10]        [Akk11→Ai11] [Akk21]      [Akk31]
  row 32-47: [Ai20]        [Ai21]       [Akk22→Ai22] [Akk32]
  row 48-63: [Ai30]        [Ai31]       [Ai32]       [Akk33→Ai33]

Stages:
  0. cp.async load fp32 64×64 → sAkk
  1. Invert 4 diagonal blocks in-place on sAkk
  2. Warps 0-2: Ai10, Ai21, Ai32 via chain MMA (C→A swap)
  3. Warps 0+2 → Ai20, warps 1+3 → Ai31 (parallel pairs, sTemp)
  4. Warps 0+1+2 → Ai30 (sTemp aggregation)
  5. All warps: load fp32 → bf16, store to global

Inputs:  A_in  [B, T, H, BT] fp32 (block-transposed layout, T = NT * BT)
Outputs: A_out [B, T, H, BT] bf16
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
import torch

# ===========================================================================
# Constants
# ===========================================================================
BS = 64
SB = 16
THREADS = 128
TEMP_PAD = 8
TEMP_COLS = SB + TEMP_PAD   # 24
NUM_TEMPS = 2
AKK_PAD = 8
AKK_STRIDE = BS + AKK_PAD   # 72


# ===========================================================================
# TF32 MMA m16n8k8 inline PTX
# ===========================================================================
@dsl_user_op
def mma_tf32_m16n8k8(
    a0, a1, a2, a3, b0, b1, c0, c1, c2, c3,
    *, loc=None, ip=None,
):
    a0b = llvm.bitcast(T.i32(), a0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a1b = llvm.bitcast(T.i32(), a1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a2b = llvm.bitcast(T.i32(), a2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a3b = llvm.bitcast(T.i32(), a3.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b0b = llvm.bitcast(T.i32(), b0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b1b = llvm.bitcast(T.i32(), b1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [a0b, a1b, a2b, a3b, b0b, b1b,
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
# Diagonal 16×16 unit-lower-triangular inversion (half-warp, in-place)
# ===========================================================================
@dsl_user_op
def _invert_diag(sAkk: cute.Tensor, block_rc, lane_id, *, loc=None, ip=None):
    my_row = lane_id % 16
    halfwarp_base = (lane_id // 16) * 16
    r_off = block_rc * 16
    c_off = block_rc * 16

    rInv = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    rInv[0] = cutlass.Float32(1.0)
    for x in range(1, 16):
        rInv[x] = cutlass.Float32(0.0)

    for d in range(1, 16):
        col_d = my_row - d
        valid = cutlass.Float32(col_d >= 0)
        a_val = cutlass.Float32(sAkk[r_off + my_row, c_off + col_d]) * valid
        acc = cutlass.Float32(0.0)
        for j in range(1, d):
            a_re = cutlass.Float32(sAkk[r_off + my_row, c_off + my_row - (d - j)])
            inv_shfl = cute.arch.shuffle_sync(rInv[j], halfwarp_base + my_row - d + j)
            acc = acc + a_re * inv_shfl
        rInv[d] = (-a_val - acc) * valid

    rInv[0] = cutlass.Float32(1.0)
    sAkk[r_off + my_row, c_off + my_row] = rInv[0]
    for d in range(1, 16):
        sAkk[r_off + my_row, c_off + (my_row + 16 - d) % 16] = rInv[d] * cutlass.Float32(my_row >= d)


# ===========================================================================
# 16×16 matmul: load A & B from sAkk, MMA, return C registers
# ===========================================================================
@dsl_user_op
def _matmul_AB(sAkk: cute.Tensor, br_A, bc_A, br_B, bc_B, lane_id,
               *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 16
    rB = br_B * 16
    cB = bc_B * 16

    a0k0 = cutlass.Float32(sAkk[rA + gid,     cA + 2 * tid])
    a1k0 = cutlass.Float32(sAkk[rA + gid + 8, cA + 2 * tid])
    a2k0 = cutlass.Float32(sAkk[rA + gid,     cA + 2 * tid + 1])
    a3k0 = cutlass.Float32(sAkk[rA + gid + 8, cA + 2 * tid + 1])
    a0k1 = cutlass.Float32(sAkk[rA + gid,     cA + 8 + 2 * tid])
    a1k1 = cutlass.Float32(sAkk[rA + gid + 8, cA + 8 + 2 * tid])
    a2k1 = cutlass.Float32(sAkk[rA + gid,     cA + 8 + 2 * tid + 1])
    a3k1 = cutlass.Float32(sAkk[rA + gid + 8, cA + 8 + 2 * tid + 1])

    b0_k0n0 = cutlass.Float32(sAkk[rB + 2 * tid,     cB + gid])
    b1_k0n0 = cutlass.Float32(sAkk[rB + 2 * tid + 1, cB + gid])
    b0_k0n1 = cutlass.Float32(sAkk[rB + 2 * tid,     cB + 8 + gid])
    b1_k0n1 = cutlass.Float32(sAkk[rB + 2 * tid + 1, cB + 8 + gid])
    b0_k1n0 = cutlass.Float32(sAkk[rB + 8 + 2 * tid,     cB + gid])
    b1_k1n0 = cutlass.Float32(sAkk[rB + 8 + 2 * tid + 1, cB + gid])
    b0_k1n1 = cutlass.Float32(sAkk[rB + 8 + 2 * tid,     cB + 8 + gid])
    b1_k1n1 = cutlass.Float32(sAkk[rB + 8 + 2 * tid + 1, cB + 8 + gid])

    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_k0n0, b1_k0n0, _z, _z, _z, _z)
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_k1n0, b1_k1n0, cn0_0, cn0_1, cn0_2, cn0_3)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_k0n1, b1_k0n1, _z, _z, _z, _z)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_k1n1, b1_k1n1, cn1_0, cn1_1, cn1_2, cn1_3)

    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# ===========================================================================
# Chain MMA: pre-loaded A (from C→A swap), load B from sAkk  (Stage 2)
# ===========================================================================
@dsl_user_op
def _chain_mma_B(sAkk: cute.Tensor, br_B, bc_B,
                 a0k0, a1k0, a2k0, a3k0, a0k1, a1k1, a2k1, a3k1,
                 lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)
    rB = br_B * 16
    cB = bc_B * 16

    b0_k0n0 = cutlass.Float32(sAkk[rB + 2 * tid,     cB + gid])
    b1_k0n0 = cutlass.Float32(sAkk[rB + 2 * tid + 1, cB + gid])
    b0_k0n1 = cutlass.Float32(sAkk[rB + 2 * tid,     cB + 8 + gid])
    b1_k0n1 = cutlass.Float32(sAkk[rB + 2 * tid + 1, cB + 8 + gid])
    b0_k1n0 = cutlass.Float32(sAkk[rB + 8 + 2 * tid,     cB + gid])
    b1_k1n0 = cutlass.Float32(sAkk[rB + 8 + 2 * tid + 1, cB + gid])
    b0_k1n1 = cutlass.Float32(sAkk[rB + 8 + 2 * tid,     cB + 8 + gid])
    b1_k1n1 = cutlass.Float32(sAkk[rB + 8 + 2 * tid + 1, cB + 8 + gid])

    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_k0n0, b1_k0n0, _z, _z, _z, _z)
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_k1n0, b1_k1n0, cn0_0, cn0_1, cn0_2, cn0_3)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_k0n1, b1_k0n1, _z, _z, _z, _z)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_k1n1, b1_k1n1, cn1_0, cn1_1, cn1_2, cn1_3)

    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# ===========================================================================
# Chain MMA: load A from sAkk, pre-loaded B (from C→B shuffle)  (Stages 3-4)
# ===========================================================================
@dsl_user_op
def _chain_mma_A(sAkk: cute.Tensor, br_A, bc_A,
                 b0_k0n0, b1_k0n0, b0_k0n1, b1_k0n1,
                 b0_k1n0, b1_k1n0, b0_k1n1, b1_k1n1,
                 lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 16

    a0k0 = cutlass.Float32(sAkk[rA + gid,     cA + 2 * tid])
    a1k0 = cutlass.Float32(sAkk[rA + gid + 8, cA + 2 * tid])
    a2k0 = cutlass.Float32(sAkk[rA + gid,     cA + 2 * tid + 1])
    a3k0 = cutlass.Float32(sAkk[rA + gid + 8, cA + 2 * tid + 1])
    a0k1 = cutlass.Float32(sAkk[rA + gid,     cA + 8 + 2 * tid])
    a1k1 = cutlass.Float32(sAkk[rA + gid + 8, cA + 8 + 2 * tid])
    a2k1 = cutlass.Float32(sAkk[rA + gid,     cA + 8 + 2 * tid + 1])
    a3k1 = cutlass.Float32(sAkk[rA + gid + 8, cA + 8 + 2 * tid + 1])

    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_k0n0, b1_k0n0, _z, _z, _z, _z)
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_k1n0, b1_k1n0, cn0_0, cn0_1, cn0_2, cn0_3)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_k0n1, b1_k0n1, _z, _z, _z, _z)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_k1n1, b1_k1n1, cn1_0, cn1_1, cn1_2, cn1_3)

    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# ===========================================================================
# Store negated C result (16×16) to sAkk
# ===========================================================================
@dsl_user_op
def _store_neg_C(sAkk: cute.Tensor, br, bc,
                 c0, c1, c2, c3, c4, c5, c6, c7,
                 lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    r = br * 16
    c = bc * 16
    sAkk[r + gid,     c + 2 * tid]         = -c0
    sAkk[r + gid,     c + 2 * tid + 1]     = -c1
    sAkk[r + gid + 8, c + 2 * tid]         = -c2
    sAkk[r + gid + 8, c + 2 * tid + 1]     = -c3
    sAkk[r + gid,     c + 8 + 2 * tid]     = -c4
    sAkk[r + gid,     c + 8 + 2 * tid + 1] = -c5
    sAkk[r + gid + 8, c + 8 + 2 * tid]     = -c6
    sAkk[r + gid + 8, c + 8 + 2 * tid + 1] = -c7


# ===========================================================================
# Shuffle C-accumulator layout → B-operand layout (warp-level, no SMEM)
# ===========================================================================
@dsl_user_op
def _shuffle_C_to_B(
    c0, c1, c2, c3, c4, c5, c6, c7,
    lane_id,
    *, loc=None, ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    src_a = 8 * tid + gid // 2
    src_b = src_a + 4
    f_odd = cutlass.Float32(gid % 2)
    f_even = cutlass.Float32(1) - f_odd

    c0_a = cute.arch.shuffle_sync(c0, src_a)
    c1_a = cute.arch.shuffle_sync(c1, src_a)
    c2_a = cute.arch.shuffle_sync(c2, src_a)
    c3_a = cute.arch.shuffle_sync(c3, src_a)
    c4_a = cute.arch.shuffle_sync(c4, src_a)
    c5_a = cute.arch.shuffle_sync(c5, src_a)
    c6_a = cute.arch.shuffle_sync(c6, src_a)
    c7_a = cute.arch.shuffle_sync(c7, src_a)

    c0_b = cute.arch.shuffle_sync(c0, src_b)
    c1_b = cute.arch.shuffle_sync(c1, src_b)
    c2_b = cute.arch.shuffle_sync(c2, src_b)
    c3_b = cute.arch.shuffle_sync(c3, src_b)
    c4_b = cute.arch.shuffle_sync(c4, src_b)
    c5_b = cute.arch.shuffle_sync(c5, src_b)
    c6_b = cute.arch.shuffle_sync(c6, src_b)
    c7_b = cute.arch.shuffle_sync(c7, src_b)

    b0_00 = c0_a * f_even + c1_a * f_odd
    b1_00 = c0_b * f_even + c1_b * f_odd
    b0_10 = c2_a * f_even + c3_a * f_odd
    b1_10 = c2_b * f_even + c3_b * f_odd
    b0_01 = c4_a * f_even + c5_a * f_odd
    b1_01 = c4_b * f_even + c5_b * f_odd
    b0_11 = c6_a * f_even + c7_a * f_odd
    b1_11 = c6_b * f_even + c7_b * f_odd

    return b0_00, b1_00, b0_10, b1_10, b0_01, b1_01, b0_11, b1_11


# ===========================================================================
# sTemp helpers (non-swizzled, direct indexing OK)
# ===========================================================================
@dsl_user_op
def _store_C_temp(
    sT: cute.Tensor, buf,
    c0, c1, c2, c3, c4, c5, c6, c7,
    lane_id,
    *, loc=None, ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    sT[gid,     2 * tid,         buf] = c0
    sT[gid,     2 * tid + 1,     buf] = c1
    sT[gid + 8, 2 * tid,         buf] = c2
    sT[gid + 8, 2 * tid + 1,     buf] = c3
    sT[gid,     8 + 2 * tid,     buf] = c4
    sT[gid,     8 + 2 * tid + 1, buf] = c5
    sT[gid + 8, 8 + 2 * tid,     buf] = c6
    sT[gid + 8, 8 + 2 * tid + 1, buf] = c7


@dsl_user_op
def _load_C_temp(sT: cute.Tensor, buf, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    c0 = cutlass.Float32(sT[gid,     2 * tid,         buf])
    c1 = cutlass.Float32(sT[gid,     2 * tid + 1,     buf])
    c2 = cutlass.Float32(sT[gid + 8, 2 * tid,         buf])
    c3 = cutlass.Float32(sT[gid + 8, 2 * tid + 1,     buf])
    c4 = cutlass.Float32(sT[gid,     8 + 2 * tid,     buf])
    c5 = cutlass.Float32(sT[gid,     8 + 2 * tid + 1, buf])
    c6 = cutlass.Float32(sT[gid + 8, 8 + 2 * tid,     buf])
    c7 = cutlass.Float32(sT[gid + 8, 8 + 2 * tid + 1, buf])
    return c0, c1, c2, c3, c4, c5, c6, c7


# ===========================================================================
# Main kernel
# ===========================================================================
@cute.kernel
def akk_inv_kernel(
    g2s_copy: cute.TiledCopy,
    gA_tensor: cute.Tensor,
    mOut: cute.Tensor,
    akk_smem_layout: cute.Layout,
    temp_layout: cute.Layout,
    NT: int,
    H: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32
    h_idx, nt_idx, b_idx = cute.arch.block_idx()

    # ===== SMEM allocation =====
    smem = cutlass.utils.SmemAllocator()
    sAkk = smem.allocate_tensor(cutlass.Float32, akk_smem_layout, 128)
    sTemp = smem.allocate_tensor(cutlass.Float32, temp_layout, 128)

    # ===== Stage 0: cp.async load fp32 64×64 → sAkk =====
    gA_batch = gA_tensor[(None, None, h_idx, nt_idx, b_idx)]

    thr_g2s = g2s_copy.get_slice(tidx)
    thr_gSrc = thr_g2s.partition_S(gA_batch)
    thr_sDst = thr_g2s.partition_D(sAkk)
    cute.copy(g2s_copy, thr_gSrc, thr_sDst)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # ===== Stage 1: Diagonal block inversion (in-place on sAkk) =====
    if warp_idx == 0:
        _invert_diag(sAkk, lane_id // 16, lane_id)
    if warp_idx == 1:
        _invert_diag(sAkk, 2 + lane_id // 16, lane_id)

    cute.arch.barrier()

    # ===== Stage 2: First batch — Ai10, Ai21, Ai32 =====
    # Warp 0: Ai10 = -(Ai11 @ Akk10) @ Ai00
    if warp_idx == 0:
        t0,t1,t2,t3,t4,t5,t6,t7 = _matmul_AB(sAkk, 1, 1, 0, 1, lane_id)
        a0k0,a1k0,a2k0,a3k0 = t0,t2,t1,t3
        a0k1,a1k1,a2k1,a3k1 = t4,t6,t5,t7
        r0,r1,r2,r3,r4,r5,r6,r7 = _chain_mma_B(
            sAkk, 0, 0,
            a0k0,a1k0,a2k0,a3k0, a0k1,a1k1,a2k1,a3k1, lane_id)
        _store_neg_C(sAkk, 1, 0, r0,r1,r2,r3,r4,r5,r6,r7, lane_id)

    # Warp 1: Ai21 = -(Ai22 @ Akk21) @ Ai11
    if warp_idx == 1:
        t0,t1,t2,t3,t4,t5,t6,t7 = _matmul_AB(sAkk, 2, 2, 1, 2, lane_id)
        a0k0,a1k0,a2k0,a3k0 = t0,t2,t1,t3
        a0k1,a1k1,a2k1,a3k1 = t4,t6,t5,t7
        r0,r1,r2,r3,r4,r5,r6,r7 = _chain_mma_B(
            sAkk, 1, 1,
            a0k0,a1k0,a2k0,a3k0, a0k1,a1k1,a2k1,a3k1, lane_id)
        _store_neg_C(sAkk, 2, 1, r0,r1,r2,r3,r4,r5,r6,r7, lane_id)

    # Warp 2: Ai32 = -(Ai33 @ Akk32) @ Ai22
    if warp_idx == 2:
        t0,t1,t2,t3,t4,t5,t6,t7 = _matmul_AB(sAkk, 3, 3, 2, 3, lane_id)
        a0k0,a1k0,a2k0,a3k0 = t0,t2,t1,t3
        a0k1,a1k1,a2k1,a3k1 = t4,t6,t5,t7
        r0,r1,r2,r3,r4,r5,r6,r7 = _chain_mma_B(
            sAkk, 2, 2,
            a0k0,a1k0,a2k0,a3k0, a0k1,a1k1,a2k1,a3k1, lane_id)
        _store_neg_C(sAkk, 3, 2, r0,r1,r2,r3,r4,r5,r6,r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 3: Second batch — Ai20, Ai31 (warp pairs via sTemp) =====
    _z = cutlass.Float32(0.0)
    t0=_z; t1=_z; t2=_z; t3=_z; t4=_z; t5=_z; t6=_z; t7=_z

    # --- Ai20 = -Ai22 @ (Akk20 @ Ai00 + Akk21 @ Ai10) ---
    if warp_idx == 0:
        t0,t1,t2,t3,t4,t5,t6,t7 = _matmul_AB(sAkk, 0, 2, 0, 0, lane_id)

    if warp_idx == 2:
        s0,s1,s2,s3,s4,s5,s6,s7 = _matmul_AB(sAkk, 1, 2, 1, 0, lane_id)
        _store_C_temp(sTemp, 0, s0,s1,s2,s3,s4,s5,s6,s7, lane_id)

    # --- Ai31 = -Ai33 @ (Akk31 @ Ai11 + Akk32 @ Ai21) ---
    if warp_idx == 1:
        t0,t1,t2,t3,t4,t5,t6,t7 = _matmul_AB(sAkk, 1, 3, 1, 1, lane_id)

    if warp_idx == 3:
        s0,s1,s2,s3,s4,s5,s6,s7 = _matmul_AB(sAkk, 2, 3, 2, 1, lane_id)
        _store_C_temp(sTemp, 1, s0,s1,s2,s3,s4,s5,s6,s7, lane_id)

    cute.arch.barrier()

    # Warp 0: accumulate T1+T2, shuffle→B, multiply by Ai22
    if warp_idx == 0:
        e0,e1,e2,e3,e4,e5,e6,e7 = _load_C_temp(sTemp, 0, lane_id)
        t0=t0+e0; t1=t1+e1; t2=t2+e2; t3=t3+e3
        t4=t4+e4; t5=t5+e5; t6=t6+e6; t7=t7+e7
        sb = _shuffle_C_to_B(t0,t1,t2,t3,t4,t5,t6,t7, lane_id)
        r0,r1,r2,r3,r4,r5,r6,r7 = _chain_mma_A(
            sAkk, 2, 2,
            sb[0],sb[1], sb[4],sb[5], sb[2],sb[3], sb[6],sb[7], lane_id)
        _store_neg_C(sAkk, 2, 0, r0,r1,r2,r3,r4,r5,r6,r7, lane_id)

    # Warp 1: accumulate T1'+T2', shuffle→B, multiply by Ai33
    if warp_idx == 1:
        e0,e1,e2,e3,e4,e5,e6,e7 = _load_C_temp(sTemp, 1, lane_id)
        t0=t0+e0; t1=t1+e1; t2=t2+e2; t3=t3+e3
        t4=t4+e4; t5=t5+e5; t6=t6+e6; t7=t7+e7
        sb = _shuffle_C_to_B(t0,t1,t2,t3,t4,t5,t6,t7, lane_id)
        r0,r1,r2,r3,r4,r5,r6,r7 = _chain_mma_A(
            sAkk, 3, 3,
            sb[0],sb[1], sb[4],sb[5], sb[2],sb[3], sb[6],sb[7], lane_id)
        _store_neg_C(sAkk, 3, 1, r0,r1,r2,r3,r4,r5,r6,r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 4: Third batch — Ai30 =====
    t0=_z; t1=_z; t2=_z; t3=_z; t4=_z; t5=_z; t6=_z; t7=_z

    if warp_idx == 0:
        t0,t1,t2,t3,t4,t5,t6,t7 = _matmul_AB(sAkk, 0, 3, 0, 0, lane_id)

    if warp_idx == 1:
        s0,s1,s2,s3,s4,s5,s6,s7 = _matmul_AB(sAkk, 1, 3, 1, 0, lane_id)
        _store_C_temp(sTemp, 0, s0,s1,s2,s3,s4,s5,s6,s7, lane_id)

    if warp_idx == 2:
        s0,s1,s2,s3,s4,s5,s6,s7 = _matmul_AB(sAkk, 2, 3, 2, 0, lane_id)
        _store_C_temp(sTemp, 1, s0,s1,s2,s3,s4,s5,s6,s7, lane_id)

    cute.arch.barrier()

    # Warp 0: accumulate all three, shuffle→B, multiply by Ai33
    if warp_idx == 0:
        e0,e1,e2,e3,e4,e5,e6,e7 = _load_C_temp(sTemp, 0, lane_id)
        t0=t0+e0; t1=t1+e1; t2=t2+e2; t3=t3+e3
        t4=t4+e4; t5=t5+e5; t6=t6+e6; t7=t7+e7
        e0,e1,e2,e3,e4,e5,e6,e7 = _load_C_temp(sTemp, 1, lane_id)
        t0=t0+e0; t1=t1+e1; t2=t2+e2; t3=t3+e3
        t4=t4+e4; t5=t5+e5; t6=t6+e6; t7=t7+e7
        sb = _shuffle_C_to_B(t0,t1,t2,t3,t4,t5,t6,t7, lane_id)
        r0,r1,r2,r3,r4,r5,r6,r7 = _chain_mma_A(
            sAkk, 3, 3,
            sb[0],sb[1], sb[4],sb[5], sb[2],sb[3], sb[6],sb[7], lane_id)
        _store_neg_C(sAkk, 3, 0, r0,r1,r2,r3,r4,r5,r6,r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 5: Store sAkk fp32 → bf16 to global [B, T, H, BT] =====
    row_start = warp_idx * SB

    for ri in cutlass.range_constexpr(SB):
        row = row_start + ri
        c0 = lane_id * 2
        c1 = lane_id * 2 + 1
        v0 = cutlass.Float32(sAkk[row, c0]) * cutlass.Float32(row >= c0)
        v1 = cutlass.Float32(sAkk[row, c1]) * cutlass.Float32(row >= c1)
        t_row = nt_idx * BS + row
        mOut[b_idx, t_row, h_idx, c0] = v0.to(cutlass.BFloat16)
        mOut[b_idx, t_row, h_idx, c1] = v1.to(cutlass.BFloat16)


# ===========================================================================
# Host JIT function
# ===========================================================================
@cute.jit
def akk_inv_host(
    A_in: cute.Tensor,
    A_out: cute.Tensor,
    B: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
):
    T_val = NT * BS
    view_layout = cute.make_layout(
        (BS, BS, H, NT, B),
        stride=(H * BS, 1, BS, BS * H * BS, T_val * H * BS))
    gA_view = cute.make_tensor(A_in.iterator, view_layout)

    # sAkk: non-swizzled padded layout (stride 72)
    akk_smem_2d = cute.make_layout((BS, BS), stride=(AKK_STRIDE, 1))

    # cp.async G→S copy: 128-bit (4×fp32) vectorised
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    g2s_copy = cute.make_tiled_copy_tv(
        copy_atom,
        thr_layout=cute.make_layout((16, 8), stride=(8, 1)),
        val_layout=cute.make_layout((1, 4)),
    )

    # sTemp layout (non-swizzled)
    temp_layout = cute.make_layout(
        (SB, TEMP_COLS, NUM_TEMPS),
        stride=(TEMP_COLS, 1, SB * TEMP_COLS))

    smem_bytes = (BS * AKK_STRIDE * 4
                  + SB * TEMP_COLS * NUM_TEMPS * 4
                  + 256)

    akk_inv_kernel(
        g2s_copy, gA_view, A_out,
        akk_smem_2d, temp_layout,
        NT, H,
    ).launch(
        grid=(H, NT, B),
        block=(THREADS, 1, 1),
        smem=smem_bytes,
    )


# ===========================================================================
# Input preparation: block-transpose lower triangle to upper triangle
# ===========================================================================
def prepare_input(M):
    """Take unit lower-triangular M [batch,64,64] and produce the
    block-transposed layout expected by the kernel."""
    B = M.shape[0]
    M_in = torch.zeros_like(M)
    for i in range(4):
        r0, r1 = i * SB, (i + 1) * SB
        M_in[:, r0:r1, r0:r1] = M[:, r0:r1, r0:r1]
    for i in range(4):
        for j in range(i):
            ir0, ir1 = i * SB, (i + 1) * SB
            jr0, jr1 = j * SB, (j + 1) * SB
            M_in[:, jr0:jr1, ir0:ir1] = M[:, ir0:ir1, jr0:jr1]
    return M_in


# ===========================================================================
# Test
# ===========================================================================
def test_akk_inv():
    cutlass.cuda.initialize_cuda_context()

    B_TEST = 1
    H_TEST = 96
    NT_TEST = 128
    T_TEST = NT_TEST * BS
    inv_batch = B_TEST * NT_TEST * H_TEST
    WARMUP = 5
    BENCH = 100

    print("=" * 60)
    print("Akk 64x64 Inverse — TF32 MMA, FP32 SMEM (padded+cp.async)")
    print("=" * 60)
    print(f"  B={B_TEST}, NT={NT_TEST}, H={H_TEST}, T={T_TEST}")
    print(f"  inv_batch={inv_batch},  Matrix: {BS}x{BS},  Threads: {THREADS}")
    print(f"  AKK_STRIDE={AKK_STRIDE} (pad={AKK_PAD})")

    torch.manual_seed(42)

    L = torch.randn(inv_batch, BS, BS, device="cuda", dtype=torch.float32) * 0.1
    L = L.tril(-1)
    M = torch.eye(BS, device="cuda", dtype=torch.float32).unsqueeze(0) + L

    M_bt = prepare_input(M)

    M_input = (M_bt
               .reshape(B_TEST, NT_TEST, H_TEST, BS, BS)
               .permute(0, 1, 3, 2, 4)
               .contiguous()
               .reshape(B_TEST, T_TEST, H_TEST, BS))

    M_out = torch.zeros(B_TEST, T_TEST, H_TEST, BS, device="cuda", dtype=torch.bfloat16)

    M_in_ct = from_dlpack(M_input, assumed_align=16)
    M_in_ct.element_type = cutlass.Float32
    M_out_ct = from_dlpack(M_out, assumed_align=16)
    M_out_ct.element_type = cutlass.BFloat16

    print("\nCompiling ...")
    compiled = cute.compile(akk_inv_host, M_in_ct, M_out_ct, B_TEST, NT_TEST, H_TEST)
    torch.cuda.synchronize()
    print("Done.")

    print(f"\nWarmup ({WARMUP} iters) ...")
    for _ in range(WARMUP):
        compiled(M_in_ct, M_out_ct)
    torch.cuda.synchronize()

    compiled(M_in_ct, M_out_ct)
    torch.cuda.synchronize()

    # --- Correctness ---
    M_inv_ref = torch.linalg.inv(M)
    mask = torch.tril(torch.ones(BS, BS, device="cuda", dtype=torch.bool))

    out_3d = (M_out
              .reshape(B_TEST, NT_TEST, BS, H_TEST, BS)
              .permute(0, 1, 3, 2, 4)
              .contiguous()
              .reshape(inv_batch, BS, BS))
    out_f = out_3d.float()
    ref_f = M_inv_ref.float()
    diff = (out_f - ref_f).abs()
    diff_lower = diff[:, mask]

    n = diff_lower.numel()
    n_1e2 = (diff_lower < 1e-2).sum().item()
    n_1e3 = (diff_lower < 1e-3).sum().item()
    max_d = diff_lower.max().item()
    mean_d = diff_lower.mean().item()

    print(f"\nCorrectness (lower triangle):")
    print(f"  max={max_d:.6f}  mean={mean_d:.6f}")
    print(f"  |diff|<1e-2: {n_1e2 / n * 100:.3f}%  |diff|<1e-3: {n_1e3 / n * 100:.3f}%")
    print(f"  nan: out={torch.isnan(out_f).sum().item()} ref={torch.isnan(ref_f).sum().item()}")

    for i in range(4):
        for j in range(i + 1):
            blk = diff[:, i * SB:(i + 1) * SB, j * SB:(j + 1) * SB]
            print(f"  Block({i},{j}): max={blk.max().item():.6e}")

    # --- Benchmark (warm L2) ---
    print(f"\nBenchmark warm L2 ({BENCH} iters) ...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH):
        compiled(M_in_ct, M_out_ct)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / BENCH
    data_mb = inv_batch * BS * BS * (4 + 2) / 1e6
    bw = data_mb / ms * 1e3 / 1e3
    print(f"  Time: {ms:.4f} ms   BW: {bw:.1f} GB/s")

    # --- Benchmark (cold L2) ---
    l2_flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.int8)
    print(f"\nBenchmark cold L2 ({BENCH} iters) ...")
    start.record()
    for _ in range(BENCH):
        l2_flush.fill_(0)
        compiled(M_in_ct, M_out_ct)
    end.record()
    torch.cuda.synchronize()
    ms_cold = start.elapsed_time(end) / BENCH
    data_mb = inv_batch * BS * BS * (4 + 2) / 1e6
    bw_cold = data_mb / ms_cold * 1e3 / 1e3
    print(f"  Time: {ms_cold:.4f} ms   BW: {bw_cold:.1f} GB/s")
    print(f"\n  Delta (cold - warm): {(ms_cold - ms) * 1000:.1f} us")
    print("=" * 60)


if __name__ == "__main__":
    test_akk_inv()
