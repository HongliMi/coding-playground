"""
Akk 64×64 Lower Triangular Block Inversion — TF32 MMA, FP32 SMEM.

Architecture:
  - 4 warps (128 threads) per CTA, each CTA processes one 64×64 Akk matrix
  - sAkk [64,64] fp32 with swizzle<2,5,2>: input in UPPER triangle (block-transposed),
    output written to LOWER triangle → no read/write conflicts
  - sTemp [16, 24, 2] fp32: inter-warp communication buffer (C-layout only)
  - C→B layout conversion via warp shuffles (16 shfl + 8 selp, no SMEM)
  - TF32 MMA m16n8k8 for all 16×16 block matmuls, FP32 accumulators
  - Diagonal inversion via anti-diagonal sweep with shuffles

Block layout in sAkk (4×4 sub-blocks of 16×16):
  Upper tri = INPUT, Diagonal = in-place inversion, Lower tri = OUTPUT
         col 0-15    col 16-31   col 32-47   col 48-63
  row 0-15:  [Akk00→Ai00] [Akk10]      [Akk20]      [Akk30]
  row 16-31: [Ai10]        [Akk11→Ai11] [Akk21]      [Akk31]
  row 32-47: [Ai20]        [Ai21]       [Akk22→Ai22] [Akk32]
  row 48-63: [Ai30]        [Ai31]       [Ai32]       [Akk33→Ai33]

Stages:
  1. TMA load fp32 64×64 → sAkk; warps 0-1 invert 4 diagonal blocks
  2. Warps 0-2: Ai10, Ai21, Ai32 via chain MMA (C→A swap)
  3. Warps 0+2 → Ai20, warps 1+3 → Ai31 (parallel pairs, sTemp)
  4. Warps 0+1+2 → Ai30 (sTemp aggregation)
  5. All warps: convert fp32 → bf16, store to global

Inputs:  A_in  [batch, 64, 64] fp32 (block-transposed layout)
Outputs: A_out [batch, 64, 64] bf16
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
def _invert_diag(sAkk: cute.Tensor, diag_offset, lane_id, *, loc=None, ip=None):
    """Anti-diagonal sweep: reads lower-tri from sAkk, writes inverse back."""
    my_row = lane_id % 16
    halfwarp_base = (lane_id // 16) * 16
    off = diag_offset

    rInv = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    rInv[0] = cutlass.Float32(1.0)
    for x in range(1, 16):
        rInv[x] = cutlass.Float32(0.0)

    for d in range(1, 16):
        col_d = my_row - d
        valid = cutlass.Float32(col_d >= 0)
        a_val = cutlass.Float32(sAkk[off + my_row, off + col_d]) * valid
        acc = cutlass.Float32(0.0)
        for j in range(1, d):
            a_re = cutlass.Float32(sAkk[off + my_row, off + my_row - (d - j)])
            inv_shfl = cute.arch.shuffle_sync(rInv[j], halfwarp_base + my_row - d + j)
            acc = acc + a_re * inv_shfl
        rInv[d] = (-a_val - acc) * valid

    rInv[0] = cutlass.Float32(1.0)
    sAkk[off + my_row, off + my_row] = rInv[0]
    for d in range(1, 16):
        sAkk[off + my_row, off + (my_row + 16 - d) % 16] = rInv[d] * cutlass.Float32(my_row >= d)


# ===========================================================================
# Helper: 16×16 matmul  C = sAkk[A_block] @ sAkk[B_block]
# A loaded row-major, B loaded col-major.  Returns 8 C regs (two n-tiles).
# ===========================================================================
@dsl_user_op
def _matmul_AB(
    sAkk: cute.Tensor,
    a_rb, a_cb, b_rb, b_cb,
    lane_id,
    *, loc=None, ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)

    # A k-tile 0
    a0k0 = cutlass.Float32(sAkk[a_rb + gid,     a_cb + 2 * tid])
    a1k0 = cutlass.Float32(sAkk[a_rb + gid + 8, a_cb + 2 * tid])
    a2k0 = cutlass.Float32(sAkk[a_rb + gid,     a_cb + 2 * tid + 1])
    a3k0 = cutlass.Float32(sAkk[a_rb + gid + 8, a_cb + 2 * tid + 1])
    # A k-tile 1
    a0k1 = cutlass.Float32(sAkk[a_rb + gid,     a_cb + 8 + 2 * tid])
    a1k1 = cutlass.Float32(sAkk[a_rb + gid + 8, a_cb + 8 + 2 * tid])
    a2k1 = cutlass.Float32(sAkk[a_rb + gid,     a_cb + 8 + 2 * tid + 1])
    a3k1 = cutlass.Float32(sAkk[a_rb + gid + 8, a_cb + 8 + 2 * tid + 1])

    # B 4 (k,n) sub-tiles
    b0_00 = cutlass.Float32(sAkk[b_rb + 2 * tid,     b_cb + gid])
    b1_00 = cutlass.Float32(sAkk[b_rb + 2 * tid + 1, b_cb + gid])
    b0_01 = cutlass.Float32(sAkk[b_rb + 2 * tid,     b_cb + 8 + gid])
    b1_01 = cutlass.Float32(sAkk[b_rb + 2 * tid + 1, b_cb + 8 + gid])
    b0_10 = cutlass.Float32(sAkk[b_rb + 8 + 2 * tid,     b_cb + gid])
    b1_10 = cutlass.Float32(sAkk[b_rb + 8 + 2 * tid + 1, b_cb + gid])
    b0_11 = cutlass.Float32(sAkk[b_rb + 8 + 2 * tid,     b_cb + 8 + gid])
    b1_11 = cutlass.Float32(sAkk[b_rb + 8 + 2 * tid + 1, b_cb + 8 + gid])

    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_00, b1_00, _z, _z, _z, _z)
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_10, b1_10, cn0_0, cn0_1, cn0_2, cn0_3)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_01, b1_01, _z, _z, _z, _z)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_11, b1_11, cn1_0, cn1_1, cn1_2, cn1_3)

    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# ===========================================================================
# Helper: chain MMA — uses existing C regs as A (C→A swap), loads B from sAkk
# Result = T @ B  where T is passed via (t_n0, t_n1) in C layout.
# ===========================================================================
@dsl_user_op
def _chain_mma_B(
    sAkk: cute.Tensor,
    b_rb, b_cb,
    lane_id,
    t0, t1, t2, t3, t4, t5, t6, t7,
    *, loc=None, ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)

    # C→A swap: a0=c0, a1=c2, a2=c1, a3=c3
    # t_n0 → A_k0,  t_n1 → A_k1
    ak0_0, ak0_1, ak0_2, ak0_3 = t0, t2, t1, t3
    ak1_0, ak1_1, ak1_2, ak1_3 = t4, t6, t5, t7

    b0_00 = cutlass.Float32(sAkk[b_rb + 2 * tid,     b_cb + gid])
    b1_00 = cutlass.Float32(sAkk[b_rb + 2 * tid + 1, b_cb + gid])
    b0_01 = cutlass.Float32(sAkk[b_rb + 2 * tid,     b_cb + 8 + gid])
    b1_01 = cutlass.Float32(sAkk[b_rb + 2 * tid + 1, b_cb + 8 + gid])
    b0_10 = cutlass.Float32(sAkk[b_rb + 8 + 2 * tid,     b_cb + gid])
    b1_10 = cutlass.Float32(sAkk[b_rb + 8 + 2 * tid + 1, b_cb + gid])
    b0_11 = cutlass.Float32(sAkk[b_rb + 8 + 2 * tid,     b_cb + 8 + gid])
    b1_11 = cutlass.Float32(sAkk[b_rb + 8 + 2 * tid + 1, b_cb + 8 + gid])

    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        ak0_0, ak0_1, ak0_2, ak0_3, b0_00, b1_00, _z, _z, _z, _z)
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        ak1_0, ak1_1, ak1_2, ak1_3, b0_10, b1_10, cn0_0, cn0_1, cn0_2, cn0_3)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        ak0_0, ak0_1, ak0_2, ak0_3, b0_01, b1_01, _z, _z, _z, _z)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        ak1_0, ak1_1, ak1_2, ak1_3, b0_11, b1_11, cn1_0, cn1_1, cn1_2, cn1_3)

    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# ===========================================================================
# Helper: store negated C fragment to sAkk (16×16 = two 16×8 n-tiles)
# ===========================================================================
@dsl_user_op
def _store_neg_C(
    sAkk: cute.Tensor, rb, cb,
    c0, c1, c2, c3, c4, c5, c6, c7,
    lane_id,
    *, loc=None, ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    sAkk[rb + gid,     cb + 2 * tid]         = -c0
    sAkk[rb + gid,     cb + 2 * tid + 1]     = -c1
    sAkk[rb + gid + 8, cb + 2 * tid]         = -c2
    sAkk[rb + gid + 8, cb + 2 * tid + 1]     = -c3
    sAkk[rb + gid,     cb + 8 + 2 * tid]     = -c4
    sAkk[rb + gid,     cb + 8 + 2 * tid + 1] = -c5
    sAkk[rb + gid + 8, cb + 8 + 2 * tid]     = -c6
    sAkk[rb + gid + 8, cb + 8 + 2 * tid + 1] = -c7


# ===========================================================================
# sTemp helpers: store C, load C (inter-warp communication only)
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
# Shuffle C-accumulator layout → B-operand layout (warp-level, no SMEM)
#
# C layout (16×16): thread (gid,tid) holds c0=M[gid,2t], c1=M[gid,2t+1],
#   c2=M[gid+8,2t], c3=M[gid+8,2t+1], c4..c7 for cols 8-15.
# B layout (k16×n16): thread needs b[2t,gid], b[2t+1,gid], etc.
#
# Source lanes:  src_a = 8*tid + gid//2  (rows 0-7 of B)
#                src_b = src_a + 4        (rows 1,3,5,7 of B)
# Select:        gid even → c_even reg,  gid odd → c_odd reg
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
# Helper: load A from sAkk + B from shuffled C regs, MMA, return C
# Replaces _mma_A_smem_B_temp — no SMEM roundtrip for B operand.
# ===========================================================================
@dsl_user_op
def _mma_A_smem_B_shfl(
    sAkk: cute.Tensor, a_rb, a_cb,
    c0, c1, c2, c3, c4, c5, c6, c7,
    lane_id,
    *, loc=None, ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)

    a0k0 = cutlass.Float32(sAkk[a_rb + gid,     a_cb + 2 * tid])
    a1k0 = cutlass.Float32(sAkk[a_rb + gid + 8, a_cb + 2 * tid])
    a2k0 = cutlass.Float32(sAkk[a_rb + gid,     a_cb + 2 * tid + 1])
    a3k0 = cutlass.Float32(sAkk[a_rb + gid + 8, a_cb + 2 * tid + 1])
    a0k1 = cutlass.Float32(sAkk[a_rb + gid,     a_cb + 8 + 2 * tid])
    a1k1 = cutlass.Float32(sAkk[a_rb + gid + 8, a_cb + 8 + 2 * tid])
    a2k1 = cutlass.Float32(sAkk[a_rb + gid,     a_cb + 8 + 2 * tid + 1])
    a3k1 = cutlass.Float32(sAkk[a_rb + gid + 8, a_cb + 8 + 2 * tid + 1])

    b0_00, b1_00, b0_10, b1_10, b0_01, b1_01, b0_11, b1_11 = \
        _shuffle_C_to_B(c0, c1, c2, c3, c4, c5, c6, c7, lane_id)

    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_00, b1_00, _z, _z, _z, _z)
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_10, b1_10, cn0_0, cn0_1, cn0_2, cn0_3)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0_01, b1_01, _z, _z, _z, _z)
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0_11, b1_11, cn1_0, cn1_1, cn1_2, cn1_3)

    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# ===========================================================================
# Main kernel
# ===========================================================================
@cute.kernel
def akk_inv_kernel(
    tma_load_atom: cute.CopyAtom,
    tma_load_tensor: cute.Tensor,
    mOut: cute.Tensor,
    akk_smem_layout,
    temp_layout: cute.Layout,
    batch_size: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32
    batch_idx, _, _ = cute.arch.block_idx()

    # ===== SMEM allocation =====
    smem = cutlass.utils.SmemAllocator()
    sAkk = smem.allocate_tensor(
        cutlass.Float32, akk_smem_layout.outer, 128,
        swizzle=akk_smem_layout.inner)
    sTemp = smem.allocate_tensor(cutlass.Float32, temp_layout, 128)
    mbar_ptr = smem.allocate_array(cutlass.Int64, 1)

    tile_bytes = BS * BS * 4

    if tidx == 0:
        cute.arch.mbarrier_init(mbar_ptr, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # ===== Stage 0: TMA Load fp32 64×64 → sAkk =====
    # Global view is (BS, BS, batch); select batch, then single tile
    gA_batch = tma_load_tensor[(None, None, batch_idx)]
    gA_tile = cute.local_tile(gA_batch, (BS, BS), (0, 0))

    if warp_idx == 0:
        if tidx == 0:
            cute.arch.mbarrier_expect_tx(mbar_ptr, tile_bytes)
        ts_ld, tg_ld = cpasync.tma_partition(
            tma_load_atom, 0, cute.make_layout(1),
            cute.group_modes(sAkk, 0, 2),
            cute.group_modes(gA_tile, 0, 2))
        cute.copy(tma_load_atom, tg_ld, ts_ld, tma_bar_ptr=mbar_ptr)
        if tidx == 0:
            cute.arch.mbarrier_arrive(mbar_ptr)

    cute.arch.mbarrier_wait(mbar_ptr, 0)

    # ===== Stage 1: Diagonal block inversion (warps 0-1, half-warps) =====
    # warp 0 lanes 0-15 → block(0,0)=0, lanes 16-31 → block(1,1)=16
    # warp 1 lanes 0-15 → block(2,2)=32, lanes 16-31 → block(3,3)=48
    if warp_idx == 0:
        diag = (lane_id // 16) * SB
        _invert_diag(sAkk, diag, lane_id)
    if warp_idx == 1:
        diag = 32 + (lane_id // 16) * SB
        _invert_diag(sAkk, diag, lane_id)

    cute.arch.barrier()

    # ===== Stage 2: First batch — independent chain MMAs =====
    # Warp 0: Ai10 = -(Ai11 @ Akk10) @ Ai00
    #   A=Ai11(16,16), B=Akk10(0,16), chain B=Ai00(0,0) → out Ai10(16,0)
    if warp_idx == 0:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
            sAkk, 16, 16, 0, 16, lane_id)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
            sAkk, 0, 0, lane_id, t0, t1, t2, t3, t4, t5, t6, t7)
        _store_neg_C(sAkk, 16, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    # Warp 1: Ai21 = -(Ai22 @ Akk21) @ Ai11
    #   A=Ai22(32,32), B=Akk21(16,32), chain B=Ai11(16,16) → out Ai21(32,16)
    if warp_idx == 1:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
            sAkk, 32, 32, 16, 32, lane_id)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
            sAkk, 16, 16, lane_id, t0, t1, t2, t3, t4, t5, t6, t7)
        _store_neg_C(sAkk, 32, 16, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    # Warp 2: Ai32 = -(Ai33 @ Akk32) @ Ai22
    #   A=Ai33(48,48), B=Akk32(32,48), chain B=Ai22(32,32) → out Ai32(48,32)
    if warp_idx == 2:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
            sAkk, 48, 48, 32, 48, lane_id)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
            sAkk, 32, 32, lane_id, t0, t1, t2, t3, t4, t5, t6, t7)
        _store_neg_C(sAkk, 48, 32, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 3: Second batch — warp pairs via sTemp =====
    _z = cutlass.Float32(0.0)
    t0 = _z; t1 = _z; t2 = _z; t3 = _z
    t4 = _z; t5 = _z; t6 = _z; t7 = _z

    # --- Ai20 = -Ai22 @ (Akk20 @ Ai00 + Akk21 @ Ai10) ---
    # Warp 0: T1 = Akk20(0,32) @ Ai00(0,0)
    if warp_idx == 0:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
            sAkk, 0, 32, 0, 0, lane_id)

    # Warp 2: T2 = Akk21(16,32) @ Ai10(16,0) → sTemp[0]
    if warp_idx == 2:
        s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(
            sAkk, 16, 32, 16, 0, lane_id)
        _store_C_temp(sTemp, 0, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)

    # --- Ai31 = -Ai33 @ (Akk31 @ Ai11 + Akk32 @ Ai21) ---
    # Warp 1: T1' = Akk31(16,48) @ Ai11(16,16)
    if warp_idx == 1:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
            sAkk, 16, 48, 16, 16, lane_id)

    # Warp 3: T2' = Akk32(32,48) @ Ai21(32,16) → sTemp[1]
    if warp_idx == 3:
        s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(
            sAkk, 32, 48, 32, 16, lane_id)
        _store_C_temp(sTemp, 1, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)

    cute.arch.barrier()

    # Warp 0: accumulate T3 = T1 + T2, final multiply Ai22 @ T3
    if warp_idx == 0:
        e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 0, lane_id)
        t0 = t0 + e0; t1 = t1 + e1; t2 = t2 + e2; t3 = t3 + e3
        t4 = t4 + e4; t5 = t5 + e5; t6 = t6 + e6; t7 = t7 + e7
        r0, r1, r2, r3, r4, r5, r6, r7 = _mma_A_smem_B_shfl(
            sAkk, 32, 32, t0, t1, t2, t3, t4, t5, t6, t7, lane_id)
        _store_neg_C(sAkk, 32, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    # Warp 1: accumulate T3' = T1' + T2', final multiply Ai33 @ T3'
    if warp_idx == 1:
        e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 1, lane_id)
        t0 = t0 + e0; t1 = t1 + e1; t2 = t2 + e2; t3 = t3 + e3
        t4 = t4 + e4; t5 = t5 + e5; t6 = t6 + e6; t7 = t7 + e7
        r0, r1, r2, r3, r4, r5, r6, r7 = _mma_A_smem_B_shfl(
            sAkk, 48, 48, t0, t1, t2, t3, t4, t5, t6, t7, lane_id)
        _store_neg_C(sAkk, 48, 16, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 4: Third batch — Ai30 via 3 warps =====
    # Ai30 = -Ai33 @ (Akk30 @ Ai00 + Akk31 @ Ai10 + Akk32 @ Ai20)
    t0 = _z; t1 = _z; t2 = _z; t3 = _z
    t4 = _z; t5 = _z; t6 = _z; t7 = _z

    # Warp 0: T1 = Akk30(0,48) @ Ai00(0,0)
    if warp_idx == 0:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
            sAkk, 0, 48, 0, 0, lane_id)

    # Warp 1: T2 = Akk31(16,48) @ Ai10(16,0) → sTemp[0]
    if warp_idx == 1:
        s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(
            sAkk, 16, 48, 16, 0, lane_id)
        _store_C_temp(sTemp, 0, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)

    # Warp 2: T3 = Akk32(32,48) @ Ai20(32,0) → sTemp[1]
    if warp_idx == 2:
        s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(
            sAkk, 32, 48, 32, 0, lane_id)
        _store_C_temp(sTemp, 1, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)

    cute.arch.barrier()

    # Warp 0: accumulate and final multiply
    if warp_idx == 0:
        e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 0, lane_id)
        t0 = t0 + e0; t1 = t1 + e1; t2 = t2 + e2; t3 = t3 + e3
        t4 = t4 + e4; t5 = t5 + e5; t6 = t6 + e6; t7 = t7 + e7
        e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 1, lane_id)
        t0 = t0 + e0; t1 = t1 + e1; t2 = t2 + e2; t3 = t3 + e3
        t4 = t4 + e4; t5 = t5 + e5; t6 = t6 + e6; t7 = t7 + e7
        r0, r1, r2, r3, r4, r5, r6, r7 = _mma_A_smem_B_shfl(
            sAkk, 48, 48, t0, t1, t2, t3, t4, t5, t6, t7, lane_id)
        _store_neg_C(sAkk, 48, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 5: Store sAkk fp32 → bf16 to global =====
    row_start = warp_idx * SB
    for ri in cutlass.range_constexpr(SB):
        row = row_start + ri
        c0 = lane_id * 2
        c1 = lane_id * 2 + 1
        v0 = cutlass.Float32(sAkk[row, c0])
        v1 = cutlass.Float32(sAkk[row, c1])
        mOut[batch_idx, row, c0] = v0.to(cutlass.BFloat16)
        mOut[batch_idx, row, c1] = v1.to(cutlass.BFloat16)


# ===========================================================================
# Host JIT function
# ===========================================================================
@cute.jit
def akk_inv_host(
    A_in: cute.Tensor,
    A_out: cute.Tensor,
    batch_size: cutlass.Constexpr[int],
):
    view_layout = cute.make_layout(
        (BS, BS, batch_size),
        stride=(BS, 1, BS * BS))
    akk_view = cute.make_tensor(A_in.iterator, view_layout)

    sw = cute.make_swizzle(2, 5, 2)
    outer = cute.make_layout((8, 32), stride=(32, 1))
    smem_atom = cute.make_composed_layout(sw, 0, outer)
    akk_smem_2d = cute.tile_to_shape(smem_atom, (BS, BS), order=(0, 1))

    tma_op = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
    tma_load_atom, tma_load_tensor = cpasync.make_tiled_tma_atom(
        tma_op, akk_view, akk_smem_2d,
        cute.product_each(akk_smem_2d.shape), num_multicast=1)

    temp_layout = cute.make_layout(
        (SB, TEMP_COLS, NUM_TEMPS),
        stride=(TEMP_COLS, 1, SB * TEMP_COLS))

    smem_bytes = (BS * BS * 4
                  + SB * TEMP_COLS * NUM_TEMPS * 4
                  + 8 + 256)

    akk_inv_kernel(
        tma_load_atom, tma_load_tensor, A_out,
        akk_smem_2d, temp_layout, batch_size,
    ).launch(
        grid=(batch_size, 1, 1),
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

    BATCH = 96 * 128
    WARMUP = 5
    BENCH = 100

    print("=" * 60)
    print("Akk 64×64 Inverse — TF32 MMA, FP32 SMEM")
    print("=" * 60)
    print(f"  Batch: {BATCH},  Matrix: {BS}×{BS},  Threads: {THREADS}")

    torch.manual_seed(42)

    L = torch.randn(BATCH, BS, BS, device="cuda", dtype=torch.float32) * 0.1
    L = L.tril(-1)
    M = torch.eye(BS, device="cuda", dtype=torch.float32).unsqueeze(0) + L

    M_input = prepare_input(M).contiguous()
    M_out = torch.zeros(BATCH, BS, BS, device="cuda", dtype=torch.bfloat16)

    M_in_ct = from_dlpack(M_input, assumed_align=16)
    M_in_ct.element_type = cutlass.Float32
    M_out_ct = from_dlpack(M_out, assumed_align=16)
    M_out_ct.element_type = cutlass.BFloat16

    print("\nCompiling ...")
    compiled = cute.compile(akk_inv_host, M_in_ct, M_out_ct, BATCH)
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

    out_f = M_out.float()
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

    # --- Benchmark ---
    print(f"\nBenchmark ({BENCH} iters) ...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH):
        compiled(M_in_ct, M_out_ct)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / BENCH
    data_mb = BATCH * BS * BS * (4 + 2) / 1e6
    bw = data_mb / ms * 1e3 / 1e3
    print(f"  Time: {ms:.4f} ms   BW: {bw:.1f} GB/s")
    print("=" * 60)


if __name__ == "__main__":
    test_akk_inv()
