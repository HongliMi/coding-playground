# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
TF32 MMA 16x16 with CuTe ldmatrix/stmatrix API.

Algorithm:
1. Global → SMEM: Load A (direct), Load B (transpose to col-major)
2. K=0..7: ldmatrix A[:, 0:8], ldmatrix B tiles, TF32 MMA #1 #2
3. K=8..15: ldmatrix A[:, 8:16], ldmatrix B tiles, TF32 MMA #3 #4
4. Store: stmatrix C to SMEM
5. SMEM → Global
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


THREADS_PER_CTA = 32  # 1 warp


# ===========================================================================
# TF32 MMA Inline PTX (from mma_tf32_16x16.py)
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
# Main Kernel
# ===========================================================================
@cute.kernel
def tf32_gemm_kernel(
    mA: cute.Tensor,   # (16, 16) FP16 global
    mB: cute.Tensor,   # (16, 16) FP16 global  
    mC: cute.Tensor,   # (16, 16) FP16 global (output)
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
    sC_layout: cute.Layout,
    tiled_mma: cute.TiledMma,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_copy_C: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    
    # Allocate shared memory
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float16, sA_layout, 16)
    sB = smem.allocate_tensor(cutlass.Float16, sB_layout, 16)
    sC = smem.allocate_tensor(cutlass.Float16, sC_layout, 16)
    
    # =========================================================================
    # Step 1: Global → SMEM
    # A: direct copy, B: transpose to col-major
    # =========================================================================
    elems_per_thread = 16 * 16 // 32
    for i in range(elems_per_thread):
        idx = tidx * elems_per_thread + i
        row = idx // 16
        col = idx % 16
        sA[row, col] = mA[row, col]
        sB[col, row] = mB[row, col]  # Transpose B
    
    cute.arch.sync_threads()
    
    # =========================================================================
    # Step 2: Setup partitions
    # =========================================================================
    thr_mma = tiled_mma.get_slice(tidx)
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)
    
    # Initialize accumulators (FP32)
    c0_0, c0_1, c0_2, c0_3 = cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0)
    c1_0, c1_1, c1_2, c1_3 = cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0)
    
    # =========================================================================
    # Step 3: K=0..7 iteration
    # =========================================================================
    # Load A[:, 0:8] using ldmatrix.x2
    sA_k0 = cute.local_tile(sA, tiler=(16, 8), coord=(0, 0))
    tCsA_k0 = thr_mma.partition_A(sA_k0)
    tCrA_k0 = tiled_mma.make_fragment_A(tCsA_k0)
    tCsA_k0_view = thr_copy_A.partition_S(sA_k0)
    tCrA_k0_view = thr_copy_A.retile(tCrA_k0)
    cute.copy(tiled_copy_A, tCsA_k0_view, tCrA_k0_view)
    
    # ldmatrix [0,1,2,3] -> TF32 MMA [0,2,1,3]
    a0_k0 = tCrA_k0[0].to(cutlass.Float32)
    a1_k0 = tCrA_k0[2].to(cutlass.Float32)
    a2_k0 = tCrA_k0[1].to(cutlass.Float32)
    a3_k0 = tCrA_k0[3].to(cutlass.Float32)
    
    # Load B[0:8, 0:8] using ldmatrix.x1 (col-major in sB)
    sB_00 = cute.local_tile(sB, tiler=(8, 8), coord=(0, 0))
    tCsB_00 = thr_mma.partition_B(sB_00)
    tCrB_00 = tiled_mma.make_fragment_B(tCsB_00)
    tCsB_00_view = thr_copy_B.partition_S(sB_00)
    tCrB_00_view = thr_copy_B.retile(tCrB_00)
    cute.copy(tiled_copy_B, tCsB_00_view, tCrB_00_view)
    
    b0_00 = tCrB_00[0].to(cutlass.Float32)
    b1_00 = tCrB_00[1].to(cutlass.Float32)
    
    # MMA #1: C0 += A[:, 0:8] @ B[0:8, 0:8]
    c0_0, c0_1, c0_2, c0_3 = mma_tf32_m16n8k8(
        a0_k0, a1_k0, a2_k0, a3_k0,
        b0_00, b1_00,
        c0_0, c0_1, c0_2, c0_3
    )
    
    # Load B[0:8, 8:16]
    sB_01 = cute.local_tile(sB, tiler=(8, 8), coord=(1, 0))
    tCsB_01 = thr_mma.partition_B(sB_01)
    tCrB_01 = tiled_mma.make_fragment_B(tCsB_01)
    tCsB_01_view = thr_copy_B.partition_S(sB_01)
    tCrB_01_view = thr_copy_B.retile(tCrB_01)
    cute.copy(tiled_copy_B, tCsB_01_view, tCrB_01_view)
    
    b0_01 = tCrB_01[0].to(cutlass.Float32)
    b1_01 = tCrB_01[1].to(cutlass.Float32)
    
    # MMA #2: C1 += A[:, 0:8] @ B[0:8, 8:16]
    c1_0, c1_1, c1_2, c1_3 = mma_tf32_m16n8k8(
        a0_k0, a1_k0, a2_k0, a3_k0,
        b0_01, b1_01,
        c1_0, c1_1, c1_2, c1_3
    )
    
    # =========================================================================
    # Step 4: K=8..15 iteration
    # =========================================================================
    # Load A[:, 8:16]
    sA_k1 = cute.local_tile(sA, tiler=(16, 8), coord=(0, 1))
    tCsA_k1 = thr_mma.partition_A(sA_k1)
    tCrA_k1 = tiled_mma.make_fragment_A(tCsA_k1)
    tCsA_k1_view = thr_copy_A.partition_S(sA_k1)
    tCrA_k1_view = thr_copy_A.retile(tCrA_k1)
    cute.copy(tiled_copy_A, tCsA_k1_view, tCrA_k1_view)
    
    # ldmatrix [0,1,2,3] -> TF32 MMA [0,2,1,3]
    a0_k1 = tCrA_k1[0].to(cutlass.Float32)
    a1_k1 = tCrA_k1[2].to(cutlass.Float32)
    a2_k1 = tCrA_k1[1].to(cutlass.Float32)
    a3_k1 = tCrA_k1[3].to(cutlass.Float32)
    
    # Load B[8:16, 0:8]
    sB_10 = cute.local_tile(sB, tiler=(8, 8), coord=(0, 1))
    tCsB_10 = thr_mma.partition_B(sB_10)
    tCrB_10 = tiled_mma.make_fragment_B(tCsB_10)
    tCsB_10_view = thr_copy_B.partition_S(sB_10)
    tCrB_10_view = thr_copy_B.retile(tCrB_10)
    cute.copy(tiled_copy_B, tCsB_10_view, tCrB_10_view)
    
    b0_10 = tCrB_10[0].to(cutlass.Float32)
    b1_10 = tCrB_10[1].to(cutlass.Float32)
    
    # MMA #3
    c0_0, c0_1, c0_2, c0_3 = mma_tf32_m16n8k8(
        a0_k1, a1_k1, a2_k1, a3_k1,
        b0_10, b1_10,
        c0_0, c0_1, c0_2, c0_3
    )
    
    # Load B[8:16, 8:16]
    sB_11 = cute.local_tile(sB, tiler=(8, 8), coord=(1, 1))
    tCsB_11 = thr_mma.partition_B(sB_11)
    tCrB_11 = tiled_mma.make_fragment_B(tCsB_11)
    tCsB_11_view = thr_copy_B.partition_S(sB_11)
    tCrB_11_view = thr_copy_B.retile(tCrB_11)
    cute.copy(tiled_copy_B, tCsB_11_view, tCrB_11_view)
    
    b0_11 = tCrB_11[0].to(cutlass.Float32)
    b1_11 = tCrB_11[1].to(cutlass.Float32)
    
    # MMA #4
    c1_0, c1_1, c1_2, c1_3 = mma_tf32_m16n8k8(
        a0_k1, a1_k1, a2_k1, a3_k1,
        b0_11, b1_11,
        c1_0, c1_1, c1_2, c1_3
    )
    
    # =========================================================================
    # Step 5: Store C using stmatrix (FP32 -> FP16)
    # =========================================================================
    # Store C[:, 0:8]
    sC_0 = cute.local_tile(sC, tiler=(16, 8), coord=(0, 0))
    tCrC_0_fp16 = tiled_mma.make_fragment_A(tCsA_k0)
    tCrC_0_fp16[0] = c0_0.to(cutlass.Float16)
    tCrC_0_fp16[1] = c0_1.to(cutlass.Float16)
    tCrC_0_fp16[2] = c0_2.to(cutlass.Float16)
    tCrC_0_fp16[3] = c0_3.to(cutlass.Float16)
    
    tCsC_0_view = thr_copy_C.partition_D(sC_0)
    tCrC_0_copy = thr_copy_C.retile(tCrC_0_fp16)
    cute.copy(tiled_copy_C, tCrC_0_copy, tCsC_0_view)
    
    # Store C[:, 8:16]
    sC_1 = cute.local_tile(sC, tiler=(16, 8), coord=(0, 1))
    tCrC_1_fp16 = tiled_mma.make_fragment_A(tCsA_k1)
    tCrC_1_fp16[0] = c1_0.to(cutlass.Float16)
    tCrC_1_fp16[1] = c1_1.to(cutlass.Float16)
    tCrC_1_fp16[2] = c1_2.to(cutlass.Float16)
    tCrC_1_fp16[3] = c1_3.to(cutlass.Float16)
    
    tCsC_1_view = thr_copy_C.partition_D(sC_1)
    tCrC_1_copy = thr_copy_C.retile(tCrC_1_fp16)
    cute.copy(tiled_copy_C, tCrC_1_copy, tCsC_1_view)
    
    cute.arch.sync_threads()
    
    # =========================================================================
    # Step 6: SMEM → Global
    # =========================================================================
    for i in range(elems_per_thread):
        idx = tidx * elems_per_thread + i
        row = idx // 16
        col = idx % 16
        mC[row, col] = sC[row, col]


@cute.jit
def tf32_gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # SMEM layouts
    sA_layout = cute.make_layout((16, 16), stride=(16, 1))  # row-major
    sB_layout = cute.make_layout((16, 16), stride=(1, 16))  # col-major
    sC_layout = cute.make_layout((16, 16), stride=(16, 1))  # row-major
    
    # Helper MMA for ldmatrix/stmatrix layout (m16n8k8)
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 8))
    tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 8))
    
    # LdMatrix: x2 for A (16x8), x1 for B (8x8), StMatrix: x2 for C (16x8)
    atom_copy_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.Float16)
    atom_copy_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 1), cutlass.Float16)
    atom_copy_C = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(False, 2), cutlass.Float16)
    
    tiled_copy_A = cute.make_tiled_copy_A(atom_copy_A, tiled_mma)
    tiled_copy_B = cute.make_tiled_copy_B(atom_copy_B, tiled_mma)
    tiled_copy_C = cute.make_tiled_copy_C(atom_copy_C, tiled_mma)
    
    smem_size = (
        cute.size_in_bytes(cutlass.Float16, sA_layout) +
        cute.size_in_bytes(cutlass.Float16, sB_layout) +
        cute.size_in_bytes(cutlass.Float16, sC_layout)
    )
    
    tf32_gemm_kernel(
        mA, mB, mC,
        sA_layout, sB_layout, sC_layout,
        tiled_mma, tiled_copy_A, tiled_copy_B, tiled_copy_C,
    ).launch(grid=(1, 1, 1), block=(THREADS_PER_CTA, 1, 1), smem=smem_size)


# ===========================================================================
# Test
# ===========================================================================
def test():
    import torch
    
    print("=" * 60)
    print("TF32 MMA 16x16 with CuTe ldmatrix/stmatrix")
    print("=" * 60)
    
    cutlass.cuda.initialize_cuda_context()
    
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    
    def run_test(name, A, B, tol=0.01):
        A = A.to(device="cuda", dtype=torch.float16).contiguous()
        B = B.to(device="cuda", dtype=torch.float16).contiguous()
        C = torch.zeros(16, 16, device="cuda", dtype=torch.float16).contiguous()
        C_ref = A.float() @ B.float()
        
        compiled = cute.compile(tf32_gemm_host, from_dlpack(A), from_dlpack(B), from_dlpack(C))
        compiled(from_dlpack(A), from_dlpack(B), from_dlpack(C))
        torch.cuda.synchronize()
        
        diff = (C.float() - C_ref).abs().max().item()
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: max_diff={diff:.6f} [{status}]")
        return diff < tol
    
    all_passed = True
    
    # Case 1: Identity matrices
    print("\n[Case 1] Identity matrices")
    A = torch.eye(16)
    B = torch.eye(16)
    all_passed &= run_test("I @ I = I", A, B)
    
    # Case 2: All ones
    print("\n[Case 2] All ones")
    A = torch.ones(16, 16)
    B = torch.ones(16, 16)
    all_passed &= run_test("ones @ ones", A, B, tol=0.1)
    
    # Case 3: Small uniform values [0, 1]
    print("\n[Case 3] Uniform [0, 1]")
    torch.manual_seed(42)
    A = torch.rand(16, 16)
    B = torch.rand(16, 16)
    all_passed &= run_test("rand [0,1]", A, B)
    
    # Case 4: Normalized random values
    print("\n[Case 4] Normalized randn")
    torch.manual_seed(123)
    A = normalize(torch.randn(16, 16))
    B = normalize(torch.randn(16, 16))
    all_passed &= run_test("normalized randn", A, B)
    
    # Case 5: Small values [0, 0.1]
    print("\n[Case 5] Small values [0, 0.1]")
    torch.manual_seed(456)
    A = torch.rand(16, 16) * 0.1
    B = torch.rand(16, 16) * 0.1
    all_passed &= run_test("rand [0,0.1]", A, B)
    
    # Case 6: Diagonal matrix
    print("\n[Case 6] Diagonal matrix")
    A = torch.diag(torch.arange(1, 17, dtype=torch.float32) / 16)
    B = torch.ones(16, 16)
    all_passed &= run_test("diag @ ones", A, B)
    
    # Case 7: Lower triangular
    print("\n[Case 7] Lower triangular")
    A = torch.tril(torch.ones(16, 16))
    B = torch.tril(torch.ones(16, 16))
    all_passed &= run_test("tril @ tril", A, B, tol=0.1)
    
    # Case 8: Multiple random seeds
    print("\n[Case 8] Multiple random tests")
    for seed in [1, 42, 100, 999, 2024]:
        torch.manual_seed(seed)
        A = normalize(torch.randn(16, 16))
        B = normalize(torch.randn(16, 16))
        all_passed &= run_test(f"seed={seed}", A, B)
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    test()