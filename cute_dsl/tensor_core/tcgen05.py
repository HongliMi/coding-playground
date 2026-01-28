# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
最简单的 tcgen05 MMA kernel: 128x256x16 (FP16)
使用 Blackwell tcgen05 MMA 指令
参考 fp16_gemm_0.py 的实现，使用标准的 tile size

关键点：
1. tcgen05 MMA 的累加器存储在 TMEM (Tensor Memory) 中，不是普通寄存器
2. 需要使用 TmemAllocator 分配 TMEM
3. MMA 执行后需要用 tcgen05.make_tmem_copy 从 TMEM 加载到寄存器
4. 使用 TMA 加载 A/B 到 shared memory
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack


# Problem dimensions - use standard fp16 tile size like fp16_gemm_0.py
M = 128
N = 256
K = 64

# Data types
IO_DTYPE = cutlass.Float16
ACC_DTYPE = cutlass.Float32

# Kernel configuration
THREADS_PER_CTA = 128
AB_STAGES = 4
ACC_STAGE = 1
MMA_INST_SHAPE_MNK = (128, 256, 16)
MMA_TILER_MNK = (128, 256, 64)


@cute.struct
class SharedStorage:
    """Shared memory structure for barriers and TMEM allocation"""
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGE * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def tcgen05_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_a: cpasync.TmaInfo,
    tma_b: cpasync.TmaInfo,
    mC: cute.Tensor,
):
    """
    tcgen05 GEMM kernel 实现：
    1. 使用 TMA 加载 A/B 到 shared memory
    2. 执行 MMA (累加器在 TMEM)
    3. 从 TMEM 加载累加器到寄存器
    4. 存储到全局内存
    """
    # Extract TMA tensors
    mA = tma_a.tma_tensor
    mB = tma_b.tma_tensor

    # Thread/warp indices
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    # =========================================================================
    # Step 1: Allocate shared memory
    # =========================================================================
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # Allocate A/B shared memory with proper swizzle layout
    sA = smem.allocate_tensor(
        element_type=IO_DTYPE,
        layout=tma_a.smem_layout.outer,
        byte_alignment=128,
        swizzle=tma_a.smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=IO_DTYPE,
        layout=tma_b.smem_layout.outer,
        byte_alignment=128,
        swizzle=tma_b.smem_layout.inner,
    )

    # =========================================================================
    # Step 2: Setup TMEM allocator
    # =========================================================================
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=THREADS_PER_CTA
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512  # Same as fp16_gemm_0.py
    tmem.allocate(num_tmem_cols)

    # Prefetch TMA descriptors
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_a.atom)
        cpasync.prefetch_descriptor(tma_b.atom)

    # =========================================================================
    # Step 3: Setup pipelines
    # =========================================================================
    num_tma_copy_bytes = cute.size_in_bytes(
        IO_DTYPE, cute.select(tma_a.smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(
        IO_DTYPE, cute.select(tma_b.smem_layout, mode=[0, 1, 2])
    )

    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=ACC_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            THREADS_PER_CTA,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # =========================================================================
    # Step 4: Partition tensors for MMA
    # =========================================================================
    # (bM, bK, RestK)
    gA = cute.local_tile(mA, MMA_TILER_MNK, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB, MMA_TILER_MNK, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC, MMA_TILER_MNK, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_K, RestK)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K, RestK)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)

    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)

    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    # =========================================================================
    # Step 5: Partition for TMA
    # =========================================================================
    tAsA, tAgA = cpasync.tma_partition(
        tma_a.atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_b.atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # =========================================================================
    # Step 6: Wait for TMEM allocation
    # =========================================================================
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # =========================================================================
    # Step 7: Setup epilogue - same as fp16_gemm_0.py
    # =========================================================================
    subtile_cnt = 4
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    # Every thread loads 64 x fp32
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        ACC_DTYPE,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # (TmemCpy, NumTmemCpy, NumTiles)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    # (TmemCpy, NumTmemCpy)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, ACC_DTYPE)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, IO_DTYPE)

    # =========================================================================
    # Step 8: Main loop
    # =========================================================================
    num_k_tiles = cute.size(gA, mode=[2])

    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()

        for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=AB_STAGES - 2):
            # Issue TMA loads
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_a.atom,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_b.atom,
                tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )

            # Wait for TMA loads
            ab_full = ab_consumer.wait_and_advance()

            # Execute MMA
            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
            tile_crd = (None, None, None, ab_full.index)
            cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

            # Release A/B buffers
            ab_full.release()

        # Signal accumulator ready
        acc_empty.commit()

    # =========================================================================
    # Step 9: Epilogue
    # =========================================================================
    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()

    # TMEM -> RMEM -> GMEM
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(IO_DTYPE))
        cute.autovec_copy(tCrC, tDgC[None, None, i])

    acc_full.release()

    # =========================================================================
    # Step 10: Cleanup
    # =========================================================================
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def tcgen05_gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    """
    Host function to launch tcgen05 GEMM kernel
    """
    # Create MMA operation
    op = tcgen05.MmaF16BF16Op(
        IO_DTYPE,
        ACC_DTYPE,
        MMA_INST_SHAPE_MNK,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    # Create shared memory layouts
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, MMA_TILER_MNK, IO_DTYPE, AB_STAGES
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, MMA_TILER_MNK, IO_DTYPE, AB_STAGES
    )

    # Create TMA atoms
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, mA, a_smem_layout, MMA_TILER_MNK, tiled_mma,
    )
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, mB, b_smem_layout, MMA_TILER_MNK, tiled_mma,
    )

    # Compute grid shape
    grid_shape = cute.ceil_div((*mC.layout.shape, 1), MMA_TILER_MNK[:2])

    # Launch kernel
    tcgen05_gemm_kernel(tiled_mma, tma_a, tma_b, mC).launch(
        grid=grid_shape,
        block=(THREADS_PER_CTA, 1, 1),
    )


# ===========================================================================
# Test
# ===========================================================================
def test():
    import torch

    print("=" * 60)
    print(f"tcgen05 FP16 GEMM: {M}x{K} @ {K}x{N} = {M}x{N}")
    print("Using Float16 precision on Blackwell")
    print("参考 fp16_gemm_0.py 的实现")
    print("=" * 60)

    # Initialize CUDA
    from cuda.bindings import driver as cu_driver
    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run this example")

    def run_test(name, m, n, k, tol=1.0):
        """Run a GEMM test with given dimensions"""
        torch.manual_seed(42)
        
        # Create tensors in FP16
        # A: (M, K) row-major
        # B: (N, K) - CuTe convention
        A = torch.randn(m, k, device="cuda", dtype=torch.float16)
        B_nk = torch.randn(n, k, device="cuda", dtype=torch.float16)  # Already in CuTe convention
        C = torch.zeros(m, n, device="cuda", dtype=torch.float16)

        # Reference: A @ B^T (since B is NxK, we need A @ B^T to get MxN)
        C_ref = torch.einsum("mk,nk->mn", A.float(), B_nk.float()).to(torch.float16)

        # Convert to CuTe tensors
        A_cute = from_dlpack(A, assumed_align=32).mark_layout_dynamic(leading_dim=1)
        B_cute = from_dlpack(B_nk, assumed_align=32).mark_layout_dynamic(leading_dim=1)
        C_cute = from_dlpack(C, assumed_align=32).mark_layout_dynamic(leading_dim=1)

        # Compile and run
        compiled = cute.compile(tcgen05_gemm_host, A_cute, B_cute, C_cute)
        compiled(A_cute, B_cute, C_cute)
        torch.cuda.synchronize()

        # Check result
        diff = (C.float() - C_ref.float()).abs().max().item()
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name} ({m}x{k} @ {k}x{n}): max_diff={diff:.6f} [{status}]")

        if diff >= tol:
            print(f"    Expected (top-left 4x4):\n{C_ref[:4, :4]}")
            print(f"    Got (top-left 4x4):\n{C[:4, :4]}")

        return diff < tol

    all_passed = True

    # Test with the configured tile size
    print(f"\n[Test] GEMM with tile size {M}x{N}x{K}")
    all_passed &= run_test("fp16_gemm", M, N, K, tol=1.0)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    test()
