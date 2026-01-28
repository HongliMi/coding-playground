# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
最简单的 tcgen05 MMA kernel: 32x32x8 (TF32)
使用 Blackwell tcgen05 MMA 指令
参考 dense_gemm.py 的实现，使用 sm100_utils.make_trivial_tiled_mma

关键点：
1. TF32 使用 MmaTF32Op，K 维度 inst shape 是 8
2. 实际数据存储是 Float32，使用 internal_type=TFloat32 进行 TMA
3. tcgen05 MMA 的累加器存储在 TMEM (Tensor Memory) 中
4. 需要使用 TmemAllocator 分配 TMEM
5. MMA 执行后需要用 tcgen05.make_tmem_copy 从 TMEM 加载到寄存器
6. 使用 sm100_utils.get_tmem_load_op 获取正确的 TMEM load 操作
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack


# Problem dimensions - TF32 M-mode supports 64 or 128
M = 32   # Experiment: Try with M=32 and SMEM=32 rows
N = 32
K = 8

# Data types
IO_DTYPE = cutlass.Float32  # Actual storage type
ACC_DTYPE = cutlass.Float32

# Kernel configuration
THREADS_PER_CTA = 128
AB_STAGES = 2
ACC_STAGE = 1
MMA_TILER_MNK = (64, 32, 8)  # Testing minimum M=64 for TF32
# CTA tile shape (for epilogue) - for CtaGroup.ONE, atom_thr_size = 1
CTA_TILE_SHAPE_MNK = (64, 32, 8)

# Experiment: Try allocating only 32 rows in SMEM
SMEM_TILER_MNK = (32, 32, 8)  # Only 32 rows for SMEM (half of MMA requirement)


@cute.struct
class SharedStorage:
    """Shared memory structure for barriers and TMEM allocation"""
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGE * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def tcgen05_tf32_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_a: cpasync.TmaInfo,
    tma_b: cpasync.TmaInfo,
    mC: cute.Tensor,
    c_layout: cutlass.Constexpr,  # utils.LayoutEnum
):
    """
    tcgen05 TF32 GEMM kernel 实现：
    1. 使用 TMA 加载 A/B 到 shared memory (使用 TFloat32 internal type)
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
    num_tmem_cols = 32
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
    # Step 7: Setup epilogue - following dense_gemm.py pattern
    # =========================================================================
    # For non-TMA store, epi_tile = cta_tile_shape_mnk[:2]
    epi_tile = CTA_TILE_SHAPE_MNK[:2]

    # Get TMEM load copy atom using sm100_utils.get_tmem_load_op
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        CTA_TILE_SHAPE_MNK,
        c_layout,
        IO_DTYPE,  # c_dtype
        ACC_DTYPE,  # acc_dtype
        epi_tile,
        False,  # use_2cta_instrs
    )

    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
    tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
    # (EPI_TILE_M, EPI_TILE_N)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
    )

    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
    gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_gC = thr_copy_t2r.partition_D(gC_epi)
    # (T2R, T2R_M, T2R_N)
    tTR_rAcc = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0)].shape, ACC_DTYPE
    )

    # Group modes for iteration
    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

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
    # Step 9: Epilogue - TMEM -> RMEM -> GMEM
    # =========================================================================
    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()

    # Store accumulator to global memory in sub-tiles
    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    for subtile_idx in cutlass.range(subtile_cnt):
        # Load accumulator from tensor memory buffer to register
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        # Store to global memory (TF32 output is Float32, no type conversion)
        acc_vec = tTR_rAcc.load().to(IO_DTYPE)
        tTR_gC[(None, None, None, subtile_idx)].store(acc_vec)

    acc_full.release()

    # =========================================================================
    # Step 10: Cleanup
    # =========================================================================
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def tcgen05_tf32_gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    """
    Host function to launch tcgen05 TF32 GEMM kernel.
    Uses sm100_utils.make_trivial_tiled_mma as in dense_gemm.py
    """
    # Get major modes (assume K-major for both A and B)
    a_major_mode = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
    b_major_mode = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
    c_layout = utils.LayoutEnum.from_tensor(mC)

    # Create tiled MMA using sm100_utils.make_trivial_tiled_mma
    # For TF32, ab_dtype should be TFloat32 or Float32
    # The function will automatically use MmaTF32Op
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.TFloat32,  # ab_dtype - use TFloat32 for TF32 MMA
        a_major_mode,
        b_major_mode,
        ACC_DTYPE,
        tcgen05.CtaGroup.ONE,
        MMA_TILER_MNK[:2],  # (M, N)
    )

    # Create shared memory layouts
    # Experiment: Use SMEM_TILER_MNK (32 rows) instead of MMA_TILER_MNK (64 rows)
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, SMEM_TILER_MNK, cutlass.TFloat32, AB_STAGES  # Changed to SMEM_TILER_MNK
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, SMEM_TILER_MNK, cutlass.TFloat32, AB_STAGES  # Changed to SMEM_TILER_MNK
    )

    # Create TMA atoms with internal_type=TFloat32 for TF32 precision
    # Experiment: Use SMEM_TILER_MNK to load only 32 rows
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, mA, a_smem_layout, SMEM_TILER_MNK, tiled_mma,  # Changed to SMEM_TILER_MNK
        internal_type=cutlass.TFloat32,  # Key: use TFloat32 for TF32 TMA
    )
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, mB, b_smem_layout, SMEM_TILER_MNK, tiled_mma,  # Changed to SMEM_TILER_MNK
        internal_type=cutlass.TFloat32,  # Key: use TFloat32 for TF32 TMA
    )

    # Compute grid shape
    grid_shape = cute.ceil_div((*mC.layout.shape, 1), MMA_TILER_MNK[:2])

    # =========================================================================
    # Debug: Print shapes and configurations
    # =========================================================================
    print("\n" + "=" * 70)
    print("HOST CONFIGURATION DEBUG INFO")
    print("=" * 70)
    
    # Input tensor shapes
    print(f"\n[Input Tensors]")
    print(f"  mA layout: {mA.layout}")
    print(f"  mB layout: {mB.layout}")
    print(f"  mC layout: {mC.layout}")
    
    # Layout modes
    print(f"\n[Layout Modes]")
    print(f"  a_major_mode: {a_major_mode}")
    print(f"  b_major_mode: {b_major_mode}")
    print(f"  c_layout: {c_layout}")
    
    # MMA configuration
    print(f"\n[MMA Configuration]")
    print(f"  MMA_TILER_MNK: {MMA_TILER_MNK}")
    print(f"  tiled_mma: {tiled_mma}")
    
    # Shared memory layouts
    print(f"\n[Shared Memory Layouts]")
    print(f"  a_smem_layout: {a_smem_layout}")
    print(f"  b_smem_layout: {b_smem_layout}")
    
    # TMA info
    print(f"\n[TMA Configuration]")
    print(f"  tma_a.smem_layout: {tma_a.smem_layout}")
    print(f"  tma_b.smem_layout: {tma_b.smem_layout}")
    print(f"  tma_a.tma_tensor layout: {tma_a.tma_tensor.layout}")
    print(f"  tma_b.tma_tensor layout: {tma_b.tma_tensor.layout}")
    
    # Grid/Block configuration
    print(f"\n[Launch Configuration]")
    print(f"  grid_shape: {grid_shape}")
    print(f"  block: ({THREADS_PER_CTA}, 1, 1)")
    print(f"  AB_STAGES: {AB_STAGES}")
    print(f"  ACC_STAGE: {ACC_STAGE}")
    
    print("=" * 70 + "\n")

    # Launch kernel
    tcgen05_tf32_gemm_kernel(tiled_mma, tma_a, tma_b, mC, c_layout).launch(
        grid=grid_shape,
        block=(THREADS_PER_CTA, 1, 1),
    )


# ===========================================================================
# Test
# ===========================================================================
def test():
    import torch

    print("=" * 60)
    print(f"tcgen05 TF32 GEMM: {M}x{K} @ {K}x{N} = {M}x{N}")
    print("Using TFloat32 precision on Blackwell")
    print("参考 dense_gemm.py 的实现，使用 sm100_utils.make_trivial_tiled_mma")
    print("=" * 60)

    # Initialize CUDA
    from cuda.bindings import driver as cu_driver
    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run this example")

    def run_test(name, m, n, k, tol=0.1):
        """Run a GEMM test with given dimensions"""
        torch.manual_seed(42)
        
        # Create tensors in Float32 (TF32 uses Float32 storage)
        # A: (M, K) row-major
        # B: (N, K) - CuTe convention
        A = torch.randn(m, k, device="cuda", dtype=torch.float32)
        B_nk = torch.randn(n, k, device="cuda", dtype=torch.float32)  # Already in CuTe convention
        C = torch.zeros(m, n, device="cuda", dtype=torch.float32)

        # Reference: A @ B^T (since B is NxK, we need A @ B^T to get MxN)
        C_ref = torch.einsum("mk,nk->mn", A, B_nk)

        # Convert to CuTe tensors with force_tf32=True
        A_cute = from_dlpack(A, assumed_align=32, force_tf32=True).mark_layout_dynamic(leading_dim=1)
        B_cute = from_dlpack(B_nk, assumed_align=32, force_tf32=True).mark_layout_dynamic(leading_dim=1)
        C_cute = from_dlpack(C, assumed_align=32, force_tf32=True).mark_layout_dynamic(leading_dim=1)

        # Compile and run
        compiled = cute.compile(tcgen05_tf32_gemm_host, A_cute, B_cute, C_cute)
        compiled(A_cute, B_cute, C_cute)
        torch.cuda.synchronize()

        # Check result - TF32 has ~1e-3 precision
        diff = (C - C_ref).abs().max().item()
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name} ({m}x{k} @ {k}x{n}): max_diff={diff:.6f} [{status}]")

        
        print(f"    Expected (top-left 4x4):\n{C_ref[:4, :4]}")
        print(f"    Got (top-left 4x4):\n{C[:4, :4]}")

        return diff < tol

    all_passed = True

    # Test with the configured tile size
    print(f"\n[Test] GEMM with tile size {M}x{N}x{K}")
    # TF32 has ~10-bit mantissa precision, so tolerance should be higher
    all_passed &= run_test("tf32_gemm", M, N, K, tol=0.1)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    test()
