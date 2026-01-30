# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Batched cp.async Pipeline - 简化版
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


# 全局配置
TILE_M = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128


@cute.kernel
def cpasync_kernel(
    tiled_copy: cute.TiledCopy,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    smem_layout_staged: cute.Layout,
    num_m_tiles: cutlass.Constexpr[int],
):
    """使用 cp.async pipeline 加载所有 M tiles"""
    
    tidx, _, _ = cute.arch.thread_idx()
    batch_idx, _, _ = cute.arch.block_idx()
    
    smem = cutlass.utils.SmemAllocator()
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
    
    # 选择当前 batch
    gSrc_batch = src_tensor[(batch_idx, None, None)]  # (M, K)
    gDst_batch = dst_tensor[(batch_idx, None, None)]  # (M, K)
    
    # M 方向分 tiles
    gSrc = cute.local_tile(gSrc_batch, (TILE_M, TILE_K), (None, 0))  # (TILE_M, TILE_K, num_m_tiles)
    gDst = cute.local_tile(gDst_batch, (TILE_M, TILE_K), (None, 0))
    
    # Partition
    thr_copy = tiled_copy.get_slice(tidx)
    
    # Prefetch
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_m_tiles)
    for m_tile in range(prefetch_count):
        stage = m_tile % NUM_STAGES
        
        gSrc_tile = gSrc[(None, None, m_tile)]
        sData_stage = sData[(None, None, stage)]
        
        thr_gSrc = thr_copy.partition_S(gSrc_tile)
        thr_sData = thr_copy.partition_D(sData_stage)
        
        cute.copy(tiled_copy, thr_gSrc, thr_sData)
        cute.arch.cp_async_commit_group()
    
    # Mainloop
    for m_tile in range(num_m_tiles):
        stage = m_tile % NUM_STAGES
        
        # 等待当前 stage
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
        cute.arch.barrier()
        
        # 先发起下一个加载（异步，与写回重叠）
        next_m_tile = m_tile + prefetch_count
        if next_m_tile < num_m_tiles:
            next_stage = next_m_tile % NUM_STAGES
            
            gSrc_next = gSrc[(None, None, next_m_tile)]
            sData_next = sData[(None, None, next_stage)]
            
            thr_gSrc = thr_copy.partition_S(gSrc_next)
            thr_sData = thr_copy.partition_D(sData_next)
            
            cute.copy(tiled_copy, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()
        
        # 写回当前 stage（与加载重叠）
        for row in range(TILE_M):
            reg = sData[(row, tidx, stage)]
            gDst[(row, tidx, m_tile)] = reg


@cute.jit
def run_cpasync(src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
    batch, m, k = src.layout.shape[0], src.layout.shape[1], src.layout.shape[2]
    
    # 创建 cp.async copy（4 元素向量化）
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128
    )
    
    # Thread layout：4 行 × 32 个线程/行
    thread_layout = cute.make_layout(
        (4, 32),      # 4 行，32 线程/行
        stride=(32, 1)
    )
    val_layout = cute.make_layout((1, 4))
    
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
    
    # 计算 tiles 数量
    num_m_tiles = cute.ceil_div(m, TILE_M)
    
    # SMEM layout
    smem_layout_staged = cute.make_layout(
        (TILE_M, TILE_K, NUM_STAGES),
        stride=(TILE_K, 1, TILE_M * TILE_K)
    )
    smem_bytes = 4 * TILE_M * TILE_K * NUM_STAGES + 32
    
    print(f"cp.async: {batch} batches, {m}x{k}, {num_m_tiles} M-tiles/batch\n")
    
    cpasync_kernel(tiled_copy, src, dst, smem_layout_staged, num_m_tiles).launch(
        grid=(batch, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream
    )


if __name__ == "__main__":
    print("cp.async Test")
    print("-" * 40)
    
    batch, m, k = 4096, 128, 128
    warmup_iters = 5
    test_iters = 100
    
    src_list = [torch.full((m, k), float(i), dtype=torch.float32, device='cuda') for i in range(min(batch, 10))]
    if batch > 10:
        src_list += [torch.randn(m, k, dtype=torch.float32, device='cuda') for _ in range(batch - 10)]
    src = torch.stack(src_list, dim=0)
    dst = torch.full((batch, m, k), -1.0, dtype=torch.float32, device='cuda')
    
    src_tensor = from_dlpack(src, assumed_align=16)
    src_tensor.element_type = cutlass.Float32
    
    dst_tensor = from_dlpack(dst, assumed_align=16)
    dst_tensor.element_type = cutlass.Float32
    
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(run_cpasync, src_tensor, dst_tensor, stream)
    
    # Warmup
    for _ in range(warmup_iters):
        compiled(src_tensor, dst_tensor, stream)
    torch.cuda.synchronize()
    
    # 验证前 5 个
    print("\nFirst 5 batches (5x5):")
    for i in range(min(5, batch)):
        print(f"\nBatch {i}:")
        print("  src:", src[i, :5, :5].cpu())
        print("  dst:", dst[i, :5, :5].cpu())
        diff = torch.max(torch.abs(src[i] - dst[i])).item()
        print(f"  diff: {diff:.6f}")
        if diff > 1e-5:
            print("  ✗ FAIL!")
            exit(1)
    print("\n✓ Verification passed!\n")
    
    # Benchmark
    print(f"Benchmarking: {test_iters} iterations...")
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(test_iters)]
    dummy_buffer = torch.empty(int(80 * 1024 * 1024 / 4), dtype=torch.float32, device='cuda')
    
    times = []
    for i in range(test_iters):
        _ = dummy_buffer.sum()
        torch.cuda.synchronize()
        
        events[i][0].record()
        compiled(src_tensor, dst_tensor, stream)
        events[i][1].record()
        torch.cuda.synchronize()
    
    for start, end in events:
        times.append(start.elapsed_time(end))
    
    times = torch.tensor(times)
    data_mb = m * k * batch * 4 / 1024 / 1024
    
    print("\n" + "=" * 70)
    print(f"✓ Results:")
    print(f"  Mean: {times.mean():.3f} ms (±{times.std():.3f})")
    print(f"  Min:  {times.min():.3f} ms")
    print(f"  BW: {data_mb * 2 / (times.mean() / 1000) / 1024:.1f} GB/s (mean)")
    print("=" * 70)
    print("PASS")
    
    del compiled, src_tensor, dst_tensor

