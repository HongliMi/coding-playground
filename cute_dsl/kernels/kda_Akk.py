# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
KDA Akk/Aqk Kernel - cp.async Pipeline
Load g (float32), q (fp16), k (fp16) with async copy
Tile size: 16x64 (BC x BK), double buffer
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import json
import os


# 全局配置
BC = 16          # sub-chunk size (rows)
BK = 64          # key dimension (columns)
NUM_STAGES = 2   # double buffer
NUM_THREADS = 128


@cute.kernel
def kda_Akk_kernel(
    tiled_copy_g: cute.TiledCopy,
    tiled_copy_qk: cute.TiledCopy,
    g_tensor: cute.Tensor,       # (B, T, H, K) float32
    q_tensor: cute.Tensor,       # (B, T, H, K) fp16
    k_tensor: cute.Tensor,       # (B, T, H, K) fp16
    Akk_tensor: cute.Tensor,     # (B, T, H, BC) output (debug: write back loaded g[:, :BC])
    Aqk_tensor: cute.Tensor,     # (B, T, H, BT) output (debug: write back loaded q[:, :BC] as fp16 into [:BC])
    g_smem_layout: cute.Layout,  # (BC, BK, NUM_STAGES)
    qk_smem_layout: cute.Layout, # (BC, BK, NUM_STAGES * 2) for q and k
    BT: cutlass.Constexpr[int],
    num_k_tiles: cutlass.Constexpr[int],
):
    """
    Debug kernel:
    - Each block handles one (batch, chunk, head)
    - The block owns a [BT, K] chunk, tiled by (BC, BK)
      BT=64, BC=16 => 4 tiles along T; K=128, BK=64 => 2 tiles along K; total 8 tiles.
    - We use cp.async pipeline to load g/q/k into smem, then write back ONLY the first BC columns
      for correctness validation:
        - Akk: g[..., :BC] -> (B, T, H, BC) fp32
        - Aqk: q[..., :BC] -> (B, T, H, BT) fp16  (only [:BC] is written)
    """
    
    tidx, _, _ = cute.arch.thread_idx()
    i_b, i_t, i_h = cute.arch.block_idx()  # batch, chunk, head
    
    # Allocate shared memory
    smem = cutlass.utils.SmemAllocator()
    sG = smem.allocate_tensor(cutlass.Float32, g_smem_layout, 128)
    sQK = smem.allocate_tensor(cutlass.Float16, qk_smem_layout, 128)
    
    # Chunk start and tile base along T (each T tile is BC rows)
    t_start = i_t * BT
    t_tile_base = i_t * (BT // BC)  # base tile index in (T/BC) dimension
    
    # 获取当前 batch 和 head 的数据 slice
    # g_tensor shape: (B, T, H, K) -> select (T, K) for current batch and head
    gG_batch = g_tensor[(i_b, None, i_h, None)]   # (T, K)
    gQ_batch = q_tensor[(i_b, None, i_h, None)]   # (T, K)
    gK_batch = k_tensor[(i_b, None, i_h, None)]   # (T, K)
    
    # 对 T 和 K 维度分 tiles: (BC, BK, T/BC, K/BK) - 4D tensor
    gG = cute.local_tile(gG_batch, (BC, BK), (None, None))
    gQ = cute.local_tile(gQ_batch, (BC, BK), (None, None))
    gK = cute.local_tile(gK_batch, (BC, BK), (None, None))
    
    # Get thread-level copy partitions
    thr_copy_g = tiled_copy_g.get_slice(tidx)
    thr_copy_qk = tiled_copy_qk.get_slice(tidx)
    
    # Iterate all tiles in the chunk:
    #   iter -> (t_sub, k_tile)
    #   t_sub in [0, BT/BC), k_tile in [0, num_k_tiles)
    num_t_tiles = BT // BC
    total_tiles = num_t_tiles * num_k_tiles

    # =============== Prefetch first stages ===============
    prefetch_count = cutlass.min(NUM_STAGES - 1, total_tiles)
    for it in range(prefetch_count):
        stage = it % NUM_STAGES
        t_sub = it // num_k_tiles
        k_tile = it - t_sub * num_k_tiles

        t_tile_idx = t_tile_base + t_sub

        # Load g tile
        gG_tile = gG[(None, None, t_tile_idx, k_tile)]
        sG_stage = sG[(None, None, stage)]
        thr_gG = thr_copy_g.partition_S(gG_tile)
        thr_sG = thr_copy_g.partition_D(sG_stage)
        cute.copy(tiled_copy_g, thr_gG, thr_sG)

        # Load q tile (stages 0..NUM_STAGES-1)
        gQ_tile = gQ[(None, None, t_tile_idx, k_tile)]
        sQ_stage = sQK[(None, None, stage)]
        thr_gQ = thr_copy_qk.partition_S(gQ_tile)
        thr_sQ = thr_copy_qk.partition_D(sQ_stage)
        cute.copy(tiled_copy_qk, thr_gQ, thr_sQ)

        # Load k tile (stages NUM_STAGES..2*NUM_STAGES-1)
        gK_tile = gK[(None, None, t_tile_idx, k_tile)]
        sK_stage = sQK[(None, None, stage + NUM_STAGES)]
        thr_gK = thr_copy_qk.partition_S(gK_tile)
        thr_sK = thr_copy_qk.partition_D(sK_stage)
        cute.copy(tiled_copy_qk, thr_gK, thr_sK)

        cute.arch.cp_async_commit_group()

    # =============== Main loop ===============
    for it in range(total_tiles):
        stage = it % NUM_STAGES

        # Wait for current stage data to be ready
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
        cute.arch.barrier()

        # Issue next async loads (overlapped with writeback)
        next_it = it + prefetch_count
        if next_it < total_tiles:
            next_stage = next_it % NUM_STAGES
            t_sub_n = next_it // num_k_tiles
            k_tile_n = next_it - t_sub_n * num_k_tiles
            t_tile_idx_n = t_tile_base + t_sub_n

            # Load next g tile
            gG_next = gG[(None, None, t_tile_idx_n, k_tile_n)]
            sG_next = sG[(None, None, next_stage)]
            thr_gG = thr_copy_g.partition_S(gG_next)
            thr_sG = thr_copy_g.partition_D(sG_next)
            cute.copy(tiled_copy_g, thr_gG, thr_sG)

            # Load next q tile
            gQ_next = gQ[(None, None, t_tile_idx_n, k_tile_n)]
            sQ_next = sQK[(None, None, next_stage)]
            thr_gQ = thr_copy_qk.partition_S(gQ_next)
            thr_sQ = thr_copy_qk.partition_D(sQ_next)
            cute.copy(tiled_copy_qk, thr_gQ, thr_sQ)

            # Load next k tile
            gK_next = gK[(None, None, t_tile_idx_n, k_tile_n)]
            sK_next = sQK[(None, None, next_stage + NUM_STAGES)]
            thr_gK = thr_copy_qk.partition_S(gK_next)
            thr_sK = thr_copy_qk.partition_D(sK_next)
            cute.copy(tiled_copy_qk, thr_gK, thr_sK)

            cute.arch.cp_async_commit_group()

        # =============== Debug writeback (current stage) ===============
        t_sub = it // num_k_tiles
        k_tile = it - t_sub * num_k_tiles
        t_abs_base = t_start + t_sub * BC
        # Only write "prefix columns" on K dimension -> only k_tile==0 and c in [0, BC)
        if k_tile == 0:
            # Write Akk/Aqk: [BC rows, BC cols]
            linear = tidx
            while linear < (BC * BC):
                r = linear // BC
                c = linear - r * BC
                Akk_tensor[(i_b, t_abs_base + r, i_h, c)] = sG[(r, c, stage)]
                Aqk_tensor[(i_b, t_abs_base + r, i_h, c)] = sQK[(r, c, stage)]
                linear = linear + NUM_THREADS

        cute.arch.barrier()


@cute.jit
def run_kda_Akk(
    g_tensor: cute.Tensor,
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    Akk_tensor: cute.Tensor,
    Aqk_tensor: cute.Tensor,
    stream: cuda.CUstream
):
    B, T, H, K = g_tensor.layout.shape
    BT = 64  # chunk size
    
    # Number of tiles in K dimension
    num_k_tiles = cute.ceil_div(K, BK)
    NT = cute.ceil_div(T, BT)  # number of chunks
    
    # =============== Create tiled copy for g (float32) ===============
    # 128 bits = 4 floats per copy
    # Thread layout: (8, 16) -> 128 threads
    # Val layout: (1, 4) -> 4 floats per thread
    # Coverage per iteration: 8 rows x 64 cols, need 2 iterations for 16 rows
    copy_atom_g = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128
    )
    thread_layout_g = cute.make_layout(
        (8, 16),       # 8 rows x 16 threads per row
        stride=(16, 1)
    )
    val_layout_g = cute.make_layout((1, 4))  # 4 floats per thread
    tiled_copy_g = cute.make_tiled_copy_tv(copy_atom_g, thread_layout_g, val_layout_g)
    
    # =============== Create tiled copy for q/k (fp16) ===============
    # 128 bits = 8 halfs per copy
    # Thread layout: (16, 8) -> 128 threads
    # Val layout: (1, 8) -> 8 halfs per thread
    # Coverage: 16 rows x 64 cols in one pass
    copy_atom_qk = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float16,
        num_bits_per_copy=128
    )
    thread_layout_qk = cute.make_layout(
        (16, 8),       # 16 rows x 8 threads per row
        stride=(8, 1)
    )
    val_layout_qk = cute.make_layout((1, 8))  # 8 halfs per thread
    tiled_copy_qk = cute.make_tiled_copy_tv(copy_atom_qk, thread_layout_qk, val_layout_qk)
    
    # =============== SMEM layouts ===============
    # g: (BC, BK, NUM_STAGES) = (16, 64, 2)
    g_smem_layout = cute.make_layout(
        (BC, BK, NUM_STAGES),
        stride=(BK, 1, BC * BK)
    )
    # q and k share smem with different stages: (BC, BK, NUM_STAGES * 2)
    # stages 0,1 for q, stages 2,3 for k
    qk_smem_layout = cute.make_layout(
        (BC, BK, NUM_STAGES * 2),
        stride=(BK, 1, BC * BK)
    )
    
    # SMEM size calculation
    # g: 16 * 64 * 2 * 4 bytes = 8192 bytes
    # qk: 16 * 64 * 4 * 2 bytes = 8192 bytes
    smem_bytes = 4 * BC * BK * NUM_STAGES + 2 * BC * BK * NUM_STAGES * 2 + 128
    
    print(f"KDA Akk: B={B}, T={T}, H={H}, K={K}")
    print(f"  Tiles: NT={NT}, num_k_tiles={num_k_tiles}")
    print(f"  SMEM: {smem_bytes} bytes\n")
    
    kda_Akk_kernel(
        tiled_copy_g,
        tiled_copy_qk,
        g_tensor,
        q_tensor,
        k_tensor,
        Akk_tensor,
        Aqk_tensor,
        g_smem_layout,
        qk_smem_layout,
        BT,
        num_k_tiles,
    ).launch(
        grid=(B, NT, H),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


if __name__ == "__main__":
    print("KDA Akk cp.async Test")
    print("=" * 50)
    
    # Test parameters
    # NOTE: For debug writeback, Akk/Aqk are (B, T, H, BC) float32, keep sizes small.
    B, T, H, K = 1, 8192, 96, 128  # batch, seq_len, heads, head_dim
    BT = 64  # chunk size
    warmup_iters = 5
    test_iters = 100
    use_profiler = True
    
    # Create test tensors
    g = torch.randn(B, T, H, K, dtype=torch.float32, device='cuda')
    q = torch.randn(B, T, H, K, dtype=torch.float16, device='cuda')
    k = torch.randn(B, T, H, K, dtype=torch.float16, device='cuda')
    
    # Output tensors (debug writeback buffers)
    # - Akk: (B, T, H, BC) fp32   <- g[..., :BC]
    # - Aqk: (B, T, H, BT) fp16   <- q[..., :BC] written into Aqk[..., :BC]
    Akk = torch.empty(B, T, H, BC, dtype=torch.float32, device='cuda')
    Aqk = torch.empty(B, T, H, BT, dtype=torch.float16, device='cuda')
    
    # Create cute tensors
    g_tensor = from_dlpack(g, assumed_align=16)
    g_tensor.element_type = cutlass.Float32
    
    q_tensor = from_dlpack(q, assumed_align=16)
    q_tensor.element_type = cutlass.Float16
    
    k_tensor = from_dlpack(k, assumed_align=16)
    k_tensor.element_type = cutlass.Float16
    
    Akk_tensor = from_dlpack(Akk, assumed_align=16)
    Akk_tensor.element_type = cutlass.Float32
    
    Aqk_tensor = from_dlpack(Aqk, assumed_align=16)
    Aqk_tensor.element_type = cutlass.Float16
    
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    print("Compiling kernel...")
    compiled = cute.compile(run_kda_Akk, g_tensor, q_tensor, k_tensor, Akk_tensor, Aqk_tensor, stream)
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup_iters):
        compiled(g_tensor, q_tensor, k_tensor, Akk_tensor, Aqk_tensor, stream)
    torch.cuda.synchronize()
    
    print("\n✓ Kernel executed successfully!")

    # Correctness check (debug writeback)
    max_diff_g = (Akk - g[..., :BC]).abs().max().item()
    max_diff_q = (Aqk[..., :BC] - q[..., :BC]).abs().max().item()
    print(f"\nCorrectness:")
    print(f"  max|Akk - g| = {max_diff_g:.6e}")
    print(f"  max|Aqk - q| = {max_diff_q:.6e}")
    if max_diff_g > 1e-4 or max_diff_q > 1e-2:
        print("✗ FAIL (writeback mismatch)")
        raise SystemExit(1)
    print("✓ PASS (writeback matches g/q)")

    # Print a small slice for manual inspection (b=0, h=0): first 5 rows x first 5 cols
    print("\nInspect (b=0, h=0) first 5x5:")
    print("  Akk[0, :5, 0, :5]:")
    print(Akk[0, :5, 0, :5].detach().cpu())
    print("  g  [0, :5, 0, :5]:")
    print(g[0, :5, 0, :5].detach().cpu())
    print("  Aqk[0, :5, 0, :5]:")
    print(Aqk[0, :5, 0, :5].detach().cpu())
    print("  q  [0, :5, 0, :5]:")
    print(q[0, :5, 0, :5].detach().cpu())
    
    # Benchmark
    print(f"\nBenchmarking: {test_iters} iterations...")

    # L2 cache eviction buffer (match benchmark_all.py idea)
    dummy_buffer = torch.empty(int(80 * 1024 * 1024 / 4), dtype=torch.float32, device='cuda')

    # Data size (bytes moved by this debug kernel)
    # - Read:  g(fp32) + q(fp16) + k(fp16)  -> full K
    # - Write: Akk(fp32) -> only first BC columns
    #          Aqk(fp16) -> only first BC columns (even though Aqk is allocated with BT columns)
    read_bytes = B * T * H * K * (4 + 2 + 2)
    write_bytes = (B * T * H * BC * 4) + (B * T * H * BC * 2)
    data_bytes = read_bytes + write_bytes
    data_mb = data_bytes / 1024 / 1024
    data_gib = data_bytes / 1000 / 1000 / 1000
    
    # 峰值带宽 (GB/s) - 根据不同 GPU 平台修改
    # B200: 8000, H100 SXM: 3350, H100 PCIe: 2000, A100 SXM: 2039, A100 PCIe: 1555
    PEAK_BW_GBS = 7672  # B200

    # =============== CUDA Event Timing ===============
    print(f"\nCUDA Event Timing: {test_iters} iterations...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    cuda_event_times_ms = []
    for i in range(test_iters):
        # Evict L2 cache
        _ = dummy_buffer.sum()
        torch.cuda.synchronize()
        
        start_event.record()
        compiled(g_tensor, q_tensor, k_tensor, Akk_tensor, Aqk_tensor, stream)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        cuda_event_times_ms.append(elapsed_ms)
    
    cuda_event_times = torch.tensor(cuda_event_times_ms, dtype=torch.float64)
    cuda_event_mean_ms = cuda_event_times.mean().item()
    cuda_event_min_ms = cuda_event_times.min().item()
    cuda_event_std_ms = cuda_event_times.std().item()
    
    print(f"  ✓ CUDA Event Mean: {cuda_event_mean_ms:.4f} ms")
    print(f"  ✓ CUDA Event Min:  {cuda_event_min_ms:.4f} ms")
    print(f"  ✓ CUDA Event Std:  {cuda_event_std_ms:.4f} ms")

    # =============== torch.profiler Timing ===============
    if use_profiler:
        print(f"\nProfiling with torch.profiler: {test_iters} iterations...")
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        )

        with profiler:
            for i in range(test_iters):
                _ = dummy_buffer.sum()
                torch.cuda.synchronize()
                with torch.profiler.record_function(f"kda_Akk_iter{i}"):
                    compiled(g_tensor, q_tensor, k_tensor, Akk_tensor, Aqk_tensor, stream)
                torch.cuda.synchronize()

        trace_dir = "profiler_traces"
        os.makedirs(trace_dir, exist_ok=True)
        trace_file = os.path.join(trace_dir, "kda_Akk.json")
        profiler.export_chrome_trace(trace_file)
        print(f"  ✓ Profiler trace saved to {trace_file}")

        profiler_times_us = []
        try:
            with open(trace_file, "r") as f:
                trace_data = json.load(f)
            for event in trace_data.get("traceEvents", []):
                if event.get("cat") != "kernel":
                    continue
                name = event.get("name", "")
                dur = event.get("dur", 0)
                if dur > 0 and "kernel_cutlass" in name:
                    profiler_times_us.append(dur)
            if profiler_times_us:
                print(f"  ✓ Parsed {len(profiler_times_us)} CUTLASS kernel timings from profiler trace")
            else:
                print("  ⚠ No CUTLASS kernel timings found in profiler trace")
        except Exception as e:
            print(f"  ✗ Failed to parse profiler trace: {e}")
            profiler_times_us = []
    else:
        profiler_times_us = []
    
    # =============== Calculate metrics ===============
    # CUDA Event metrics
    cute_bw_cuda = data_gib / (cuda_event_mean_ms / 1000.0)
    cute_peak_cuda = cute_bw_cuda / PEAK_BW_GBS * 100.0
    
    # Profiler metrics
    cute_profiler_mean_ms = 0.0
    cute_bw_profiler = 0.0
    cute_peak_profiler = 0.0
    if len(profiler_times_us) > 0:
        cute_profiler_mean_ms = torch.tensor(profiler_times_us, dtype=torch.float64).mean().item() / 1000.0
        cute_bw_profiler = data_gib / (cute_profiler_mean_ms / 1000.0)
        cute_peak_profiler = cute_bw_profiler / PEAK_BW_GBS * 100.0
    
    # =============== Summary Table ===============
    print("\n" + "=" * 60)
    print(f"Performance Summary (B={B}, T={T}, H={H}, K={K})")
    print(f"Data: {data_mb:.1f} MB, Peak BW: {PEAK_BW_GBS} GB/s")
    print("=" * 60)
    print(f"{'Metric':<20} | {'Value':<18}")
    print("-" * 60)
    
    # CUDA Event row
    print(f"{'CUDA Event (ms)':<20} | {cuda_event_mean_ms:<18.4f}")
    print(f"{'  → Min (ms)':<20} | {cuda_event_min_ms:<18.4f}")
    print(f"{'  → Std (ms)':<20} | {cuda_event_std_ms:<18.4f}")
    print(f"{'  → BW (GB/s)':<20} | {cute_bw_cuda:<18.1f}")
    print(f"{'  → Peak%':<20} | {cute_peak_cuda:<18.2f}")
    
    # Profiler row
    if cute_profiler_mean_ms > 0:
        print("-" * 60)
        print(f"{'Profiler (ms)':<20} | {cute_profiler_mean_ms:<18.4f}")
        print(f"{'  → BW (GB/s)':<20} | {cute_bw_profiler:<18.1f}")
        print(f"{'  → Peak%':<20} | {cute_peak_profiler:<18.2f}")
    
    print("=" * 60)
    print("DONE")
    
    del compiled
