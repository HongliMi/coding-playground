"""
Memory Transaction Comparison: 64B vs 128B per warp.

Copy a (1, 8192, 96, 128) bf16 tensor from src to dst using BT=64 chunking.

Kernel A (64B/warp):
  4 warps split 128 cols → each warp handles 32 cols
  Per thread: 1 bf16 (2 bytes) → warp transaction = 32×2 = 64 bytes = 2 sectors
  4 warps × 1 row per iteration → 64 iterations

Kernel B (128B/warp):
  2 warps per row, each handles 64 cols via vec2
  Per thread: 1 vec2 = 2 bf16 (4 bytes) → warp transaction = 32×4 = 128 bytes = 4 sectors = 1 cache line
  4 warps → 2 rows per iteration → 32 iterations
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

BT = 64
DIM = 128
NUM_WARPS = 4
WARP_SIZE = 32
THREADS = NUM_WARPS * WARP_SIZE


# =========================================================================
# Kernel A: 64 bytes per warp transaction (scalar bf16 per thread)
# =========================================================================
@cute.kernel
def copy_64B_kernel(
    mSrc: cute.Tensor,    # (B, T, H, K) bf16
    mDst: cute.Tensor,    # (B, T, H, K) bf16
):
    tidx, _, _ = cute.arch.thread_idx()
    i_b, i_t, i_h = cute.arch.block_idx()

    warp_idx = tidx // WARP_SIZE
    lane_idx = tidx % WARP_SIZE

    chunk_start = i_t * BT
    # 4 warps × 32 cols each: warp0→[0:32], warp1→[32:64], warp2→[64:96], warp3→[96:128]
    col = warp_idx * 32 + lane_idx

    for row in cutlass.range_constexpr(BT):
        global_row = chunk_start + row
        val = mSrc[(i_b, global_row, i_h, col)]
        mDst[(i_b, global_row, i_h, col)] = val


@cute.jit
def launch_64B(
    mSrc: cute.Tensor,
    mDst: cute.Tensor,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    H: cutlass.Constexpr,
    seq_len: cutlass.Constexpr,
):
    NT = seq_len // BT
    copy_64B_kernel(mSrc, mDst).launch(
        grid=(B, NT, H),
        block=(THREADS, 1, 1),
        smem=0,
        stream=stream,
    )


# =========================================================================
# Kernel B: 128 bytes per warp transaction (vec2 bf16 per thread)
# =========================================================================
@cute.kernel
def copy_128B_kernel(
    mSrc: cute.Tensor,    # (B, T, H, K//2, 2) bf16 vec2 view
    mDst: cute.Tensor,    # (B, T, H, K//2, 2) bf16 vec2 view
):
    tidx, _, _ = cute.arch.thread_idx()
    i_b, i_t, i_h = cute.arch.block_idx()

    warp_idx = tidx // WARP_SIZE
    lane_idx = tidx % WARP_SIZE

    # 2 warps per row: warp%2 selects col half, warp//2 selects row offset
    row_pair = warp_idx // 2    # 0 or 1 → which of the 2 rows per iteration
    col_half = warp_idx % 2     # 0 → cols [0:64), 1 → cols [64:128)

    chunk_start = i_t * BT
    col_vec = col_half * 32 + lane_idx  # vec2 index: 32 threads × 2 bf16 = 64 scalar cols

    rVec = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.BFloat16)

    # 4 warps handle 2 rows per iteration → BT/2 = 32 iterations
    for iter_idx in cutlass.range_constexpr(BT // 2):
        row = iter_idx * 2 + row_pair
        global_row = chunk_start + row
        cute.autovec_copy(mSrc[(i_b, global_row, i_h, col_vec, None)], rVec)
        cute.autovec_copy(rVec, mDst[(i_b, global_row, i_h, col_vec, None)])


@cute.jit
def launch_128B(
    mSrc: cute.Tensor,
    mDst: cute.Tensor,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    H: cutlass.Constexpr,
    seq_len: cutlass.Constexpr,
):
    NT = seq_len // BT
    copy_128B_kernel(mSrc, mDst).launch(
        grid=(B, NT, H),
        block=(THREADS, 1, 1),
        smem=0,
        stream=stream,
    )


# =========================================================================
# Benchmark
# =========================================================================
def benchmark():
    B, T, H, K = 1, 8192, 96, 128
    device = torch.device("cuda")
    warmup_iters = 1
    test_iters = 100

    print(f"Memory Transaction Benchmark: 64B vs 128B per warp")
    print(f"Tensor: ({B}, {T}, {H}, {K}) bf16")
    print(f"Chunk: BT={BT}, Block={THREADS} threads ({NUM_WARPS} warps)")
    print(f"Grid: ({B}, {T // BT}, {H}) = {B * (T // BT) * H} blocks")
    print("=" * 70)

    src = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    data_bytes = B * T * H * K * 2 * 2  # read + write, bf16 = 2 bytes
    data_gb = data_bytes / 1e9

    # ===== Compile Kernel A (64B) =====
    print("\nCompiling Kernel A (64B per warp)...")
    dst_a = torch.empty_like(src)
    src_ct = from_dlpack(src, assumed_align=16)
    src_ct.element_type = cutlass.BFloat16
    dst_a_ct = from_dlpack(dst_a, assumed_align=16)
    dst_a_ct.element_type = cutlass.BFloat16

    compiled_64B = cute.compile(
        launch_64B, src_ct, dst_a_ct, stream,
        B, H, T,
    )

    # ===== Compile Kernel B (128B) =====
    print("Compiling Kernel B (128B per warp)...")
    dst_b = torch.empty_like(src)
    src_vec2 = src.view(B, T, H, K // 2, 2)
    dst_b_vec2 = dst_b.view(B, T, H, K // 2, 2)
    src_v_ct = from_dlpack(src_vec2, assumed_align=16)
    src_v_ct.element_type = cutlass.BFloat16
    dst_v_ct = from_dlpack(dst_b_vec2, assumed_align=16)
    dst_v_ct.element_type = cutlass.BFloat16

    compiled_128B = cute.compile(
        launch_128B, src_v_ct, dst_v_ct, stream,
        B, H, T,
    )

    # ===== Correctness check =====
    print("\nCorrectness check...")
    dst_a.zero_()
    dst_b.zero_()

    src_ct = from_dlpack(src, assumed_align=16)
    src_ct.element_type = cutlass.BFloat16
    dst_a_ct = from_dlpack(dst_a, assumed_align=16)
    dst_a_ct.element_type = cutlass.BFloat16
    compiled_64B(src_ct, dst_a_ct, stream)

    src_v_ct = from_dlpack(src_vec2, assumed_align=16)
    src_v_ct.element_type = cutlass.BFloat16
    dst_v_ct = from_dlpack(dst_b_vec2, assumed_align=16)
    dst_v_ct.element_type = cutlass.BFloat16
    compiled_128B(src_v_ct, dst_v_ct, stream)
    torch.cuda.synchronize()

    assert torch.equal(src, dst_a), "Kernel A correctness FAILED!"
    assert torch.equal(src, dst_b), "Kernel B correctness FAILED!"
    print("  Both kernels PASSED correctness check.")

    # ===== L2 eviction buffer =====
    evict_buf = torch.empty(int(80 * 1024 * 1024 / 4), dtype=torch.float32, device=device)

    # ===== Benchmark Kernel A (64B) =====
    print(f"\nBenchmarking Kernel A (64B/warp): {test_iters} iterations...")
    for _ in range(warmup_iters):
        src_ct = from_dlpack(src, assumed_align=16)
        src_ct.element_type = cutlass.BFloat16
        dst_a_ct = from_dlpack(dst_a, assumed_align=16)
        dst_a_ct.element_type = cutlass.BFloat16
        compiled_64B(src_ct, dst_a_ct, stream)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    times_a = []
    for _ in range(test_iters):
        _ = evict_buf.sum()
        torch.cuda.synchronize()
        start_ev.record()
        src_ct = from_dlpack(src, assumed_align=16)
        src_ct.element_type = cutlass.BFloat16
        dst_a_ct = from_dlpack(dst_a, assumed_align=16)
        dst_a_ct.element_type = cutlass.BFloat16
        compiled_64B(src_ct, dst_a_ct, stream)
        end_ev.record()
        torch.cuda.synchronize()
        times_a.append(start_ev.elapsed_time(end_ev))

    # ===== Benchmark Kernel B (128B) =====
    print(f"Benchmarking Kernel B (128B/warp): {test_iters} iterations...")
    for _ in range(warmup_iters):
        src_v_ct = from_dlpack(src_vec2, assumed_align=16)
        src_v_ct.element_type = cutlass.BFloat16
        dst_v_ct = from_dlpack(dst_b_vec2, assumed_align=16)
        dst_v_ct.element_type = cutlass.BFloat16
        compiled_128B(src_v_ct, dst_v_ct, stream)
    torch.cuda.synchronize()

    times_b = []
    for _ in range(test_iters):
        _ = evict_buf.sum()
        torch.cuda.synchronize()
        start_ev.record()
        src_v_ct = from_dlpack(src_vec2, assumed_align=16)
        src_v_ct.element_type = cutlass.BFloat16
        dst_v_ct = from_dlpack(dst_b_vec2, assumed_align=16)
        dst_v_ct.element_type = cutlass.BFloat16
        compiled_128B(src_v_ct, dst_v_ct, stream)
        end_ev.record()
        torch.cuda.synchronize()
        times_b.append(start_ev.elapsed_time(end_ev))

    # ===== Results =====
    t_a = torch.tensor(times_a)
    t_b = torch.tensor(times_b)
    mean_a = t_a.mean().item()
    mean_b = t_b.mean().item()
    min_a = t_a.min().item()
    min_b = t_b.min().item()
    bw_a = data_gb / (mean_a / 1000.0)
    bw_b = data_gb / (mean_b / 1000.0)

    print("\n" + "=" * 70)
    print(f"{'Metric':<25} | {'64B/warp (A)':<20} | {'128B/warp (B)':<20}")
    print("-" * 70)
    print(f"{'Mean (ms)':<25} | {mean_a:<20.4f} | {mean_b:<20.4f}")
    print(f"{'Min (ms)':<25} | {min_a:<20.4f} | {min_b:<20.4f}")
    print(f"{'Bandwidth (GB/s)':<25} | {bw_a:<20.1f} | {bw_b:<20.1f}")
    print(f"{'Bytes/warp/transaction':<25} | {'64':<20} | {'128':<20}")
    print(f"{'Sectors/warp/transaction':<25} | {'2':<20} | {'4 (1 cache line)':<20}")
    print(f"{'Data moved (GB)':<25} | {data_gb:<20.3f} | {data_gb:<20.3f}")
    print("=" * 70)
    speedup = mean_a / mean_b
    if speedup > 1.01:
        print(f"128B/warp is {speedup:.2f}x faster than 64B/warp")
    elif speedup < 0.99:
        print(f"64B/warp is {1/speedup:.2f}x faster than 128B/warp")
    else:
        print(f"No significant difference ({speedup:.3f}x)")
    print("=" * 70)


if __name__ == "__main__":
    benchmark()
