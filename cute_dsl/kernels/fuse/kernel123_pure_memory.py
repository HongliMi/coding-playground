"""
Fused K1+K2+K3 Pure Memory Bandwidth Test.

Each block processes 4 chunks with TMA double buffering.
Pure copies through SMEM to measure theoretical GMEM bandwidth.

Grid: (NT/4, H, B)
Block: 256 threads (8 warps)
SMEM: ~96KB (q+k+g × [64,128] bf16 × 2 stages, no swizzle)

Per-chunk SMEM→GMEM copies:
  q → k_scaled, q_scaled  [B,T,H,K] bf16
  q[:,:64] → A_qk         [B,T,H,64] bf16
  k → kg                  [B,T,H,K] bf16
  k[:,:64] → A_kk         [B,T,H,64] bf16
  g[row0] → gk_last_exp   [B,NT,H,K] fp32
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import torch
import time

B200_PEAK_BW_GBS = 7672  # GB/s

BT = 64
K_DIM = 128
K_HALF = 64
CHUNKS_PER_BLOCK = 4
NUM_WARPS = 8
THREADS = NUM_WARPS * 32
ROWS_PER_WARP = BT // NUM_WARPS  # 8
VEC = 4
NUM_STAGES = 2


@cute.kernel
def fused_memory_kernel(
    tma_atom_Q: cute.CopyAtom, tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom, tma_tensor_K: cute.Tensor,
    tma_atom_G: cute.CopyAtom, tma_tensor_G: cute.Tensor,
    mKscaled: cute.Tensor,
    mKg: cute.Tensor,
    mQscaled: cute.Tensor,
    mGkLast: cute.Tensor,
    mAqk: cute.Tensor,
    mAkk: cute.Tensor,
    smem_layout,
    num_chunks: int,
):
    i_cg, i_h, i_b = cute.arch.block_idx()
    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32

    chunk_base = i_cg * CHUNKS_PER_BLOCK

    # =====================================================================
    # SMEM: 3 tensors × [64, 128, 2 stages] bf16, no swizzle
    # Mbarrier array of 2 (one per stage)
    # =====================================================================
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.BFloat16, smem_layout, 128)
    sK = smem.allocate_tensor(cutlass.BFloat16, smem_layout, 128)
    sG = smem.allocate_tensor(cutlass.BFloat16, smem_layout, 128)
    mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)

    bytes_per_stage = BT * K_DIM * 2 * 3

    if tidx == 0:
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_init(mbars + s, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # =====================================================================
    # Prologue: TMA load chunk 0 → stage 0
    # =====================================================================
    if warp_idx == 0:
        if lane_id == 0:
            cute.arch.mbarrier_expect_tx(mbars, bytes_per_stage)

        i_bnt_0 = i_b * num_chunks + chunk_base

        sQ_s0 = sQ[(None, None, 0)]
        gQ = cute.local_tile(tma_tensor_Q, (BT, K_DIM, 1, 1), (0, 0, i_bnt_0, i_h))
        ts, tg = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1),
            cute.group_modes(sQ_s0, 0, 2), cute.group_modes(gQ[(None, None, 0, 0)], 0, 2))
        cute.copy(tma_atom_Q, tg, ts, tma_bar_ptr=mbars)

        sK_s0 = sK[(None, None, 0)]
        gK = cute.local_tile(tma_tensor_K, (BT, K_DIM, 1, 1), (0, 0, i_bnt_0, i_h))
        ts, tg = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1),
            cute.group_modes(sK_s0, 0, 2), cute.group_modes(gK[(None, None, 0, 0)], 0, 2))
        cute.copy(tma_atom_K, tg, ts, tma_bar_ptr=mbars)

        sG_s0 = sG[(None, None, 0)]
        gG = cute.local_tile(tma_tensor_G, (BT, K_DIM, 1, 1), (0, 0, i_bnt_0, i_h))
        ts, tg = cpasync.tma_partition(tma_atom_G, 0, cute.make_layout(1),
            cute.group_modes(sG_s0, 0, 2), cute.group_modes(gG[(None, None, 0, 0)], 0, 2))
        cute.copy(tma_atom_G, tg, ts, tma_bar_ptr=mbars)

        if lane_id == 0:
            cute.arch.mbarrier_arrive(mbars)

    # =====================================================================
    # Mainloop: 4 chunks, double-buffered via stage index
    # =====================================================================
    for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
        stage = chunk_iter % 2       # compile-time: 0,1,0,1
        phase = chunk_iter // 2      # compile-time: 0,0,1,1
        chunk_idx = chunk_base + chunk_iter
        chunk_start = chunk_idx * BT

        if chunk_iter > 0:
            cute.arch.barrier()

        # --- Issue NEXT TMA load (compile-time guarded) ---
        nxt_iter = chunk_iter + 1
        if nxt_iter < CHUNKS_PER_BLOCK:
            nxt_stage = nxt_iter % 2
            nxt_bnt = i_b * num_chunks + chunk_idx + 1

            if warp_idx == 0:
                if lane_id == 0:
                    cute.arch.mbarrier_expect_tx(mbars + nxt_stage, bytes_per_stage)

                sQ_nxt = sQ[(None, None, nxt_stage)]
                gQ = cute.local_tile(tma_tensor_Q, (BT, K_DIM, 1, 1), (0, 0, nxt_bnt, i_h))
                ts, tg = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1),
                    cute.group_modes(sQ_nxt, 0, 2), cute.group_modes(gQ[(None, None, 0, 0)], 0, 2))
                cute.copy(tma_atom_Q, tg, ts, tma_bar_ptr=mbars + nxt_stage)

                sK_nxt = sK[(None, None, nxt_stage)]
                gK = cute.local_tile(tma_tensor_K, (BT, K_DIM, 1, 1), (0, 0, nxt_bnt, i_h))
                ts, tg = cpasync.tma_partition(tma_atom_K, 0, cute.make_layout(1),
                    cute.group_modes(sK_nxt, 0, 2), cute.group_modes(gK[(None, None, 0, 0)], 0, 2))
                cute.copy(tma_atom_K, tg, ts, tma_bar_ptr=mbars + nxt_stage)

                sG_nxt = sG[(None, None, nxt_stage)]
                gG = cute.local_tile(tma_tensor_G, (BT, K_DIM, 1, 1), (0, 0, nxt_bnt, i_h))
                ts, tg = cpasync.tma_partition(tma_atom_G, 0, cute.make_layout(1),
                    cute.group_modes(sG_nxt, 0, 2), cute.group_modes(gG[(None, None, 0, 0)], 0, 2))
                cute.copy(tma_atom_G, tg, ts, tma_bar_ptr=mbars + nxt_stage)

                if lane_id == 0:
                    cute.arch.mbarrier_arrive(mbars + nxt_stage)

        # --- Wait for CURRENT stage ---
        cute.arch.mbarrier_wait(mbars + stage, phase)

        # --- Select current SMEM buffers (compile-time stage index) ---
        csQ = sQ[(None, None, stage)]
        csK = sK[(None, None, stage)]
        csG = sG[(None, None, stage)]

        # --- Process: SMEM → GMEM (coalesced: 32 threads × consecutive cols) ---
        for ri in cutlass.range_constexpr(ROWS_PER_WARP):
            row = warp_idx * ROWS_PER_WARP + ri
            t = chunk_start + row

            for vi in cutlass.range_constexpr(VEC):
                c = vi * 32 + lane_id

                q_val = csQ[row, c]
                mKscaled[i_b, t, i_h, c] = q_val
                mQscaled[i_b, t, i_h, c] = q_val
                if vi < K_HALF // 32:
                    mAqk[i_b, t, i_h, c] = q_val

                k_val = csK[row, c]
                mKg[i_b, t, i_h, c] = k_val
                if vi < K_HALF // 32:
                    mAkk[i_b, t, i_h, c] = k_val

            if ri == 0 and warp_idx == 0:
                for vi in cutlass.range_constexpr(VEC):
                    c = vi * 32 + lane_id
                    g_val = csG[0, c]
                    mGkLast[i_b, chunk_idx, i_h, c] = g_val.to(cutlass.Float32)

    cute.arch.barrier()


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
    def host_fn(mQ, mK, mG, mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk):
        view_layout = cute.make_layout(
            (BT, K_DIM, _BNT, _H),
            stride=(s_row, s_col, s_bnt, s_h),
        )
        mQ_view = cute.make_tensor(mQ.iterator, view_layout)
        mK_view = cute.make_tensor(mK.iterator, view_layout)
        mG_view = cute.make_tensor(mG.iterator, view_layout)

        smem_layout_2d = cute.make_layout((BT, K_DIM), stride=(K_DIM, 1))
        smem_layout_3d = cute.make_layout((BT, K_DIM, NUM_STAGES), stride=(K_DIM, 1, BT * K_DIM))

        tma_op = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_op, mQ_view, smem_layout_2d,
            cute.product_each(smem_layout_2d.shape), num_multicast=1)
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_op, mK_view, smem_layout_2d,
            cute.product_each(smem_layout_2d.shape), num_multicast=1)
        tma_atom_G, tma_tensor_G = cpasync.make_tiled_tma_atom(
            tma_op, mG_view, smem_layout_2d,
            cute.product_each(smem_layout_2d.shape), num_multicast=1)

        smem_size = BT * K_DIM * 2 * 6 + 256

        fused_memory_kernel(
            tma_atom_Q, tma_tensor_Q,
            tma_atom_K, tma_tensor_K,
            tma_atom_G, tma_tensor_G,
            mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk,
            smem_layout_3d, _NT,
        ).launch(
            grid=(_NT // CHUNKS_PER_BLOCK, _H, _B),
            block=(THREADS, 1, 1),
            smem=smem_size,
        )

    return host_fn


# =========================================================================
# Test
# =========================================================================
def test():
    cutlass.cuda.initialize_cuda_context()

    B, H, K = 1, 96, 128
    NT = 128
    T = NT * BT

    print("=" * 60)
    print("Fused K1+K2+K3 Pure Memory Bandwidth Test")
    print("=" * 60)
    smem_kb = BT * K_DIM * 2 * 6 / 1024
    print(f"B={B}, T={T}, H={H}, K={K}, NT={NT}")
    print(f"Grid: ({NT // CHUNKS_PER_BLOCK}, {H}, {B}), Block: {THREADS}")
    print(f"SMEM: {smem_kb:.0f} KB (3 × [{BT}×{K_DIM}] bf16 × {NUM_STAGES} stages, no swizzle)")
    print(f"TMA tile: [{BT}×{K_DIM}] bf16 = {BT*K_DIM*2} bytes")
    print()

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)

    k_scaled = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    kg = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    q_scaled = torch.empty(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    gk_last = torch.empty(B, NT, H, K, device="cuda", dtype=torch.float32)
    A_qk = torch.empty(B, T, H, K_HALF, device="cuda", dtype=torch.bfloat16)
    A_kk = torch.empty(B, T, H, K_HALF, device="cuda", dtype=torch.bfloat16)

    host_fn = make_host_function(B, NT, H)

    dl = from_dlpack
    print("Compiling...")
    compiled = cute.compile(
        host_fn,
        dl(q), dl(k), dl(g),
        dl(k_scaled), dl(kg), dl(q_scaled),
        dl(gk_last), dl(A_qk), dl(A_kk),
    )

    print("Running...")
    compiled(dl(q), dl(k), dl(g), dl(k_scaled), dl(kg), dl(q_scaled),
             dl(gk_last), dl(A_qk), dl(A_kk))
    torch.cuda.synchronize()

    ok = True
    def check(name, got, expected):
        nonlocal ok
        if torch.equal(got, expected):
            print(f"  {name}: PASS")
        else:
            diff = (got.float() - expected.float()).abs().max().item()
            print(f"  {name}: FAIL (max diff={diff})")
            ok = False

    print("\n[Correctness]")
    check("k_scaled == q", k_scaled, q)
    check("q_scaled == q", q_scaled, q)
    check("kg == k", kg, k)
    check("A_qk == q[:,:,:,:64]", A_qk, q[:, :, :, :K_HALF])
    check("A_kk == k[:,:,:,:64]", A_kk, k[:, :, :, :K_HALF])
    gk_ref = g[:, ::BT, :, :].float()
    check("gk_last == g[::64] fp32", gk_last, gk_ref)

    if not ok:
        print("\nSome checks FAILED!")
        return

    num_warmup, num_iters = 20, 200

    for _ in range(num_warmup):
        compiled(dl(q), dl(k), dl(g), dl(k_scaled), dl(kg), dl(q_scaled),
                 dl(gk_last), dl(A_qk), dl(A_kk))
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        compiled(dl(q), dl(k), dl(g), dl(k_scaled), dl(kg), dl(q_scaled),
                 dl(gk_last), dl(A_qk), dl(A_kk))
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) / num_iters * 1e6

    read_bytes = B * T * H * K * 2 * 3
    write_bytes = (B * T * H * K * 2 * 3 +
                   B * NT * H * K * 4 +
                   B * T * H * K_HALF * 2 * 2)
    total_bytes = read_bytes + write_bytes
    bw = total_bytes / (elapsed_us * 1e-6) / 1e9

    print(f"\n[Benchmark] {num_iters} iterations")
    print(f"  Kernel time: {elapsed_us:.2f} us")
    print(f"  Read:  {read_bytes / 1e6:.1f} MB")
    print(f"  Write: {write_bytes / 1e6:.1f} MB")
    print(f"  Total: {total_bytes / 1e6:.1f} MB")
    print(f"  Bandwidth: {bw:.1f} GB/s")
    print(f"  Utilization: {bw / B200_PEAK_BW_GBS * 100:.1f}% of B200 peak ({B200_PEAK_BW_GBS} GB/s)")
    print("=" * 60)


if __name__ == "__main__":
    test()
