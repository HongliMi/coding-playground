"""
tcgen05.mma 用 32 行 SMEM 测试

目标：SMEM A 只分配 32 行，看能否正确计算 M=32 的矩阵乘法
策略：用 PTX 直接调用 tcgen05.mma，但 SMEM 只分配 32 行
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
import torch


# 配置：只计算 32x32x8
M = 32
N = 32
K = 8
THREADS = 128

IO_DTYPE = cutlass.Float32
ACC_DTYPE = cutlass.Float32


# =============================================================================
# PTX 内嵌汇编：tcgen05.mma 和 tcgen05.ld
# =============================================================================

@dsl_user_op
def tcgen05_mma_tf32_ptx(tmem_addr, desc_a, desc_b, *, loc=None, ip=None):
    """用 PTX 调用 tcgen05.mma.cta_group::1.kind::tf32"""
    tmem_val = tmem_addr.value if hasattr(tmem_addr, "value") else tmem_addr
    desc_a_val = desc_a.value if hasattr(desc_a, "value") else desc_a
    desc_b_val = desc_b.value if hasattr(desc_b, "value") else desc_b
    
    zero = llvm.mlir_constant(ir.IntegerAttr.get(T.i32(), 0), loc=loc, ip=ip)
    one = llvm.mlir_constant(ir.IntegerAttr.get(T.i32(), 1), loc=loc, ip=ip)
    
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [tmem_val, desc_a_val, desc_b_val, zero, one, zero, zero, zero, zero],
        """
        {
            .reg .pred p;
            setp.ne.b32 p, $4, 0;
            tcgen05.mma.cta_group::1.kind::tf32 [$0], $1, $2, $3, {$5, $6, $7, $8}, p;
        }
        """,
        "r,l,l,r,r,r,r,r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


@dsl_user_op
def tcgen05_tmem_load_tf32_ptx(tmem_addr, row_idx, col_idx, *, loc=None, ip=None):
    """从 TMEM 指定位置加载 4 个 float32 到寄存器"""
    tmem_val = tmem_addr.value if hasattr(tmem_addr, "value") else tmem_addr
    row_val = row_idx.ir_value(loc=loc, ip=ip)
    col_val = col_idx.ir_value(loc=loc, ip=ip)
    
    # 结果类型：4 个 f32 的结构体
    res_type = ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>")
    
    # 我们直接把 tmem_val (指针) 当做基地址，在 PTX 内部计算偏移
    res = llvm.inline_asm(
        res_type,
        [tmem_val, row_val, col_val],
        """
        {
            .reg .b32 t<4>;
            .reg .b64 row_ptr;
            .reg .b64 row_off;
            
            // 计算行偏移: row_idx * 128 (每个 row 有 32 个 TF32)
            cvt.u64.u32 row_off, $2;
            mul.wide.u32 row_off, $2, 128;
            add.u64 row_ptr, $1, row_off;
            
            // 执行加载
            tcgen05.ld.cta_group::1.kind::tf32.major::mn {t0, t1, t2, t3}, [row_ptr], $3;
            
            // 将结果写回到输出结构体
            mov.b32 $0, t0;
            mov.b32 $1, t1;
            mov.b32 $2, t2;
            mov.b32 $3, t3;
        }
        """,
        "=r,=r,=r,=r,l,r,r", # 4个输出寄存器, 1个64位指针输入, 2个32位整数输入
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    
    # 从结构体中提取结果
    r0 = llvm.extractvalue(ir.FloatType.get_f32(), res, ir.ArrayAttr.get([ir.IntegerAttr.get(T.i64(), 0)]), loc=loc, ip=ip)
    r1 = llvm.extractvalue(ir.FloatType.get_f32(), res, ir.ArrayAttr.get([ir.IntegerAttr.get(T.i64(), 1)]), loc=loc, ip=ip)
    r2 = llvm.extractvalue(ir.FloatType.get_f32(), res, ir.ArrayAttr.get([ir.IntegerAttr.get(T.i64(), 2)]), loc=loc, ip=ip)
    r3 = llvm.extractvalue(ir.FloatType.get_f32(), res, ir.ArrayAttr.get([ir.IntegerAttr.get(T.i64(), 3)]), loc=loc, ip=ip)
    
    return r0, r1, r2, r3


# =============================================================================
# SharedStorage
# =============================================================================

@cute.struct  
class SharedStorage:
    tmem_holding_buf: cutlass.Int32


# =============================================================================
# Kernel：只分配 32x8 SMEM A
# =============================================================================

@cute.kernel
def mma_32row_kernel(
    tiled_mma: cute.TiledMma,
    mA: cute.Tensor,  # [32, 8] 输入
    mB: cute.Tensor,  # [32, 8] 输入
    mC: cute.Tensor,  # [32, 32] 输出
):
    tidx, _, _ = cute.arch.thread_idx()
    
    # 1. 分配 SMEM
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(IO_DTYPE, cute.make_layout((M, K), stride=(K, 1)), 128)
    sB = smem.allocate_tensor(IO_DTYPE, cute.make_layout((N, K), stride=(K, 1)), 128)
    
    # 2. 加载数据
    for i in cutlass.range_constexpr(2): 
        idx = tidx + i * THREADS
        if idx < M * K:
            sA[idx // K, idx % K] = mA[idx // K, idx % K]
        if idx < N * K:
            sB[idx // K, idx % K] = mB[idx // K, idx % K]
    
    cute.arch.sync_threads()
    
    # 3. 分配 TMEM
    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(N)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
    
    # 4. 手动构造 SMEM 描述符 (欺骗指令有 64 行)
    sA_64_layout = cute.make_layout((64, K), stride=(K, 1))
    desc_a = tcgen05.make_umma_smem_desc(sA.iterator, sA_64_layout, "k")
    desc_b = tcgen05.make_umma_smem_desc(sB.iterator, sB.layout, "k")
    
    # 5. 执行 MMA
    tcgen05_mma_tf32_ptx(tmem_ptr, desc_a, desc_b)
    cute.arch.sync_threads()
    
    # 6. 读取结果 (前 32 个线程负责前 32 行)
    if tidx < M:
        row = tidx
        # 32 列，每次读 4 个
        for j_quad in cutlass.range_constexpr(8):
            col = j_quad * 4
            r0, r1, r2, r3 = tcgen05_tmem_load_tf32_ptx(tmem_ptr, cutlass.Int32(row), cutlass.Int32(col))
            mC[row, col + 0] = r0
            mC[row, col + 1] = r1
            mC[row, col + 2] = r2
            mC[row, col + 3] = r3
            
    cute.arch.sync_threads()
    tmem.deallocate()


@cute.jit
def launch_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # 创建 Tiled MMA 用于 API 兼容，虽然我们主要用 PTX
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.TFloat32, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        ACC_DTYPE, tcgen05.CtaGroup.ONE, (64, 32))
    
    mma_32row_kernel(tiled_mma, mA, mB, mC).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def test_32row_mma():
    print("tcgen05.mma PTX Test: SMEM A = 32 rows ONLY")
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(N, K, device='cuda', dtype=torch.float32)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    C_ref = torch.einsum('mk,nk->mn', A, B)
    
    A_cute = from_dlpack(A, assumed_align=128, force_tf32=True).mark_layout_dynamic(leading_dim=1)
    B_cute = from_dlpack(B, assumed_align=128, force_tf32=True).mark_layout_dynamic(leading_dim=1)
    C_cute = from_dlpack(C, assumed_align=128, force_tf32=True).mark_layout_dynamic(leading_dim=1)
    
    try:
        compiled = cute.compile(launch_kernel, A_cute, B_cute, C_cute)
        compiled(A_cute, B_cute, C_cute)
        torch.cuda.synchronize()
        diff = (C - C_ref).abs().max().item()
        print(f"Max diff: {diff:.6f}, C[0,0]: GPU={C[0,0]:.4f}, Ref={C_ref[0,0]:.4f}")
        
        if diff < 0.1:
            print("✅ PASS!")
            return True
        else:
            print("❌ FAIL!")
            return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    from cuda.bindings import driver as cu_driver
    cu_driver.cuInit(0)
    test_32row_mma()
