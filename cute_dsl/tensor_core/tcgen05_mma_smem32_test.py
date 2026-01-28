"""
tcgen05.mma 用 32 行 SMEM 测试

目标：SMEM A 只分配 32x8，然后调用 M=64 的 MMA 指令，看结果。
关键：手动用 PTX 构造 SMEM 描述符，绕过 CuTe 的布局验证。
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
import torch


# =============================================================================
# PTX 封装
# =============================================================================

@dsl_user_op
def make_smem_desc_manual(smem_ptr, stride_bytes, *, loc=None, ip=None):
    """
    手动构造 SMEM 描述符 (SWIZZLE_NONE, K-major)
    """
    ptr_int = smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    stride_val = stride_bytes.ir_value(loc=loc, ip=ip)
    
    res = llvm.inline_asm(
        T.i64(),
        [ptr_int, stride_val],
        """
        {
            .reg .b64 desc, addr, lbo, ver;
            shr.b64 addr, $1, 4;
            and.b64 desc, addr, 0x3FFF;
            cvt.u64.u32 lbo, $2;
            shl.b64 lbo, lbo, 16;
            or.b64 desc, desc, lbo;
            mov.b64 ver, 0x400000000000;
            or.b64 desc, desc, ver;
            mov.b64 $0, desc;
        }
        """,
        "=l,l,r",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return res


@dsl_user_op
def tcgen05_mma_tf32_ptx(tmem_addr, desc_a, desc_b, *, loc=None, ip=None):
    """用 PTX 调用 tcgen05.mma.cta_group::1.kind::tf32"""
    tmem_val = tmem_addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
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
            .reg .b32 tmem32;
            cvt.u32.u64 tmem32, $0;
            setp.ne.b32 p, $4, 0;
            tcgen05.mma.cta_group::1.kind::tf32 [tmem32], $1, $2, $3, {$5, $6, $7, $8}, p;
        }
        """,
        "l,l,l,r,r,r,r,r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


@dsl_user_op
def tcgen05_ld_16x32b_ptx(tmem_addr, *, loc=None, ip=None):
    """
    从 TMEM 加载 16x32b 的数据（每线程 1 个 f32）
    tcgen05.ld.sync.aligned.16x32b.x1.b32 每个 warp 加载 16 行，每行 1 个 32-bit
    """
    tmem_val = tmem_addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    
    res = llvm.inline_asm(
        T.f32(),
        [tmem_val],
        """
        {
            .reg .b32 tmem32;
            cvt.u32.u64 tmem32, $1;
            tcgen05.ld.sync.aligned.16x32b.x1.b32 $0, [tmem32];
        }
        """,
        "=f,l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return res


# =============================================================================
# Kernel
# =============================================================================

@cute.struct
class SharedStorage:
    tmem_buf: cutlass.Int32


@cute.kernel
def mma_32row_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    
    # 1. SMEM 分配 - 只有 32x8！
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    
    SMEM_A_BYTES = 32 * 8 * 4
    SMEM_B_BYTES = 32 * 8 * 4
    
    ptr_sA_raw = smem.allocate(SMEM_A_BYTES, 128)
    ptr_sB_raw = smem.allocate(SMEM_B_BYTES, 128)
    
    ptr_sA = cute.recast_ptr(ptr_sA_raw, dtype=cutlass.TFloat32)
    ptr_sB = cute.recast_ptr(ptr_sB_raw, dtype=cutlass.TFloat32)
    
    layout_32x8 = cute.make_layout((32, 8), stride=(8, 1))
    sA = cute.make_tensor(ptr_sA, layout_32x8)
    sB = cute.make_tensor(ptr_sB, layout_32x8)
    
    # 2. 加载数据到 SMEM (32x8)
    for i in cutlass.range_constexpr(2):
        idx = tidx + i * 128
        if idx < 32 * 8:
            row = idx // 8
            col = idx % 8
            sA[row, col] = mA[row, col]
            sB[row, col] = mB[row, col]
    
    cute.arch.sync_threads()
    
    # Step 1: TMEM 分配
    barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128)
    tmem = utils.TmemAllocator(storage.tmem_buf, barrier_for_retrieve=barrier)
    tmem.allocate(64)  # 64 列
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    
    # Step 2: SMEM 描述符构造
    stride_A = cutlass.Int32(32)  # 8 cols * 4 bytes
    stride_B = cutlass.Int32(32)
    desc_a = make_smem_desc_manual(ptr_sA_raw, stride_A)
    desc_b = make_smem_desc_manual(ptr_sB_raw, stride_B)
    
    # Step 3: 执行 MMA (M=64, N=32, K=8) - 但 SMEM 只有 32 行！
    tcgen05_mma_tf32_ptx(tmem_ptr, desc_a, desc_b) # 加上这句话就有内存访问错误！
    
    cute.arch.sync_threads()
    
    # 从 TMEM 读取结果（需要先执行 MMA）
    # tcgen05.ld 是 warp 级操作，每个 warp 的线程分别读取对应数据
    # 注意：如果 MMA 没执行，读出来的是未定义值
    # val = tcgen05_ld_16x32b_ptx(tmem_ptr)
    
    # 暂时先写入一个简单位置验证读取是否成功
    if tidx < 32:
        row = tidx
        mC[row, 0] = sA[row, 0]  # 每个线程把读到的值写入 C 的第一列
    
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def launch_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    mma_32row_kernel(mA, mB, mC).launch(grid=(1, 1, 1), block=(128, 1, 1))


def test():
    print("=" * 60)
    print("tcgen05.mma PTX 测试：SMEM A = 32x8 (物理分配)")
    print("MMA 指令要求 M=64，我们只分配 32 行看看会发生什么")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    A = torch.randn(32, 8, device='cuda', dtype=torch.float32)
    B = torch.randn(32, 8, device='cuda', dtype=torch.float32)
    C = torch.zeros(32, 32, device='cuda', dtype=torch.float32)
    
    C_ref = A @ B.T
    
    A_cute = from_dlpack(A, assumed_align=128, force_tf32=True).mark_layout_dynamic(leading_dim=1)
    B_cute = from_dlpack(B, assumed_align=128, force_tf32=True).mark_layout_dynamic(leading_dim=1)
    C_cute = from_dlpack(C, assumed_align=128, force_tf32=True).mark_layout_dynamic(leading_dim=1)
    
    try:
        compiled = cute.compile(launch_kernel, A_cute, B_cute, C_cute)
        compiled(A_cute, B_cute, C_cute)
        torch.cuda.synchronize()
        
        diff = (C - C_ref).abs().max().item()
        print(f"\n结果:")
        print(f"  Max diff: {diff:.6f}")
        print(f"  C[0,0]: GPU={C[0,0]:.4f}, Ref={C_ref[0,0]:.4f}")
        print(f"  C[31,31]: GPU={C[31,31]:.4f}, Ref={C_ref[31,31]:.4f}")
        
        if diff < 0.1:
            print("\n✓ 成功！")
        else:
            print("\n✗ 结果不正确")
            
        return diff < 0.1
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    from cuda.bindings import driver as cu_driver
    cu_driver.cuInit(0)
    test()
