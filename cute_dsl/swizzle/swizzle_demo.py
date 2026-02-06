"""
Swizzle 地址映射
make_swizzle(B, M, S):
    swizzle(addr) = addr XOR (((addr >> S) & ((1 << B) - 1)) << M)
"""

def make_swizzle(B, M, S):
    """
    创建一个 swizzle 函数
    
    Args:
        B: 参与 XOR 的位数
        M: XOR 目标的起始位置
        S: XOR 源的起始位置
    
    Returns:
        swizzle 函数
    """
    mask = (1 << B) - 1
    
    def swizzle(addr):
        # 取 bit[M+S+B-1:M+S]，XOR 到 bit[M+B-1:M]
        xor_bits = (addr >> (M + S)) & mask
        return addr ^ (xor_bits << M)
    
    return swizzle


def show_mapping(B, M, S, num_elements=64, element_size=2):
    """显示逻辑地址到物理地址的映射
    
    Args:
        B, M, S: swizzle 参数
        num_elements: 元素个数
        element_size: 每个元素的字节数（默认2字节）
    """
    swizzle = make_swizzle(B, M, S)
    mask = (1 << B) - 1
    
    print(f"make_swizzle(B={B}, M={M}, S={S})")
    print(f"公式: addr XOR (((addr >> {S}) & {mask}) << {M})")
    print(f"即: 取 bits[{S+B-1}:{S}], XOR 到 bits[{M+B-1}:{M}]")
    print(f"元素大小: {element_size} 字节")
    print()
    print(f"{'id':>4} | {'逻辑':>6} | {'物理':>6} | {'bank':>4}")
    print("-" * 32)
    
    for i in range(num_elements):
        addr = i * element_size
        phys = swizzle(addr)
        bank = (phys // 4) % 32
        print(f"{i:>4} | {addr:>6} | {phys:>6} | {bank:>4}")
    
    print("bank:")
    for i in range(num_elements):
        if i % 32 == 0:
            j = i // 32
            print()
            print(f"row={j}", end="")
        addr = i * element_size
        phys = swizzle(addr)
        bank = (phys // 4) % 32
        print(f"{bank:>4}", end="")
    print()


if __name__ == "__main__":
    # 修改这三个参数来测试不同的 swizzle 配置
    B = 3  # 参与 XOR 的位数
    M = 4  # XOR 目标起始位置
    S = 3  # XOR 源起始位置
    
    show_mapping(B, M, S, num_elements=8*32, element_size=4)
