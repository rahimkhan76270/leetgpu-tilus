import math
import torch
import pandas as pd
import tilus
from tilus import float16, float32, int32
from tilus.utils import cdiv
from tilus.utils import benchmark_func

try:
    import triton
    import triton.language as tl
except ImportError:
    raise ImportError("Please install triton: pip install triton")


@tilus.autotune("num_warps", [4, 8])
@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128), (32, 256)])
@tilus.autotune("block_k", [16, 32])
@tilus.autotune("num_stages", [3, 4, 5])
@tilus.autotune("split_k_factor", [1, 4, 12, 16])
class MatmulV5(tilus.Script):
    def __init__(
        self,
        num_warps,
        block_m,
        block_n,
        block_k,
        num_stages,
        split_k_factor,
    ):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.split_k_factor = split_k_factor

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        self.attrs.blocks = [
            cdiv(m_size, self.block_m),
            cdiv(n_size, self.block_n),
            self.split_k_factor,
        ]
        self.attrs.warps = self.num_warps

        # the k_size for each thread block
        block_k_size = (
            cdiv(cdiv(k_size, self.split_k_factor), self.block_k) * self.block_k
        )
        start_offset_k = self.blockIdx.z * block_k_size
        end_offset_k = min(start_offset_k + block_k_size, k_size)

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = start_offset_k + stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(
            start_offset_k, end_offset_k, block_k, unroll=self.num_stages
        ):
            # computation for current tile
            a = self.load_shared(sa[current_stage])
            b = self.load_shared(sb[current_stage])
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            if preload_offset_k < end_offset_k:
                self.copy_async(
                    src=ga,
                    dst=sa[preload_stage],
                    offsets=[offset_m, preload_offset_k],
                )
                self.copy_async(
                    src=gb,
                    dst=sb[preload_stage],
                    offsets=[preload_offset_k, offset_n],
                )
            self.copy_async_commit_group()

            # update the stage
            current_stage = (current_stage + 1) % self.num_stages
            preload_stage = (preload_stage + 1) % self.num_stages
            self.copy_async_wait_group(n=self.num_stages - 2)
            self.sync()

        # free the shared memory tensors for A and B
        self.free_shared(sa)
        self.free_shared(sb)

        # cast the accumulator to float16 and change the register tensor's layout
        sc = self.shared_tensor(dtype=float16, shape=[block_m, block_n])
        casted_acc = self.cast(acc, dtype=float16)
        self.store_shared(sc, casted_acc)
        self.sync()
        rc = self.load_shared(sc)
        self.free_shared(sc)

        m_blocks, n_blocks = cdiv(m_size, block_m), cdiv(n_size, block_n)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        if self.split_k_factor == 0:
            self.store_global(gc, rc, offsets=[offset_m, offset_n])
        else:
            semaphores = self.global_tensor(
                dtype=int32, shape=[m_blocks, n_blocks], requires_clean=True
            )
            semaphore: ~int32 = ~semaphores[self.blockIdx.x, self.blockIdx.y]

            # load and accumulate the partial result in global memory
            if self.blockIdx.z > 0:
                self.lock_semaphore(semaphore, value=self.blockIdx.z)
                partial_rc = self.load_global(
                    gc, offsets=[offset_m, offset_n], shape=[block_m, block_n]
                )
                self.add(rc, partial_rc, out=rc)

            # store the result to global memory and release the semaphore
            self.store_global(gc, rc, offsets=[offset_m, offset_n])

            # release the semaphore
            self.sync()  # we need to make sure the previous store_global is finished
            self.release_semaphore(
                semaphore, value=(self.blockIdx.z + 1) % self.split_k_factor
            )


# Triton matmul kernel
@triton.jit
def triton_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
    acc = acc.to(tl.float16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

def triton_matmul(a, b, c):
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N))
    triton_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    workloads = [
        [4096, 4096, 4096],
        [4096, 4096, 14336],
    ]
    rows = []
    for m, n, k in workloads:
        matmul_tilus = MatmulV5()
        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b

        # Correctness check
        matmul_tilus(m, n, k, a, b, c_actual)
        torch.testing.assert_close(c_expect, c_actual)

        c_triton = torch.empty_like(c_actual)
        triton_matmul(a, b, c_triton)
        torch.testing.assert_close(c_expect, c_triton)

        for name, func in [
            ("torch", lambda: torch.matmul(a, b, out=c_expect)),
            ("tilus", lambda: matmul_tilus(m, n, k, a, b, c_actual)),
            ("triton", lambda: triton_matmul(a, b, c_triton)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            tflops = 2 * m * n * k / latency * 1e-9
            rows.append([m, n, k, name, latency, tflops])

    df = pd.DataFrame(rows, columns=headers)
    print(df)

if __name__ == "__main__":
    main()