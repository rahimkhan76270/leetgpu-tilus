import tilus
from tilus import float16, int32
from tilus.utils import cdiv
import torch
import math
from tilus.utils import benchmark_func
import pandas


class MatrixAdd(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 64
        self.block_n = 64

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        self.attrs.blocks = [
            cdiv(m_size, self.block_m),
            cdiv(n_size, self.block_n),
        ]
        self.attrs.warps = 4

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, n_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[m_size, n_size])
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])

        # Load tiles
        a = self.load_global(
            ga, offsets=[offset_m, offset_n], shape=[self.block_m, self.block_n]
        )
        b = self.load_global(
            gb, offsets=[offset_m, offset_n], shape=[self.block_m, self.block_n]
        )

        # Add
        c = self.register_tensor(dtype=float16, shape=[self.block_m, self.block_n])
        self.add(a, b, out=c)

        # Store
        self.store_global(gc, c, offsets=[offset_m, offset_n])


def main():
    headers = ["m", "n", "name", "latency (ms)", "tflops"]
    workloads = [[4096, 4096]]

    rows = []
    for m, n in workloads:
        kernel = MatrixAdd()

        a = torch.rand(m, n, dtype=torch.float16).cuda() - 0.5
        b = torch.rand(m, n, dtype=torch.float16).cuda() - 0.5
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a + b
        torch.cuda.synchronize()

        # launch
        kernel(m, n, a, b, c_actual)
        torch.cuda.synchronize()

        # check correctness
        torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)
        print("Correctness check passed!")

        # benchmark
        for name, func in [
            ("torch", lambda: torch.add(a, b, out=c_expect)),
            ("tilus", lambda: kernel(m, n, a, b, c_actual)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            # TFLOPS for addition is roughly (m*n) / latency? No, usually 1 FLOP per element.
            # But let's just stick to latency or simple throughput.
            # Actually, let's just print latency.
            # If we want TFLOPS, it's m*n / latency / 1e9 / 1000 (if ms) -> m*n / latency * 1e-9 * 1000?
            # Let's just use GB/s for bandwidth bound ops usually, but TFLOPS is fine for consistency with other examples if adapted.
            # Addition is 1 FLOP per element.
            tflops = m * n / latency * 1e-9
            rows.append([m, n, name, latency, tflops])

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


if __name__ == "__main__":
    main()
