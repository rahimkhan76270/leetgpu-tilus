import tilus
import torch
from tilus.utils import cdiv
from tilus import bfloat16, int32


class VectorAddition(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_x = 256

    def __call__(
        self, m_size: int, a_ptr: ~bfloat16, b_ptr: ~bfloat16, c_ptr: ~bfloat16
    ):
        self.attrs.blocks = [cdiv(m_size, self.block_x)]
        self.attrs.warps = 8

        offset_x: int32 = self.block_x * self.blockIdx.x

        ga = self.global_view(a_ptr, dtype=bfloat16, shape=[m_size])
        gb = self.global_view(b_ptr, dtype=bfloat16, shape=[m_size])
        gc = self.global_view(c_ptr, dtype=bfloat16, shape=[m_size])

        a_tile = self.load_global(ga, offsets=[offset_x], shape=[self.block_x])
        b_tile = self.load_global(gb, offsets=[offset_x], shape=[self.block_x])
        c_tile = self.register_tensor(dtype=bfloat16, shape=[self.block_x])
        self.add(a_tile, b_tile, out=c_tile)

        self.store_global(gc, c_tile, offsets=[offset_x])


if __name__ == "__main__":
    kernel = VectorAddition()
    n = 10000
    a_vector = torch.rand(size=[n], dtype=torch.bfloat16, device="cuda")
    b_vector = torch.rand(size=[n], dtype=torch.bfloat16, device="cuda")
    c_vector = torch.empty(size=[n], dtype=torch.bfloat16, device="cuda")

    kernel(n, a_vector, b_vector, c_vector)
    torch.cuda.synchronize()

    addition_torch = a_vector + b_vector
    torch.cuda.synchronize()

    torch.testing.assert_close(c_vector, addition_torch, atol=1e-5, rtol=1e-5)
    print("correct")
