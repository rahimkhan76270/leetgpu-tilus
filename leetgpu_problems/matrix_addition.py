import tilus
from tilus.utils import cdiv
from tilus import float16, int32
import torch


class Matrixaddition(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_x = 16
        self.block_y = 16

    def __call__(
        self,
        m_size: int,
        n_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        self.attrs.warps = 1
        self.attrs.blocks = [cdiv(m_size, self.block_x), cdiv(n_size, self.block_y)]

        offset_x = self.block_x * self.blockIdx.x
        offset_y = self.block_y * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, n_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[m_size, n_size])
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])

        a_tile = self.load_global(
            ga, offsets=[offset_x, offset_y], shape=[self.block_x, self.block_y]
        )
        b_tile = self.load_global(
            gb, offsets=[offset_x, offset_y], shape=[self.block_x, self.block_y]
        )
        acc = self.register_tensor(shape=[self.block_x, self.block_y], dtype=float16)
        self.add(a_tile, b_tile, out=acc)

        self.store_global(gc, acc, offsets=[offset_x, offset_y])


if __name__ == "__main__":
    m_size = 100
    n_size = 250
    a_matrix = torch.rand(size=[m_size, n_size], dtype=torch.float16, device="cuda")
    b_matrix = torch.rand(size=[m_size, n_size], dtype=torch.float16, device="cuda")
    c_matrix = torch.empty(size=[m_size, n_size], dtype=torch.float16, device="cuda")
    kernel = Matrixaddition()

    kernel(m_size, n_size, a_matrix, b_matrix, c_matrix)
    torch.cuda.synchronize()

    actual_torch = a_matrix + b_matrix
    torch.cuda.synchronize()

    torch.testing.assert_close(c_matrix, actual_torch, atol=1e-5, rtol=1e-5)
    print("correct")
