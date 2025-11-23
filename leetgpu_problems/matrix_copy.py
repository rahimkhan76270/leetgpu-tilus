import tilus
import torch
from tilus.utils import cdiv
from tilus import float16


class MatrixCopy(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_x = 32
        self.block_y = 16

    def __call__(self, m_size: int, n_size: int, a_ptr: ~float16, b_ptr: ~float16):
        self.attrs.blocks = [cdiv(m_size, self.block_x), cdiv(n_size, self.block_y)]
        self.attrs.warps = 16

        offset_x = self.block_x * self.blockIdx.x
        offset_y = self.block_y * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, n_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[m_size, n_size])

        a_tile = self.load_global(
            ga, offsets=[offset_x, offset_y], shape=[self.block_x, self.block_y]
        )
        self.store_global(gb, a_tile, offsets=[offset_x, offset_y])


if __name__ == "__main__":
    m = 5
    n = 4
    array_1 = torch.rand(size=[m, n], dtype=torch.float16, device="cuda")
    array_2 = torch.empty_like(array_1)

    kernel = MatrixCopy()
    kernel(m,n, array_1, array_2)
    torch.cuda.synchronize()
    torch.testing.assert_close(array_1, array_2, rtol=1e-6, atol=1e-6)
    print("correct")
