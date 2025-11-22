from tilus import float16, int32
import tilus
import torch
from tilus.utils import cdiv


class MatrixTranspose(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_x = 16
        self.block_y = 32

    def __call__(self, m_size: int, n_size: int, a_ptr: ~float16, b_ptr: ~float16):
        self.attrs.blocks = [cdiv(m_size, self.block_x), cdiv(n_size, self.block_y)]
        self.attrs.warps = 16

        offset_x: int32 = self.block_x * self.blockIdx.x
        offset_y: int32 = self.block_y * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, n_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[n_size, m_size])

        a_tile = self.load_global(
            ga, shape=[self.block_x, self.block_y], offsets=[offset_x, offset_y]
        )
        b_tile = self.register_tensor(dtype=float16, shape=[self.block_y, self.block_x])

        self.transpose(a_tile, out=b_tile)

        self.store_global(gb, b_tile, offsets=[offset_y, offset_x])


if __name__ == "__main__":
    m = 1000
    n = 1000
    mat_a = torch.rand(size=[m, n], dtype=torch.float16, device="cuda")
    mat_b = torch.empty(size=[n, m], dtype=torch.float16, device="cuda")

    kernel = MatrixTranspose()

    kernel(m, n, mat_a, mat_b)
    torch.cuda.synchronize()

    torch.testing.assert_close(mat_b, mat_a.T, rtol=1e-6, atol=1e-6)
    print("correct")
