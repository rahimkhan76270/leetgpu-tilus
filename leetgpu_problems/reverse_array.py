import tilus
from tilus.utils import cdiv
import torch
from tilus import float16


class ReverseArray(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_x = 1

    def __call__(self, m_size: int, a_ptr: ~float16, b_ptr: ~float16):
        self.attrs.blocks = [cdiv(m_size, self.block_x)]
        self.attrs.warps = 1

        offset = self.block_x * self.blockIdx.x

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[m_size])

        a_tile = self.load_global(ga, offsets=[offset], shape=[self.block_x])

        self.store_global(gb, a_tile, offsets=[m_size - offset - 1])


if __name__ == "__main__":
    arr_size = 10
    arr = torch.rand(size=[arr_size], dtype=torch.float16, device="cuda")
    brr = torch.empty_like(arr)
    kernel = ReverseArray()

    kernel(arr_size, arr, brr)
    torch.cuda.synchronize()
    print(arr)
    print(brr)
