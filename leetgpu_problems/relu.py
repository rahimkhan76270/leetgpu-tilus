import tilus
import torch
from tilus.utils import cdiv
from tilus import float16


class ReLU(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_x = 1

    def __call__(self, n_size: int, a_ptr: ~float16, b_ptr: ~float16):
        self.attrs.blocks = [cdiv(n_size, self.block_x)]
        self.attrs.warps = 1

        offset = self.block_x * self.blockIdx.x

        ga = self.global_view(a_ptr, dtype=float16, shape=[n_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[n_size])

        a_tile = self.load_global(ga, offsets=[offset], shape=[self.block_x])
        zero_tile = self.register_tensor(dtype=float16, shape=[self.block_x], init=0.0)

        # a_tile = self.maximum(a_tile, zero_tile)
        mask = a_tile < 0
        # self.print_tensor(" ", mask)
        a_tile = self.where(mask, zero_tile, a_tile)
        self.store_global(gb, a_tile, offsets=[offset])


if __name__ == "__main__":
    n = 10
    array_1 = torch.rand(size=[n], dtype=torch.float16, device="cuda")
    array_2 = torch.empty_like(array_1)

    kernel = ReLU()
    kernel(n, array_1, array_2)
    torch.cuda.synchronize()
    print(array_1)
    print(array_2)
