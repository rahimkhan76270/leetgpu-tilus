import tilus
import torch
from tilus.utils import cdiv
from tilus import float32, int32


class Convolution1D(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_x = 1

    def __call__(
        self,
        out_size: int,
        kernel_size: int,
        arr_size: int,
        arr_ptr: ~float32,
        kernel_ptr: ~float32,
        out_ptr: ~float32,
    ):
        self.attrs.blocks = [cdiv(out_size, self.block_x)]
        self.attrs.warps = 1

        offset: int32 = self.block_x * self.blockIdx.x

        arr = self.global_view(arr_ptr, dtype=float32, shape=[arr_size])
        kernel = self.global_view(kernel_ptr, dtype=float32, shape=[kernel_size])
        out_arr = self.global_view(out_ptr, dtype=float32, shape=[out_size])

        accum = self.register_tensor(dtype=float32, shape=[self.block_x], init=0.0)

        for i in range(kernel_size):
            a_tile = self.load_global(arr, offsets=[offset + i], shape=[self.block_x])
            k_tile = self.load_global(kernel, offsets=[i], shape=[self.block_x])
            accum = accum + a_tile * k_tile

        self.store_global(out_arr, accum, offsets=[offset])


if __name__ == "__main__":
    array_size = 4
    k_size = 2
    o_size = array_size - k_size + 1
    array = torch.tensor([2, 4, 6, 8], dtype=torch.float32, device="cuda")
    kernel_ = torch.tensor([0.5, 0.2], dtype=torch.float32, device="cuda")
    output = torch.empty(size=[o_size], dtype=torch.float32, device="cuda")

    conv1d_kernel = Convolution1D()
    conv1d_kernel(o_size, k_size, array_size, array, kernel_, output)
    torch.cuda.synchronize()
    print(output)
