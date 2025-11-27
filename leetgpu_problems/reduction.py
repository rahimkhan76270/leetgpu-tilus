import tilus
import torch
from tilus import int32


class Reduce(tilus.Script):
    def __init__(self, block=128):
        super().__init__()
        self.block = block

    def __call__(self, m_size: int, a_ptr: ~int32, b_ptr: ~int32):
        # Launch 1 block to compute the sum of the entire array
        self.attrs.blocks = [1]
        self.attrs.warps = 4

        ga = self.global_view(a_ptr, dtype=int32, shape=[m_size])
        gb = self.global_view(b_ptr, dtype=int32, shape=[1])

        # Accumulator register
        r_sum = self.register_tensor(dtype=int32, shape=[self.block], init=0)

        # Loop over the array in chunks of block size
        for offset in range(0, m_size, self.block):
            a_tile = self.load_global(ga, offsets=[offset], shape=[self.block])
            r_sum = r_sum + a_tile

        # Reduce the accumulator to a single value
        total_sum = self.sum(r_sum, dim=0, keepdim=True)

        # Store the result
        self.store_global(gb, total_sum, offsets=[0])


if __name__ == "__main__":
    n = 1024 * 1024  # Large array
    block_size = 1024

    print(f"Reducing array of size {n} with block size {block_size}")

    array_1 = torch.randint(0, 10, size=[n], device="cuda", dtype=torch.int32)
    array_2 = torch.zeros(size=[1], dtype=torch.int32, device="cuda")

    kernel = Reduce(block_size)

    kernel(n, array_1, array_2)
    torch.cuda.synchronize()

    tilus_sum = array_2.item()
    torch_sum = torch.sum(array_1).item()

    print(f"Tilus Sum: {tilus_sum}")
    print(f"Torch Sum: {torch_sum}")

    if tilus_sum == torch_sum:
        print("Success! Results match.")
    else:
        print("Mismatch!")
