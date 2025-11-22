import tilus
import torch
from tilus.utils import cdiv
from tilus import int32, uint8


class ColorInversion(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_x = 1

    def __call__(self, height: int, width: int, img_ptr: ~uint8, inv_img_ptr: ~uint8):
        self.attrs.blocks = [cdiv(height * width * 4, self.block_x)]
        self.attrs.warps = 1

        offset_x: int32 = self.block_x * self.blockIdx.x

        img = self.global_view(img_ptr, dtype=uint8, shape=[height * width * 4])
        inv_img = self.global_view(inv_img_ptr, dtype=uint8, shape=[height * width * 4])

        img_tile = self.load_global(img, offsets=[offset_x], shape=[self.block_x])

        inv_tile = img_tile
        if offset_x == 0 or offset_x % 3 != 0:
            inv_tile = (
                self.register_tensor(dtype=uint8, shape=[self.block_x], init=255)
                - inv_tile
            )

        self.store_global(inv_img, inv_tile, offsets=[offset_x])


def reference_impl(image: torch.Tensor, width: int, height: int):
    assert image.shape == (height * width * 4,)
    assert image.dtype == torch.uint8

    # Reshape to (height, width, 4) for easier processing
    image_reshaped = image.view(height, width, 4)

    # Invert RGB channels (first 3 channels), keep alpha unchanged
    image_reshaped[:, :, :3] = 255 - image_reshaped[:, :, :3]
    return image_reshaped.flatten()


if __name__ == "__main__":
    height = 5
    width = 2
    img1 = torch.randint(
        0, 256, size=(height * width * 4,), dtype=torch.uint8, device="cuda"
    )
    inv_img1 = torch.empty_like(img1)
    torch_solve = reference_impl(img1, width, height)
    kernel = ColorInversion()
    kernel(height, width, img1, inv_img1)
    torch.cuda.synchronize()
    print(img1)
    print(inv_img1)
    print(torch_solve)
    torch.testing.assert_close(inv_img1, torch_solve, atol=1e-5, rtol=1e-5)
    print("correct")
