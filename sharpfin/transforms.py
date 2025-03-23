import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Transform
from . import functional as SFF
from .cms import apply_srgb
import math
from typing import Any, Dict, Tuple
from PIL import Image
from .functional import ResizeKernel, SharpenKernel, QuantHandling, _get_resize_kernel

__all__ = ["ResizeKernel", "SharpenKernel", "QuantHandling"]

class Scale(Transform):
    """Rescaling transform that supports multiple scaling algorithms and provides appropriate options for output data types.

    The input should be a `torch.Tensor` of shape `[B, C, H, W]` or a `TVTensor`. Images are assumed to be in the sRGB color space.

    Args:
        out_res (tuple[int, int], int): Output resolution of the transform. [H, W]. Single int input will be used as resolution for both axes.
        device (torch.device): The device to perform computations on. Defaults to CUDA if available, otherwise CPU.
        dtype (torch.dtype): The data type for computations.
            - `torch.float16` has plenty enough accuracy and is fast, and is recommended for CUDA backends.
            - `torch.float32` has more than enough accuracy, and is recommended for CPU backends (because most CPUs do not have FPUs that handle float16)
            - `torch.bfloat16` is equally as fast as float16 on CUDA but is inaccurate. Only use if it is the only reasonably performant one for your backend.
            - `torch.float8` formats are not allowed due to severe accuracy issues.
            - `torch.float64` is supported but generally unnecessary.

        out_dtype (torch.dtype, optional): The data type of the output image. Accepts any floating-point or unsigned integer `torch.dtype`. Defaults to the same as `dtype`.
            - A floating-point format is recommended for images intended for model input.
            - Use an unsigned integer format only when saving images.
            - When using a uint format, output quantization is required.

        quantization (QuantHandling, optional): The quantization/dithering method when converting to an integer format. Has no effect if `out_dtype` is floating-point or `None`.
            - `QuantHandling.TRUNCATE`: Default PyTorch behavior; always rounds down. Not recommended.
            - `QuantHandling.ROUND`: Default option; rounds to the nearest integer without dithering. Can cause color banding.
            - `QuantHandling.STOCHASTIC_ROUND`: Uses stochastic rounding by sampling a Bernoulli distribution based on quantization error. May introduce noise that reduces detail.
            - `QuantHandling.BAYER`: Uses a tiled Bayer matrix for dithering. Can create a grid-like pattern in smooth gradient areas.

        generator (torch.Generator, optional): RNG generator used for stochastic rounding quantization.

        resize_kernel (ResizeKernel): The resampling kernel to use.
            - `ResizeKernel.NEAREST`: Nearest-neighbor interpolation.
            - `ResizeKernel.BILINEAR`: Bilinear interpolation.
            - `ResizeKernel.CATMULL_ROM`: Bicubic resampling using the Catmull-Rom spline (B=0, C=0.5).
            - `ResizeKernel.MITCHELL`: Bicubic resampling using the Mitchell-Netravali spline (B=1/3, C=1/3).
            - `ResizeKernel.B_SPLINE`: Bicubic resampling using the B-spline (B=1, C=0).
            - `ResizeKernel.LANCZOS2`: Lanczos resampling with a kernel size of 2.
            - `ResizeKernel.LANCZOS3`: Lanczos resampling with a kernel size of 3.
            - `ResizeKernel.MAGIC_KERNEL`: Magic Kernel resampling (https://johncostella.com/magic/). Offers excellent anti-aliasing but may introduce excessive blurring.
            - `ResizeKernel.MAGIC_KERNEL_SHARP_2013`: Magic Kernel combined with a sharpening filter.
            - `ResizeKernel.MAGIC_KERNEL_SHARP_2021`: Default and recommended resampling algorithm. Provides superior quality compared to Lanczos-3.
                More details: https://johncostella.com/magic/mks.pdf

        sharpen_kernel (SharpenKernel, optional): A sharpening kernel, primarily used with `ResizeKernel.MAGIC_KERNEL`.
            - Options: `SharpenKernel.SHARP_2013` or `SharpenKernel.SHARP_2021`.
            - Typically, using `MAGIC_KERNEL_SHARP_2013` or `MAGIC_KERNEL_SHARP_2021` directly is preferable.
            - This separate sharpening step is mostly retained for testing purposes against the reference Magic Kernel Sharp implementation.
            - Unlike the combined kernel approach, separating the sharpening step is less efficient, especially since resampling is already performed as a large sparse matrix multiplication.

        do_srgb_conversion (bool, optional): Whether to convert images from sRGB to linear RGB before resampling. Defaults to `True`.
            - This is recommended, as rescaling in a linear color space produces correct results.
            - Only disable if your images are not in the sRGB color space.
    """
    _transformed_types = (torch.Tensor,)
    def __init__(self, 
        out_res: tuple[int, int] | int,
        device: torch.device | str = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        out_dtype: torch.dtype | None = None,
        quantization: QuantHandling = QuantHandling.ROUND,
        generator: torch.Generator | None = None,
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        sharpen_kernel: SharpenKernel | None = None,
        do_srgb_conversion: bool = True,
    ):
        super().__init__()

        if not dtype.is_floating_point: 
            raise ValueError("dtype must be a floating point type")
        if dtype.itemsize == 1: 
            raise ValueError("float8 types are not supported due to severe accuracy issues and limited function support. float16 or float32 is recommended.")
        if dtype == torch.float16 and device in [torch.device('cpu'), 'cpu']: 
            print("Warning: float16 will most likely perform poorly on most CPUs. float32 is recommended for CPU backends.")
        if dtype == torch.bfloat16: 
            print("Warning: bfloat16 can be expected to lead to noticeable accuracy issues. float16 or float32 is recommended.")
        if out_dtype is not None and not out_dtype.is_floating_point and out_dtype not in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
            raise ValueError("out_dtype must be a torch float format or a torch unsigned int format")

        if isinstance(out_res, int):
            out_res = (out_res, out_res)
        self.out_res = out_res
        self.device = device
        self.dtype = dtype
        self.out_dtype = out_dtype if out_dtype is not None else dtype
        self.do_srgb_conversion = do_srgb_conversion

        if self.out_dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
            match quantization:
                case QuantHandling.TRUNCATE:
                    self.quantize_function = lambda x: x.mul(torch.iinfo(self.out_dtype).max).to(self.out_dtype)
                case QuantHandling.ROUND:
                    self.quantize_function = lambda x: x.mul(torch.iinfo(self.out_dtype).max).round().to(self.out_dtype)
                case QuantHandling.STOCHASTIC_ROUND:
                    if generator is not None:
                        self.generator = torch.Generator(self.device)
                    else:
                        self.generator = generator
                    self.quantize_function = lambda x: SFF.stochastic_round(x, self.out_dtype, self.generator)
                case QuantHandling.BAYER:
                    self.bayer_matrix = torch.tensor(SFF.generate_bayer_matrix(16), dtype=self.dtype, device=self.device) / 255
                    self.quantize_function = lambda x: self.apply_bayer_matrix(x)
                case _:
                    raise ValueError(f"Unknown quantization handling type {quantization}")
        else:
            self.quantize_function = lambda x: x

        self.resize_kernel, self.kernel_window = _get_resize_kernel(resize_kernel)

        match sharpen_kernel:
            case SharpenKernel.SHARP_2013:
                kernel = torch.tensor([-1, 6, -1], dtype=dtype, device=device) / 4
                self.sharp_2013_kernel = torch.outer(kernel, kernel).view(1, 1, 3, 3).expand(3, -1, -1, -1)
                self.sharpen_step = lambda x: SFF.sharpen_conv2d(x, self.sharp_2013_kernel, 1)
            case SharpenKernel.SHARP_2021:
                kernel = torch.tensor([-1, 6, -35, 204, -35, 6, -1], dtype=dtype, device=device) / 144
                self.sharp_2021_kernel = torch.outer(kernel, kernel).view(1, 1, 7, 7).expand(3, -1, -1, -1)
                self.sharpen_step = lambda x: SFF.sharpen_conv2d(x, self.sharp_2021_kernel, 3)
            case None:
                self.sharpen_step = lambda x: x
            case _:
                raise ValueError(f"Unknown sharpen kernel {sharpen_kernel}")

    def apply_bayer_matrix(self, x: torch.Tensor):
        H, W = x.shape[-2:]
        b = self.bayer_matrix.repeat(1,1,math.ceil(H/16),math.ceil(W/16))[:,:,:H,:W]
        return (x*255 + b).to(self.out_dtype)

    @torch.compile(disable=False)
    def downscale(self, image: torch.Tensor, out_res: tuple[int, int]):
        H, W = out_res
        image = image.to(device=self.device, dtype=self.dtype)
        if self.do_srgb_conversion:
            image = SFF.srgb_to_linear(image)

        image = SFF._downscale_axis(image, W, self.kernel_window, self.resize_kernel, self.device, self.dtype)
        image = SFF._downscale_axis(image.mT, H, self.kernel_window, self.resize_kernel, self.device, self.dtype).mT

        image = self.sharpen_step(image)

        if self.do_srgb_conversion:
            image = SFF.linear_to_srgb(image)
        image = image.clamp(0,1)
        image = self.quantize_function(image)
        return image

    @torch.compile(disable=False)
    def upscale(self, image: torch.Tensor, out_res: tuple[int, int]):
        H, W = out_res
        image = image.to(device=self.device, dtype=self.dtype)
        if self.do_srgb_conversion:
            image = SFF.srgb_to_linear(image)

        image = self.sharpen_step(image)

        image = SFF._upscale_axis(image, W, self.kernel_window, self.resize_kernel, self.device, self.dtype)
        image = SFF._upscale_axis(image.mT, H, self.kernel_window, self.resize_kernel, self.device, self.dtype).mT

        if self.do_srgb_conversion:
            image = SFF.linear_to_srgb(image)
        image = image.clamp(0,1)
        image = self.quantize_function(image)
        return image

    def _transform(self, inpt: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        image = inpt
        if image.shape[-1] <= self.out_res[-1] and image.shape[-2] <= self.out_res[-2]:
            return self.upscale(image, self.out_res)
        elif image.shape[-1] >= self.out_res[-1] and image.shape[-2] >= self.out_res[-2]:
            return self.downscale(image, self.out_res)
        else:
            raise ValueError("Mixed axis resizing (e.g. scaling one axis up and the other down) is not supported. File a bug report with your use case if needed.")


class ApplyCMS(Transform):
    """Apply color management to a PIL Image to standardize it to sRGB color space.

    This transform does not support torchscript.

    Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    """
    _transformed_types = (Image.Image,)

    def _transform(self, inpt: Image.Image, params: Dict[str, Any]) -> Image.Image:
        if not isinstance(inpt, Image.Image):
            raise TypeError(f"inpt should be PIL Image. Got {type(inpt)}")

        return apply_srgb(inpt)

class AlphaComposite(Transform):
    _transformed_types = (Image.Image,)
    def __init__(
        self,
        background: Tuple[int,int,int] = (255, 255, 255)
    ):
        super().__init__()
        self.background = background

    def _transform(self, inpt: Image.Image, params: Dict[str, Any]) -> Image.Image:
        if not isinstance(inpt, Image.Image):
            raise TypeError(f"inpt should be PIL Image. Got {type(inpt)}")
        if not inpt.has_transparency_data:
            return inpt

        bg = Image.new("RGB", inpt.size, self.background).convert('RGBa')
        return Image.alpha_composite(bg, inpt)
