import torch
import numpy as np

### Color management and conversion functions
def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        # Clamping is for protection against NaNs during backwards passes.
        ((torch.clamp(image, min=0.04045) + 0.055) / 1.055) ** 2.4
    )

def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        torch.clamp(1.055 * image ** (1 / 2.4) - 0.055, min=0.0, max=1.0)
    )

### Resampling kernels
def nearest(x: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)

    weights = torch.where(x <= 0.5, 1., 0.)

    return weights

def bilinear(x: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)

    weights = torch.where(x <= 1.0, 1 - x, 0.)

    return weights

def mitchell(x: torch.Tensor, B: float = 1 / 3, C: float = 1 / 3) -> torch.Tensor:
    x = torch.abs(x)

    weights = torch.where(x <= 2, (-B - 6 * C) * x**3 + (6 * B + 30 * C) * x**2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C), 0)
    weights = torch.where(x <= 1, (12 - 9 * B - 6 * C) * x**3 + (-18 + 12 * B + 6 * C) * x**2 + (6 - 2 * B), weights)

    return weights

def magic_kernel(x: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)

    weights = torch.where(x <= 1.5, (1/2) * (x - 3/2) ** 2, 0)
    weights = torch.where(x <= 0.5, (3/4) - x ** 2, weights)

    return weights

def magic_kernel_sharp_2013(x: torch.Tensor):
    x = torch.abs(x)

    weights = torch.where(x <= 2.5, (-1/8) * (x - 5/2) ** 2, 0)
    weights = torch.where(x <= 1.5, (1 - x) * (7/4 - x), weights)
    weights = torch.where(x <= 0.5, (17/16) - (7/4) * x ** 2, weights)

    return weights

def magic_kernel_sharp_2021(x: torch.Tensor):
    x = torch.abs(x)

    weights = torch.where(x <= 4.5, (-1/288) * (x - 9/2) ** 2, 0)
    weights = torch.where(x <= 3.5, (1/36) * (x - 3) * (x - 15/4), weights)
    weights = torch.where(x <= 3.5, (1/36) * (x - 3) * (x - 15/4), weights)
    weights = torch.where(x <= 2.5, (1/6) * (x - 2) * (65/24 - x), weights)
    weights = torch.where(x <= 1.5, (35/36) * (x - 1) * (x - 239/140), weights)
    weights = torch.where(x <= 0.5, (577/576) - (239/144) * x ** 2, weights)

    return weights

def lanczos(x: torch.Tensor, n: int):
    return torch.where(torch.abs(x) < n, torch.sinc(x) * torch.sinc(x/n), 0)

### Dithering and related functions.
def stochastic_round(
        x: torch.Tensor,
        out_dtype: torch.dtype,
        generator: torch.Generator = torch.Generator(),
    ):
    image = x * torch.iinfo(out_dtype).max
    image_quant = image.to(out_dtype)
    quant_error = image - image_quant.to(image.dtype)
    dither = torch.empty_like(image_quant, dtype=torch.bool)
    torch.bernoulli(quant_error, generator=generator, out=dither)
    return image_quant + dither

def generate_bayer_matrix(n):
    """Generate an n x n Bayer matrix where n is a power of 2."""
    assert (n & (n - 1)) == 0 and n > 0, "n must be a power of 2"

    if n == 1:
        return np.array([[0]])  # Base case

    smaller_matrix = generate_bayer_matrix(n // 2)
    
    return np.block([
        [4 * smaller_matrix + 0, 4 * smaller_matrix + 2],
        [4 * smaller_matrix + 3, 4 * smaller_matrix + 1]
    ])
