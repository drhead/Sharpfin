import torch
import math
import triton
import triton.language as tl

from sharpfin.util import ResizeKernel
from typing import Tuple
import torch.nn.functional as F

from sharpfin.util import linear_to_srgb, srgb_to_linear


# Magic Kernel Sharp modified to operate in-place on negative values
# For some reason doing it this way is slightly faster, but less readable.
@triton.jit
def magic_kernel_sharp_2021_triton(x):
    x = -tl.abs(x)
    x = tl.where(x >= -0.5, (577/576) - (239/144) * (x * x), x)
    x = tl.where(x < -0.5 and x >= -1.5, (35/36) * (x + 1) * (x + 239/140), x)
    x = tl.where(x < -1.5 and x >= -2.5, -(1/6) * (x + 2) * (65/24 + x), x)
    x = tl.where(x < -2.5 and x >= -3.5, (1/36) * (x + 3) * (x + 15/4), x)
    x = tl.where(x < -3.5 and x >= -4.5, -(1/288) * ((x + 9/2) * (x + 9/2)), x)
    x = tl.where(x < -4.5, 0, x)
    return x

@triton.jit
def linear_to_srgb_triton(x):
    return tl.where(x <= 0.0031308, 
                    x * 12.92, 
                    1.055 * tl.math.exp(tl.math.log(tl.cast(x, tl.float32)) / 2.4) - 0.055)

@triton.jit
def srgb_to_linear_triton(x):
    return tl.where(x <= 0.04045, 
                    x / 12.92, 
                    tl.math.exp(tl.math.log(tl.cast((x + 0.055) / 1.055, tl.float32)) * 2.4))

from sharpfin.sparse_backend import dds, Matrix

def _get_resize_kernel_triton(k: ResizeKernel):
    match k:
        case ResizeKernel.NEAREST:
            raise NotImplementedError
            kernel_window = 1.
        case ResizeKernel.BILINEAR:
            raise NotImplementedError
            kernel_window = 1.
        case ResizeKernel.MITCHELL:
            raise NotImplementedError
            kernel_window = 2.
        case ResizeKernel.CATMULL_ROM:
            raise NotImplementedError
            kernel_window = 2.
        case ResizeKernel.B_SPLINE:
            raise NotImplementedError
            kernel_window = 2.
        case ResizeKernel.LANCZOS2:
            raise NotImplementedError
            kernel_window = 2.
        case ResizeKernel.LANCZOS3:
            raise NotImplementedError
            kernel_window = 3.
        case ResizeKernel.MAGIC_KERNEL:
            raise NotImplementedError
            kernel_window = 1.5
        case ResizeKernel.MAGIC_KERNEL_SHARP_2013:
            raise NotImplementedError
            kernel_window = 2.5
        case ResizeKernel.MAGIC_KERNEL_SHARP_2021:
            resize_kernel = magic_kernel_sharp_2021_triton
            kernel_window = 4.5
        case _:
            raise ValueError(f"Unknown resize kernel {k}")
    return resize_kernel, kernel_window

# Sparse Downscale and support functions.

def create_tile_mask(
        dest_size: int,
        src_size: int,
        kernel_window: float = 4.5,
        tile_size: int = 64
    ):
    k = dest_size / src_size
    PAD = math.ceil((kernel_window) / k)
    coords_source = torch.arange(
        (-PAD + (tile_size / 2)) * k,
        (src_size + PAD + (tile_size / 2)) * k,
        tile_size * k,
        dtype=torch.float32
    )
    coords_dest = torch.arange(
        tile_size / 2,
        dest_size + (tile_size / 2),
        tile_size,
        dtype=torch.float32
    )

    coord_grid = torch.sub(coords_source.unsqueeze(-1), other=coords_dest)
    coord_grid.abs_() # inplace operation
    return torch.lt(coord_grid, tile_size + 0.5)

def create_tensor_metadata(
        num_sparse_blocks: int, 
        tile_mask: torch.Tensor,
        indices: torch.Tensor,
        block_offsets_t: torch.Tensor,
        col_indices_t: torch.Tensor,
        offsets: torch.Tensor,
        offsets_t: torch.Tensor,
    ):

    # could this run on cuda? yes (on pytorch 2.6). is it worth running on cuda? *no*.
    torch.nonzero_static(tile_mask, size=num_sparse_blocks, out=indices)

    # these are contiguous and pinned because we transposed the buffer
    row_indices = indices[:,0] # type: torch.Tensor
    col_indices = indices[:,1] # type: torch.Tensor

    # Strangely, being unstable is better for kernel performance.
    # Stable will cause warp stalls.
    torch.argsort(col_indices, stable=False, out=block_offsets_t)
    torch.take(row_indices, block_offsets_t, out=col_indices_t)

    # reusing the offsets buffer here helps performance
    torch.sum(tile_mask, dim=1, out=offsets[1:])
    torch.sum(tile_mask, dim=0, out=offsets_t[1:])
    torch.cumsum(offsets, dim=0, out=offsets)
    torch.cumsum(offsets_t, dim=0, out=offsets_t)
    return row_indices, col_indices, block_offsets_t, col_indices_t, offsets, offsets_t

# for isolating the one mandatory graph break
@torch.compiler.disable
def _get_nnz_and_buffers(tile_mask):
    num_sparse_blocks = torch.sum(tile_mask).item()
    return (
        num_sparse_blocks,
        [
            torch.empty((2, num_sparse_blocks), dtype=torch.int64, pin_memory=True).T, # indices
            torch.empty((num_sparse_blocks,), dtype=torch.int64, pin_memory=True), #block_offsets_t
            torch.empty((num_sparse_blocks,), dtype=torch.int64, pin_memory=True), #col_indices_t
            torch.zeros((tile_mask.shape[0] + 1,), dtype=torch.int32, pin_memory=True), #offsets
            torch.zeros((tile_mask.shape[1] + 1,), dtype=torch.int32, pin_memory=True) #offsets_t
        ]
    )

def generate_sparse_matrix(dest_size, src_size, kernel_window=4.5, tile_size=64):
    """
    Generate a sparse resampling matrix based on a block-wise mask.
    
    Parameters:
        dest_size (int): Number of output pixels.
        src_size (int): Number of input pixels.
        kernel_window (float): Effective width of the resampling kernel.
        tile_size (int): Size of each sparse block.

    Returns:
        Matrix: A sparse representation of the computed coordinate grid.
    """

    tile_mask = create_tile_mask(dest_size, src_size, kernel_window, tile_size)

    # Needs to be item since used to create tensor shapes. Unavoidable.
    # Inductor also does not support pinned memory, need to create buffers outside of compile.
    num_sparse_blocks, buffers = _get_nnz_and_buffers(tile_mask)

    # Pinned memory buffers created here will be the in the same memory locations ultimately used.
    # The transpose on the indices matrix is done so that row_indices and col_indices are contiguous
    # when assigned from the results of nonzero, which makes the following copy to GPU faster.
    row_indices, col_indices, block_offsets_t, col_indices_t, offsets, offsets_t = create_tensor_metadata(
        num_sparse_blocks,
        tile_mask,
        *buffers
    )

    # Create sparse tensor
    return Matrix(
        (tile_mask.shape[0] * tile_size, tile_mask.shape[1] * tile_size),
        torch.empty(num_sparse_blocks, tile_size, tile_size, dtype=torch.float16, device='cuda'),
        row_indices.to(device='cuda', dtype=torch.int32, non_blocking=True),
        col_indices.to(device='cuda', dtype=torch.int32, non_blocking=True),
        offsets.to(device='cuda', dtype=torch.int32, non_blocking=True),
        column_indices_t=col_indices_t.to(device='cuda', dtype=torch.int32, non_blocking=True),
        offsets_t=offsets_t.to(device='cuda', dtype=torch.int32, non_blocking=True),
        block_offsets_t=block_offsets_t.to(device='cuda', dtype=torch.int32, non_blocking=True)
    )

@triton.jit
def compute_sparse_coord_grid_kernel(
    coords_source_ptr, coords_dest_ptr, sparse_data_ptr,
    row_indices_ptr, col_indices_ptr,
    k, M, N, BLOCK_SIZE: tl.constexpr, SPARSE_BLOCK_SIZE: tl.constexpr
):
    SPARSE_BLOCK_NUMEL = SPARSE_BLOCK_SIZE * SPARSE_BLOCK_SIZE
    sparse_block = tl.program_id(0)

    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tl.load(row_indices_ptr + sparse_block) * SPARSE_BLOCK_SIZE + tile_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.load(col_indices_ptr + sparse_block) * SPARSE_BLOCK_SIZE + tile_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_row = row_offsets < M
    mask_col = col_offsets < N

    coord_source = tl.load(coords_source_ptr + row_offsets, mask=mask_row, other=0.0)
    coord_dest = tl.load(coords_dest_ptr + col_offsets, mask=mask_col, other=0.0)

    x = tl.cast(coord_source[:, None] - coord_dest[None, :], tl.float16)

    # Magic Kernel Sharp modified to operate in-place on negative values
    # For some reason doing it this way is slightly faster, but less readable.
    x = magic_kernel_sharp_2021_triton(x)

    # Strangely, this is far more accurate in Triton than it is in Torch.
    x /= (1/k)

    sparse_block_ptr = sparse_data_ptr + sparse_block * SPARSE_BLOCK_NUMEL

    local_row_start = tile_row * BLOCK_SIZE
    local_col_start = tile_col * BLOCK_SIZE

    local_rows = local_row_start + tl.arange(0, BLOCK_SIZE)
    local_cols = local_col_start + tl.arange(0, BLOCK_SIZE)

    local_rows_2d = local_rows[:, None]
    local_cols_2d = local_cols[None, :]

    store_offset = local_rows_2d * SPARSE_BLOCK_SIZE + local_cols_2d

    tl.store(sparse_block_ptr + store_offset, x)

def compute_sparse_coord_grid(target_size, source_size, kernel_window, BLOCK_SIZE=32, SPARSE_BLOCK_SIZE = 64):
    
    assert SPARSE_BLOCK_SIZE % BLOCK_SIZE == 0

    k = target_size / source_size
    PAD = math.ceil((kernel_window - 0.5) / k)

    coords_source = torch.arange((-PAD + 0.5)*k, (source_size + PAD + 0.5)*k, k, dtype=torch.float32, device='cuda')
    coords_dest = torch.arange(0.5, target_size + 0.5, 1, dtype=torch.float32, device='cuda')

    M, N = coords_source.shape[0], coords_dest.shape[0]
    x = generate_sparse_matrix(target_size, source_size, kernel_window, SPARSE_BLOCK_SIZE)

    SPARSE_NUM_BLOCKS = x.data.shape[0]

    grid = lambda meta: (SPARSE_NUM_BLOCKS, triton.cdiv(SPARSE_BLOCK_SIZE, meta['BLOCK_SIZE']), triton.cdiv(SPARSE_BLOCK_SIZE, meta['BLOCK_SIZE']))
    compute_sparse_coord_grid_kernel[grid](
        coords_source, coords_dest, x.data,
        x.row_indices, x.column_indices,
        k, M, N, BLOCK_SIZE, SPARSE_BLOCK_SIZE)
    return x

# Dense kernel for downsampling coord_grids

@triton.jit
def compute_coord_grid_kernel(
    coords_source_ptr, coords_dest_ptr, coord_grid_ptr, k,
    M, N, BLOCK_SIZE: tl.constexpr, 
):
    row_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_row = row_offsets < M
    mask_col = col_offsets < N
    
    coord_source = tl.load(coords_source_ptr + row_offsets, mask=mask_row)
    coord_dest = tl.load(coords_dest_ptr + col_offsets, mask=mask_col)

    x = tl.cast(coord_source[:, None] - coord_dest[None, :], tl.float16)

    x = magic_kernel_sharp_2021_triton(x)

    x /= (1/k)

    tl.store(coord_grid_ptr + row_offsets[:, None] * N + col_offsets[None, :], x, mask=mask_row[:, None] & mask_col[None, :])

def compute_coord_grid(target_size, source_size, kernel_window=4.5, BLOCK_SIZE=32):
    k = target_size / source_size
    PAD = math.ceil((kernel_window - 0.5) / k)

    coords_source = torch.arange((-PAD + 0.5)*k, (source_size + PAD + 0.5)*k, k, dtype=torch.float32, device='cuda')
    coords_dest = torch.arange(0.5, target_size + 0.5, 1, dtype=torch.float32, device='cuda')

    M, N = coords_source.shape[0], coords_dest.shape[0]
    coord_grid = torch.empty((M, N), dtype=torch.float16, device='cuda')

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    compute_coord_grid_kernel[grid](coords_source, coords_dest, coord_grid, k, M, N, BLOCK_SIZE)
    return coord_grid

@torch.compile
def downscale_sparse(
        image: torch.Tensor,
        target_size: Tuple[int, int],
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        BLOCK_SIZE: int = 32,
        SPARSE_BLOCK_SIZE: int = 64
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel_triton(resize_kernel)

    y_s_w = compute_sparse_coord_grid(target_size[-1], image.shape[-1], window, BLOCK_SIZE, SPARSE_BLOCK_SIZE)
    y_s_h = compute_sparse_coord_grid(target_size[-2], image.shape[-2], window, BLOCK_SIZE, SPARSE_BLOCK_SIZE)

    PAD_W = math.ceil((window - 0.5) / (target_size[-1] / image.shape[-1]))
    PAD_H = math.ceil((window - 0.5) / (target_size[-2] / image.shape[-2]))

    image = srgb_to_linear(image)

    image = F.pad(image, (
        PAD_W,
        y_s_w.shape[0] - image.shape[-1] - PAD_W,
        PAD_H,
        y_s_h.shape[0] - image.shape[-2] - PAD_H
    ), mode='replicate')

    image = image.view(-1, image.shape[-1])
    image = dds(image, y_s_w)
    image = image.view(3, -1, image.shape[-1])
    image = image.mT
    image = image.reshape(-1, image.shape[-1])
    image = dds(image, y_s_h, fuse_srgb=True)
    image = image.view(3, -1, image.shape[-1])
    image = image.mT
    image = image[:, :target_size[0], :target_size[1]]
    image.clamp_(0.,1.)
    return image

@torch.compile
def downscale_triton(
        image: torch.Tensor,
        target_size: Tuple[int, int],
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel_triton(resize_kernel)

    y_s_w = compute_coord_grid(target_size[-1], image.shape[-1], window)
    y_s_h = compute_coord_grid(target_size[-2], image.shape[-2], window)

    PAD_W = math.ceil((window - 0.5) / (target_size[-1] / image.shape[-1]))
    PAD_H = math.ceil((window - 0.5) / (target_size[-2] / image.shape[-2]))

    image = srgb_to_linear(image)

    image = F.pad(image, (
        PAD_W,
        y_s_w.shape[0] - image.shape[-1] - PAD_W,
        PAD_H,
        y_s_h.shape[0] - image.shape[-2] - PAD_H
    ), mode='replicate')

    image = image.view(-1, image.shape[-1])
    image = image @ y_s_w
    image = image.view(3, -1, image.shape[-1])
    image = image.mT
    image = image.reshape(-1, image.shape[-1])
    image = image @ y_s_h
    image = image.view(3, -1, image.shape[-1])
    image = image.mT
    image = linear_to_srgb(image[:, :target_size[0], :target_size[1]])
    image.clamp_(0.,1.)
    return image