import torch
import math
import triton
import triton.language as tl

from sharpfin.util import ResizeKernel
from typing import Tuple
import torch.nn.functional as F
from triton.language.extra import libdevice
from sharpfin.util import linear_to_srgb, srgb_to_linear

# Magic Kernel Sharp modified to operate in-place on negative values
# For some reason doing it this way is slightly faster, but less readable.
@triton.jit
def magic_kernel_sharp_2021_triton(x: tl.tensor) -> tl.tensor:
    x = libdevice.copysign(x, -1.) # equivalent to -tl.abs(x), less ops
    x = tl.where(x >= -0.5, (577/576) - (239/144) * (x * x), x)
    x = tl.where(x < -0.5 and x >= -1.5, (35/36) * (x + 1) * (x + 239/140), x)
    x = tl.where(x < -1.5 and x >= -2.5, -(1/6) * (x + 2) * (65/24 + x), x)
    x = tl.where(x < -2.5 and x >= -3.5, (1/36) * (x + 3) * (x + 15/4), x)
    x = tl.where(x < -3.5 and x >= -4.5, -(1/288) * ((x + 9/2) * (x + 9/2)), x)
    x = tl.where(x < -4.5, 0, x)
    return x

# NOTE: libdevice.fast_powf mostly just clips subnormals compared to libdevice.pow
@triton.jit
def linear_to_srgb_triton(x):
    return tl.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * libdevice.fast_powf(x, 1/2.4) - 0.055
    )

@triton.jit
def srgb_to_linear_triton(x):
    return tl.where(
        x <= 0.04045,
        x / 12.92,
        libdevice.fast_powf((x + 0.055) / 1.055, 2.4)
    )

from sharpfin.sparse_backend import triton_dds, Matrix

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

# Amanatides, John and Woo, Andrew -- Fast Voxel Traversal
def grid_line_tiles(x0, y0, x1, y1, grid_width, grid_height):
    tiles = set()

    dx = x1 - x0
    dy = y1 - y0

    x = math.floor(x0)
    y = math.floor(y0)

    end_x = math.floor(x1)
    end_y = math.floor(y1)

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1

    t_max_x = ((x + (step_x > 0)) - x0) / dx if dx != 0 else float('inf')
    t_max_y = ((y + (step_y > 0)) - y0) / dy if dy != 0 else float('inf')

    t_delta_x = abs(1 / dx) if dx != 0 else float('inf')
    t_delta_y = abs(1 / dy) if dy != 0 else float('inf')

    while True:
        if 0 <= x < grid_width and 0 <= y < grid_height:
            tiles.add((y,x))
        if x == end_x and y == end_y:
            break
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            x += step_x
        else:
            t_max_y += t_delta_y
            y += step_y

    return tiles

def tile_mask_function(dest_size, src_size, kernel_window=4.5, tile_size=64):
    k = dest_size / src_size
    PAD = math.ceil((kernel_window-0.5) / k)

    grid_size = math.ceil((src_size + 2*PAD)/tile_size), math.ceil(dest_size/tile_size)

    line_1 = 0, 0.5/tile_size, (dest_size)/tile_size, (src_size+0.5)/tile_size
    line_2 = 0, (2*PAD - 0.5)/tile_size, (dest_size)/tile_size, (src_size + 2*PAD - 0.5)/tile_size
    lines = line_1, line_2

    mask = torch.zeros(grid_size, dtype=torch.bool)

    tiles = set()

    for (x0, y0, x1, y1) in lines:
        tiles.update(grid_line_tiles(x0, y0, x1, y1, grid_size[1], grid_size[0]))

    tiles = torch.tensor(list(tiles))

    mask[tiles[:,0], tiles[:,1]] = True

    return mask, tiles

def create_tensor_metadata(
        tile_mask: torch.Tensor,
        tiles: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        offsets_t: torch.Tensor,
    ):

    indices[:,:2] = tiles

    torch.argsort(indices[:,1], stable=True, out=indices[:,2]) # block_offsets_t
    torch.take(indices[:,0], indices[:,2], out=indices[:,3]) # col_indices_t

    # reusing the offsets buffer here helps performance
    torch.sum(tile_mask, dim=1, out=offsets[1:])
    torch.sum(tile_mask, dim=0, out=offsets_t[1:])
    torch.cumsum(offsets, dim=0, out=offsets)
    torch.cumsum(offsets_t, dim=0, out=offsets_t)

    return indices, offsets, offsets_t

# for isolating the one mandatory graph break
@torch.compiler.disable
def _get_nnz_and_buffers(tile_mask):
    num_sparse_blocks = torch.sum(tile_mask).item()

        # Creating all the index buffers as a single tensor helps greatly with memory transfers.
        # Since we are transposing it at the start, each index will still be contiguous once sliced.
        # offsets/offsets_t could be joined too despite their mismatched shapes, but it makes 
        # torch.compile unhappy and yields no noticeable performance benefit.
    return [
        torch.empty((4, num_sparse_blocks), dtype=torch.int64, pin_memory=True).T, # indices
        torch.zeros((tile_mask.shape[0] + 1,), dtype=torch.int32, pin_memory=True), # offsets
        torch.zeros((tile_mask.shape[1] + 1,), dtype=torch.int32, pin_memory=True) # offsets_t
    ]


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

    tile_mask, tiles = tile_mask_function(dest_size, src_size, kernel_window, tile_size)

    # Needs to be item since used to create tensor shapes. Unavoidable.
    # Inductor also does not support pinned memory, need to create buffers outside of compile.
    buffers = _get_nnz_and_buffers(tile_mask)
    num_sparse_blocks = buffers[0].shape[0]

    # Pinned memory buffers created here will be the in the same memory locations ultimately used.
    # The transpose on the indices matrix is done so that row_indices and col_indices are contiguous
    # when assigned from the results of nonzero, which makes the following copy to GPU faster.
    indices, offsets, offsets_t = create_tensor_metadata(
        tile_mask,
        tiles,
        *buffers
    )

    indices = indices.to(device='cuda', dtype=torch.int32, non_blocking=True)

    # Create sparse tensor
    return Matrix(
        (tile_mask.shape[0] * tile_size, tile_mask.shape[1] * tile_size),
        torch.empty(num_sparse_blocks, tile_size, tile_size, dtype=torch.float16, device='cuda'),
        row_indices=indices[:,0],
        column_indices=indices[:,1],
        offsets=offsets.to(device='cuda', non_blocking=True),
        column_indices_t=indices[:,3],
        offsets_t=offsets_t.to(device='cuda', non_blocking=True),
        block_offsets_t=indices[:,2]
    )

@triton.jit
def compute_sparse_coord_grid_kernel(
    coords_source_ptr, coords_dest_ptr, sparse_data_ptr,
    row_indices_ptr, col_indices_ptr,
    k: float, M: int, N: int, BLOCK_SIZE: tl.constexpr, SPARSE_BLOCK_SIZE: tl.constexpr
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

    x = magic_kernel_sharp_2021_triton(x)

    x *= k

    sparse_block_ptr = sparse_data_ptr + sparse_block * SPARSE_BLOCK_NUMEL

    local_row_start = tile_row * BLOCK_SIZE
    local_col_start = tile_col * BLOCK_SIZE

    local_rows = local_row_start + tl.arange(0, BLOCK_SIZE)
    local_cols = local_col_start + tl.arange(0, BLOCK_SIZE)

    local_rows_2d = local_rows[:, None]
    local_cols_2d = local_cols[None, :]

    store_offset = local_rows_2d * SPARSE_BLOCK_SIZE + local_cols_2d

    tl.store(sparse_block_ptr + store_offset, x)

def compute_sparse_coord_grid(target_size, source_size, kernel_window, BLOCK_SIZE=32, SPARSE_BLOCK_SIZE=64):
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

    x *= k

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

@triton.jit
def srgb_to_linear_pad_replicate_kernel(
    x_ptr, y_ptr,
    M_X, N_X,
    M_Y, N_Y,
    M_PAD, N_PAD,
    stride_xc, stride_xm, stride_xn,
    stride_yc, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    
    pid_c = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_cl = tl.maximum(offs_m, M_PAD) - M_PAD
    offs_m_cl = tl.minimum(offs_m_cl, M_X - 1)
    offs_n_cl = tl.maximum(offs_n, N_PAD) - N_PAD
    offs_n_cl = tl.minimum(offs_n_cl, N_X - 1)

    mask_m = offs_m < M_Y
    mask_n = offs_n < N_Y

    x_block_ptrs = x_ptr + pid_c * stride_xc + offs_m_cl[:, None] * stride_xm + offs_n_cl[None, :] * stride_xn
    y_block_ptrs = y_ptr + pid_c * stride_yc + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    x_vals = tl.load(x_block_ptrs)
    y_vals = srgb_to_linear_triton(x_vals)

    tl.store(y_block_ptrs, y_vals, mask=mask_m[:, None] & mask_n[None, :])

def srgb_to_linear_pad_replicate(
        img: torch.Tensor,
        pad_h: int,
        pad_w: int,
        sparse_block_size: int = 0,
    ):
    C = img.shape[0]

    M_PAD = pad_h
    N_PAD = pad_w

    if sparse_block_size != 0:
        out_H = img.shape[-2] + M_PAD + (-(img.shape[-2] + M_PAD)) % sparse_block_size     # output image height
        out_W = img.shape[-1] + N_PAD + (-(img.shape[-1] + N_PAD)) % sparse_block_size     # output image width
    else:
        out_H = img.shape[-2] + M_PAD + M_PAD
        out_W = img.shape[-1] + N_PAD + N_PAD

    out = torch.empty(C, out_H, out_W, dtype=img.dtype, device=img.device)

    BLOCK_M = 1
    BLOCK_N = 512

    grid = lambda META: (
        C,
        (out.shape[1] + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        (out.shape[2] + META['BLOCK_N'] - 1) // META['BLOCK_N'],
    )

    srgb_to_linear_pad_replicate_kernel[grid](
        img, out,
        img.shape[1], img.shape[2],
        out.shape[1], out.shape[2],
        M_PAD, N_PAD,
        img.stride(0), img.stride(1), img.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out

def downscale_sparse(
        image: torch.Tensor,
        target_size: Tuple[int, int],
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        BLOCK_SIZE: int = 32,
        SPARSE_BLOCK_SIZE: int = 64,

    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel_triton(resize_kernel)

    T_W = target_size[-1]
    T_H = target_size[-2]
    S_W = image.shape[-1]
    S_H = image.shape[-2]

    y_s_w = compute_sparse_coord_grid(T_W, S_W, window, BLOCK_SIZE, SPARSE_BLOCK_SIZE)
    y_s_h = compute_sparse_coord_grid(T_H, S_H, window, BLOCK_SIZE, SPARSE_BLOCK_SIZE)

    PAD_W = math.ceil((window - 0.5) / (T_W / S_W))
    PAD_H = math.ceil((window - 0.5) / (T_H / S_H))

    image = srgb_to_linear_pad_replicate(image, PAD_H, PAD_W, SPARSE_BLOCK_SIZE)

    image = triton_dds(image, y_s_w, output_mt=True)

    image = triton_dds(image, y_s_h, fuse_srgb=True, output_mt=True, output_slice=(T_H, T_W))

    return image

def downscale_triton(
        image: torch.Tensor,
        target_size: torch.Size,
        resize_kernel: ResizeKernel = ResizeKernel.MAGIC_KERNEL_SHARP_2021,
    ) -> torch.Tensor:
    kernel, window = _get_resize_kernel_triton(resize_kernel)

    y_s_w = compute_coord_grid(target_size[-1], image.shape[-1], window)
    y_s_h = compute_coord_grid(target_size[-2], image.shape[-2], window)

    PAD_W = math.ceil((window - 0.5) / (target_size[-1] / image.shape[-1]))
    PAD_H = math.ceil((window - 0.5) / (target_size[-2] / image.shape[-2]))

    image = srgb_to_linear_pad_replicate(image, PAD_H, PAD_W)

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