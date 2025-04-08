import torch
import math
import triton
import triton.language as tl
import sharpfin.sparse_backend as SFSB

def create_tile_mask(dest_size, src_size, kernel_window=4.5, tile_size=64):
    k = dest_size / src_size
    PAD = math.ceil(kernel_window / k)
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
    # Force casting during tensor writing
    return torch.lt(coord_grid, tile_size + 0.5, out=torch.empty_like(coord_grid, dtype=torch.bool))

def create_tensor_metadata(
        num_sparse_blocks: int, 
        tile_mask: torch.Tensor,
        indices: torch.Tensor,
        block_offsets_t: torch.Tensor,
        col_indices_t: torch.Tensor,
        offsets: torch.Tensor,
        offsets_t: torch.Tensor,
    ):

    # writes to indices buffer
    # could this run on cuda? yes (on pytorch 2.6). is it worth running on cuda? *no*.
    torch.nonzero_static(tile_mask, size=num_sparse_blocks, out=indices)

    row_indices = indices[:,0] # type: torch.Tensor
    col_indices = indices[:,1] # type: torch.Tensor

    block_offsets_t = torch.argsort(col_indices, stable=False, out=block_offsets_t)
    # this might normally be done as an indexing op but we need to write to the pinned buffer we made
    # underlying operation is the same
    col_indices_t = torch.gather(row_indices, 0, block_offsets_t, out=col_indices_t)

    offsets[1:] = torch.cumsum(torch.sum(tile_mask, dim=1), dim=0)
    offsets_t[1:] = torch.cumsum(torch.sum(tile_mask, dim=0), dim=0)
    return row_indices, col_indices, block_offsets_t, col_indices_t, offsets, offsets_t

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
    num_sparse_blocks = torch.sum(tile_mask).item()

    # Pinned memory buffers created here will be the in the same memory locations ultimately used.
    # The transpose on the indices matrix is done so that row_indices and col_indices are contiguous
    # when assigned from the results of nonzero.
    row_indices, col_indices, block_offsets_t, col_indices_t, offsets, offsets_t = create_tensor_metadata(
        num_sparse_blocks,
        tile_mask,
        torch.empty((2, num_sparse_blocks), dtype=torch.int64, pin_memory=True).T, # indices
        torch.empty((num_sparse_blocks,), dtype=torch.int64, pin_memory=True), #block_offsets_t
        torch.empty((num_sparse_blocks,), dtype=torch.int64, pin_memory=True), #col_indices_t
        torch.zeros((tile_mask.shape[0] + 1,), dtype=torch.int32, pin_memory=True), #offsets
        torch.zeros((tile_mask.shape[1] + 1,), dtype=torch.int32, pin_memory=True) #offsets_t
    )

    # Create sparse tensor
    return SFSB.Matrix(
        (tile_mask.shape[0] * tile_size, tile_mask.shape[1] * tile_size),
        torch.empty(num_sparse_blocks, tile_size, tile_size, dtype=torch.float16, device='cuda'),
        row_indices.to(device='cuda', non_blocking=True),
        col_indices.to(device='cuda', non_blocking=True),
        offsets.to(device='cuda', non_blocking=True),
        column_indices_t=col_indices_t.to(device='cuda', non_blocking=True),
        offsets_t=offsets_t.to(device='cuda', non_blocking=True),
        block_offsets_t=block_offsets_t.to(device='cuda', non_blocking=True)
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

    x = tl.abs(x)
    x = tl.where(x <= 0.5, -(577/576) + (239/144) * (x * x), x)
    x = tl.where(x > 0.5 and x <= 1.5, -(35/36) * (x - 1) * (x - 239/140), x)
    x = tl.where(x > 1.5 and x <= 2.5, -(1/6) * (x - 2) * (65/24 - x), x)
    x = tl.where(x > 2.5 and x <= 3.5, -(1/36) * (x - 3) * (x - 15/4), x)
    x = tl.where(x > 3.5 and x <= 4.5, (1/288) * ((x - 9/2) * (x - 9/2)), x)
    x = tl.where(x > 4.5, 0, x)
    x = -x

    # Accuracy may need to be investigated on x /= (1/k). The original torch code is x /= x.sum(dim=0,keepdim=True).
    # This seems to return identical results in every case and avoids needing more kernel launches.
    x /= (1/k)

    # Compute location in sparse tensor memory
    sparse_block_ptr = sparse_data_ptr + sparse_block * SPARSE_BLOCK_NUMEL

    # Compute local offsets within sparse block
    local_row_start = tile_row * BLOCK_SIZE
    local_col_start = tile_col * BLOCK_SIZE

    local_rows = local_row_start + tl.arange(0, BLOCK_SIZE)
    local_cols = local_col_start + tl.arange(0, BLOCK_SIZE)

    local_rows_2d = local_rows[:, None]
    local_cols_2d = local_cols[None, :]

    # Flattened storage offset inside sparse block
    store_offset = local_rows_2d * SPARSE_BLOCK_SIZE + local_cols_2d

    # Combine masks
    store_mask = mask_row[:, None] & mask_col[None, :]

    # Store result into sparse block
    tl.store(sparse_block_ptr + store_offset, x, mask=store_mask)

def compute_sparse_coord_grid(size, image_shape, kernel_window, BLOCK_SIZE=32, SPARSE_BLOCK_SIZE = 64):
    
    assert SPARSE_BLOCK_SIZE % BLOCK_SIZE == 0

    k = size / image_shape[-1]
    PAD = math.ceil(kernel_window / k)

    coords_source = torch.arange((-PAD + 0.5)*k, (image_shape[-1] + PAD + 0.5)*k, k, dtype=torch.float32, device='cuda')
    coords_dest = torch.arange(0.5, size + 0.5, 1, dtype=torch.float32, device='cuda')

    M, N = coords_source.shape[0], coords_dest.shape[0]
    x = generate_sparse_matrix(size, image_shape[-1], kernel_window, SPARSE_BLOCK_SIZE)

    SPARSE_NUM_BLOCKS = x.data.shape[0]

    grid = lambda meta: (SPARSE_NUM_BLOCKS, triton.cdiv(SPARSE_BLOCK_SIZE, meta['BLOCK_SIZE']), triton.cdiv(SPARSE_BLOCK_SIZE, meta['BLOCK_SIZE']))
    compute_sparse_coord_grid_kernel[grid](
        coords_source, coords_dest, x.data,
        x.row_indices, x.column_indices,
        k, M, N, BLOCK_SIZE, SPARSE_BLOCK_SIZE)
    return x