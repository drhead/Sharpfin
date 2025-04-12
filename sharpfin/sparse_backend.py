import numpy as np
import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from .triton_functional import linear_to_srgb_triton

# Code is all adapted from https://github.com/stanford-futuredata/stk, licensed under Apache-2.0
# Very reduced set of functions for handling DDS (Dense = Dense @ Sparse) matmul only, with the
# DDS kernel modified to be more flexible on input shapes.

def _validate_matrix(shape, data, row_indices, column_indices, offsets):
    # Data should be [nnz, block_size, block_size]
    if data.dim() == 1:
        data = torch.reshape(data, [data.numel(), 1, 1])

    # Blocks should be square.
    if data.shape[-2] != data.shape[-1]:
        raise ValueError(
            "Expected square blocking in data. "
            f"Got block shape {[data.shape[-2], data.shape[-1]]}")

    # Flatten batch dimensions on data - original shape preserved
    # in shape argument.
    block_size = data.shape[-1]
    data = data.view([-1, block_size, block_size])

    if data.dim() != 3:
        raise ValueError(
            "Expected 3D shape for data (nnz, block, block). "
            f"Got shape {data.dim()}D shape.")

    block_size = data.shape[1]
    if shape[-2] % block_size != 0 or shape[-1] % block_size != 0:
        raise ValueError(
            "Matrix shape must be dividible by blocking. "
            f"Got shape {shape} with "
            f"{[block_size, block_size]} blocking.")

    if np.prod(shape) < data.numel():
        raise ValueError(
            "Invalid matrix. Number of nonzeros exceeds matrix capacity "
            f"({data.numel()} v. {np.prod(shape)})")

    if row_indices.dim() != 1:
        raise ValueError(
            f"Expected 1D row_indices. Got {row_indices.dim()}D row_indices.")

    if column_indices.dim() != 1:
        raise ValueError(
            f"Expected 1D column_indices. Got {column_indices.dim()}D column_indices.")

    if offsets.dim() != 1:
        raise ValueError(
            f"Expected 1D offsets. Got {offsets.dim()}D offsets.")

    if row_indices.numel() != data.shape[0]:
        raise ValueError(
            "Expected 1 index per nonzero block. "
            f"Got {row_indices.numel()} row_indices for {data.shape[0]} blocks")

    if column_indices.numel() != data.shape[0]:
        raise ValueError(
            "Expected 1 index per nonzero block. "
            f"Got {column_indices.numel()} column_indices for {data.shape[0]} blocks")

    block_rows = np.prod(shape[:-1]) / block_size
    if offsets.numel() != block_rows + 1:
        raise ValueError(
            "Expected one offset per block row plus one. "
            f"Got {offsets.numel()} offsets with {block_rows} block rows.")

    is_cuda = (data.is_cuda and
               row_indices.is_cuda and
               column_indices.is_cuda and
               offsets.is_cuda)
    is_cpu = (not data.is_cuda and
              not row_indices.is_cuda and
              not column_indices.is_cuda and
              not offsets.is_cuda)
    if not (is_cuda or is_cpu):
        raise ValueError(
            "Expected data & meta-data on common device. "
            f"Got data on {data.device}, row_indices on {row_indices.device} "
            f"column_indices on {column_indices.device} and "
            f"offsets on {offsets.device}.")

    if data.dtype != torch.float16:
        raise ValueError(
            f"Expected float16 data. Got {data.dtype} data.")
    if row_indices.dtype != torch.int16:
        raise ValueError(
            f"Expected int16 row_indices. Got {row_indices.dtype} row_indices.")
    if column_indices.dtype != torch.int16:
        raise ValueError(
            f"Expected int16 column_indices. Got {column_indices.dtype} column_indices.")
    if offsets.dtype != torch.int32:
        raise ValueError(
            f"Expected int32 offsets. Got {offsets.dtype} offsets.")
    return data


def _transpose(size, data: torch.Tensor, row_indices: torch.Tensor, column_indices: torch.Tensor, offsets):
    block_columns = size[1] // data.shape[1]

    # Sort row indices by column indices to get the transposed matrix's
    # column indices.
    gather_indices = column_indices.argsort()
    column_indices_t = row_indices.gather(0, gather_indices)
    block_offsets_t = gather_indices.int()

    # NOTE: Histogram is not implemented for any integer type on CPU. Do
    # the histogram in 32-bit float, which can exactly represent 16-bit
    # integers.
    column_indices_float = column_indices.float()

    zero = torch.zeros((1,), dtype=torch.int32, device=data.device)
    nnz_per_column = column_indices_float.histc(block_columns, 0, block_columns)
    nnz_per_column = nnz_per_column.int()
    offsets_t = torch.cat([zero, nnz_per_column.cumsum(0, dtype=torch.int32)])
    return column_indices_t, offsets_t, block_offsets_t


class Matrix(torch.nn.Module):
    """A matrix stored in sparse format.

    Underlying format is block compressed sparse row (BCSR).

    TODO(tgale): Make this mirror torch.Tensor API as much as possible.
    """

    def __init__(self,
                 size,
                 data,
                 row_indices,
                 column_indices,
                 offsets,
                 column_indices_t=None,
                 offsets_t=None,
                 block_offsets_t=None):
        super().__init__()
        self._size = size
        self._data = data
        self._row_indices = row_indices
        self._column_indices = column_indices
        self._offsets = offsets

        # Produce the transpose meta-data if it is not passed in.
        if ((column_indices_t is None) or (offsets_t is None) or
            (block_offsets_t is None)):
            column_indices_t, offsets_t, block_offsets_t = _transpose(
                size, data, row_indices, column_indices, offsets)
        self._column_indices_t = column_indices_t
        self._offsets_t = offsets_t
        self._block_offsets_t = block_offsets_t

        self._transposed = False

        # Validate that our metadata will not overflow.
        max_dim = np.iinfo(np.int16).max * self.blocking
        if column_indices.dtype == torch.int16:
            if size[0] > max_dim or size[1] > max_dim:
                raise ValueError(
                    "Sparse matrix with shape {size} exceeds representable "
                    "size with 16-bit indices.")

    def validate(self):
        _validate_matrix(self._size,
                         self._data,
                         self._row_indices,
                         self._column_indices,
                         self._offsets)

        # TODO(tgale): Add heavyweight data validation.

    def to(self, device):
        # TODO(tgale): Handle type conversions here. We
        # need to set the appropriate meta-data type for
        # the given floating-point type.
        self._data = self._data.to(device)
        self._row_indices = self._row_indices.to(device)
        self._column_indices = self._column_indices.to(device)
        self._offsets = self._offsets.to(device)
        self._column_indices_t = self._column_indices_t.to(device)
        self._offsets_t = self._offsets_t.to(device)
        self._block_offsets_t = self._block_offsets_t.to(device)
        return self

    def cuda(self):
        return self.to(torch.cuda.current_device())

    def clone(self):
        return Matrix(
            self.size(),
            self.data.clone(),
            self.row_indices.clone(),
            self.column_indices.clone(),
            self.offsets.clone(),
            self.column_indices_t.clone(),
            self.offsets_t.clone(),
            self.block_offsets_t.clone())

    def t(self):
        if self.dim() != 2:
            raise ValueError(
                "t() expects a tensor with <= 2 dimensions, "
                f"but self is {self.dim()}D.")
        out = Matrix(self.size(),
                     self.data,
                     self.row_indices,
                     self.column_indices,
                     self.offsets,
                     self.column_indices_t,
                     self.offsets_t,
                     self.block_offsets_t)
        out._transposed = not self._transposed
        out._size = torch.Size((self._size[1], self._size[0]))
        return out

    def contiguous(self):
        raise ValueError("Not yet implemented.")

    def is_contiguous(self):
        return not self._transposed

    @property
    def is_cuda(self):
        return self._data.is_cuda

    @property
    def device(self):
        return self._data.device

    def size(self):
        return self._size

    @property
    def shape(self):
        return self.size()

    def dim(self):
        return len(self._size)

    @property
    def data(self):
        return self._data

    @property
    def row_indices(self):
        return self._row_indices

    @property
    def column_indices(self):
        return self._column_indices

    @property
    def offsets(self):
        return self._offsets

    @property
    def offsets_t(self):
        return self._offsets_t

    @property
    def column_indices_t(self):
        return self._column_indices_t

    @property
    def block_offsets_t(self):
        return self._block_offsets_t

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def nnz(self):
        return self.data.numel()

    @property
    def blocking(self):
        return self.data.shape[1]

    @property
    def requires_grad(self):
        return self.data.requires_grad

    def requires_grad_(self, x):
        self.data.requires_grad_(x)
        return self

    def view(self, *shape):
        assert self.is_contiguous()
        if shape[-1] != self.size()[-1]:
            raise ValueError(
                "Can't change view on compressed dimension. "
                f"{self.size()[-1]} v. {shape[-1]}.")
        if np.prod(shape) != np.prod(self.size()):
            raise ValueError(
                "Mismatch in numel of Matrix and new shape. "
                f"{np.prod(self.size())} v. {np.prod(shape)}")
        return Matrix(shape,
                      self.data,
                      self.row_indices,
                      self.column_indices,
                      self.offsets,
                      self.column_indices_t,
                      self.offsets_t,
                      self.block_offsets_t)

    @property
    def grad(self):
        # TODO(tgale): Make sure this mirrors torch.Tensor
        # behavior in the case where we ask for the gradient
        # of a non-contiguous tensor.
        size = self.size()
        if not self.is_contiguous():
            size = torch.Size((size[1], size[0]))
        out = Matrix(size,
                     self.data.grad,
                     self.row_indices,
                     self.column_indices,
                     self.offsets,
                     self.column_indices_t,
                     self.offsets_t,
                     self.block_offsets_t)
        return out if self.is_contiguous() else out.t()

# TODO(tgale): Replace this helper with a custom kernel. This operation
# is much simpler to do than how it's currently implemented.
@torch.no_grad()
def _expand_for_blocking(idxs, blocking):
    # Duplicate for block column dimension.
    idxs = torch.reshape(idxs, [idxs.size()[0], 1, 2]).repeat(1, blocking, 1)

    # Update the column indices.
    idxs[:, :, 1] *= blocking
    idxs[:, :, 1] += torch.reshape(torch.arange(blocking, device=idxs.device), [1, blocking])

    # Duplicate for block row dimension.
    idxs = torch.reshape(idxs, [idxs.size()[0], 1, blocking, 2])
    idxs = idxs.repeat(1, blocking, 1, 1)

    # Update the row indices.
    idxs[:, :, :, 0] *= blocking
    idxs[:, :, :, 0] += torch.reshape(torch.arange(blocking, device=idxs.device), [1, blocking, 1])
    idxs = torch.reshape(idxs, [-1, 2])
    return idxs


# TODO(tgale): Add input type checking.
@torch.no_grad()
def to_dense(x):
    assert isinstance(x, Matrix)

    shape = (np.prod(x.shape[:-1]), x.shape[-1])
    row_idxs = x.row_indices.type(torch.int32)
    col_idxs = x.column_indices.type(torch.int32)
    indices = _expand_for_blocking(torch.stack([row_idxs, col_idxs], dim=1), x.blocking)
    indices = (indices[:, 0] * shape[1] + indices[:, 1]).type(torch.int64)

    out = torch.zeros(shape[0] * shape[1], dtype=x.dtype, device=x.device)
    out.scatter_(0, indices, x.data.flatten())
    return out.reshape(x.size())


@dataclass
class TritonConfig:
    BLOCK_M: int = 128
    BLOCK_N: int = 128
    BLOCK_K: int = 32
    BLOCK_SIZE: int = 64
    NUM_STAGES: int = 4
    NUM_WARPS: int = 4

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({}, num_stages=TritonConfig.NUM_STAGES, num_warps=TritonConfig.NUM_WARPS),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dds_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A: tl.constexpr, trans_B: tl.constexpr, fuse_srgb: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):

    # matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)

    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)
    start_inx = tl.load(offsets + pid_n)
    end_inx = tl.load(offsets + pid_n + 1)

    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE

    # block pointer to dense matrix improves L1 cache efficiency
    A_block_ptr = tl.make_block_ptr(
        base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1)
    )

    # pointers to sparse matrix
    rn = tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)

    B += (rbk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    ak_sub_incr = BLOCK_K * stride_ak
    ak_block_incr = BLOCK_SIZE * stride_ak
    bk_sub_incr = BLOCK_K * stride_bk

    for block_inx in range(end_inx - start_inx):
        a_col_idx = tl.load(column_indices + start_inx + block_inx)
        A_ptr = tl.advance(A_block_ptr, (0, a_col_idx * ak_block_incr))

        if trans_B:
            b_block_offset = (start_inx + block_inx)
        else:
            b_block_offset = tl.load(block_offsets_t + start_inx + block_inx)

        for sub_block_inx in range(nsub_blocks):
            ptr_B = B + b_block_offset * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr

            a = tl.load(A_ptr)
            b = tl.load(ptr_B)
            acc = tl.dot(a, b, acc, out_dtype=tl.float16)

            A_ptr = tl.advance(A_ptr, (0, ak_sub_incr))

    acc = acc.to(C.dtype.element_ty)
    if fuse_srgb:
        acc = linear_to_srgb_triton(acc)
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_C = (cm < M)[:, None] & (cn < N)[None, :]
    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)

    tl.store(C, acc, mask=mask_C)

def triton_dds(lhs: torch.Tensor,
        shape,
        data: torch.Tensor,
        offsets: torch.Tensor,
        row_indices: torch.Tensor,
        column_indices: torch.Tensor,
        offsets_t: torch.Tensor,
        column_indices_t: torch.Tensor,
        block_offsets_t: torch.Tensor,
        transpose_b: bool,
        fuse_srgb: bool,
        out: torch.Tensor
    ):

    device = lhs.device
    trans_B = transpose_b
    trans_A = (lhs.stride(0) > 1 and lhs.stride(1) > 1)

    # checks constraints
    assert lhs.shape[1] <= shape[0], "incompatible dimensions" # changed to LTE to allow padded RHS
    M, K = lhs.shape
    _, N = shape

    if trans_A:
        stride_am, stride_ak = lhs.stride(1), lhs.stride(0)
    else:
        stride_am, stride_ak = lhs.stride(0), lhs.stride(1)

    if trans_B:
        stride_bk, stride_bn = data.stride(2), data.stride(1)
        b_column_indices, b_offsets = column_indices, offsets
    else:
        stride_bk, stride_bn = data.stride(1), data.stride(2)
        b_column_indices, b_offsets = column_indices_t, offsets_t

    # launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    _dds_kernel[grid](
        lhs, data, out, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
        row_indices, b_column_indices, b_offsets,
        block_offsets_t, trans_A, trans_B, fuse_srgb,
        GROUP_M=128, ACC_TYPE=tl.float16, BLOCK_M=64,
        BLOCK_N=data.shape[1], BLOCK_SIZE=data.shape[1], BLOCK_K=min(data.shape[1], 32)
    )
    return out

def dds(a, b, fuse_srgb = False):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, Matrix)
    out = torch.empty((a.size()[0], b.size()[1]),
                    dtype=a.dtype,
                    device=a.device)
    return triton_dds(
        a,
        b.size(),
        b.data, b.offsets,
        b.row_indices,
        b.column_indices,
        b.offsets_t,
        b.column_indices_t,
        b.block_offsets_t,
        not b.is_contiguous(),
        fuse_srgb,
        out
        )
