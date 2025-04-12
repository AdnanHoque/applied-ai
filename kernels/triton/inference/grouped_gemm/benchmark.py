import torch
import torch.utils.benchmark as benchmark


# Deepseek Grouped GEMM
import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
from persistent_kernel_fp8 import grouped_gemm_fp8_rowwise_persistent
from persistent_kernel_fp8_tma import grouped_gemm_fp8_rowwise_persistent as grouped_gemm_fp8_rowwise_persistent_tma

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def construct_grouped_deep_gemm(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


# Construct Triton Kernel prep data

def construct_grouped_triton_gemm(
    M: int,
    K: int,
    N: int,
    num_experts: int,
    group_size_m: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test data with proper block alignment.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension (K)
        output_dim: Output dimension (N)
        num_experts: Number of experts
        group_size_m: Size of expert groups
        device: Device to create tensors on
        dtype: Data type for inputs and weights

    Returns:
        Tuple of (inputs, expert_weights, expert_indices)
    """
    # Calculate total number of tokens
    M_total = M * num_experts

    # Ensure M_total is a multiple of group_size_m
    padded_M = ((M_total + group_size_m - 1) // group_size_m) * group_size_m
    padding_needed = padded_M - M_total

    if padding_needed > 0:
        print(f"Padding input from {M_total} to {padded_M} to ensure group alignment")
        M_total = padded_M

    # Create inputs
    inputs = torch.randn((M_total, K), dtype=dtype, device=device)

    # Create expert weights
    expert_weights = torch.randn(
        (num_experts, N, K), dtype=dtype, device=device
    )

    # Create expert indices with proper group alignment
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)

    # Create fake scales for rowwise FP8 GG
    a_scale = torch.ones((M_total,)).to(dtype=torch.float32, device='cuda') 
    b_scale = torch.ones((num_experts, N)).to(dtype=torch.float32, device='cuda')

    # Assign experts in contiguous blocks of group_size_m
    num_groups = M_total // group_size_m

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size_m
        end_idx = start_idx + group_size_m

        # Assign this entire group to one expert
        expert_idx = group_idx % num_experts
        expert_indices[start_idx:end_idx] = expert_idx
    

    return inputs.to(torch.float8_e4m3fn), expert_weights.to(torch.float8_e4m3fn), expert_indices, a_scale, b_scale


def deep_gemm_func(x_fp8, y_fp8, out, m_indices):
    return deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

def triton_gemm_func(a_fp8, b_fp8, expert_indices, a_scale, b_scale):
    return grouped_gemm_fp8_rowwise_persistent(a_fp8, b_fp8, expert_indices, a_scale, b_scale)

def triton_tma_gemm_func(a_fp8, b_fp8, expert_indices, a_scale, b_scale):
    return grouped_gemm_fp8_rowwise_persistent_tma(a_fp8, b_fp8, expert_indices, a_scale, b_scale)


num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')
results = []

for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):

    # DeepGEMM
    x_fp8_dg, y_fp8_dg, out, ref_out = construct_grouped_deep_gemm(num_groups, m, k, n, is_masked=False)
    m_indices_dg = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
    m_indices_dg = m_indices_dg.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)

    # Persistent Triton GEMM
    a_fp8, b_fp8, expert_indices, a_scale, b_scale = construct_grouped_triton_gemm(m, k, n, num_groups)

    label = 'FP8 Grouped GEMM Kernel Comparison'
    sub_label = f'num_groups: {num_groups}, m: {m}, n: {n}, k: {k}'

    results.append(benchmark.Timer(
        stmt='deep_gemm_func(a, b, out, m_indices)',
        setup='from __main__ import deep_gemm_func',
        globals={'a': x_fp8_dg, 'b' : y_fp8_dg, 'out': out, 'm_indices': m_indices_dg},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='DeepGEMM FP8 Group GEMM').blocked_autorange(min_run_time=1))

    results.append(benchmark.Timer(
        stmt='triton_gemm_func(a, b, expert_indices, a_scale, b_scale)',
        setup='from __main__ import triton_gemm_func',
        globals={'a': a_fp8, 'b' : b_fp8, 'expert_indices': expert_indices, 'a_scale' : a_scale, 'b_scale' : b_scale},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='Triton FP8 Group GEMM').blocked_autorange(min_run_time=1))
    
    results.append(benchmark.Timer(
        stmt='triton_tma_gemm_func(a, b, expert_indices, a_scale, b_scale)',
        setup='from __main__ import triton_tma_gemm_func',
        globals={'a': a_fp8, 'b' : b_fp8, 'expert_indices': expert_indices, 'a_scale' : a_scale, 'b_scale' : b_scale},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='Triton TMA FP8 Group GEMM').blocked_autorange(min_run_time=1))
    
    
compare = benchmark.Compare(results)
compare.print()