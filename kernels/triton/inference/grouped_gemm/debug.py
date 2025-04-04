# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import time
from typing import Tuple

import torch
import triton
from persistent_kernel_fp8 import grouped_gemm_fp8_rowwise_persistent

def create_aligned_test_data(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    output_dim: int,
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
    M_total = batch_size * seq_len

    # Ensure M_total is a multiple of group_size_m
    padded_M = ((M_total + group_size_m - 1) // group_size_m) * group_size_m
    padding_needed = padded_M - M_total

    if padding_needed > 0:
        print(f"Padding input from {M_total} to {padded_M} to ensure group alignment")
        M_total = padded_M

    # Create inputs
    inputs = torch.randn((M_total, hidden_dim), dtype=dtype, device=device)

    # Create expert weights
    expert_weights = torch.randn(
        (num_experts, output_dim, hidden_dim), dtype=dtype, device=device
    )

    # Create expert indices with proper group alignment
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)

    # Assign experts in contiguous blocks of group_size_m
    num_groups = M_total // group_size_m

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size_m
        end_idx = start_idx + group_size_m

        # Assign this entire group to one expert
        expert_idx = group_idx % num_experts
        expert_indices[start_idx:end_idx] = expert_idx

    return inputs, expert_weights, expert_indices


def create_aligned_test_data_fp8(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    output_dim: int,
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
    M_total = batch_size * seq_len

    # Ensure M_total is a multiple of group_size_m
    padded_M = ((M_total + group_size_m - 1) // group_size_m) * group_size_m
    padding_needed = padded_M - M_total

    if padding_needed > 0:
        print(f"Padding input from {M_total} to {padded_M} to ensure group alignment")
        M_total = padded_M

    # Create inputs
    inputs = torch.randn((M_total, hidden_dim), dtype=dtype, device=device)

    # Create expert weights
    expert_weights = torch.randn(
        (num_experts, output_dim, hidden_dim), dtype=dtype, device=device
    )

    # Create expert indices with proper group alignment
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)

    # Create fake scales for rowwise FP8 GG
    a_scale = torch.ones((M_total,)).to(dtype=torch.float32, device='cuda') 
    b_scale = torch.ones((num_experts, output_dim)).to(dtype=torch.float32, device='cuda')

    # Assign experts in contiguous blocks of group_size_m
    num_groups = M_total // group_size_m

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size_m
        end_idx = start_idx + group_size_m

        # Assign this entire group to one expert
        expert_idx = group_idx % num_experts
        expert_indices[start_idx:end_idx] = expert_idx
    

    return inputs.to(torch.float8_e4m3fn), expert_weights.to(torch.float8_e4m3fn), expert_indices, a_scale, b_scale

def pytorch_reference(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Reference implementation using PyTorch for verification.
    """
    M_total, K = inputs.shape
    num_experts, N, _ = expert_weights.shape

    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Process each group
    for i in range(0, M_total, group_size_m):
        end_idx = min(i + group_size_m, M_total)

        # Get expert index for this group
        expert_idx = expert_indices[i].item()

        # Get expert weights
        expert_weight = expert_weights[expert_idx]

        # Compute output for this group
        output[i:end_idx] = torch.matmul(inputs[i:end_idx], expert_weight.t())

    return output


def pytorch_reference_fp8_rowwise(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    x_scale: torch.Tensor, 
    w_scale: torch.Tensor, # [K, N]
    group_size_m: int = 128,

):

    """
    Reference implementation using PyTorch for verification.
    """
    M_total, K = inputs.shape
    num_experts, N, _ = expert_weights.shape

    output = torch.zeros((M_total, N), device=inputs.device, dtype=torch.bfloat16)

    # Process each group
    for i in range(0, M_total, group_size_m):
        end_idx = min(i + group_size_m, M_total)

        # Get expert index for this group
        expert_idx = expert_indices[i].item()

        # Get expert weights
        expert_weight = expert_weights[expert_idx]

        # Get expert index scale
        n_start = expert_idx * N
        n_end = (expert_idx + 1) * N
        b_scale = w_scale[n_start:n_end][None, :]

        # Get input scale
        a_scale = x_scale[i:end_idx][:, None]

        # Compute output for this group
        output[i:end_idx] = (inputs[i:end_idx, :].to(torch.float32) 
                            @ expert_weight.t().to(torch.float32)
                             # * a_scale
                             # * b_scale
        ).to(torch.bfloat16)

        # print(f"{output=}")

        return output


def verify_results(
    output_triton: torch.Tensor,
    output_reference: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """
    Verify that the Triton output matches the reference output.
    """

    print(f"{output_triton.dtype=}")
          
    print(f"{output_reference.dtype=}")

    is_close = torch.allclose(output_triton, output_reference, rtol=rtol, atol=atol)

    if not is_close:
        # Compute error statistics
        abs_diff = torch.abs(output_triton - output_reference)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()

        # Find location of maximum difference
        flat_idx = torch.argmax(abs_diff.view(-1))
        row = flat_idx // output_triton.shape[1]
        col = flat_idx % output_triton.shape[1]

        print(f"Results do not match!")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Max difference at [{row}, {col}]")
        print(f"Triton: {output_triton[row, col].item():.6f}")
        print(f"Reference: {output_reference[row, col].item():.6f}")

        return False

    return True


def test_large_fp8():
    """Test with large dimensions (similar to paper configurations)."""
    print("\nRunning large FP8 test...")

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory < 20e9
    ):
        print("Skipping large test - insufficient GPU memory")
        return True

    batch_size, seq_len = 32, 128  # 4096 tokens
    hidden_dim, output_dim = 4096, 7168
    num_experts = 8
    group_size_m = 128

    inputs, expert_weights, expert_indices, x_scale, w_scale = create_aligned_test_data_fp8(
        batch_size, seq_len, hidden_dim, output_dim, num_experts, group_size_m
    )

    # Run our implementation
    output_triton = grouped_gemm_fp8_rowwise_persistent(
        inputs, expert_weights, expert_indices, x_scale, w_scale
    )

    # Run reference
    output_reference = pytorch_reference(
        inputs.to(torch.bfloat16), expert_weights.to(torch.bfloat16), expert_indices
    )

    print(f"{output_triton=}")
    print(f"{output_reference=}")

    # Verify results
    is_correct = verify_results(output_triton, output_reference)
    print(f"Large test {'passed' if is_correct else 'failed'}")

    return is_correct


def benchmark_performance_fp8():
    """Benchmark performance against PyTorch reference."""
    print("\nRunning performance benchmark...")

    # Use dimensions from the paper
    batch_size, seq_len = 32, 1024  # 4096 tokens
    hidden_dim, output_dim = 4096, 7168
    num_experts = 8
    group_size_m = 128

    inputs, expert_weights, expert_indices, x_scale, w_scale = create_aligned_test_data_fp8(
        batch_size, seq_len, hidden_dim, output_dim, num_experts, group_size_m
    )

    # Warmup
    for _ in range(5):
        output_triton = grouped_gemm_fp8_rowwise_persistent(
            inputs,
            expert_weights,
            expert_indices,
            x_scale,
            w_scale,
        )
        output_pytorch = pytorch_reference(
            inputs, expert_weights, expert_indices,
        )
        torch.cuda.synchronize()

    # Benchmark Triton
    num_runs = 10
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        output_triton = grouped_gemm_fp8_rowwise_persistent(
            inputs,
            expert_weights,
            expert_indices,
            x_scale,
            w_scale,
        )
        torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        output_pytorch = pytorch_reference_fp8_rowwise(
            inputs, expert_weights, expert_indices, x_scale, w_scale,
        )
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # Calculate TFLOPS
    M = batch_size * seq_len
    flops = 2 * M * hidden_dim * output_dim  # Multiply-adds
    triton_tflops = flops / (triton_time / 1000) / 1e12
    pytorch_tflops = flops / (pytorch_time / 1000) / 1e12

    speedup = pytorch_time / triton_time

    print(f"\nPerformance Results:")
    print(f"  Dimensions: {batch_size}x{seq_len}x{hidden_dim} -> {output_dim}")
    print(f"  Triton: {triton_time:.2f} ms ({triton_tflops:.2f} TFLOPS)")
    print(f"  PyTorch: {pytorch_time:.2f} ms ({pytorch_tflops:.2f} TFLOPS)")
    print(f"  Speedup: {speedup:.2f}x")

    # Format for paper table
    num_groups = M // group_size_m
    m_per_group = M / num_groups
    print(f"\nPaper table format:")
    print(
        f"{num_experts}\t{num_groups}\t{int(m_per_group)}\t{hidden_dim}\t{output_dim}\t{int(triton_tflops)} TFLOPS\t{int(triton_time)} ms\t{speedup:.1f}x"
    )

    return (
        speedup > 0.9
    )  # Consider it a success if performance is at least 90% of PyTorch
    


def run_all_tests():
    """Run all tests and return overall success."""
    test_results = []

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return False

    test_results.append(test_large_fp8())
    test_results.append(benchmark_performance_fp8())

    # Overall success
    all_passed = all(test_results)
    print(
        f"\nOverall test result: {'All tests passed!' if all_passed else 'Some tests failed!'}"
    )

    return all_passed


if __name__ == "__main__":
    run_all_tests()
