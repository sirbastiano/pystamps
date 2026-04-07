"""Kernel implementations and backend selectors."""

from pystamps.kernels.accelerated import (
    BackendUnavailableError,
    describe_backend_matrix,
    run_stage4_edge_stats_kernel,
    run_stage2_grid_accumulate_kernel,
    run_stage2_histogram_kernel,
    run_stage2_topofit_coh_row_invariant_kernel,
    run_stage2_topofit_kernel,
    run_stage2_topofit_row_invariant_kernel,
    run_stage7_scla_kernel,
    run_stage8_edge_noise_kernel,
    stage2_native_available,
)
from pystamps.kernels.registry import (
    DEFAULT_REGISTRY,
    BackendProvider,
    KernelImplementation,
    KernelRegistry,
    KernelResolutionError,
    ResolvedKernel,
)

__all__ = [
    "BackendUnavailableError",
    "BackendProvider",
    "DEFAULT_REGISTRY",
    "KernelImplementation",
    "KernelRegistry",
    "KernelResolutionError",
    "ResolvedKernel",
    "describe_backend_matrix",
    "run_stage4_edge_stats_kernel",
    "run_stage2_grid_accumulate_kernel",
    "run_stage2_histogram_kernel",
    "run_stage2_topofit_coh_row_invariant_kernel",
    "run_stage2_topofit_kernel",
    "run_stage2_topofit_row_invariant_kernel",
    "run_stage7_scla_kernel",
    "run_stage8_edge_noise_kernel",
    "stage2_native_available",
]
