# xopt

[![R-CMD-check](https://github.com/alrobles/xopt/workflows/R-CMD-check/badge.svg)](https://github.com/alrobles/xopt/actions)
[![Probe Tests](https://github.com/alrobles/xopt/workflows/Probe%20Tests/badge.svg)](https://github.com/alrobles/xopt/actions)

## State-of-the-Art Optimization for R

`xopt` is a modern optimization framework that combines:

- **ucminfcpp** — Quasi-Newton BFGS optimizer with line search
- **XAD** — Automatic differentiation (forward + adjoint/reverse mode)
- **xtensor-r** — Zero-copy R↔C++ tensor bridge with broadcasting

## Features

- **Automatic gradients** via XAD tape-based autodiff
- **Efficient tensor operations** with zero-copy xtensor integration
- **C++20 compiled objectives** for maximum performance
- **Species Distribution Modeling** optimized workflows
- **AGPL-3 licensed** to match XAD dependency

## Installation

### From GitHub (Development Version)

```r
# install.packages("devtools")
devtools::install_github("alrobles/xopt")
```

### System Requirements

- R >= 4.2.0 (for C++20 support)
- C++20 compatible compiler (GCC >= 10, Clang >= 12, MSVC >= 19.29)

## Quick Start

```r
library(xopt)

# Run Phase 1 tests (core abstractions)
test_logistic_sdm_gradient()   # Verify analytical gradients
test_logistic_sdm_endtoend()   # Full SDM workflow
test_masked_sum()              # Raster masking

# Run Phase 2 tests (L-BFGS solvers)
test_lbfgs_quadratic()         # L-BFGS on simple problem
test_lbfgs_rosenbrock()        # L-BFGS on Rosenbrock
test_lbfgs_largescale()        # L-BFGS on n=1000 problem

test_lbfgsb_bounds()           # L-BFGS-B with box constraints
test_lbfgsb_rosenbrock()       # Constrained Rosenbrock

test_maxent_gradient()         # MaxEnt gradient accuracy
test_maxent_endtoend()         # MaxEnt optimization
test_chunked_processing()      # Large raster (250k cells)
test_chunked_with_na()         # NA handling
```

## Project Status

**Current Phase: Phase 2 - Algorithm Breadth** ✓

### Completed Phases

**Phase 0 - Project Scaffolding** ✓
- [x] Package skeleton with proper metadata
- [x] C++20 build configuration
- [x] AGPL-3 license
- [x] Probe test files validating core concepts
- [x] GitHub Actions CI
- [x] Comprehensive documentation

**Phase 1 - Core Abstractions** ✓
- [x] Problem/RasterProblem API with compile-time policies
- [x] Gradient computation policies (XAD, User, FiniteDiff)
- [x] Tensor shape metadata for raster support
- [x] Logistic SDM reference model
- [x] Chunking and masking infrastructure
- [x] Comprehensive test coverage

**Phase 2 - Algorithm Breadth** ✓
- [x] L-BFGS solver for large-scale optimization
- [x] L-BFGS-B solver with box constraints
- [x] Box constraint support in Problem API
- [x] MaxEnt (Poisson point process) reference model
- [x] Validated chunking on large rasters (>250k cells)
- [x] 9 new tests (C++ and R integration)

**Next Phase: Phase 3 - Advanced Features**

See [ENHANCED_IMPLEMENTATION_PLAN.md](docs/ENHANCED_IMPLEMENTATION_PLAN.md) for the complete roadmap.

## Documentation

- [xopt Analysis](docs/xopt_analysis.md) — Technical analysis and design decisions
- [Enhanced Implementation Plan](docs/ENHANCED_IMPLEMENTATION_PLAN.md) — Phased development roadmap
- [Phase 1 Implementation](docs/PHASE1_IMPLEMENTATION.md) — Core abstractions documentation
- [Phase 2 Implementation](docs/PHASE2_IMPLEMENTATION.md) — L-BFGS solvers and MaxEnt model
- [Integration Plan](docs/INTEGRATION_PLAN.md) — Detailed integration strategy

## Use Cases

### Species Distribution Modeling

```r
# Environmental raster stack
env_stack <- array(dim = c(100, 100, 5))  # 100x100 grid, 5 variables

# Optimize SDM parameters with automatic gradients (Phase 1+)
# result <- xopt::minimize(
#   par = initial_params,
#   fn = sdm_objective,
#   method = "ucminf",
#   gradient = "auto"  # XAD provides exact gradients
# )
```

### General Optimization

```r
# Simple example (Phase 1+)
# objective <- function(x) sum(x^2)
# result <- xopt::minimize(
#   par = c(1, 2, 3),
#   fn = objective,
#   method = "ucminf"
# )
```

## Development Roadmap

### Phase 0: Scaffolding ✓ (Complete)
Package skeleton, CI, probe tests, documentation

### Phase 1: Core Abstractions ✓ (Complete)
Problem API, raster support, logistic SDM, masking/chunking

### Phase 2: Algorithm Breadth ✓ (Complete)
L-BFGS, L-BFGS-B, box constraints, MaxEnt model

### Phase 3: Advanced Features (4-6 weeks)
Exact Hessians, least squares, R-side AD

### Phase 4+: Production Ready
Constraints, multi-start, statistical models, CRAN submission

## Related Projects

- [ucminfcpp](https://github.com/alrobles/ucminfcpp) — BFGS optimizer for R
- [xad-r](https://github.com/alrobles/xad-r) — Automatic differentiation for R
- [xtensor-r](https://github.com/alrobles/xtensor-r) — R/C++ tensor bridge
- [XAD](https://github.com/auto-differentiation/xad) — C++ autodiff library

## Contributing

Contributions are welcome! Please:

1. Check existing issues and PRs
2. Follow the phased implementation plan
3. Add tests for new features
4. Update documentation

## License

AGPL-3 (GNU Affero General Public License v3.0)

This license is chosen to match the XAD dependency. See [LICENSE](LICENSE) for details.

## Citation

If you use xopt in research, please cite:

```bibtex
@misc{xopt,
  title = {xopt: State-of-the-Art Optimization for R},
  author = {Robles, Angel},
  year = {2026},
  url = {https://github.com/alrobles/xopt}
}
```

## Acknowledgments

- XAD team for the excellent autodiff library
- xtensor developers for the tensor library
- R Core Team for R itself
