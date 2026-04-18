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

# Phase 0: Probe tests demonstrate basic functionality
probe_xad_xtensor()  # Test XAD + xtensor integration
probe_sdm()          # Test SDM optimization workflow
```

## Project Status

**Current Phase: Phase 0 - Project Scaffolding** ✓

Phase 0 establishes the foundational structure:

- [x] Package skeleton with proper metadata
- [x] C++20 build configuration
- [x] AGPL-3 license
- [x] Probe test files validating core concepts
- [x] GitHub Actions CI
- [x] Comprehensive documentation

**Next Phase: Phase 1 - Minimal Viable Integration**

See [ENHANCED_IMPLEMENTATION_PLAN.md](docs/ENHANCED_IMPLEMENTATION_PLAN.md) for the complete roadmap.

## Documentation

- [xopt Analysis](docs/xopt_analysis.md) — Technical analysis and design decisions
- [Enhanced Implementation Plan](docs/ENHANCED_IMPLEMENTATION_PLAN.md) — Phased development roadmap
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

### Phase 1: Basic Integration (2-3 weeks)
Minimal viable `minimize()` with XAD gradients

### Phase 2: Algorithm Breadth (3-4 weeks)
L-BFGS, L-BFGS-B, parameter transformations

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
