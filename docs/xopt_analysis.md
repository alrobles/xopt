# xopt Analysis

This document provides analysis and context for the xopt package development.

## Overview

The xopt package is a state-of-the-art R optimization framework that integrates three powerful C++ libraries:

1. **ucminfcpp** - Quasi-Newton BFGS optimizer with line search
2. **XAD** - Automatic differentiation (forward + adjoint/reverse mode)
3. **xtensor-r** - Zero-copy R↔C++ tensor bridge with broadcasting

## Integration Strategy

### Key Technical Decisions

**C++20 Standard**: Required for modern language features and optimal performance
- R >= 4.2.0 supports C++20
- SystemRequirements declared in DESCRIPTION

**AGPL-3 License**: Reflects dependency on XAD
- XAD is AGPL-3 licensed
- ucminfcpp can be relicensed to AGPL-3 (owner's package)
- xtensor-r is BSD-3 (compatible)

**Header-Only Dependencies**: Prefer LinkingTo over vendoring
- Reduces maintenance burden
- Clear license chain for CRAN
- Enables downstream packages to use xopt headers

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   R User API                        │
│  xopt::minimize(par, fn, method="ucminf", ...)     │
└──────────────────────┬──────────────────────────────┘
                       │ Rcpp
┌──────────────────────▼──────────────────────────────┐
│              C++20 Integration Layer                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ xtensor-r   │  │  XAD autodiff│  │ ucminf     │ │
│  │ rarray<T>   │  │  adjoint tape│  │ minimize   │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         └────────────────┼────────────────┘         │
└─────────────────────────────────────────────────────┘
```

## Phase 0 Deliverables

### 1. Package Skeleton ✓
- DESCRIPTION with proper metadata
- NAMESPACE with export patterns
- LICENSE file (AGPL-3)
- Directory structure (R/, src/, inst/include/, tests/, docs/)

### 2. Build Configuration ✓
- Makevars with C++20 standard (CXX_STD = CXX20)
- SystemRequirements declaration
- Rcpp integration

### 3. Probe Tests ✓
- `src/tests/probe_xad_xtensor.cpp` - Demonstrates XAD + xtensor works
- `src/tests/probe_sdm.cpp` - Species distribution model use case
- Exported via Rcpp for CI validation

### 4. CI Setup (In Progress)
- GitHub Actions workflow for R CMD check
- Test on Linux with R >= 4.2.0
- Validate probe tests pass

### 5. Documentation ✓
- This analysis document
- ENHANCED_IMPLEMENTATION_PLAN.md for roadmap
- README updates

## Use Cases

### Primary Target: Statistical & Ecological Modeling

The package owner's research focuses on:
- Species distribution modeling (SDM)
- Ecological niche modeling (e.g., maxentcpp, rxbioclim)
- Optimization over raster/tensor environmental data

**Why xopt is uniquely suited:**
- xtensor handles multi-dimensional environmental data naturally
- XAD provides exact gradients for complex statistical models
- ucminfcpp offers robust BFGS optimization

### Example: SDM Optimization

```r
# Environmental data as multi-dimensional array
env_stack <- array(dim = c(100, 100, 5))  # 100x100 grid, 5 variables

# Optimize SDM parameters with automatic gradients
result <- xopt::minimize(
  par = initial_params,
  fn = sdm_objective,     # XAD traces through this
  method = "ucminf",
  gradient = "auto"       # XAD provides exact gradient
)
```

## Next Phases

**Phase 1** (2 weeks): Minimal viable integration
- Add LinkingTo dependencies
- Implement basic minimize() with XAD gradients
- Gradient checking diagnostics

**Phase 2** (3-4 weeks): Algorithm breadth
- L-BFGS for large problems
- L-BFGS-B for box constraints
- Parameter transformations

**Phase 3** (4-6 weeks): Advanced features
- Exact Hessians via XAD
- Least squares optimization
- R-side AD (RTMB-style)

**Phase 4+**: Constraints, multi-start, statistical models

## Technical Notes

### Gradient Computation Strategies

Three paths based on what user provides:

1. **Compiled XAD objective** (fastest)
   - C++ function using XAD active types
   - Tape-based adjoint mode
   - Near-zero overhead

2. **R function with xad-r** (moderate)
   - R function using xad_adj_real types
   - S3 dispatch overhead
   - Still exact gradients

3. **Finite differences** (fallback)
   - Plain R function
   - Numerical approximation
   - Slower but always works

### Memory and Performance

- Tape-based AD has O(n) memory for n operations
- BFGS has O(p²) memory for p parameters
- For large p (>10,000), L-BFGS reduces to O(m·p) where m ≈ 10

### Licensing Considerations

AGPL-3 requires:
- Source code availability for users
- Network use triggers distribution requirements
- Compatible with GPL-3+, incompatible with proprietary

This is acceptable for academic/research use and matches the ecological modeling community's open-source ethos.

## References

- ucminfcpp: https://github.com/alrobles/ucminfcpp
- XAD: https://github.com/auto-differentiation/xad
- xad-r: https://github.com/alrobles/xad-r
- xtensor: https://github.com/xtensor-stack/xtensor
- xtensor-r: https://github.com/alrobles/xtensor-r

## Validation Strategy

### Correctness Tests
- Compare gradients: XAD vs. finite differences
- Benchmark problems: Rosenbrock, extended Rosenbrock, Powell
- Match stats::optim() results on test cases

### Performance Tests
- Profile against optim(), ucminf, nloptr
- Test scaling with parameter dimension
- Memory usage validation

### Statistical Tests
- GLM fitting matches glm() coefficients
- SDM optimization on real ecological data
- Convergence diagnostics

## Success Criteria for Phase 0

✓ Package installs and loads
✓ Probe tests compile and run
✓ CI validates on Linux R 4.2+
✓ AGPL-3 license properly declared
✓ C++20 compilation works
✓ Documentation provides context

**Status**: Ready for Phase 1 implementation
