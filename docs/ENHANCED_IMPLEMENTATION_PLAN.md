# Enhanced Implementation Plan for xopt

## Executive Summary

This document outlines a phased approach to building xopt, a state-of-the-art R optimization package. Each phase builds on previous work, with clear acceptance criteria and deliverables.

## Design Philosophy

### Core Principles

1. **Zero-overhead abstraction**: Template-based C++ for inlining and optimization
2. **Progressive enhancement**: Start simple, add complexity incrementally
3. **User choice**: Multiple gradient modes, fall back gracefully
4. **Testing-first**: Each feature ships with tests and benchmarks

### Problem Abstraction

Instead of a single callback function, we use a compile-time `Problem` trait:

```cpp
template <class UserObj,
          GradKind Grad = GradKind::XadAdj,
          HessKind Hess = HessKind::BfgsApprox>
struct Problem {
    int n_par;                              // Dimension
    std::vector<double> lower, upper;       // Bounds
    UserObj obj;                            // User's objective

    double value(const double* x) const;
    void gradient(const double* x, double* g) const;
    void hessian(const double* x, double* H) const;
};
```

This allows:
- Compile-time dispatch (no virtual functions)
- Solver can query only what it needs
- Easy to extend with new features

## Phased Roadmap

### Phase 0: Project Scaffolding ✓ (Current)

**Timeline**: 1 week
**Status**: COMPLETE

**Deliverables**:
- [x] Package skeleton (DESCRIPTION, NAMESPACE, LICENSE)
- [x] Directory structure (R/, src/, inst/include/, tests/, docs/)
- [x] Makevars with C++20 configuration
- [x] AGPL-3 license declaration
- [x] Probe test files (probe_xad_xtensor.cpp, probe_sdm.cpp)
- [x] CI workflow (GitHub Actions)
- [x] Documentation (xopt_analysis.md, this file)

**Acceptance Criteria**:
- Package installs on R >= 4.2.0
- Probe tests compile and run
- CI passes on Linux
- All documentation committed

---

### Phase 1: Minimal Viable Integration (2-3 weeks)

**Goal**: Ship a working optimizer with automatic gradients

**Dependencies**:
- Add to DESCRIPTION: `LinkingTo: Rcpp, ucminfcpp, xadr, xtensorR`
- Verify all three packages export headers via inst/include/

**Implementation Tasks**:

1. **Problem abstraction** (inst/include/xopt/problem.hpp)
   - Template-based Problem struct
   - Gradient policies: XadAdj, UserFn, FiniteDiff
   - Tensor shape metadata for rtensor support

2. **R API** (R/minimize.R, src/minimize.cpp)
   ```r
   xopt::minimize(
     par,                # numeric vector OR array
     fn,                 # R function or XPtr
     gr = NULL,          # optional gradient
     method = "ucminf",
     gradient = "auto",  # auto, xad, user, finite
     control = list()
   )
   ```

3. **Gradient dispatch**:
   - If fn is XPtr<ObjFun>: use compiled path
   - If gradient = "user" and gr provided: use user gradient
   - If gradient = "xad": attempt XAD trace (may fall back)
   - Default: finite differences via ucminfcpp

4. **Tests** (tests/testthat/):
   - test-rosenbrock.R: Compare to optim("BFGS")
   - test-gradients.R: Verify XAD vs finite diff
   - test-simple-glm.R: Fit logistic regression

5. **Documentation**:
   - Vignette: "Getting Started with xopt"
   - Function documentation via roxygen2

**Acceptance Criteria**:
- [ ] minimize() works for R function objectives
- [ ] Gradients match finite differences (< 1e-6 error)
- [ ] Rosenbrock test matches optim() result
- [ ] CI passes on Linux, macOS, Windows
- [ ] Vignette renders successfully

**Risks & Mitigations**:
- *Risk*: LinkingTo packages not yet on CRAN
- *Mitigation*: Document manual installation, prepare for CRAN submission pipeline

---

### Phase 2: Algorithm Breadth (3-4 weeks)

**Goal**: Add methods beyond BFGS, handle constraints

**Implementation Tasks**:

1. **L-BFGS algorithm** (inst/include/xopt/lbfgs.hpp)
   - Two-loop recursion for memory-limited BFGS
   - Configurable history size (m = 5-20)
   - Same Problem interface as ucminf

2. **L-BFGS-B** for box constraints
   - Byrd-Lu-Nocedal algorithm
   - Handles lower/upper bounds
   - Active set management

3. **Parameter transformations** (inst/include/xopt/transform.hpp)
   - log transform: (0, ∞) → ℝ
   - logit transform: (0, 1) → ℝ
   - softmax: simplex → ℝⁿ⁻¹
   - Compose with AD automatically

4. **Method dispatch**:
   ```r
   xopt::minimize(..., method = c("ucminf", "lbfgs", "lbfgsb"))
   ```

5. **Extended tests**:
   - Large-scale problems (n = 1000, 10000) for L-BFGS
   - Box-constrained problems
   - Transformed parameters

**Acceptance Criteria**:
- [ ] L-BFGS handles 10,000 parameters efficiently
- [ ] L-BFGS-B respects bounds exactly
- [ ] Transformations preserve gradient correctness
- [ ] Benchmark vignette compares all methods

---

### Phase 3: Exact Hessians & Least Squares (4-6 weeks)

**Goal**: Leverage XAD's higher-order AD for Newton methods

**Implementation Tasks**:

1. **Hessian computation** via xad::fwd_adj
   - Forward-over-adjoint for exact Hessian
   - Optional: Hessian-vector products only (for CG)

2. **Trust-region Newton** (inst/include/xopt/trust_region.hpp)
   - Steihaug CG for large problems
   - Dogleg for small-medium problems
   - Uses HVP when available

3. **Levenberg-Marquardt** for nonlinear least squares
   - Specialized for f(x) = ½Σrᵢ(x)²
   - Requires Jacobian (XAD forward mode)
   - Much faster than BFGS for NLS

4. **R-side AD** (research):
   - RTMB-style: mask base functions with xad-r versions
   - Trace plain R functions automatically
   - Proof of concept in vignette

5. **Tests**:
   - Verify Hessian accuracy vs finite differences
   - NLS benchmarks (curve fitting, GLM)
   - Compare LM vs BFGS on NLS problems

**Acceptance Criteria**:
- [ ] Exact Hessian matches finite diff (< 1e-5)
- [ ] TR-Newton converges faster than BFGS near optimum
- [ ] LM solves NLS 3-10× faster than BFGS
- [ ] Vignette: "Automatic Differentiation in xopt"

---

### Phase 4: Constraints & Advanced Features (6-8 weeks)

**Goal**: Handle general nonlinear constraints, scale to production

**Implementation Tasks**:

1. **Augmented Lagrangian** for general constraints
   - Equality and inequality constraints
   - Uses any inner solver (BFGS, L-BFGS-B, etc.)

2. **Multi-start optimization**:
   - Accept par as (n_starts, n_par) matrix
   - Parallel solves with RcppParallel
   - Return best + all results

3. **Sparsity support**:
   - Sparse Hessian via Matrix::dgCMatrix
   - Sparse linear algebra (RcppEigen)

4. **JIT compilation** (experimental):
   - XAD's JITCompiler for static graphs
   - Compile once, evaluate many times
   - CasADi-style optimization

5. **Checkpointing**:
   - XAD CheckpointCallback for deep objectives
   - Bound tape memory usage

**Acceptance Criteria**:
- [ ] AL solves constrained problems correctly
- [ ] Multi-start finds global optimum on multimodal problems
- [ ] Sparse problems (n > 10,000) solve efficiently
- [ ] JIT shows speedup on repeated evaluations

---

### Phase 5: Statistical Models & Polish (Ongoing)

**Goal**: Production-ready for CRAN, compete with TMB/RTMB

**Implementation Tasks**:

1. **Laplace approximation**:
   - Integrate over random effects
   - Marginal likelihood via Hessian
   - TMB/RTMB territory

2. **Broom tidiers**:
   - tidy(), glance(), augment() methods
   - Integration with tidyverse workflows

3. **pkgdown site**:
   - Complete API documentation
   - Tutorials and case studies
   - Benchmark comparisons

4. **CRAN submission**:
   - Pass R CMD check --as-cran
   - Vignettes build on all platforms
   - Response to reviewer feedback

**Acceptance Criteria**:
- [ ] Package on CRAN
- [ ] pkgdown site deployed
- [ ] Case study: real SDM optimization
- [ ] Performance comparison published

---

## Implementation Guidelines

### Code Style

- C++ headers: inst/include/xopt/ (public API)
- C++ implementation: src/ (compiled code)
- R functions: R/ with roxygen2 docs
- Tests: tests/testthat/ with descriptive names

### Testing Strategy

**Unit tests** (testthat):
- Each function has corresponding test file
- Test edge cases and error handling
- Fast tests (<1s each)

**Integration tests**:
- End-to-end workflows
- Benchmark problems (Rosenbrock, Powell, etc.)
- Statistical models (GLM, NLS)

**Performance tests** (manual):
- Profile with Rprof
- Benchmark against competitors
- Document in vignettes

### Documentation Requirements

Each function needs:
- @title and @description
- @param for all arguments
- @return describing output structure
- @examples with working code
- @export if user-facing

Each vignette needs:
- Clear learning objective
- Motivating example
- Step-by-step explanation
- Performance comparison (where relevant)

### CI/CD Pipeline

GitHub Actions workflows:
1. **R-CMD-check**: Linux, macOS, Windows
2. **test-coverage**: Codecov integration
3. **pkgdown**: Auto-deploy on main branch
4. **CRAN-prep**: Check before submission

---

## Risk Assessment

### High Priority Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dependency packages not on CRAN | Blocks CRAN submission | Coordinate with package authors, prepare joint submission |
| C++20 not available on older R | Limits user base | Document R >= 4.2.0 requirement, provide fallback |
| AGPL license limits adoption | Fewer users | Clear license documentation, confirm with legal |
| XAD tape overhead for R functions | Poor performance | Optimize tape usage, document fast path |

### Medium Priority Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Windows compilation issues | CI failures | Test early, use win-builder |
| Memory leaks in tape code | Crashes | Valgrind testing, ASAN builds |
| Numerical stability issues | Incorrect results | Extensive testing on ill-conditioned problems |

### Low Priority Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Competition from existing packages | Lower adoption | Focus on unique features (tensor support, XAD) |
| Documentation incomplete | User confusion | Prioritize vignettes, examples |

---

## Success Metrics

### Phase 0 (Current)
- ✓ CI passing
- ✓ Probe tests working
- ✓ Documentation complete

### Phase 1
- Package installs successfully
- Basic optimization works
- Test coverage > 80%

### Phase 2
- Benchmark shows competitive performance
- All methods converge correctly
- No memory leaks (valgrind)

### Phase 3
- Hessian accuracy demonstrated
- LM faster than BFGS on NLS
- R-side AD proof of concept

### Phase 4
- Constrained optimization validated
- Scales to 10,000+ parameters
- Production use cases documented

### Phase 5
- CRAN acceptance
- 100+ CRAN downloads/month
- Community contributions

---

## Open Research Questions

1. **GPU acceleration**: Can xtensor + CUDA provide speedup?
2. **Probabilistic programming**: Can xopt back a lightweight PPL?
3. **Implicit differentiation**: Useful for optimization layers?
4. **Stochastic optimization**: SGD, Adam for large-scale?
5. **Auto-conditioning**: Detect and fix scaling issues?

These are Phase 6+ explorations after core features are stable.

---

## Conclusion

This plan provides a clear path from Phase 0 (scaffolding, complete) to Phase 5 (CRAN-ready package). Each phase:

- Builds incrementally on previous work
- Has concrete deliverables and acceptance criteria
- Balances ambition with pragmatism
- Maintains backward compatibility

The next immediate step is **Phase 1**: implementing the basic minimize() function with automatic gradients via XAD.

**Estimated timeline to CRAN**: 4-6 months from Phase 0 completion, depending on dependency package availability.
