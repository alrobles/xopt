# roptim — State-of-the-Art Optimization R Package

## Integration Plan: xad-r + ucminfcpp + xtensor-r

This document describes the concrete plan to build a new R optimization package that combines:

- **ucminfcpp** — Quasi-Newton BFGS optimizer with line search
- **xad-r** — Automatic differentiation (forward + adjoint/reverse mode)
- **xtensor-r** — Zero-copy R↔C++ tensor bridge with broadcasting & lazy evaluation

---

## 1. What Each Component Brings

| Component | Role | Key C++ Interface |
|---|---|---|
| **ucminfcpp** | Quasi-Newton BFGS optimizer with line search | `ucminf::minimize_direct<F>(x0, fdf, ctrl)` — templated, zero-overhead, header-only |
| **xad-r** | Automatic differentiation (forward + adjoint/reverse mode) | XAD C++ library (git submodule at `auto-differentiation/xad`), tape-based adjoint mode |
| **xtensor-r** | Zero-copy R↔C++ tensor bridge with broadcasting & lazy eval | `xt::rarray<double>`, `xt::rtensor<double, N>` — wraps R SEXP arrays in-place |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   R User API                        │
│  roptim::minimize(par, fn, method="ucminf",         │
│                   gradient="auto", ...)              │
└──────────────────────┬──────────────────────────────┘
                       │ Rcpp
┌──────────────────────▼──────────────────────────────┐
│              C++17 Integration Layer                │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ xtensor-r   │  │  XAD autodiff│  │ ucminf     │ │
│  │ rarray<T>   │  │  adjoint tape│  │ minimize   │ │
│  │ rtensor<T,N>│  │  forward mode│  │ _direct<F> │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                │                │         │
│         └────────────────┼────────────────┘         │
│              Unified fdf callable                   │
└─────────────────────────────────────────────────────┘
```

---

## 3. The Key Integration Point

The critical insight is that `ucminf::minimize_direct<F>` accepts **any callable** with signature:

```cpp
void(const std::vector<double>& x, std::vector<double>& g, double& f)
```

XAD's adjoint mode can compute the full gradient in a single backward sweep. The integration fuses them into a single `fdf` lambda that:

1. Receives `x` as `std::vector<double>`
2. Records a tape via XAD
3. Evaluates `f(x)` using XAD active types
4. Calls `computeAdjoints()` to get the full gradient
5. Writes `f` and `g` back

---

## 4. Concrete C++ Prototype — Rcpp Interface

```cpp
// src/roptim_autodiff.cpp
#define STRICT_R_HEADERS
#include <Rcpp.h>

// xtensor-r for array operations
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor-r/rarray.hpp"
#include "xtensor-r/rtensor.hpp"

// XAD autodiff
#include "XAD/XAD.hpp"

// ucminf optimizer (header-only)
#include "ucminf_core.hpp"

using AD = xad::adj<double>;       // adjoint (reverse) mode type
using Tape = AD::tape_type;

// [[Rcpp::export]]
Rcpp::List minimize_autodiff_cpp(
    Rcpp::NumericVector par,
    Rcpp::Function fn_r,           // R function f(x) -> scalar
    Rcpp::List control)
{
    int n = par.size();

    // Extract control parameters
    ucminf::Control ctrl;
    ctrl.grtol   = Rcpp::as<double>(control["grtol"]);
    ctrl.xtol    = Rcpp::as<double>(control["xtol"]);
    ctrl.stepmax = Rcpp::as<double>(control["stepmax"]);
    ctrl.maxeval = Rcpp::as<int>(control["maxeval"]);

    // The fdf lambda: uses XAD to autodiff any R function
    auto fdf = [&fn_r, n](const std::vector<double>& xv,
                           std::vector<double>& gv,
                           double& f)
    {
        Rcpp::NumericVector xr(xv.begin(), xv.end());
        f = Rcpp::as<double>(fn_r(xr));
        // For R functions, finite-difference fallback
        // For C++ objectives, replace with XAD tape (see Section 5)
    };

    std::vector<double> x0(par.begin(), par.end());
    ucminf::Result res = ucminf::minimize_direct(std::move(x0), fdf, ctrl);

    // Convert result to xtensor for any post-processing
    xt::rtensor<double, 1> x_out = xt::adapt(res.x, {static_cast<size_t>(n)});

    return Rcpp::List::create(
        Rcpp::Named("par")         = Rcpp::NumericVector(res.x.begin(), res.x.end()),
        Rcpp::Named("value")       = res.f,
        Rcpp::Named("convergence") = static_cast<int>(res.status),
        Rcpp::Named("neval")       = res.n_eval,
        Rcpp::Named("maxgradient") = res.max_gradient
    );
}
```

---

## 5. Pure C++ Autodiff Path (The Real Power)

The true state-of-the-art path is when the objective is defined **in C++**, so XAD can differentiate it exactly via tape:

```cpp
// src/roptim_xad_native.hpp
#pragma once
#include "XAD/XAD.hpp"
#include "ucminf_core.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

// Template: user defines objective as a function of xad active types
// The optimizer gets exact gradients for free via adjoint mode
template<typename UserObjFn>
ucminf::Result minimize_with_autodiff(
    std::vector<double> x0,
    UserObjFn obj_fn,              // obj_fn(std::vector<AD>& x) -> AD
    const ucminf::Control& ctrl = {})
{
    using AD = xad::adj<double>;
    using Tape = AD::tape_type;

    int n = static_cast<int>(x0.size());

    auto fdf = [&obj_fn, n](const std::vector<double>& xv,
                              std::vector<double>& gv,
                              double& f)
    {
        Tape tape;
        tape.activate();

        // Create XAD active variables
        std::vector<AD> x_ad(xv.begin(), xv.end());
        for (auto& xi : x_ad) tape.registerInput(xi);

        tape.newRecording();

        // Evaluate objective using active types
        AD y = obj_fn(x_ad);

        tape.registerOutput(y);
        xad::value(y) = 1.0;  // seed

        // Backward sweep — computes all gradients in O(1) * cost(f)
        tape.computeAdjoints();

        f = xad::value(y);
        for (int i = 0; i < n; ++i)
            gv[i] = xad::derivative(x_ad[i]);

        tape.deactivate();
    };

    return ucminf::minimize_direct(std::move(x0), fdf, ctrl);
}
```

---

## 6. R-Level API — Rosenbrock Example

```cpp
// src/roptim_rosenbrock_example.cpp
#include <Rcpp.h>
#include "roptim_xad_native.hpp"

// Compiled C++ + autodiff + optimizer — zero R overhead
// [[Rcpp::export]]
Rcpp::List minimize_rosenbrock(Rcpp::NumericVector par) {
    auto rosenbrock = [](std::vector<xad::adj<double>>& x) -> xad::adj<double> {
        auto dx = 1.0 - x[0];
        auto dy = x[1] - x[0] * x[0];
        return dx * dx + 100.0 * dy * dy;
    };

    auto res = minimize_with_autodiff(
        std::vector<double>(par.begin(), par.end()),
        rosenbrock
    );

    return Rcpp::List::create(
        Rcpp::Named("par") = Rcpp::NumericVector(res.x.begin(), res.x.end()),
        Rcpp::Named("value") = res.f,
        Rcpp::Named("convergence") = static_cast<int>(res.status)
    );
}
```

---

## 7. Package Structure

```
roptim/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── minimize.R              # High-level R API
│   └── zzz.R                   # Package load
├── src/
│   ├── Makevars                # -std=c++17 -I flags
│   ├── Makevars.win
│   ├── include/
│   │   ├── ucminf_core.hpp     # vendored from ucminfcpp
│   │   ├── ucminf_core_impl.hpp
│   │   └── roptim_xad_native.hpp
│   ├── xad/                    # git submodule → auto-differentiation/xad
│   ├── xtensor/                # vendored headers (or via inst/include)
│   ├── xtensor-r/              # vendored headers
│   ├── roptim_minimize.cpp     # Rcpp exports
│   └── RcppExports.cpp
├── inst/
│   └── include/                # Public headers for LinkingTo users
├── tests/
│   └── testthat/
└── vignettes/
```

---

## 8. Build Configuration

### Makevars

```makefile
CXX_STD = CXX17
PKG_CXXFLAGS = -I../inst/include \
               -I./xad/src \
               -I./include \
               -DSTRICT_R_HEADERS
PKG_LIBS = $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)
```

### DESCRIPTION

```
Package: roptim
Title: High-Performance Optimization with Automatic Differentiation for R
Version: 0.1.0
Depends: R (>= 3.5.0)
Imports: Rcpp (>= 1.0.0)
LinkingTo: Rcpp
SystemRequirements: C++17
License: GPL (>= 3)
```

---

## 9. R-Level API Design

```r
# R/minimize.R

#' High-level optimization with automatic gradient selection
#'
#' @param par     Numeric vector of starting values.
#' @param fn      Objective function to minimize.
#' @param gr      Optional gradient function. If NULL, auto-selected.
#' @param gradient Strategy: "auto" (XAD adjoint), "user" (use gr),
#'                 "forward" (finite-diff), "central" (finite-diff).
#' @param control  Control parameters (see ucminf_control).
#' @param ...      Additional arguments passed to fn and gr.
#' @return A list with par, value, convergence, and diagnostics.
#' @export
minimize <- function(par, fn, gr = NULL, ...,
                     gradient = c("auto", "user", "forward", "central"),
                     control = list()) {
    gradient <- match.arg(gradient)

    if (gradient == "auto" && is.null(gr)) {
        # Use XAD adjoint mode autodiff
        message("Using automatic differentiation (XAD adjoint mode)")
        # Route to minimize_autodiff_cpp(...)
    } else if (gradient == "user" || !is.null(gr)) {
        # Use user-supplied gradient
        # Route to ucminf_cpp(...)
    } else {
        # Use finite-difference
        # Route to ucminf_cpp(...) with grad_type = forward/central
    }
}
```

---

## 10. Where xtensor-r Adds Value

| Use Case | How xtensor-r Helps |
|---|---|
| **Post-processing** | Return Hessian/covariance as `xt::rtensor<double,2>` with zero-copy to R matrix |
| **Batch optimization** | Process multiple starting points as an xtensor matrix, broadcast operations |
| **Tensor objectives** | Users defining objectives on matrices/tensors get numpy-like syntax in C++ |
| **Lazy evaluation** | `xt::sin(m)`, `xt::sum()` etc. compute lazily — efficient for large-scale problems |
| **Future algorithms** | L-BFGS, trust-region methods benefit from efficient matrix operations |

---

## 11. Development Roadmap

### Phase 1 — Package Skeleton
- [ ] Create new `roptim` repository
- [ ] Vendor ucminfcpp headers (`ucminf_core.hpp`, `ucminf_core_impl.hpp`)
- [ ] Vendor xtensor-r headers
- [ ] Basic R pass-through to `minimize_direct`
- [ ] Tests: Rosenbrock, quadratic, and sphere functions

### Phase 2 — XAD Autodiff Integration
- [ ] Add XAD as git submodule
- [ ] Implement `minimize_with_autodiff<>` template
- [ ] Expose compiled C++ objectives via `Rcpp::XPtr` path
- [ ] Tests: verify exact gradients match analytical ones

### Phase 3 — High-Level R API
- [ ] Implement `minimize()` with `gradient="auto"` selection
- [ ] Auto-detect: user gradient → XAD adjoint → finite-difference fallback
- [ ] S3 print/summary methods for results
- [ ] Vignette: getting started

### Phase 4 — New Algorithms & Features
- [ ] Add L-BFGS algorithm option
- [ ] Add trust-region method
- [ ] Constrained optimization (box constraints)
- [ ] Batch optimization via xtensor
- [ ] Hessian computation via XAD forward-over-reverse

### Phase 5 — Polish & Release
- [ ] Performance benchmarks vs optim(), ucminf, nloptr
- [ ] CRAN-ready documentation
- [ ] CI/CD workflows (R CMD check on Linux/macOS/Windows)
- [ ] CRAN submission

---

## 12. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Tensor library | xtensor-r (not RcppArmadillo) | Zero-copy R arrays, lazy eval, numpy-like API, header-only |
| Autodiff | XAD (not Stan Math, CppAD) | Already integrated in xad-r, high performance, supports adjoint+forward |
| Optimizer core | ucminfcpp header-only | Template `minimize_direct<F>` enables full inlining of fdf, zero overhead |
| Build system | Makevars + vendored headers | CRAN-compatible, no external dependencies needed at install time |
| Gradient default | Adjoint (reverse) mode | O(1) gradient cost regardless of dimension — ideal for high-dimensional problems |

---

## 13. References

- [ucminfcpp](https://github.com/alrobles/ucminfcpp) — C++17 UCMINF optimizer
- [xad-r](https://github.com/alrobles/xad-r) — R bindings for XAD autodiff
- [xtensor-r](https://github.com/alrobles/xtensor-r) — R bindings for xtensor
- [XAD](https://github.com/auto-differentiation/xad) — Automatic differentiation C++ library
- Nielsen, H. B. (2000). *UCMINF — An Algorithm for Unconstrained, Nonlinear Optimization*. Report IMM-REP-2000-19, DTU.
