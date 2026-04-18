
# `xopt` — Analysis of the Integration Plan and a Generalized Research Roadmap

**Author:** Devin (for Angel, @alrobles)
**Scope:** Critical analysis of `alrobles/xopt/docs/INTEGRATION_PLAN.md` and a generalized design that unifies **ucminfcpp**, **xad-r** and **xtensor-r** into a state-of-the-art R optimizer, with a concrete research path for future features.
**Inputs read:** `alrobles/xopt`, `alrobles/ucminfcpp`, `alrobles/xad-r`, `alrobles/xad`, `alrobles/xtensor-r`, `alrobles/xtensor`. Comparative references: TMB/RTMB, `optimx::optimr`, Ceres (Jet / AutoDiffCostFunction), CasADi (symbolic graph + JIT), Stan Math, JuMP / Optim.jl.

---

## 1. Executive summary

The current plan is **directionally correct** (header-only core + template `minimize_direct<F>` + XAD tape for gradients + `xtensor-r` for tensor I/O) but has four gaps that will block SOTA status as written:

1. **The "autodiff any R function" promise in §4 is not achievable with XAD.** XAD is a C++ overloaded-type AD library; it can only differentiate code expressed in its active types. It cannot trace through `Rcpp::Function` calls. The plan's `fdf` lambda in §4 falls back to finite differences — which is fine, but should not be labeled "autodiff". This conflation will confuse users and reviewers.
2. **The objective abstraction is too narrow.** `void(const std::vector<double>& x, std::vector<double>& g, double& f)` is the right *innermost* contract, but the *user-facing* abstraction should be a `Problem` with pluggable traits (gradient mode, Hessian mode, constraints, bounds, sparsity, scaling, parameter shape). Ceres, CasADi, TMB, and `optimr` all converged on such an abstraction.
3. **Only BFGS + unconstrained is planned.** "SOTA" for an R optimizer in 2026 means: BFGS, L‑BFGS, trust‑region Newton, Gauss–Newton / Levenberg–Marquardt for least squares, projected / L‑BFGS‑B for box constraints, augmented Lagrangian / SQP / interior point for general constraints, and Laplace‑approximated marginal likelihood (RTMB territory). Shipping only ucminfcpp competes with `stats::optim` and `ucminf`, not with `nloptr`, `TMB`/`RTMB`, or `nlmixr2`.
4. **xtensor-r's role is under-specified.** The plan lists "post-processing, batch, tensor objectives" but doesn't commit to an API. The real leverage of `xt::rarray` / `xt::rtensor` is (a) letting users optimize over *parameters with structure* (matrices, tensors) without manual `unlist`/`relist`, and (b) exposing the Hessian / covariance as a proper R matrix with zero copy. Both should be first-class.

The rest of this document details these gaps and proposes a concrete, phased design.

---

## 2. What we actually have today (grounded in code, not the plan)

### 2.1 `ucminfcpp` — optimizer core

- Public header: `inst/include/ucminf_core.hpp` exposes
  - `ucminf::Status` (9 codes),
  - `ucminf::Control { grtol, xtol, stepmax, maxeval, inv_hessian_lt }`,
  - `ucminf::Result { x, f, n_eval, max_gradient, last_step, status, inv_hessian_lt }`,
  - `minimize(x0, ObjFun, control)` (std::function — R/Python/Julia path),
  - `minimize_direct<F>(x0, F, control)` (template — zero-overhead inline path).
- R API: `ucminf()`, `ucminf_xptr()`, `ucminf_control()`. The `_xptr` variant already takes `Rcpp::XPtr<ucminf::ObjFun>` — *this is the correct hook for the compiled-objective fast path*, and the plan should build on it rather than redefining the boundary.
- Limitations relevant to xopt:
  - Dense BFGS update — inverse Hessian is stored as packed lower triangle, O(n²) memory and update cost. Not suitable for n ≳ 10⁴.
  - Unconstrained only.
  - Single-scalar objective; no least-squares or sum-of-terms specialization.

### 2.2 `xad-r` / XAD — automatic differentiation

- xad-r exposes two S3 classes backed by `Rcpp::XPtr`:
  - `xad_adj_real` with operator overloads (`+ - * / ^`) and `adj_*` math functions, plus `gradient_adjoint(f, x)`.
  - `xad_fwd_real` with `gradient_forward(f, x)`.
- Native XAD capabilities (`src/xad/src/XAD/`) that xad-r does **not** yet surface but which matter for xopt:
  - `xad::fwd_adj<double>` second-order forward-over-adjoint → **exact Hessian** (`XAD/Hessian.hpp`, sample in `xad/samples/Hessian`).
  - `XAD/Jacobian.hpp` → multi-output Jacobian (needed for least-squares / Gauss–Newton).
  - `CheckpointCallback.hpp` → tape memory bounded for deep / long unrolled objectives.
  - `JIT*` headers → expression-DAG JIT (record once, compile, re-evaluate many times — analogous to CasADi's JIT and Ceres `forge` proposal).
- Important constraint for R: the S3 per-operation dispatch in `xad-r` is *not* cheap. Using `gradient_adjoint` on a pure-R closure gives correct gradients but is ≈ 10–100× slower per op than operating directly on `Rcpp::XPtr<adj_type>` in compiled code. This motivates the **two-tier gradient strategy** in §4.

### 2.3 `xtensor-r` — zero-copy R↔C++ tensors

- Headers: `rarray.hpp` (dynamic rank), `rtensor.hpp` (fixed rank), `rcpp_extensions.hpp` (Rcpp traits), `roptional.hpp` (NA-aware), `rvectorize.hpp` (numpy-style ufunc adapter). All zero-copy over `SEXP`.
- What it buys xopt:
  - Parameters can be declared as `xt::rtensor<double, N>` → the user's R object keeps its shape (e.g. weight matrices, `array(dim = c(...))`). The optimizer internally flattens to a `std::vector<double>` view but restores shape on output.
  - Hessians and covariance matrices are returned as `xt::rtensor<double, 2>` → zero-copy R matrix.
  - Broadcasting / lazy expressions (`xt::sum`, `xt::sin`) make vectorized penalty / regularization terms idiomatic in compiled objectives.
  - NA-aware `xt::roptional` is uniquely useful for statistical objectives where some observations are missing.

---

## 3. What is wrong (or missing) in the plan, in order of impact

| # | Issue in the plan | Why it matters | Proposed fix |
|---|---|---|---|
| P1 | §4 labels the R-function path "autodiff" but actually falls back to finite differences. | Misleading; users will report incorrect gradients. | Rename the three paths clearly: (a) `gradient="xad"` — *compiled C++ objective traced by XAD*, (b) `gradient="user"` — user-supplied `gr`, (c) `gradient="finite"` — FD with step policy. The R-function-only autodiff path requires an *R-side* tracer (see §5 — a TMB/RTMB-style approach) and is a phase-3 research item, not phase 1. |
| P2 | Objective abstracted only as `fn(x) → f` + optional `gr`. | Cannot express least squares, constraints, bounds, tensor params, Hessians, or sparsity. | Introduce a **`Problem`** abstraction (§4) analogous to Ceres `Problem` / NLPModels.jl / `optimr`'s expanded signature. |
| P3 | Only ucminfcpp's BFGS is planned. | ucminf is great at n ≲ 500 but cannot compete with `nloptr`/`TMB`/`optimx` above that. | Plan the solver layer behind a **dispatch interface** that accepts (x₀, f, ∇f, ∇²f, J, bounds, cons). BFGS is one backend; L‑BFGS, trust‑region Newton, Gauss–Newton/LM, and a projected variant (L‑BFGS‑B) are drop-in additions once the interface is right. |
| P4 | `rtensor` is claimed to help but never actually used in the §4 prototype. | Users will not gain tensor-shaped parameter support; the `rtensor::adapt` line is cosmetic. | Commit to two `rtensor` usages from day 1: (a) `par` can be an arbitrary R array; (b) return `hessian` / `inv_hessian` / `covariance` as `rtensor<double,2>`. |
| P5 | `Rcpp::Function` is called on every iteration. | R↔C++ round-trip dominates runtime (`ucminf_core.hpp:30-38` explicitly documents this). | Make the "fast path" the default whenever `fn` is a `Rcpp::XPtr<ucminf::ObjFun>` or an xad-r compiled closure, and document the overhead for the R-closure path. |
| P6 | XAD tape is constructed *inside* the fdf lambda on every call. | Tape allocation + activation/deactivation per iteration is avoidable overhead; it also breaks reuse of recorded graphs. | Hoist the tape out of `fdf` and use `newRecording()` per iteration; for static graphs, go further and record once → compile with `XAD::JIT*` → evaluate many times (CasADi pattern). |
| P7 | No `trace` / callback hook. | Practitioners need iteration logs and early stopping; all SOTA solvers expose a per-iteration callback. | Add `control$callback` (R function called with iteration, x, f, \|\|g\|\|, α) bridged via `Rcpp::Function` — *only if present*, so there is no overhead when unused. Thread-safe for single-thread use; disable for parallel multi-start. |
| P8 | No parameter transformation / scaling. | ucminf, like BFGS generally, is sensitive to scaling and to hard-positive or hard-simplex constraints. | Provide declarative transforms: `log_lower`, `logit`, `softmax`, `cholesky`, following RTMB's `obj$par` convention. These compose with AD so gradients are automatic. |
| P9 | Thread-safety not discussed. | XAD tapes are per-thread; ucminf's `minimize_direct` holds state for one solve. | Document: one `Tape` per solve, `minimize_*` is re-entrant across threads if each thread has its own problem instance. Enables multi-start parallelism (§6.4). |
| P10 | Version pinning / vendoring not specified. | xtensor has recently moved to C++20 (`target_compile_features(... cxx_std_20)`); XAD AGPL‑3 may conflict with GPL-3+ ucminfcpp for CRAN. | Pin xtensor to the last C++17 release, and confirm license compatibility (see §7.5). |

---

## 4. Generalized C++ design: the `Problem` abstraction

Replace the single `fdf` callback with a compile-time `Problem` trait struct, so the solver can query only what the user actually provides:

```cpp
// inst/include/xopt/problem.hpp
namespace xopt {

enum class Shape { Vector, Tensor };
enum class GradKind { None, UserFn, FiniteDiff, XadAdj, XadFwd };
enum class HessKind { None, UserFn, BfgsApprox, XadFwdAdj, FD };

template <class UserObj,
          GradKind Grad = GradKind::XadAdj,
          HessKind Hess = HessKind::BfgsApprox>
struct Problem {
    // Flattened parameter dimension
    int n_par;

    // Optional shape metadata (rtensor dimensions), used only at R boundary
    std::vector<std::size_t> par_shape;

    // Bounds; empty = ±inf
    std::vector<double> lower, upper;

    // Linear / nonlinear constraints (phase 4)
    // ... LinearCons, NonlinearCons ...

    // Parameter transform pipeline (log, logit, softmax, cholesky, ...)
    // ... Transform pipeline ...

    UserObj obj;            // user's objective in "internal" (transformed) coords

    // Required: scalar evaluation
    double value(const double* x) const;

    // Provided based on Grad policy
    void gradient(const double* x, double* g) const;

    // Provided based on Hess policy (may be a no-op / delegated to solver)
    void hessian(const double* x, double* H_rowmajor) const;

    // Jacobian for least-squares variants (phase 3)
    void residual(const double* x, double* r) const;          // m-dim
    void jacobian(const double* x, double* J_rowmajor) const; // m×n
};

} // namespace xopt
```

Key properties:

- **Template policies, not virtual calls.** `GradKind`/`HessKind` select the implementation at compile time — matches ucminfcpp's `minimize_direct<F>` philosophy and preserves full inlining.
- **Uniform solver-facing API.** Each algorithm (BFGS, L‑BFGS, TR, LM, L‑BFGS‑B) is templated on `Problem` and calls only the members it needs.
- **One path for tensor params.** The R boundary flattens `xt::rtensor` / `xt::rarray` into a contiguous buffer (zero-copy when layout permits, one reshape copy otherwise); after the solve it un-flattens back into an R array of the original shape.
- **AD modes are policies, not separate code paths.** `GradKind::XadAdj` is the default when the objective is expressible in XAD active types; `FiniteDiff` is the fallback; `UserFn` wins when the user supplies `gr`.

### 4.1 The four concrete routes that the R API dispatches to

The R user sees one function `xopt::minimize()`, but internally there are four C++ routes, chosen in this priority order:

1. **Compiled XAD objective (`XPtr` → `xad::adj<double>`)** — user has a `cppFunction()` / `sourceCpp()` snippet or a `XPtr<ObjFun>` built against xopt's headers. Gradients via `xad::adj` adjoint sweep; optionally Hessian via `xad::fwd_adj`. **Peak performance path.** This is effectively Ceres `AutoDiffCostFunction` on top of ucminf.
2. **Compiled user `XPtr<ObjFun>`** — user has already written `f + g` in C++ (e.g. via the existing `ucminfcpp::ucminf_xptr`). xopt just reuses it. **Backwards compatibility with ucminfcpp.**
3. **xad-r S3 objective** — user wrote `fn` in R but in terms of `xad_adj_real`. We call `gradient_adjoint(fn, x)` each iteration. Correct gradients, moderate speed (S3 dispatch is heavy).
4. **Plain R objective + finite differences** — the existing `ucminfcpp::ucminf` behavior. Correct, slow; default fallback when nothing better is available.

A fifth research route (§5) — an R-side tracer à la RTMB — is the only way to get *exact* gradients on a *plain* R function; it is explicitly a future phase.

### 4.2 Tape lifetime and JIT (fixes P6)

Replace the tape-per-call pattern with:

```cpp
template <class UserObj>
Result solve(Problem<UserObj, GradKind::XadAdj> p,
             const std::vector<double>& x0,
             const Control& ctl) {
    using AD = xad::adj<double>;
    typename AD::tape_type tape; tape.activate();

    std::vector<AD> x_ad(p.n_par);
    for (auto& a : x_ad) tape.registerInput(a);

    // Record once per iteration; optionally hoist further if graph is static.
    auto fdf = [&](const std::vector<double>& xv, std::vector<double>& gv, double& f) {
        for (int i = 0; i < p.n_par; ++i) xad::value(x_ad[i]) = xv[i];
        tape.newRecording();
        AD y = p.obj(x_ad);
        tape.registerOutput(y);
        xad::derivative(y) = 1.0;
        tape.computeAdjoints();
        f = xad::value(y);
        for (int i = 0; i < p.n_par; ++i) gv[i] = xad::derivative(x_ad[i]);
    };
    return ucminf::minimize_direct(std::vector<double>(x0), fdf, ctl.to_ucminf());
}
```

**JIT escalation (research).** When the objective's expression graph is static (no branches depending on `x`), XAD's `JITCompiler` can compile the recorded graph to native code and replay it per iteration without re-tracing — the same trick Ceres' proposed `forge` integration and CasADi's `jit=true` option provide. Worth prototyping in phase 4.

---

## 5. The hardest generalization: exact gradients for a plain R objective

This is the feature that would genuinely make `xopt` competitive with **TMB / RTMB**. Two viable designs:

**A. RTMB-style: ADify the R interpreter for a subset of operators.** RTMB overloads `+`, `-`, `*`, `%*%`, `sum`, `log`, distributions, … on a CppAD-backed S4 class so that a plain-looking R function is automatically traced. This is exactly what `xad-r` already does for `+ - * / ^` and the `adj_*` functions — the missing piece is a *masking layer* that swaps `base::sin` → `adj_sin` inside a user-supplied `fn` at call time. Concretely: evaluate `fn` in an environment whose parent is a curated namespace of xad-r shadows of `base` / `stats`. RTMB proves this is feasible; it is also the cleanest upgrade path for existing `ucminfcpp::ucminf` users who do not want to rewrite in C++.

**B. Ceres `forge`-style: JIT-compile a recorded trace.** On the first few calls, record the expression DAG with `xad::fwd<double>` (cheap one-input warm-up). Then feed that DAG to `XAD/JITCompiler.hpp`, produce a shared object, and swap the per-iteration path to a direct call into the compiled kernel. Works only for straight-line (control-flow-free) objectives, but covers a huge fraction of statistical likelihoods.

Both are **phase 3 research**, not phase 1.

---

## 6. Algorithm matrix — what "state-of-the-art" should mean here

| Algorithm | Why include | Inputs needed | Gradient mode sweet spot | Constraints |
|---|---|---|---|---|
| UCMINF BFGS (have) | Robust default, drop-in for `stats::optim("BFGS")`. | f, ∇f | adjoint | none |
| L‑BFGS | Large n (≥ 10⁴), limited memory. | f, ∇f | adjoint | none |
| L‑BFGS‑B | Box-constrained large n. | f, ∇f, bounds | adjoint | box |
| Trust-region Newton (CG / dogleg) | Exact Hessian wins near optimum. | f, ∇f, H·v or H | adjoint + fwd_adj or HVP | none |
| Gauss–Newton / Levenberg–Marquardt | Nonlinear least squares dominates stats/ecology (and this user's `rxbioclim` / `maxentcpp` work). | r(x), J(x) | forward for small m, adjoint otherwise | optional box |
| Augmented Lagrangian (AL) | Equality + inequality constraints with any inner solver. | f, ∇f, c, ∇c | adjoint | general |
| SQP | Gold standard for medium-size NLP. | f, ∇f, H (or quasi-Newton), c, ∇c | adjoint + fwd_adj | general |
| Interior point (Mehrotra) | Large-scale NLP; needs sparsity. | f, ∇f, ∇²L, c, ∇c | adjoint + sparse | general |
| Laplace-approx marginal likelihood | Random-effects models; the thing TMB/RTMB is famous for. | f, ∇f, ∇²f | adjoint + fwd_adj | none |

Phases 1–2 should ship UCMINF + L‑BFGS + L‑BFGS‑B (all use only `f, ∇f`, so they come free with the `Problem` abstraction and the existing xad-r adjoint path). LM is a natural phase 3 because it needs a Jacobian path. AL/SQP/IP belong to phase 4–5.

### 6.1 Least-squares structure (phase 3 focus)

R users (especially in the author's `rxbioclim` / `maxentcpp` workflows) regularly fit models where the objective is ½·∑ rᵢ(x)². Giving `xopt` a specialized entry point:

```r
xopt::nls_minimize(par, residual_fn, method = "lm", ...)
```

is a big win because the Jacobian (via `xad::Jacobian`) is much cheaper than the Hessian, and LM is typically 3–10× faster than BFGS for NLS.

### 6.2 Hessian-vector products (HVP)

For large-scale Newton-CG, we never form the Hessian explicitly — we only need `v → ∇²f(x) · v`. XAD's forward-over-adjoint mode (`xad::fwd_adj`) computes HVP in O(cost(f)), which is the right complexity. This should be exposed as a first-class operation in `Problem`.

### 6.3 Sparsity

Statistical / PDE-derived Hessians are usually sparse. Medium term: hook into `Matrix::dgCMatrix` via `Rcpp::S4`, and pair with a sparse linear solver (`Eigen::SimplicialLDLT` or SuiteSparse CHOLMOD through `RcppEigen`).

### 6.4 Multi-start and batched optimization

xtensor-r makes this almost free: accept `par` as an `(n_starts, n_par)` matrix, run solves in parallel with `RcppParallel` / `std::thread`, each with its own `tape`. Return an `(n_starts, …)` tensor of results plus the global best.

---

## 7. Engineering concerns

### 7.1 The R user API

Target a single function mirroring `optimr::optimr` (which itself mirrors `optim`) so existing code ports immediately:

```r
xopt::minimize(
  par,                # numeric vector OR array (any dim) OR list of arrays
  fn,                 # R closure, Rcpp::XPtr, or a compiled C++ Problem (SEXP tag)
  gr     = NULL,      # optional gradient
  hess   = NULL,      # optional Hessian or Hessian-vector-product
  lower  = -Inf,
  upper  =  Inf,
  method = c("ucminf", "lbfgs", "lbfgsb", "tr-newton", "lm"),
  gradient = c("auto", "xad", "user", "central", "forward"),
  hessian  = c("none", "bfgs", "xad", "user"),
  control = xopt_control(...),
  problem = NULL      # fully-specified Problem XPtr, shortcut past fn/gr/hess
)
```

The result should be backwards-compatible with `optim()` *and* carry richer fields:

```r
list(par, value, counts, convergence, message,
     gradient, hessian, covariance,     # zero-copy rtensor<double,2>
     trajectory = tibble(iter, f, gnorm, step, alpha),
     diagnostics = list(...))
```

### 7.2 Control parameters

Inherit from `ucminf_control()` and extend per-method. Use a single `xopt_control()` factory with method-specific sub-lists, validated. Make `callback` a first-class entry (P7).

### 7.3 Package / build layout

Keep the plan's layout but with two adjustments:

- Do **not** vendor xtensor / xtensor-r / XAD directly. Consume them via `LinkingTo: xtensorR, xadr` (i.e. other alrobles packages that already CRAN-package those headers). This prevents triple-maintenance of the same vendored code and — crucially — lets CRAN see the license chain cleanly.
- `inst/include/xopt/` must ship the `Problem` headers so that downstream packages (e.g. a future `maxentxopt` or `rxbioclim`) can use xopt directly from C++ with `LinkingTo: xopt`.

### 7.4 Testing strategy

- **Correctness.** `testthat` suite on Rosenbrock, extended Rosenbrock (n up to 10⁴), Powell singular, Beale, Brown badly-scaled, Broyden tridiagonal, Nielsen benchmark set (already used in ucminfcpp). Gradient correctness checked via finite differences and vs xad-r.
- **Parity with `stats::optim` and `ucminf`.** Reuse ucminfcpp's `bench_r_vs_cpp.R` pattern.
- **End-to-end statistics example.** A GLM fitted via `xopt::minimize` should match `glm(..., family=)` coefficients to 1e-8, demonstrating the R-function + xad-r adjoint path.
- **Performance benchmarks** documented in a vignette, vs `optim`, `ucminf`, `ucminfcpp`, `nloptr`, `TMB`/`RTMB`.

### 7.5 Licensing

- `ucminfcpp` is GPL-3+.
- `xtensor` / `xtensor-r` are BSD-3.
- **XAD is AGPL-3** (see `xad-r/DESCRIPTION` → `License: AGPL-3`). An AGPL header compiled into a GPL package produces an AGPL binary. This is compatible on CRAN but must be declared. Alternative: gate the XAD path behind a SystemRequirements toggle so a pure ucminfcpp build remains GPL-3 for users who cannot accept AGPL. Worth confirming with Xcelerit before CRAN submission; Stan Math chose a different AD lib specifically to stay BSD.

### 7.6 Numerical hygiene — things optimizers get wrong in practice

These are table stakes and must be in phase 1 (they are *not* in the current plan):

- **Consistent initial step (`stepmax`).** Auto-scale from `||x₀||` and a short probing line search rather than the bare `1.0` default.
- **Gradient check on startup** when `trace > 0`: compare user-supplied `gr` (or xad-r gradient) against central differences at `x₀`; warn on mismatch above 1e-4.
- **Finite-difference step policy.** Use Nielsen/Press-style `h = max(1, |xᵢ|) · ε` with different `ε` for forward vs central; expose `gradstep` (ucminfcpp already does this).
- **Parameter scaling diagnostic.** Warn when `max(|xᵢ|)/min(|xᵢ|) > 1e6` on solution.
- **Reproducibility.** Record the RNG seed if multi-start; record the XAD tape stats when `trace > 1`.

---

## 8. Phased roadmap (concrete)

Each phase ends with a release candidate; none is marked done until its test suite + vignette ship.

### Phase 0 — Scaffolding (1 week)
- `xopt` package skeleton (already a stub).
- `LinkingTo: ucminfcpp, xadr, xtensorR`. Confirm all three expose their headers in `inst/include/`.
- `Makevars` with `-std=c++17` (xtensor has moved to C++20 but offers a C++17 compatibility shim; pin accordingly).
- CRAN-style CI: R-CMD-check on Linux/macOS/Windows, GitHub Actions.

### Phase 1 — Minimal viable xopt = ucminfcpp + xad-r (2 weeks)
- `Problem<UserObj, Grad=XadAdj>` abstraction (§4).
- R API: `xopt::minimize()` dispatching to
  - compiled C++ objective via `XPtr<ObjFun>` (reusing `ucminfcpp::ucminf_xptr`);
  - xad-r S3 objective via `gradient_adjoint`;
  - plain R objective via ucminfcpp finite-differences.
- `rtensor`-shaped `par` (flatten on entry, unflatten on exit).
- Gradient-check diagnostic, `trace`, `callback`, `xopt_control()`.
- Tests: Rosenbrock, quadratic, Nielsen set, GLM-as-MLE.
- Vignette: "Three ways to hand xopt your objective".

### Phase 2 — Algorithm breadth (3–4 weeks)
- Implement L‑BFGS (two-loop recursion; header-only, same `minimize_direct<F>` pattern). Memory policy `m` exposed via control.
- Implement L‑BFGS‑B (Byrd–Lu–Nocedal) for box bounds.
- Add `method=` dispatch and cross-method benchmarks.
- Extend `Problem` with `lower`, `upper`, `Transform` pipeline (log / logit / softmax / cholesky).
- Expose final Hessian approximation (ucminf's final inverse Hessian is already available) as `xt::rtensor<double,2>` covariance.

### Phase 3 — Exact Hessians, least squares, R-side AD (4–6 weeks)
- Hook `xad::fwd_adj` → Hessian and HVP; expose in `Problem`.
- Trust-region Newton (Steihaug CG) using HVP.
- `xopt::nls_minimize()` with Levenberg–Marquardt and `xad::Jacobian`.
- **Research:** R-side tracer (RTMB-style) on top of xad-r S3 to give exact gradients for plain R functions. Vignette: "Automatic differentiation for R functions, no C++ required".
- Sparsity: optional `Matrix::dgCMatrix` Hessian path via RcppEigen.

### Phase 4 — Constraints and scale (6–8 weeks)
- General nonlinear constraints: augmented Lagrangian + inner L‑BFGS‑B / SQP.
- Multi-start and batched solves over `(n_starts, n_par)` rtensor with `RcppParallel`.
- JIT escalation: prototype `xad::JITCompiler` on recorded graphs; fall back gracefully when unavailable.
- Checkpointing for deep objectives (`xad::CheckpointCallback`).

### Phase 5 — Statistical models & polish (ongoing)
- `xopt::laplace_marginal()` — Laplace approximation over integrated random effects, the defining feature of TMB/RTMB.
- `broom`-style tidiers (`tidy`, `glance`, `augment`).
- `pkgdown` site, complete benchmark vignette vs `optim`, `ucminf`, `nloptr`, `TMB`, `RTMB`.
- CRAN submission.

---

## 9. Open research directions once the above is stable

1. **GPU / SIMD objectives.** xtensor has an `xsimd` backend and there are experimental CUDA layers; XAD has an opt-in AVX-enabled tape. For PDE-constrained optimization or batched ML problems on R, this is a genuine differentiator — no R optimizer currently offers it.
2. **Probabilistic programming back-end.** With Laplace + Hessian, `xopt` could back a lightweight PPL similar to brms but with fully compiled gradients. This is where the user's `ecoagent` / `maxentcpp` pipelines could benefit structurally.
3. **Implicit differentiation.** Expose `solve(F(x,θ)=0) → ∂x/∂θ` for fixed-point / optimization-layer gradients — a growing area (JAXopt, `Optax`, `ImplicitDifferentiation.jl`).
4. **Reinforcement-learning-style stochastic optimization.** Variance-reduced SGD, Adam, RMSprop as additional methods — trivially plugged into the `Problem` interface because they only need `∇f`.
5. **Automatic problem conditioning.** Detect ill-scaling from the diagonal of the initial BFGS or Gauss–Newton approximation and auto-insert a `Transform::Scale`.
6. **Interop with `TMB::MakeADFun` output.** Users with existing TMB models should be able to pass `obj` directly to `xopt::minimize` — an adapter around `obj$fn`/`obj$gr`/`obj$he` gives instant reach into the TMB userbase.

---

## 10. What I recommend you commit to *before* starting phase 1

These are small, cheap, irreversible-in-a-good-way decisions:

1. **`LinkingTo`, don't vendor.** Depend on `ucminfcpp`, `xadr`, `xtensorR` as sibling packages with headers exported via `inst/include/`.
2. **`Problem` header is a public header under `inst/include/xopt/`.** Downstream packages can use xopt without touching its internals.
3. **Dispatch by objective *type*, not by string.** If the user passes an `XPtr<ObjFun>`, always prefer it; if they pass an `xad_adj_real`-valued closure, take the xad-r adjoint path; else finite differences. The `gradient="auto"` default just codifies that precedence.
4. **One result class, one control factory.** `xopt_result` subclasses `list` with `print.xopt_result`, `summary.xopt_result`, `coef`, `vcov`, `logLik`. `xopt_control()` is the single source of truth for every method.
5. **AGPL clarity.** Either accept AGPL for the whole package, or gate XAD behind a `USE_XAD=1` build flag so the default CRAN binary is GPL-3. Decide once, up front.

Once those five are nailed, phases 1 and 2 become almost mechanical — and phase 3 onward is where the research value of integrating three high-quality header-only C++ libraries into R actually shows up.
