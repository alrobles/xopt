# xopt Roadmap v3 — State-of-the-Art Unconstrained Optimization for R

**Status:** draft (2026-04-18) — supersedes prior `INTEGRATION_PLAN.md`,
`ENHANCED_IMPLEMENTATION_PLAN.md`, and external roadmap drafts.

This document reconciles three inputs into one authoritative plan:

1. the original `INTEGRATION_PLAN.md` (ucminfcpp + xad-r + xtensor-r),
2. the phase-2 redesign sketch (general SOTA optimizer; xtensor as shape
   vocabulary; ODE / inverse-problem competitiveness),
3. an external DeepSeek-generated "Complete Automatic Differentiation in xopt"
   roadmap.

We assess each against the **actual current state of `main`** (post PR #30) and
derive a single forward plan.

---

## 0. Guiding philosophy

xopt is a general-purpose **unconstrained** optimization package for R whose
distinguishing claims are:

1. **Anchored by ucminfcpp.** `ucminf::minimize_direct<F>` is the default
   solver and the only one with a zero-overhead header-only fast path.
2. **Backed by XAD.** Gradients, Jacobians, Hessians, and Hessian–vector
   products are computed by the XAD tape when the objective is available as
   C++ with templated Scalar type.
3. **Shape-aware via xtensor.** Parameters, residuals, Jacobians, Hessians,
   L-BFGS history, multi-start ensembles, and ODE trajectories all carry an
   xtensor identity; the R↔C++ boundary is zero-copy.
4. **Honest about what AD it actually delivers.** No magical "AD any R
   function" claim. Gradient modes are labelled explicitly; silent fallback
   to finite differences is a warning, not the default.
5. **Competes on complex problems.** ODE parameter fits, NLS on NIST, Laplace
   approximation, implicit-function differentiation — not just a `stats::optim`
   replacement.

Explicit non-goals for the entire v3 horizon:

- No MCMC sampler, no posterior inference, no DSL compiler.
- No in-package ODE / DAE / PDE solver.
- No GPU backend.
- No general nonlinear-equality/inequality constraints beyond augmented
  Lagrangian.
- **No "compile any R function to XAD at runtime" feature.** See §5.1.

---

## 1. What is already on `main` (honest audit)

Commit `029843f` + PR #30 (`c1ccdf0`). Concrete artefacts present today:

### 1.1 R surface (`R/`)

| File | Role |
|------|------|
| `R/minimize.R` | `xopt_minimize(par, fn, gr, method, gradient, lower, upper, control)`; routes `gradient = "auto" \| "user" \| "finite"`. |
| `R/multistart.R` | `xopt_multistart(par_matrix, fn, ...)` parallel multi-start driver. |
| `R/ad_tracer.R` | `xopt_ad_trace(fn)` — masks base-R math in the closure env with `xadr::adj_*` equivalents for the whitelist `sin cos tan exp log sqrt asin acos atan abs gamma lgamma digamma trigamma`. |
| `R/control.R` | `xopt_control(...)` control-struct builder. |

### 1.2 C++ headers (`inst/include/xopt/`)

| Header | Contents |
|--------|----------|
| `problem.hpp` | `ProblemBase<Scalar>`, `enum GradKind { XadAdj, XadFwd, UserFn, FiniteDiff, None }`, `enum HessKind { XadFwdAdj, BfgsApprox, LbfgsApprox, UserFn, None }`. |
| `param_spec.hpp` | `ParamSpec` with `positive`, `bounded`, `simplex`, `spd_chol`, `identity` differentiable transforms. |
| `benchmarks.hpp` | Rosenbrock, PowellSingular, BrownBadlyScaled, BroydenTridiagonal, Beale, Sphere, Quadratic. |
| `ad_reduce.hpp` | `xopt::sum_active` helper (xtensor + XAD composability shim). |
| `diagnostics.hpp` | Solver diagnostic utilities. |
| `solvers/ucminf_solver.hpp` | `ucminf_solve(problem, ...)` and `std::function` overload (now routed through `minimize_direct` — PR #30). |
| `solvers/trust_region_newton.hpp` | Steihaug-CG trust-region Newton. |
| `solvers/nls_solver.hpp` | Levenberg–Marquardt. |
| `laplace.hpp` | Laplace approximation to the marginal likelihood. |
| `second_order.hpp` | Exact Hessian-vector product machinery via XAD forward-over-adjoint. |
| `phase4.hpp` | Augmented-Lagrangian + sparsity prototypes. |

### 1.3 Exercise tests (`src/test_*.cpp`, all Rcpp-exported, return 0 on pass)

- `test_benchmarks.cpp` — eight standard unconstrained benchmarks.
- `test_param_spec.cpp` — transform round-trip tests.
- `test_nls.cpp` + `test_nls_nist.cpp` — NLS + NIST certified datasets.
- `test_multi_start.cpp` — multi-start ensemble (100/100 converged on Rosenbrock).
- `test_phase3.cpp` — trust-region Newton + Laplace.
- `test_phase4.cpp` — AL + sparse Jacobian prototype.
- `probe_xad_xtensor.cpp` — end-to-end xtensor × XAD adjoint gradient check.

### 1.4 Packaging

- `DESCRIPTION`: `License: AGPL-3`, `SystemRequirements: C++20`,
  `Depends: R (>= 4.2.0)`, `LinkingTo: Rcpp, ucminfcpp`,
  `Remotes: alrobles/ucminfcpp`.
- `Makevars` / `Makevars.win` pin `CXX_STD=CXX20` and suppress the Rcpp
  `-Wcast-function-type` noise.
- `tools/vendor-headers.sh` vendors xtensor / xad-r / ucminfcpp headers
  under `inst/include/`.
- One vignette: `vignettes/three-ways-to-optimize.Rmd`.
- CI: GitHub Actions on Ubuntu / macOS / Windows, R 4.5.

### 1.5 Upstream dependencies as they exist today

- `alrobles/ucminfcpp` — already shipped, exposes `ucminf::minimize_direct<F>`
  (header-only template) and `ucminf::minimize(...)` (non-inline, defined in
  `src/ucminf_core.cpp`). xopt now uses the former exclusively (PR #30).
- `alrobles/xad-r` (R package name `xadr`) — already shipped, exports
  `adj_Real`, tape management (`adj_createTape`, `adj_newRecording`,
  `adj_computeAdjoints`, `adj_clearDerivatives`), adjoint arithmetic
  (`adj_add*`, `adj_sub*`, `adj_mul*`, `adj_div*`), and adjoint-mode math
  for the 14-function whitelist used by `R/ad_tracer.R`. Forward-mode
  bindings via `fwd_1st.R`.
- `alrobles/xtensor`, `alrobles/xtensor-r` — already shipped; `xt::rarray`,
  `xt::rtensor`, zero-copy from R SEXP arrays.

**Bottom line:** phases 1–3 of most prior roadmaps are already landed. The
remaining work is *advanced* features and *honest API surface*, not basic
integration.

---

## 2. Audit: what DeepSeek's roadmap got right, wrong, and missed

DeepSeek's plan proposes four phases totalling ≈4 months. Against current
`main`:

### 2.1 Already done (DeepSeek's phases 1 + 2 + most of 3)

| DeepSeek task | Current state |
|---|---|
| "Integrate XAD into the build system (submodule / `FetchContent`)" | `LinkingTo:` via `xad-r` + vendored headers. `FetchContent` is CMake-only and inapplicable to R packages. |
| "Write C++ AD tests (Rosenbrock, sphere, quadratic) using `xad::FReal`/`AReal`" | `src/test_benchmarks.cpp`, `src/probe_xad_xtensor.cpp`. |
| "Create `xadr` R package skeleton" | `alrobles/xad-r` already exists as a full R package with ~70 exported functions. |
| "Implement `ad_gradient` / `ad_jacobian` / `ad_hessian` / `ad_hvp`" | Low-level primitives exist in `xad-r`; convenience wrappers are ~2 days of work. |
| "Enhance trust-region Newton to accept HVP function" | Present in `inst/include/xopt/solvers/trust_region_newton.hpp` (Steihaug-CG over HVPs). |
| "Upgrade NLS solver" | Already C++ Levenberg–Marquardt in `inst/include/xopt/solvers/nls_solver.hpp`. |
| "Integrate into augmented Lagrangian" | Prototype in `inst/include/xopt/phase4.hpp`. |

Treating these as 2–3 months of future work mischaracterises the project.

### 2.2 Architecturally infeasible as described

> "Develop a runtime compilation helper. Create `xadr::compile(fn)` that takes
> an R function, translates it to a C++ expression using XAD, and compiles it
> on the fly via `Rcpp::cppFunction()`."

This is not buildable for arbitrary R. R combines:

- **Dynamic typing with S3 / S4 dispatch.** `fn(x)` may resolve to different
  methods depending on `class(x)`; there is no single static C++ expression.
- **Non-standard evaluation and lazy evaluation.** Functions may inspect and
  rewrite their own call expressions, defer argument evaluation, or modify
  enclosing environments. XAD cannot trace through semantics that do not
  exist at the C++ level.
- **`...` forwarding, `do.call`, `Reduce`, `Map`, `sapply`, closures over
  captured mutable state.** No direct C++ translation.
- **Data-dependent branching.** `if (sum(x) > 0) branch_a(x) else branch_b(x)`
  has no single XAD graph; it has two.
- **Calls into base R internals and external packages.** Any call that ends
  up in compiled C code written without XAD types (every `stats::`, every
  Matrix op, every user-compiled package) is opaque to the tape.

Every serious project that has attempted "AD from arbitrary host-language
code" has, in the end, restricted itself to a **domain-specific subset**:

- **RTMB / TMB** — restricted tagged DSL over `advector`; years of engineering,
  and still not all of R.
- **PyTorch / JAX** — tracing works only over tensor ops from a specific
  library, not over arbitrary Python (try differentiating a function that
  calls `json.loads` — it won't work).
- **Julia Zygote / Enzyme** — operate on Julia's **typed SSA IR**, not on
  source text.

xopt should therefore **not** promise "`gradient = "auto"` just works for
any R function". §5.1 defines the honest alternative.

### 2.3 Genuinely new items buried in DeepSeek's "Phase 4 (ongoing)"

These are the items DeepSeek identified that *are* missing from `main`. They
are the entire content of v3:

1. **Checkpointing** for long adjoint tapes (XAD `CheckpointCallback`).
2. **Sparse AD** with Curtis–Powell–Reid-style graph colouring and
   compressed forward-mode Jacobian evaluation. A stub exists in
   `src/test_phase4.cpp` but is not exposed through the Problem API.
3. **Differentiation through optimisation** via implicit function theorem —
   the flagship SOTA feature. Completely missing from `main`.
4. **Benchmark report vs. analytic / FD / competitor packages.**

### 2.4 Missing from DeepSeek but critical for SOTA

- **`xopt::linalg`** — differentiable `chol`, `solve`, `logdet`, `inv`.
  Without these, Gaussian likelihoods, Laplace approximation (already in
  the repo but currently computed with hand-rolled rules), Kalman smoothers,
  and GP hyperparameter fits cannot be differentiated end-to-end.
- **L-BFGS and L-BFGS-B solvers.** UCMINF carries a dense `n × n` inverse
  Hessian; it does not scale past `n ≈ a few hundred`. L-BFGS with
  `O(n · m)` memory is mandatory for competing with `nloptr::lbfgsb3c` and
  `stats::optim(method = "L-BFGS-B")` at scale.
- **ODE sensitivity hook.** An optional `Problem::sensitivity()` method so
  users can plug SUNDIALS / CVODES forward sensitivities and skip taping
  through their own integrator. xopt itself ships no ODE solver.
- **Honest gradient-mode labelling.** See §5.1.

---

## 3. Assessment: prior INTEGRATION_PLAN.md

Still accurate at the level of *what the package is*. Specifically wrong at
the level of *what AD xopt can deliver*:

> "... autodiff any R function."

No. The current `R/ad_tracer.R` path differentiates only the closed-form
base-R math whitelist. Any R function that calls `stats::`, any compiled
external code, any control flow beyond pure math — still finite differences
or user gradient. v3 tightens this language.

The architecture diagram in `INTEGRATION_PLAN.md` §2 remains correct. The
`Problem<UserObj, GradKind, HessKind, Scalar>` template in
`inst/include/xopt/problem.hpp` is already the policy-based design the plan
described.

---

## 4. The four gradient paths (honest labelling)

xopt supports **four** gradient paths. None of them is "AD any R function".

| Path | User writes | What xopt does | Accuracy | Cost |
|------|-------------|----------------|----------|------|
| **A. compiled-XAD** | C++ `template <class Scalar> Scalar model(const xt::xtensor<Scalar,1>& p)` via `Rcpp::sourceCpp` or `LinkingTo` | Instantiate template with `xad::AReal`, adjoint sweep | machine precision | 1 forward + 1 reverse pass per grad |
| **B. user-R-gradient** | R `fn` **and** R `gr(x)` | Call `gr(x)` directly | depends on user | 1 user call |
| **C. finite-differences** | R `fn` only | Central differences in `.xopt_fd_gradient` | O(ε) | `2n` function evals |
| **D. traced-R (experimental)** | R `fn` that uses *only* whitelisted base-R math | `xopt_ad_trace(fn)` re-evaluates in a masked env with `xadr::adj_*` | machine precision, *when it applies* | ≈1 forward + 1 reverse pass |

API-level implication for `xopt_minimize(..., gradient = ...)`:

```r
gradient = c("auto", "compiled", "user", "fd", "traced")
```

where `"auto"` tries **compiled → user → traced → fd** in that order and
**emits a warning** when it drops below compiled/user (because that is the
point where the solver silently loses AD). Current `"auto"` goes
`user → traced → fd` already; v3 adds the compiled path above it and the
warnings below it. Signal the selected mode on the returned object
(`result$gradient_mode`).

---

## 5. v3 work plan

Four phases, each a discrete PR series. Target total ≈10 weeks of actual
new engineering beyond what is on `main` today.

### 5.1 Phase 5A — "SOTA anchors" (~4 weeks)

Each item is a standalone PR with unit tests and a minimal vignette section.

#### 5A-1. `xopt::linalg::{chol, solve, logdet, inv}` with AD rules (~2 weeks)

Unblocks Laplace / Gaussian / GP / Kalman workflows.

- Dense backend only; calls R's LAPACK via `xt::adapt` on raw pointers.
- Hand-coded adjoint rules (no XAD tape through LAPACK):
  - `chol(A) = L`:  `Ā = L^{-T} · Φ(L̄ᵀ L) · L^{-1}` where Φ is the
    lower-triangle-half-diagonal operator.
  - `solve(A, b) = x`: `Ā = -x̄ · xᵀ`, `b̄ = A^{-T} · x̄`.
  - `logdet(A) = 2 · Σ log(diag(L))`: `Ā = A^{-T}`.
  - `inv(A)`: rarely needed for performance, but symbolic AD rule provided.
- Header `inst/include/xopt/linalg/{chol,solve,logdet,inv}.hpp`.
- Tests: compare forward gradients against finite-difference ε-gradients on
  random SPD matrices; compare reverse gradients against user-coded closed
  forms on a 2-D Gaussian log-likelihood.
- Acceptance vignette: Laplace approximation for a small logistic GLMM
  `(y | β, u) ~ Bern(logit^{-1}(Xβ + Zu))`, integrating out `u` via
  `xopt::laplace`, differentiating through `logdet(H(β))`.

#### 5A-2. L-BFGS solver (~1.5 weeks)

- New header `inst/include/xopt/solvers/lbfgs_solver.hpp`.
- Memory as `xt::xtensor<double, 2>` of shape `(m, n)` for `s_k`/`y_k` stacks.
- Strong Wolfe line search.
- Interface: `lbfgs_solve(problem, control)` consuming the existing
  `Problem<UserObj, GradKind, HessKind, Scalar>` abstraction.
- Benchmarks on a `n = 10 000` L2-regularised logistic regression vs.
  `stats::optim("L-BFGS-B")` and `lbfgsb3c::lbfgsb3c` (wall-clock + iteration
  count).

#### 5A-3. Sparse Jacobian via colouring (~1 week)

- Graph-colouring on user-declared sparsity (or forward-mode seed detection).
- Compressed evaluation with XAD forward-mode.
- Return `Matrix::dgCMatrix` through a thin R wrapper.
- Tests: sparsity patterns from the `test_phase4.cpp` prototype; verify
  dense and compressed Jacobians match.

### 5.2 Phase 5B — "the differentiator" (~4 weeks)

The flagship SOTA feature: differentiating through the *result* of an
optimisation.

#### 5B-1. `FixedPointProblem` + `ImplicitFunction` API (~2 weeks)

Given `g(x, θ) = 0` solved for `x*(θ)`, the implicit function theorem gives

```
∂x*/∂θ = − (∂g/∂x)^{-1} · (∂g/∂θ).
```

- New header `inst/include/xopt/implicit/fixed_point.hpp`.
- Users supply (a) a solver for `g(x, θ) = 0`, (b) access to `∂g/∂x` and
  `∂g/∂θ` via XAD or user Jacobians.
- xopt returns `x*` with attached adjoint machinery that on backward
  evaluates the linear solve using `xopt::linalg::solve`.
- Tests: matrix-completion fixed point, implicit layer on a 2-D rootfinder.

#### 5B-2. `xopt_differentiate(result, hyper)` (~2 weeks)

- Treat KKT optimality conditions of any `xopt_minimize` result as a fixed
  point:   `∇_x f(x*, θ) = 0`.
- Return the Jacobian `dx*/dθ` (and via chain rule, `dL/dθ` for any downstream
  loss `L`).
- R-level convenience wrapper; C++ kernel.
- Acceptance vignette: hyperparameter gradient on a ridge-regression
  regularisation strength; compare to analytic `(XᵀX + λI)^{-1}` gradient.

### 5.3 Phase 5C — "long tapes" (~2 weeks)

#### 5C-1. Checkpointing hook on the compiled-XAD path (~1 week)

- Expose `xad::CheckpointCallback` through the `Problem` API.
- Optional `Problem::checkpoint_schedule()` returning a vector of step
  indices where the tape should be discarded and later replayed.
- Test: exponential-decay NLS fit with 10 000 residuals; compare peak tape
  memory with vs. without checkpointing.

#### 5C-2. ODE sensitivity hook (~1 week)

- `Problem::sensitivity(x, y, dydx) -> bool` optional method.
- When present, xopt uses user-supplied forward sensitivities and skips
  taping through the integrator.
- When absent, falls back to tape-through-integrator on the compiled path.
- Same pattern as `SciMLSensitivity.jl`.
- Acceptance vignette: SIR-model parameter fit `(β, γ, N)` with SUNDIALS
  CVODES forward sensitivities supplied by the user.

### 5.4 Out of scope for v3

- `xadr::compile(fn)` runtime R-to-C++ translation — §2.2.
- MCMC / posterior inference — use `stan` / `Nimble` / `RTMB` instead.
- GPU backend, JIT tape compilation, DSL compilation.
- General nonlinear constraints beyond augmented Lagrangian.
- Multi-objective / Pareto-front solvers.

---

## 6. API changes landing in v3

### 6.1 `xopt_minimize`

```r
xopt_minimize(
  par, fn,
  gr       = NULL,
  method   = c("ucminf", "lbfgs", "trust-newton", "nls"),
  gradient = c("auto", "compiled", "user", "fd", "traced"),
  lower    = -Inf, upper = Inf,
  hessian  = c("none", "xad-fwd-adj", "bfgs-approx", "lbfgs-approx", "user"),
  control  = xopt_control()
) -> xopt_result
```

- `result$gradient_mode` records which of the four paths was selected.
- A `warning()` is raised whenever `"auto"` drops from **compiled/user** to
  **traced/fd** (so users notice when they've silently lost AD).

### 6.2 New user-facing functions

- `xopt_linalg_chol(A)`, `xopt_linalg_solve(A, b)`, `xopt_linalg_logdet(A)`
  — R wrappers primarily for testing; the real use is inside user C++.
- `xopt_lbfgs(par, fn, gr, control)` — direct L-BFGS entry point.
- `xopt_sparse_jacobian(fn, x, sparsity)` — compressed-FD Jacobian.
- `xopt_fixed_point(g, x0, theta, ...)` — implicit-function API.
- `xopt_differentiate(result, hyper)` — hyperparameter gradient of an
  optimisation result.

### 6.3 Deprecations

None in v3. Everything currently on `main` stays.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `xopt::linalg` AD rules have subtle sign/transpose bugs | Property-based tests (random SPD, random RHS) comparing to FD gradients to 1e-6. |
| L-BFGS line search divergence on ill-conditioned problems | Strong Wolfe backtrack + Moré–Thuente fallback; use UCMINF's line search as reference. |
| Implicit-diff linear solve is ill-conditioned near singular Hessians | Regularise to `(H + λI)^{-1}`; return a `condition_number` diagnostic; document the caveat. |
| Checkpointing adds tape-replay bugs | Gate behind explicit opt-in on `Problem::checkpoint_schedule`. |
| AGPL-3 (XAD) propagates to dependents | Already accepted in `DESCRIPTION`. Document clearly in README. Revisit if a CRAN split becomes desirable. |
| `xopt::linalg` depends on LAPACK availability on all platforms | R itself depends on LAPACK; safe to assume. Use `R_ext/BLAS.h` / `R_ext/Lapack.h` directly. |

---

## 8. Acceptance criteria

v3 is "done" when the following end-to-end examples run in the test suite
with compiled-XAD gradients and no finite-difference fallback:

1. Logistic GLMM Laplace approximation (`5A-1`).
2. `n = 10 000` L2 logistic regression via L-BFGS, agreeing with `stats::glm`
   coefficients to 1e-8 (`5A-2`).
3. Sparse Jacobian matching dense Jacobian on a 100-residual test
   (`5A-3`).
4. Matrix-completion fixed-point implicit-diff gradient (`5B-1`).
5. Ridge-regression hyperparameter gradient (`5B-2`).
6. 10 000-residual NLS fit with checkpointed tape (`5C-1`).
7. SIR ODE fit with user SUNDIALS sensitivities (`5C-2`).

Each example ships as a section in `vignettes/` with runtime and memory
numbers against a baseline competitor (`stats::optim`, `nloptr`,
`minpack.lm`, `TMB`/`RTMB` where applicable).

---

## 9. Out-of-phase research directions (post-v3)

Tracked for later, not committed:

- **Constrained optimisation at scale** — interior-point + SQP.
- **JIT tape compilation** (Ceres `forge` / CasADi style).
- **Hooking into `torch` or `jax-for-r`** for GPU gradients of objective
  kernels.
- **Honest "traced-R" expansion** — extend the `ad_tracer.R` whitelist to
  cover vectorised ops, `matrix`, and a documented subset of `Matrix::`. Any
  token outside the whitelist raises an error rather than silently missing.
- **`xopt::torch_backend`** bridging to LibTorch for problems where the
  objective is naturally a neural-net-style computation.

---

## Appendix A — file changes this roadmap implies

Files added in v3 (not created by this docs PR):

```
inst/include/xopt/linalg/chol.hpp
inst/include/xopt/linalg/solve.hpp
inst/include/xopt/linalg/logdet.hpp
inst/include/xopt/linalg/inv.hpp
inst/include/xopt/solvers/lbfgs_solver.hpp
inst/include/xopt/implicit/fixed_point.hpp
inst/include/xopt/sparse/coloring.hpp
inst/include/xopt/sparse/compressed_jacobian.hpp
R/linalg.R
R/lbfgs.R
R/implicit.R
R/differentiate.R
src/test_linalg.cpp
src/test_lbfgs.cpp
src/test_implicit.cpp
vignettes/laplace-glmm.Rmd
vignettes/logistic-lbfgs.Rmd
vignettes/implicit-ridge.Rmd
vignettes/ode-sir-fit.Rmd
```

Files modified in v3:

```
R/minimize.R           # gradient-mode labels + warnings
inst/include/xopt/problem.hpp  # Problem::sensitivity() optional hook
DESCRIPTION            # Suggests: Matrix, knitr, rmarkdown
```

No deprecations. No removals.

---

## Appendix B — what this roadmap does NOT promise

Listed here so expectation-setting is unambiguous:

- `xopt_minimize` will not gain the ability to differentiate arbitrary R
  functions. It gains **four explicit gradient paths** with honest labels.
- xopt will not ship an ODE integrator. It provides a **sensitivity hook**
  so users plug in their own (SUNDIALS, deSolve, odin).
- xopt will not support GPUs, MCMC, general nonlinear constraints beyond
  AL, or symbolic computation.
- `xopt::linalg` is not a full LAPACK wrapper. It provides **four primitives
  with AD rules**: `chol`, `solve`, `logdet`, `inv`. That's enough for
  Laplace / Gaussian / implicit-diff workflows.
- Benchmarks compare against specific competitors on specific problem
  classes. There is no universal "faster than X" claim.
