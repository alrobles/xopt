# Forward-mode correctness: compare xopt::linalg primitives against R base
# (base::chol, base::solve, base::determinant, base::solve for inverse).

# Helper: build an SPD matrix with controlled conditioning.
.spd <- function(n, seed = 1L) {
  set.seed(seed)
  M <- matrix(rnorm(n * n), n, n)
  crossprod(M) + n * diag(n)
}

test_that("xopt_chol matches base::chol", {
  for (n in c(1L, 3L, 5L, 8L)) {
    A <- .spd(n, seed = 100L + n)
    L_xopt <- xopt_chol(A)
    L_base <- t(chol(A))  # base::chol returns upper; we return lower

    expect_equal(L_xopt, L_base, tolerance = 1e-10)
    expect_equal(L_xopt %*% t(L_xopt), A, tolerance = 1e-10, ignore_attr = TRUE)
    # Upper triangle should be zero
    expect_true(all(L_xopt[upper.tri(L_xopt)] == 0))
  }
})

test_that("xopt_solve matches base::solve", {
  for (n in c(1L, 3L, 5L, 8L)) {
    A <- .spd(n, seed = 200L + n)
    b <- rnorm(n)
    x_xopt <- xopt_solve(A, b)
    x_base <- as.numeric(solve(A, b))
    expect_equal(x_xopt, x_base, tolerance = 1e-10)
  }
})

test_that("xopt_logdet matches base::determinant(A, log = TRUE)", {
  for (n in c(1L, 3L, 5L, 8L)) {
    A <- .spd(n, seed = 300L + n)
    ld_xopt <- xopt_logdet(A)
    ld_base <- as.numeric(determinant(A, logarithm = TRUE)$modulus)
    expect_equal(ld_xopt, ld_base, tolerance = 1e-10)
  }
})

test_that("xopt_inv matches base::solve(A)", {
  for (n in c(1L, 3L, 5L, 8L)) {
    A <- .spd(n, seed = 400L + n)
    Ai_xopt <- xopt_inv(A)
    Ai_base <- solve(A)
    expect_equal(Ai_xopt, Ai_base, tolerance = 1e-9, ignore_attr = TRUE)
  }
})

test_that("xopt_chol rejects non-SPD input", {
  expect_error(xopt_chol(matrix(c(1, 2, 2, 1), 2, 2)),
               "not symmetric positive-definite")
})

# ---------------------------------------------------------------------------
# Adjoint-mode correctness: XAD adjoint gradients match central finite
# differences for a scalar-reduced objective over each primitive. We use a
# relative tolerance of 1e-5 since central-FD with step sqrt(eps) has
# ~O(eps^{2/3}) error, which comfortably dominates the adjoint sweep's
# ~O(eps) floating-point noise for n <= 8 matrices.
# ---------------------------------------------------------------------------

.central_fd_mat <- function(fn, A, eps = 1e-5) {
  G <- matrix(0, nrow(A), ncol(A))
  for (j in seq_len(ncol(A))) {
    for (i in seq_len(nrow(A))) {
      Ap <- A; Ap[i, j] <- Ap[i, j] + eps
      Am <- A; Am[i, j] <- Am[i, j] - eps
      G[i, j] <- (fn(Ap) - fn(Am)) / (2 * eps)
    }
  }
  G
}

.central_fd_vec <- function(fn, b, eps = 1e-5) {
  g <- numeric(length(b))
  for (i in seq_along(b)) {
    bp <- b; bp[i] <- bp[i] + eps
    bm <- b; bm[i] <- bm[i] - eps
    g[i] <- (fn(bp) - fn(bm)) / (2 * eps)
  }
  g
}

test_that("adjoint of sum(chol(A)) matches central FD", {
  n <- 4L
  A <- .spd(n, seed = 500L)
  g_ad <- xopt:::xopt_chol_grad(A)
  fn <- function(M) sum(xopt_chol(M))
  g_fd <- .central_fd_mat(fn, A)
  expect_equal(g_ad, g_fd, tolerance = 1e-5)
})

test_that("adjoint of sum(solve(A, b)) w.r.t. A matches central FD", {
  n <- 4L
  A <- .spd(n, seed = 600L)
  b <- rnorm(n)
  g_ad <- xopt:::xopt_solve_grad(A, b, wrt = "A")
  fn <- function(M) sum(xopt_solve(M, b))
  g_fd <- .central_fd_mat(fn, A)
  expect_equal(g_ad, g_fd, tolerance = 1e-5)
})

test_that("adjoint of sum(solve(A, b)) w.r.t. b matches central FD", {
  n <- 4L
  A <- .spd(n, seed = 700L)
  b <- rnorm(n)
  g_ad <- xopt:::xopt_solve_grad(A, b, wrt = "b")
  fn <- function(v) sum(xopt_solve(A, v))
  g_fd <- .central_fd_vec(fn, b)
  expect_equal(g_ad, g_fd, tolerance = 1e-5)
})

test_that("adjoint of logdet(A) matches base analytic gradient A^{-1}", {
  # Closed form for a general matrix: d(log|A|)/dA = (A^{-1})^T. The
  # Cholesky-based logdet only reads the lower triangle of A, so treating
  # each A[i,j] as an independent scalar yields an asymmetric gradient
  # (strict lower sees 2x the "symmetric" contribution; strict upper is
  # zero; diagonal is 1x). The canonical symmetric gradient is recovered
  # by symmetrization: (G + G^T) / 2, which must equal A^{-1}.
  n <- 4L
  A <- .spd(n, seed = 800L)
  g_ad <- xopt:::xopt_logdet_grad(A)
  g_sym <- (g_ad + t(g_ad)) / 2
  g_analytic <- solve(A)
  expect_equal(g_sym, g_analytic, tolerance = 1e-9, ignore_attr = TRUE)
})

test_that("adjoint of sum(inv(A)) matches central FD", {
  n <- 4L
  A <- .spd(n, seed = 900L)
  g_ad <- xopt:::xopt_inv_grad(A)
  fn <- function(M) sum(xopt_inv(M))
  g_fd <- .central_fd_mat(fn, A)
  expect_equal(g_ad, g_fd, tolerance = 1e-5)
})

# ---------------------------------------------------------------------------
# Tape-size growth: custom CheckpointCallback adjoints should record tape
# memory that scales sub-cubically in n. The generic elementary-op path
# (which chol(AReal) still takes) records every operation in chol and the
# triangular solves, so tape bytes scale as O(n^3) plus constants. The
# callback path for logdet / solve / inv records only the outputs, so tape
# bytes scale at most quadratically. A log-log slope fit gives a robust,
# platform-independent check.
# ---------------------------------------------------------------------------

.logfit_slope <- function(ns, bytes) {
  fit <- stats::lm(log(bytes) ~ log(ns))
  as.numeric(stats::coef(fit)[2])
}

test_that("CheckpointCallback adjoint tape grows sub-cubically in n", {
  ns <- c(10L, 20L, 40L, 80L)
  bytes_logdet <- numeric(length(ns))
  bytes_solve  <- numeric(length(ns))
  bytes_inv    <- numeric(length(ns))
  for (k in seq_along(ns)) {
    A <- .spd(ns[k], seed = 1000L + ns[k])
    bytes_logdet[k] <- xopt:::xopt_linalg_tape_bytes(A, "logdet")
    bytes_solve[k]  <- xopt:::xopt_linalg_tape_bytes(A, "solve")
    bytes_inv[k]    <- xopt:::xopt_linalg_tape_bytes(A, "inv")
  }
  # Slopes well below 3.0 confirm we escaped the elementary-op O(n^3) regime.
  expect_lt(.logfit_slope(ns, bytes_logdet), 2.0)
  expect_lt(.logfit_slope(ns, bytes_solve),  2.2)
  expect_lt(.logfit_slope(ns, bytes_inv),    2.5)
})
