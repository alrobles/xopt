context("xopt_implicit_spd — implicit function theorem adjoint")

.spd <- function(n, seed = 1L) {
  set.seed(seed)
  M <- matrix(stats::rnorm(n * n), n, n)
  crossprod(M) + n * diag(n)
}

# -----------------------------------------------------------------------------
# Ridge regression: β*(λ) = (X^T X + λ I)^{-1} X^T y.
# With L(β) = ‖β‖², dL/dλ admits a closed form:
#   dL/dλ = -2 β*^T (X^T X + λ I)^{-1} β*
# Verify the IFT callback matches (a) the analytic formula and (b) central FD.
# -----------------------------------------------------------------------------

test_that("IFT ridge gradient matches analytic -2 β^T A^{-1} β", {
  set.seed(2026L)
  n <- 50L; p <- 8L
  X <- matrix(stats::rnorm(n * p), n, p)
  y <- stats::rnorm(n)
  lambda <- 0.75
  A <- crossprod(X) + lambda * diag(p)
  beta_star <- solve(A, crossprod(X, y))
  analytic <- as.numeric(-2 * crossprod(beta_star, solve(A, beta_star)))
  ad <- xopt:::xopt_implicit_spd_grad_ridge(X, y, lambda)
  expect_equal(ad, analytic, tolerance = 1e-10)
})

test_that("IFT ridge gradient matches central finite differences", {
  set.seed(2027L)
  n <- 40L; p <- 5L
  X <- matrix(stats::rnorm(n * p), n, p)
  y <- stats::rnorm(n)
  lambda <- 0.5
  ad <- xopt:::xopt_implicit_spd_grad_ridge(X, y, lambda)
  ridge_loss <- function(lam) {
    A <- crossprod(X) + lam * diag(p)
    beta <- solve(A, crossprod(X, y))
    sum(beta^2)
  }
  eps <- 1e-6
  fd <- (ridge_loss(lambda + eps) - ridge_loss(lambda - eps)) / (2 * eps)
  expect_equal(ad, fd, tolerance = 1e-5)
})

# -----------------------------------------------------------------------------
# Generic IFT VJP: θ̄ = − B^T A^{-1} x̄
# Verify componentwise against a direct R computation of the same formula.
# -----------------------------------------------------------------------------

test_that("direct IFT cotangent matches -B^T A^{-1} x_bar", {
  set.seed(2028L)
  n <- 6L; p <- 3L
  A <- .spd(n, seed = 2028L)
  B <- matrix(stats::rnorm(n * p), n, p)
  x_star <- stats::rnorm(n)
  x_bar  <- stats::rnorm(n)
  theta_ad <- xopt:::xopt_implicit_spd(A, B, x_star, x_bar)
  theta_ref <- as.numeric(-t(B) %*% solve(A, x_bar))
  expect_equal(theta_ad, theta_ref, tolerance = 1e-10)
})

test_that("zero upstream gradient yields zero theta_bar (short-circuit)", {
  set.seed(2029L)
  n <- 4L; p <- 2L
  A <- .spd(n, seed = 2029L)
  B <- matrix(stats::rnorm(n * p), n, p)
  x_star <- stats::rnorm(n)
  x_bar  <- rep(0.0, n)
  theta_ad <- xopt:::xopt_implicit_spd(A, B, x_star, x_bar)
  expect_equal(theta_ad, rep(0.0, p))
})
