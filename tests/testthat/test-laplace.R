# Laplace approximation tests.
#
# All targets used here are Gaussian (possibly after a change of variables), so
# the Laplace approximation is exact and we can assert equality to the
# closed-form marginal up to optimizer / finite-difference tolerance.

test_that("xopt_laplace recovers log Z for a 1D Gaussian", {
  # fn = 0.5 * (theta - 3)^2 / 2 corresponds to N(3, sigma^2 = 2), so
  # Z = sqrt(2*pi*2) and log Z = 0.5 * log(2*pi*2).
  fn <- function(theta) 0.5 * (theta - 3)^2 / 2
  res <- xopt_laplace(fn, par = 0)
  expect_equal(res$log_marginal, 0.5 * log(2 * pi * 2), tolerance = 1e-4)
  expect_equal(res$par, 3, tolerance = 1e-4)
  expect_equal(res$value, 0, tolerance = 1e-6)
})

test_that("xopt_laplace recovers log Z for a 3D Gaussian with dense covariance", {
  set.seed(42)
  L <- matrix(rnorm(9), 3, 3)
  Sigma <- crossprod(L) + diag(3)
  Prec <- solve(Sigma)
  mu <- c(1, -2, 0.5)

  # fn = 0.5 * (theta - mu)^T Prec (theta - mu); target is N(mu, Sigma).
  # log Z = 0.5 * p * log(2*pi) + 0.5 * log det Sigma
  #       = 0.5 * p * log(2*pi) - 0.5 * log det Prec.
  fn <- function(theta) {
    d <- theta - mu
    0.5 * as.numeric(t(d) %*% Prec %*% d)
  }

  res <- xopt_laplace(fn, par = c(0, 0, 0))
  expected <- 0.5 * length(mu) * log(2 * pi) + 0.5 * determinant(Sigma,
                                                                 logarithm = TRUE)$modulus
  attributes(expected) <- NULL
  expect_equal(as.numeric(res$log_marginal), expected, tolerance = 1e-3)
  expect_equal(res$par, mu, tolerance = 1e-3)
})

test_that("xopt_laplace accepts a user-supplied analytical Hessian", {
  # 2D Gaussian N(0, Sigma) with known Prec = diag(2, 0.5).
  Prec <- diag(c(2, 0.5))
  Sigma <- solve(Prec)

  fn <- function(theta) 0.5 * as.numeric(t(theta) %*% Prec %*% theta)
  hess <- function(theta) Prec

  res <- xopt_laplace(fn, par = c(1, 1), hessian = hess)
  expected <- 0.5 * 2 * log(2 * pi) + 0.5 * as.numeric(determinant(Sigma,
                                                                     logarithm = TRUE)$modulus)
  expect_equal(res$log_marginal, expected, tolerance = 1e-6)
  expect_equal(res$logdet_hessian, log(det(Prec)), tolerance = 1e-10)
})

test_that("xopt_laplace with optimize = FALSE treats par as the mode", {
  # Same 2D Gaussian; pre-supply the mode.
  Prec <- diag(c(2, 0.5))
  fn <- function(theta) 0.5 * as.numeric(t(theta) %*% Prec %*% theta)
  hess <- function(theta) Prec

  res <- xopt_laplace(fn, par = c(0, 0), hessian = hess, optimize = FALSE)
  expect_null(res$fit)
  expect_equal(res$par, c(0, 0))
  expect_equal(res$value, 0, tolerance = 1e-12)
})

test_that("xopt_laplace errors when Hessian at mode is indefinite", {
  # fn(x) = -x^2 has a maximum at 0, not a minimum; Hessian is -2.
  # The C++ Cholesky inside xopt_logdet should reject it.
  fn <- function(x) -x^2
  hess <- function(x) matrix(-2, 1, 1)
  expect_error(
    xopt_laplace(fn, par = 0.0, hessian = hess, optimize = FALSE)
  )
})

test_that("print.xopt_laplace produces expected fields", {
  fn <- function(theta) 0.5 * (theta - 3)^2 / 2
  res <- xopt_laplace(fn, par = 0)
  out <- capture.output(print(res))
  expect_true(any(grepl("log Z", out)))
  expect_true(any(grepl("Mode:", out)))
})
