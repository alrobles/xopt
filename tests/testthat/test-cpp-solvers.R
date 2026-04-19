test_that("xopt_tr_newton_cpp solves a simple quadratic", {
  fn <- function(x) 0.5 * sum(x^2)
  gr <- function(x) x
  hvp <- function(x, v) v

  res <- xopt_tr_newton_cpp(
    par = c(2, -3),
    fn = fn,
    gr = gr,
    hvp = hvp,
    control = list(gtol = 1e-10, xtol = 1e-12, maxiter = 100)
  )

  expect_s3_class(res, "xopt_result")
  expect_lt(max(abs(res$par)), 1e-6)
  expect_true(res$convergence %in% c(1L, 2L, 4L))
})

test_that("xopt_minimize(method='tr_newton') routes through C++ wrapper", {
  fn <- function(x) (x[1] - 1)^2 + (x[2] + 2)^2
  gr <- function(x) c(2 * (x[1] - 1), 2 * (x[2] + 2))

  res <- xopt_minimize(
    par = c(4, 4),
    fn = fn,
    gr = gr,
    method = "tr_newton"
  )

  expect_s3_class(res, "xopt_result")
  expect_equal(res$gradient_mode, "user")
  expect_equal(res$par, c(1, -2), tolerance = 1e-6)
})

test_that("xopt_nls uses C++ LM wrapper when available", {
  x <- seq(0, 1, length.out = 10)
  y <- 1.5 + 0.7 * x

  residual_fn <- function(par) y - (par[1] + par[2] * x)
  jacobian_fn <- function(par) cbind(rep(-1, length(x)), -x)

  res <- xopt_nls(
    par = c(0, 0),
    residual_fn = residual_fn,
    jacobian_fn = jacobian_fn,
    control = list(maxiter = 50)
  )

  expect_equal(res$par, c(1.5, 0.7), tolerance = 1e-6)
  expect_true(res$convergence %in% c(1L, 2L, 4L))
})
