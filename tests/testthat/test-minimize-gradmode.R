test_that("xopt_minimize reports gradient_mode='user' when gr is supplied", {
  fn <- function(x) sum((x - 1)^2)
  gr <- function(x) 2 * (x - 1)
  res <- xopt_minimize(c(0, 0), fn, gr)
  expect_equal(res$gradient_mode, "user")
  expect_true(sqrt(sum((res$par - 1)^2)) < 1e-6)
})

test_that("xopt_minimize respects explicit gradient='fd' and its 'finite' alias", {
  fn <- function(x) sum((x - 1)^2)
  res_fd <- xopt_minimize(c(0, 0), fn, gradient = "fd")
  res_fin <- xopt_minimize(c(0, 0), fn, gradient = "finite")
  expect_equal(res_fd$gradient_mode, "fd")
  expect_equal(res_fin$gradient_mode, "fd")
  expect_true(sqrt(sum((res_fd$par - 1)^2)) < 1e-4)
  expect_true(sqrt(sum((res_fin$par - 1)^2)) < 1e-4)
})

test_that("gradient='user' errors when gr is not supplied", {
  fn <- function(x) sum((x - 1)^2)
  expect_error(
    xopt_minimize(c(0, 0), fn, gradient = "user"),
    "requires a gradient function"
  )
})

test_that("gradient='compiled' also requires gr and reports that label", {
  fn <- function(x) sum((x - 1)^2)
  gr <- function(x) 2 * (x - 1)
  expect_error(
    xopt_minimize(c(0, 0), fn, gradient = "compiled"),
    "requires a gradient function"
  )
  res <- xopt_minimize(c(0, 0), fn, gr, gradient = "compiled")
  expect_equal(res$gradient_mode, "compiled")
  expect_true(sqrt(sum((res$par - 1)^2)) < 1e-6)
})

test_that("gradient mode validation rejects unknown strings", {
  fn <- function(x) sum((x - 1)^2)
  expect_error(
    xopt_minimize(c(0, 0), fn, gradient = "bogus"),
    "Invalid gradient mode"
  )
})

test_that("gradient='auto' reports 'auto' or 'fd' depending on xadr availability", {
  fn <- function(x) sum((x - 1)^2)
  res <- xopt_minimize(c(0, 0), fn, gradient = "auto")
  expect_true(res$gradient_mode %in% c("auto", "fd"))
  expect_true(sqrt(sum((res$par - 1)^2)) < 1e-4)
})

test_that("gradient='traced' errors out cleanly without xadr", {
  skip_if(requireNamespace("xadr", quietly = TRUE),
          "xadr is installed; traced mode covered elsewhere")
  fn <- function(x) sum((x - 1)^2)
  expect_error(
    xopt_minimize(c(0, 0), fn, gradient = "traced"),
    "requires the xadr package"
  )
})

test_that("print.xopt_result surfaces the gradient_mode label", {
  fn <- function(x) sum((x - 1)^2)
  gr <- function(x) 2 * (x - 1)
  res <- xopt_minimize(c(0, 0), fn, gr)
  out <- capture.output(print(res))
  expect_true(any(grepl("Grad source: user", out, fixed = TRUE)))
})
