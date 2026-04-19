test_that("Phase 3 Hessian/HVP finite-difference checks pass", {
  expect_equal(test_phase3_hessian_hvp(), 0)
})

test_that("Phase 3 trust-region Newton converges on ill-conditioned problem", {
  expect_equal(test_phase3_trust_region_newton(), 0)
})

test_that("Phase 3 Laplace approximation matches Gaussian reference", {
  expect_equal(test_phase3_laplace(), 0)
})

test_that("R-side tracer keeps base behavior and auto gradient works", {
  fn <- function(x) sin(x[1]) + exp(x[2])
  traced <- xopt_ad_trace(fn)
  x <- c(0.3, -0.1)
  expect_equal(traced(x), fn(x), tolerance = 1e-12)

  g <- xopt_auto_gradient(fn, x, tracer = TRUE)
  g_exact <- c(cos(x[1]), exp(x[2]))
  expect_equal(as.numeric(g), g_exact, tolerance = 1e-4)
})
