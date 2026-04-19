test_that("NLS exponential decay fit works", {
  result <- test_nls_exponential_decay()
  expect_equal(result, 0)
})

test_that("NLS Misra1a problem works", {
  result <- test_nls_misra1a()
  expect_equal(result, 0)
})

test_that("NLS linear regression works", {
  result <- test_nls_linear()
  expect_equal(result, 0)
})

test_that("NLS Jacobian accuracy works", {
  result <- test_nls_jacobian_accuracy()
  expect_equal(result, 0)
})

test_that("NLS covariance computation works", {
  result <- test_nls_covariance()
  expect_equal(result, 0)
})
