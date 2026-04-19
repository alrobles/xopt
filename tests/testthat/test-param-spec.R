test_that("ParamSpec round-trip works", {
  result <- test_param_spec_roundtrip()
  expect_equal(result, 0)
})

test_that("Positive transform works", {
  result <- test_positive_transform()
  expect_equal(result, 0)
})

test_that("Bounded transform works", {
  result <- test_bounded_transform()
  expect_equal(result, 0)
})

test_that("Rosenbrock with structured params works", {
  result <- test_rosenbrock_structured()
  expect_equal(result, 0)
})

test_that("Positive-constrained MLE works", {
  result <- test_positive_constrained_mle()
  expect_equal(result, 0)
})
