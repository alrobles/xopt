test_that("L-BFGS solver works on quadratic function", {
  result <- test_lbfgs_quadratic()
  expect_equal(result, 0)
})

test_that("L-BFGS solver works on Rosenbrock function", {
  result <- test_lbfgs_rosenbrock()
  expect_equal(result, 0)
})

test_that("L-BFGS solver handles large-scale problems", {
  result <- test_lbfgs_largescale()
  expect_equal(result, 0)
})

test_that("L-BFGS-B solver respects box constraints", {
  result <- test_lbfgsb_bounds()
  expect_equal(result, 0)
})

test_that("L-BFGS-B solver works on constrained Rosenbrock", {
  result <- test_lbfgsb_rosenbrock()
  expect_equal(result, 0)
})

test_that("MaxEnt gradient is accurate", {
  result <- test_maxent_gradient()
  expect_equal(result, 0)
})

test_that("MaxEnt end-to-end optimization works", {
  result <- test_maxent_endtoend()
  expect_equal(result, 0)
})

test_that("Chunked raster processing works on large datasets", {
  result <- test_chunked_processing()
  expect_equal(result, 0)
})

test_that("Chunked processing handles NA values correctly", {
  result <- test_chunked_with_na()
  expect_equal(result, 0)
})
