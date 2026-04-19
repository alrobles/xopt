test_that("Rosenbrock benchmark works", {
  result <- test_rosenbrock_benchmark()
  expect_equal(result, 0)
})

test_that("Sphere benchmark works", {
  result <- test_sphere_benchmark()
  expect_equal(result, 0)
})

test_that("Powell Singular benchmark works", {
  result <- test_powell_singular_benchmark()
  expect_equal(result, 0)
})

test_that("Beale benchmark works", {
  result <- test_beale_benchmark()
  expect_equal(result, 0)
})

test_that("Brown Badly Scaled benchmark works", {
  result <- test_brown_badly_scaled_benchmark()
  expect_equal(result, 0)
})

test_that("Broyden Tridiagonal benchmark works", {
  result <- test_broyden_tridiagonal_benchmark()
  expect_equal(result, 0)
})

test_that("Quadratic benchmark works", {
  result <- test_quadratic_benchmark()
  expect_equal(result, 0)
})
