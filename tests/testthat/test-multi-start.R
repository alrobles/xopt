test_that("Multi-start Rosenbrock works", {
  result <- test_multistart_rosenbrock()
  expect_equal(result, 0)
})

test_that("Multi-start Rastrigin works", {
  result <- test_multistart_rastrigin()
  expect_equal(result, 0)
})

test_that("Multi-start is deterministic", {
  result <- test_multistart_deterministic()
  expect_equal(result, 0)
})

test_that("Multi-start scaling test", {
  result <- test_multistart_scaling()
  expect_equal(result, 0)
})
