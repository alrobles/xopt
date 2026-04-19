test_that("logistic SDM gradient accuracy test works", {
  result <- test_logistic_sdm_gradient()
  expect_equal(result, 0)
})

test_that("logistic SDM end-to-end test works", {
  result <- test_logistic_sdm_endtoend()
  expect_equal(result, 0)
})

test_that("masked sum reduction works", {
  result <- test_masked_sum()
  expect_equal(result, 0)
})

test_that("raster mask from NA values works", {
  result <- test_raster_mask_na()
  expect_equal(result, 0)
})

test_that("logistic SDM with NA values works", {
  result <- test_logistic_sdm_with_na()
  expect_equal(result, 0)
})

test_that("mask intersection works", {
  result <- test_mask_intersection()
  expect_equal(result, 0)
})
