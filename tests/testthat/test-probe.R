test_that("probe_xad_xtensor works", {
  result <- xopt:::probe_xad_xtensor()
  expect_equal(result, 0)
})
