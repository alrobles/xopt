test_that("probe_xad_xtensor works", {
  result <- probe_xad_xtensor()
  expect_equal(result, 0)
})

test_that("probe_sdm works", {
  result <- probe_sdm()
  expect_equal(result, 0)
})
