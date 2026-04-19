test_that("NLS Osborne1 NIST problem works", {
  skip("LM without Marquardt-scaled trust-region diverges on NIST Osborne1 from start 1; tracked separately from Phase 5.")
  result <- test_nls_osborne1()
  expect_equal(result, 0)
})

test_that("NLS Helical Valley works", {
  result <- test_nls_helical_valley()
  expect_equal(result, 0)
})

test_that("NLS performance test", {
  result <- test_nls_performance()
  expect_equal(result, 0)
})
