test_that("Phase 4 constrained optimization works", {
  expect_equal(test_phase4_constraints(), 0)
})

test_that("Phase 4 parallel multi-start is stable", {
  expect_equal(test_phase4_multistart_parallel(), 0)
})

test_that("Phase 4 sparse Jacobian path works", {
  expect_equal(test_phase4_sparse(), 0)
})

test_that("Phase 4 JIT/checkpoint prototype works", {
  expect_equal(test_phase4_jit_checkpoint(), 0)
})
