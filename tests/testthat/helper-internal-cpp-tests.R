.run_internal_cpp_test <- function(name) {
  xopt:::xopt_internal_run_test(name)
}

test_rosenbrock_benchmark <- function() .run_internal_cpp_test("test_rosenbrock_benchmark")
test_sphere_benchmark <- function() .run_internal_cpp_test("test_sphere_benchmark")
test_powell_singular_benchmark <- function() .run_internal_cpp_test("test_powell_singular_benchmark")
test_beale_benchmark <- function() .run_internal_cpp_test("test_beale_benchmark")
test_brown_badly_scaled_benchmark <- function() .run_internal_cpp_test("test_brown_badly_scaled_benchmark")
test_broyden_tridiagonal_benchmark <- function() .run_internal_cpp_test("test_broyden_tridiagonal_benchmark")
test_quadratic_benchmark <- function() .run_internal_cpp_test("test_quadratic_benchmark")

test_multistart_rosenbrock <- function() .run_internal_cpp_test("test_multistart_rosenbrock")
test_multistart_rastrigin <- function() .run_internal_cpp_test("test_multistart_rastrigin")
test_multistart_deterministic <- function() .run_internal_cpp_test("test_multistart_deterministic")
test_multistart_scaling <- function() .run_internal_cpp_test("test_multistart_scaling")

test_nls_exponential_decay <- function() .run_internal_cpp_test("test_nls_exponential_decay")
test_nls_misra1a <- function() .run_internal_cpp_test("test_nls_misra1a")
test_nls_linear <- function() .run_internal_cpp_test("test_nls_linear")
test_nls_jacobian_accuracy <- function() .run_internal_cpp_test("test_nls_jacobian_accuracy")
test_nls_covariance <- function() .run_internal_cpp_test("test_nls_covariance")
test_nls_osborne1 <- function() .run_internal_cpp_test("test_nls_osborne1")
test_nls_helical_valley <- function() .run_internal_cpp_test("test_nls_helical_valley")
test_nls_performance <- function() .run_internal_cpp_test("test_nls_performance")

test_param_spec_roundtrip <- function() .run_internal_cpp_test("test_param_spec_roundtrip")
test_positive_transform <- function() .run_internal_cpp_test("test_positive_transform")
test_bounded_transform <- function() .run_internal_cpp_test("test_bounded_transform")
test_rosenbrock_structured <- function() .run_internal_cpp_test("test_rosenbrock_structured")
test_positive_constrained_mle <- function() .run_internal_cpp_test("test_positive_constrained_mle")

test_phase3_hessian_hvp <- function() .run_internal_cpp_test("test_phase3_hessian_hvp")
test_phase3_trust_region_newton <- function() .run_internal_cpp_test("test_phase3_trust_region_newton")
test_phase3_laplace <- function() .run_internal_cpp_test("test_phase3_laplace")

test_phase4_constraints <- function() .run_internal_cpp_test("test_phase4_constraints")
test_phase4_multistart_parallel <- function() .run_internal_cpp_test("test_phase4_multistart_parallel")
test_phase4_sparse <- function() .run_internal_cpp_test("test_phase4_sparse")
test_phase4_jit_checkpoint <- function() .run_internal_cpp_test("test_phase4_jit_checkpoint")
