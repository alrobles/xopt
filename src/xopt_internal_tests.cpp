#include <Rcpp.h>

int test_rosenbrock_benchmark();
int test_sphere_benchmark();
int test_powell_singular_benchmark();
int test_beale_benchmark();
int test_brown_badly_scaled_benchmark();
int test_broyden_tridiagonal_benchmark();
int test_quadratic_benchmark();

int test_multistart_rosenbrock();
int test_multistart_rastrigin();
int test_multistart_deterministic();
int test_multistart_scaling();

int test_nls_exponential_decay();
int test_nls_misra1a();
int test_nls_linear();
int test_nls_jacobian_accuracy();
int test_nls_covariance();
int test_nls_osborne1();
int test_nls_helical_valley();
int test_nls_performance();

int test_param_spec_roundtrip();
int test_positive_transform();
int test_bounded_transform();
int test_rosenbrock_structured();
int test_positive_constrained_mle();

int test_phase3_hessian_hvp();
int test_phase3_trust_region_newton();
int test_phase3_laplace();

int test_phase4_constraints();
int test_phase4_multistart_parallel();
int test_phase4_sparse();
int test_phase4_jit_checkpoint();

// [[Rcpp::export]]
int xopt_internal_run_test(std::string name) {
    if (name == "test_rosenbrock_benchmark") return test_rosenbrock_benchmark();
    if (name == "test_sphere_benchmark") return test_sphere_benchmark();
    if (name == "test_powell_singular_benchmark") return test_powell_singular_benchmark();
    if (name == "test_beale_benchmark") return test_beale_benchmark();
    if (name == "test_brown_badly_scaled_benchmark") return test_brown_badly_scaled_benchmark();
    if (name == "test_broyden_tridiagonal_benchmark") return test_broyden_tridiagonal_benchmark();
    if (name == "test_quadratic_benchmark") return test_quadratic_benchmark();

    if (name == "test_multistart_rosenbrock") return test_multistart_rosenbrock();
    if (name == "test_multistart_rastrigin") return test_multistart_rastrigin();
    if (name == "test_multistart_deterministic") return test_multistart_deterministic();
    if (name == "test_multistart_scaling") return test_multistart_scaling();

    if (name == "test_nls_exponential_decay") return test_nls_exponential_decay();
    if (name == "test_nls_misra1a") return test_nls_misra1a();
    if (name == "test_nls_linear") return test_nls_linear();
    if (name == "test_nls_jacobian_accuracy") return test_nls_jacobian_accuracy();
    if (name == "test_nls_covariance") return test_nls_covariance();
    if (name == "test_nls_osborne1") return test_nls_osborne1();
    if (name == "test_nls_helical_valley") return test_nls_helical_valley();
    if (name == "test_nls_performance") return test_nls_performance();

    if (name == "test_param_spec_roundtrip") return test_param_spec_roundtrip();
    if (name == "test_positive_transform") return test_positive_transform();
    if (name == "test_bounded_transform") return test_bounded_transform();
    if (name == "test_rosenbrock_structured") return test_rosenbrock_structured();
    if (name == "test_positive_constrained_mle") return test_positive_constrained_mle();

    if (name == "test_phase3_hessian_hvp") return test_phase3_hessian_hvp();
    if (name == "test_phase3_trust_region_newton") return test_phase3_trust_region_newton();
    if (name == "test_phase3_laplace") return test_phase3_laplace();

    if (name == "test_phase4_constraints") return test_phase4_constraints();
    if (name == "test_phase4_multistart_parallel") return test_phase4_multistart_parallel();
    if (name == "test_phase4_sparse") return test_phase4_sparse();
    if (name == "test_phase4_jit_checkpoint") return test_phase4_jit_checkpoint();

    Rcpp::stop("Unknown internal test: %s", name);
}
