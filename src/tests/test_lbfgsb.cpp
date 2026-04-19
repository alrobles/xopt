// test_lbfgsb.cpp - Unit tests for L-BFGS-B solver with box constraints
//
// Tests the L-BFGS-B implementation on constrained optimization problems

#include <Rcpp.h>
#include "../inst/include/xopt/problem.hpp"
#include "../inst/include/xopt/solvers/lbfgsb.hpp"
#include <vector>
#include <cmath>
#include <limits>

using namespace xopt;
using namespace xopt::solvers;

// Quadratic with bounds
template <typename Scalar = double>
class BoundedQuadraticProblem : public ProblemBase<Scalar> {
public:
    explicit BoundedQuadraticProblem(int n) : ProblemBase<Scalar>(n) {}

    Scalar value(const Scalar* x) const {
        Scalar sum = 0.0;
        for (int i = 0; i < this->n_par; ++i) {
            Scalar target = static_cast<Scalar>(i + 1);  // Target: x[i] = i+1
            Scalar diff = x[i] - target;
            sum += diff * diff;
        }
        return sum;
    }

    void gradient(const Scalar* x, Scalar* g) const {
        for (int i = 0; i < this->n_par; ++i) {
            Scalar target = static_cast<Scalar>(i + 1);
            g[i] = 2.0 * (x[i] - target);
        }
    }
};

//' @title Test L-BFGS-B with box constraints
//' @description Verify L-BFGS-B respects bounds
//' @export
// [[Rcpp::export]]
int test_lbfgsb_bounds() {
    Rcpp::Rcout << "Test: L-BFGS-B with box constraints" << std::endl;

    const int n = 5;
    BoundedQuadraticProblem<double> problem(n);

    // Set bounds: unconstrained optimum would be x[i] = i+1
    // But we constrain some parameters
    std::vector<double> lower(n);
    std::vector<double> upper(n);

    lower[0] = -std::numeric_limits<double>::infinity();  // Unbounded below
    upper[0] = 0.5;  // Upper bound prevents reaching optimum (1.0)

    lower[1] = 3.0;  // Lower bound prevents reaching optimum (2.0)
    upper[1] = std::numeric_limits<double>::infinity();

    lower[2] = -std::numeric_limits<double>::infinity();
    upper[2] = std::numeric_limits<double>::infinity();  // Unconstrained

    lower[3] = 2.0;
    upper[3] = 3.0;  // Bounded on both sides, optimum (4.0) outside

    lower[4] = -std::numeric_limits<double>::infinity();
    upper[4] = std::numeric_limits<double>::infinity();

    problem.set_bounds(lower, upper);

    // Initial point
    std::vector<double> x0(n, 0.0);

    // Solve
    LBFGSBControl control;
    control.pgtol = 1e-5;
    control.max_iter = 100;
    control.trace = false;

    auto result = minimize_lbfgsb(problem, x0, control);

    Rcpp::Rcout << "  Iterations: " << result.iterations << std::endl;
    Rcpp::Rcout << "  Final value: " << result.value << std::endl;
    Rcpp::Rcout << "  Solution: [";
    for (int i = 0; i < n; ++i) {
        Rcpp::Rcout << result.par[i];
        if (i < n - 1) Rcpp::Rcout << ", ";
    }
    Rcpp::Rcout << "]" << std::endl;

    // Check bounds are respected
    bool bounds_ok = true;
    for (int i = 0; i < n; ++i) {
        if (result.par[i] < lower[i] - 1e-6 || result.par[i] > upper[i] + 1e-6) {
            Rcpp::Rcout << "  ERROR: Bounds violated at parameter " << i << std::endl;
            bounds_ok = false;
        }
    }

    // Check expected values
    // x[0] should be at upper bound (0.5)
    bool x0_correct = std::abs(result.par[0] - 0.5) < 1e-4;
    // x[1] should be at lower bound (3.0)
    bool x1_correct = std::abs(result.par[1] - 3.0) < 1e-4;
    // x[2] should be at unconstrained optimum (3.0)
    bool x2_correct = std::abs(result.par[2] - 3.0) < 1e-4;
    // x[3] should be at upper bound (3.0) since optimum is 4.0
    bool x3_correct = std::abs(result.par[3] - 3.0) < 1e-4;
    // x[4] should be at unconstrained optimum (5.0)
    bool x4_correct = std::abs(result.par[4] - 5.0) < 1e-4;

    bool converged = (result.convergence == 0);

    Rcpp::Rcout << "  Bounds respected: " << (bounds_ok ? "YES" : "NO") << std::endl;
    Rcpp::Rcout << "  x[0] at 0.5: " << (x0_correct ? "YES" : "NO") << std::endl;
    Rcpp::Rcout << "  x[1] at 3.0: " << (x1_correct ? "YES" : "NO") << std::endl;
    Rcpp::Rcout << "  x[2] at 3.0: " << (x2_correct ? "YES" : "NO") << std::endl;
    Rcpp::Rcout << "  x[3] at 3.0: " << (x3_correct ? "YES" : "NO") << std::endl;
    Rcpp::Rcout << "  x[4] at 5.0: " << (x4_correct ? "YES" : "NO") << std::endl;

    if (converged && bounds_ok && x0_correct && x1_correct &&
        x2_correct && x3_correct && x4_correct) {
        Rcpp::Rcout << "  Status: PASS" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        return 1;
    }
}

//' @title Test L-BFGS-B on Rosenbrock with bounds
//' @description Test constrained Rosenbrock optimization
//' @export
// [[Rcpp::export]]
int test_lbfgsb_rosenbrock() {
    Rcpp::Rcout << "Test: L-BFGS-B on constrained Rosenbrock" << std::endl;

    // Rosenbrock with bounds
    class BoundedRosenbrock : public ProblemBase<double> {
    public:
        BoundedRosenbrock() : ProblemBase<double>(2) {}

        double value(const double* x) const {
            double t1 = 1.0 - x[0];
            double t2 = x[1] - x[0] * x[0];
            return t1 * t1 + 100.0 * t2 * t2;
        }

        void gradient(const double* x, double* g) const {
            double t1 = 1.0 - x[0];
            double t2 = x[1] - x[0] * x[0];
            g[0] = -2.0 * t1 - 400.0 * x[0] * t2;
            g[1] = 200.0 * t2;
        }
    };

    BoundedRosenbrock problem;

    // Bounds: constrain to [0, 0.5] x [0, 2]
    // Unconstrained optimum is (1, 1), so x[0] will hit upper bound
    std::vector<double> lower = {0.0, 0.0};
    std::vector<double> upper = {0.5, 2.0};
    problem.set_bounds(lower, upper);

    // Initial point
    std::vector<double> x0 = {0.1, 0.1};

    // Solve
    LBFGSBControl control;
    control.pgtol = 1e-5;
    control.max_iter = 500;
    control.trace = false;

    auto result = minimize_lbfgsb(problem, x0, control);

    Rcpp::Rcout << "  Iterations: " << result.iterations << std::endl;
    Rcpp::Rcout << "  Final value: " << result.value << std::endl;
    Rcpp::Rcout << "  Solution: [" << result.par[0] << ", "
                << result.par[1] << "]" << std::endl;

    // Check bounds
    bool bounds_ok = (result.par[0] >= -1e-6 && result.par[0] <= 0.5 + 1e-6 &&
                      result.par[1] >= -1e-6 && result.par[1] <= 2.0 + 1e-6);

    // x[0] should be at upper bound (0.5)
    // x[1] should be near x[0]^2 = 0.25 for constrained optimum
    bool x0_at_bound = std::abs(result.par[0] - 0.5) < 1e-3;
    bool x1_near_optimum = std::abs(result.par[1] - 0.25) < 1e-2;

    bool converged = (result.convergence == 0 || result.iterations < control.max_iter);

    if (bounds_ok && x0_at_bound && x1_near_optimum) {
        Rcpp::Rcout << "  Status: PASS" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        Rcpp::Rcout << "    Bounds OK: " << bounds_ok << std::endl;
        Rcpp::Rcout << "    x[0] at bound: " << x0_at_bound << std::endl;
        Rcpp::Rcout << "    x[1] near optimum: " << x1_near_optimum << std::endl;
        return 1;
    }
}
