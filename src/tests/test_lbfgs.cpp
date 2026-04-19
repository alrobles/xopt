// test_lbfgs.cpp - Unit tests for L-BFGS solver
//
// Tests the L-BFGS implementation on standard optimization problems

#include <Rcpp.h>
#include "../inst/include/xopt/problem.hpp"
#include "../inst/include/xopt/solvers/lbfgs.hpp"
#include <vector>
#include <cmath>
#include <random>

using namespace xopt;
using namespace xopt::solvers;

// Simple quadratic objective: f(x) = sum(x^2)
template <typename Scalar = double>
class QuadraticObjective {
public:
    Scalar value(const Scalar* x, int n) const {
        Scalar sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += x[i] * x[i];
        }
        return sum;
    }

    void gradient(const Scalar* x, Scalar* g, int n) const {
        for (int i = 0; i < n; ++i) {
            g[i] = 2.0 * x[i];
        }
    }
};

// Adapter for Problem interface
template <typename Scalar = double>
class QuadraticProblem : public ProblemBase<Scalar> {
private:
    QuadraticObjective<Scalar> obj_;

public:
    explicit QuadraticProblem(int n) : ProblemBase<Scalar>(n) {}

    Scalar value(const Scalar* x) const {
        return obj_.value(x, this->n_par);
    }

    void gradient(const Scalar* x, Scalar* g) const {
        obj_.gradient(x, g, this->n_par);
    }
};

// Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
template <typename Scalar = double>
class RosenbrockProblem : public ProblemBase<Scalar> {
public:
    RosenbrockProblem() : ProblemBase<Scalar>(2) {}

    Scalar value(const Scalar* x) const {
        Scalar t1 = 1.0 - x[0];
        Scalar t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    }

    void gradient(const Scalar* x, Scalar* g) const {
        Scalar t1 = 1.0 - x[0];
        Scalar t2 = x[1] - x[0] * x[0];
        g[0] = -2.0 * t1 - 400.0 * x[0] * t2;
        g[1] = 200.0 * t2;
    }
};

//' @title Test L-BFGS on quadratic function
//' @description Verify L-BFGS finds the minimum of a simple quadratic
//' @export
// [[Rcpp::export]]
int test_lbfgs_quadratic() {
    Rcpp::Rcout << "Test: L-BFGS on quadratic function" << std::endl;

    const int n = 10;
    QuadraticProblem<double> problem(n);

    // Initial point
    std::vector<double> x0(n, 1.0);

    // Solve
    LBFGSControl control;
    control.gtol = 1e-6;
    control.max_iter = 100;
    control.trace = false;

    auto result = minimize_lbfgs(problem, x0, control);

    Rcpp::Rcout << "  Iterations: " << result.iterations << std::endl;
    Rcpp::Rcout << "  Function evals: " << result.function_evals << std::endl;
    Rcpp::Rcout << "  Final value: " << result.value << std::endl;
    Rcpp::Rcout << "  Convergence: " << result.convergence << " ("
                << result.message << ")" << std::endl;

    // Check convergence
    bool converged = (result.convergence == 0);
    bool near_zero = (result.value < 1e-10);

    // Check solution is near zero
    double max_par = 0.0;
    for (const auto& p : result.par) {
        max_par = std::max(max_par, std::abs(p));
    }
    bool solution_correct = (max_par < 1e-5);

    Rcpp::Rcout << "  Max |x|: " << max_par << std::endl;

    if (converged && near_zero && solution_correct) {
        Rcpp::Rcout << "  Status: PASS" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        return 1;
    }
}

//' @title Test L-BFGS on Rosenbrock function
//' @description Verify L-BFGS finds minimum of Rosenbrock (challenging problem)
//' @export
// [[Rcpp::export]]
int test_lbfgs_rosenbrock() {
    Rcpp::Rcout << "Test: L-BFGS on Rosenbrock function" << std::endl;

    RosenbrockProblem<double> problem;

    // Initial point (away from optimum at (1,1))
    std::vector<double> x0 = {-1.2, 1.0};

    // Solve
    LBFGSControl control;
    control.gtol = 1e-6;
    control.max_iter = 1000;
    control.trace = false;

    auto result = minimize_lbfgs(problem, x0, control);

    Rcpp::Rcout << "  Iterations: " << result.iterations << std::endl;
    Rcpp::Rcout << "  Function evals: " << result.function_evals << std::endl;
    Rcpp::Rcout << "  Final value: " << result.value << std::endl;
    Rcpp::Rcout << "  Solution: [" << result.par[0] << ", "
                << result.par[1] << "]" << std::endl;
    Rcpp::Rcout << "  Convergence: " << result.convergence << " ("
                << result.message << ")" << std::endl;

    // Check solution is near (1, 1)
    double error_x = std::abs(result.par[0] - 1.0);
    double error_y = std::abs(result.par[1] - 1.0);
    double max_error = std::max(error_x, error_y);

    Rcpp::Rcout << "  Max error from (1,1): " << max_error << std::endl;

    bool converged = (result.convergence == 0);
    bool near_optimum = (max_error < 1e-4);
    bool value_correct = (result.value < 1e-6);

    if (converged && near_optimum && value_correct) {
        Rcpp::Rcout << "  Status: PASS" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        return 1;
    }
}

//' @title Test L-BFGS on large-scale problem
//' @description Verify L-BFGS handles large problems efficiently
//' @export
// [[Rcpp::export]]
int test_lbfgs_largescale() {
    Rcpp::Rcout << "Test: L-BFGS on large-scale quadratic (n=1000)" << std::endl;

    const int n = 1000;
    QuadraticProblem<double> problem(n);

    // Random initial point
    std::vector<double> x0(n);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        x0[i] = dist(rng);
    }

    // Solve
    LBFGSControl control;
    control.gtol = 1e-5;
    control.max_iter = 100;
    control.m = 10;  // Limited memory
    control.trace = false;

    auto result = minimize_lbfgs(problem, x0, control);

    Rcpp::Rcout << "  Iterations: " << result.iterations << std::endl;
    Rcpp::Rcout << "  Function evals: " << result.function_evals << std::endl;
    Rcpp::Rcout << "  Final value: " << result.value << std::endl;
    Rcpp::Rcout << "  Convergence: " << result.convergence << " ("
                << result.message << ")" << std::endl;

    // Check convergence
    bool converged = (result.convergence == 0);
    bool near_zero = (result.value < 1e-8);
    bool efficient = (result.iterations < 50);

    if (converged && near_zero && efficient) {
        Rcpp::Rcout << "  Status: PASS" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        return 1;
    }
}
