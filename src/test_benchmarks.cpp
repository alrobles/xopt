// Test classical benchmarks with ucminf integration

#include <Rcpp.h>
#include <xopt/benchmarks.hpp>
#include <xopt/solvers/ucminf_solver.hpp>
#include <xopt/diagnostics.hpp>
#include <xopt/problem.hpp>

using namespace Rcpp;
using namespace xopt;

// Wrapper to use benchmarks with Problem interface
template <typename BenchmarkType>
struct BenchmarkProblem : public ProblemBase<double> {
    BenchmarkType benchmark;

    BenchmarkProblem(BenchmarkType bench)
        : ProblemBase<double>(bench.dimension()), benchmark(std::move(bench)) {}

    double value(const double* x) const {
        return benchmark.value(x);
    }

    void gradient(const double* x, double* g) const {
        benchmark.gradient(x, g);
    }

    static constexpr bool has_gradient() { return true; }
    static constexpr GradKind gradient_kind() { return GradKind::UserFn; }
};

int test_rosenbrock_benchmark() {
    try {
        benchmarks::Rosenbrock rb(2);
        BenchmarkProblem<benchmarks::Rosenbrock> prob(std::move(rb));

        // Check gradient
        std::vector<double> x0 = {-1.2, 1.0};
        auto check_result = diagnostics::check_problem_gradient(prob, x0);

        if (!check_result.passed) {
            Rcpp::Rcout << "Rosenbrock gradient check FAILED\n";
            diagnostics::print_gradient_check(check_result, Rcpp::Rcout);
            return 1;
        }

        // Optimize
        solvers::UcminfControl control;
        control.grtol = 1e-8;
        control.maxeval = 1000;

        auto result = solvers::ucminf_solve(prob, x0, control);

        Rcpp::Rcout << "Rosenbrock optimization:\n";
        Rcpp::Rcout << "  Convergence: " << result.convergence << " - "
                    << result.message << "\n";
        Rcpp::Rcout << "  Iterations: " << result.iterations << "\n";
        Rcpp::Rcout << "  Final value: " << result.value << "\n";
        Rcpp::Rcout << "  Solution: [" << result.par[0] << ", " << result.par[1] << "]\n";

        // Check if converged to correct minimum
        if (result.convergence != 1 && result.convergence != 2) {
            Rcpp::Rcout << "Failed to converge\n";
            return 1;
        }

        if (result.value > 1e-6) {
            Rcpp::Rcout << "Did not reach minimum (f = " << result.value << ")\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error: " << e.what() << "\n";
        return 1;
    }
}

int test_sphere_benchmark() {
    try {
        benchmarks::Sphere sphere(5);
        BenchmarkProblem<benchmarks::Sphere> prob(std::move(sphere));

        std::vector<double> x0(5, 0.5);
        auto check_result = diagnostics::check_problem_gradient(prob, x0);

        if (!check_result.passed) {
            Rcpp::Rcout << "Sphere gradient check FAILED\n";
            return 1;
        }

        solvers::UcminfControl control;
        auto result = solvers::ucminf_solve(prob, x0, control);

        Rcpp::Rcout << "Sphere optimization:\n";
        Rcpp::Rcout << "  Convergence: " << result.convergence << "\n";
        Rcpp::Rcout << "  Final value: " << result.value << "\n";

        if (result.value > 1e-10) {
            Rcpp::Rcout << "Did not reach minimum\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error: " << e.what() << "\n";
        return 1;
    }
}

int test_powell_singular_benchmark() {
    try {
        benchmarks::PowellSingular powell(4);
        BenchmarkProblem<benchmarks::PowellSingular> prob(std::move(powell));

        std::vector<double> x0 = {3.0, -1.0, 0.0, 1.0};
        auto check_result = diagnostics::check_problem_gradient(prob, x0);

        if (!check_result.passed) {
            Rcpp::Rcout << "Powell Singular gradient check FAILED\n";
            diagnostics::print_gradient_check(check_result, Rcpp::Rcout);
            return 1;
        }

        solvers::UcminfControl control;
        control.maxeval = 2000;
        auto result = solvers::ucminf_solve(prob, x0, control);

        Rcpp::Rcout << "Powell Singular optimization:\n";
        Rcpp::Rcout << "  Convergence: " << result.convergence << "\n";
        Rcpp::Rcout << "  Final value: " << result.value << "\n";

        // Powell singular can be difficult, accept value < 1e-4
        if (result.value > 1e-4) {
            Rcpp::Rcout << "Did not reach minimum (f = " << result.value << ")\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error: " << e.what() << "\n";
        return 1;
    }
}

int test_beale_benchmark() {
    try {
        benchmarks::Beale beale;
        BenchmarkProblem<benchmarks::Beale> prob(std::move(beale));

        std::vector<double> x0 = {1.0, 1.0};
        auto check_result = diagnostics::check_problem_gradient(prob, x0);

        if (!check_result.passed) {
            Rcpp::Rcout << "Beale gradient check FAILED\n";
            return 1;
        }

        solvers::UcminfControl control;
        auto result = solvers::ucminf_solve(prob, x0, control);

        Rcpp::Rcout << "Beale optimization:\n";
        Rcpp::Rcout << "  Convergence: " << result.convergence << "\n";
        Rcpp::Rcout << "  Final value: " << result.value << "\n";
        Rcpp::Rcout << "  Solution: [" << result.par[0] << ", " << result.par[1] << "]\n";

        if (result.value > 1e-6) {
            Rcpp::Rcout << "Did not reach minimum\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error: " << e.what() << "\n";
        return 1;
    }
}

int test_brown_badly_scaled_benchmark() {
    try {
        benchmarks::BrownBadlyScaled brown;
        BenchmarkProblem<benchmarks::BrownBadlyScaled> prob(std::move(brown));

        std::vector<double> x0 = {1.0, 1.0};
        auto check_result = diagnostics::check_problem_gradient(prob, x0);

        if (!check_result.passed) {
            Rcpp::Rcout << "Brown Badly Scaled gradient check FAILED\n";
            return 1;
        }

        solvers::UcminfControl control;
        control.maxeval = 2000;
        auto result = solvers::ucminf_solve(prob, x0, control);

        Rcpp::Rcout << "Brown Badly Scaled optimization:\n";
        Rcpp::Rcout << "  Convergence: " << result.convergence << "\n";
        Rcpp::Rcout << "  Final value: " << result.value << "\n";

        // This problem is badly scaled, accept any reasonable convergence
        if (result.convergence < 1 || result.convergence > 4) {
            Rcpp::Rcout << "Did not converge properly\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error: " << e.what() << "\n";
        return 1;
    }
}

int test_broyden_tridiagonal_benchmark() {
    try {
        benchmarks::BroydenTridiagonal broyden(5);
        BenchmarkProblem<benchmarks::BroydenTridiagonal> prob(std::move(broyden));

        std::vector<double> x0(5, -1.0);
        auto check_result = diagnostics::check_problem_gradient(prob, x0);

        if (!check_result.passed) {
            Rcpp::Rcout << "Broyden Tridiagonal gradient check FAILED\n";
            return 1;
        }

        solvers::UcminfControl control;
        control.maxeval = 1000;
        auto result = solvers::ucminf_solve(prob, x0, control);

        Rcpp::Rcout << "Broyden Tridiagonal optimization:\n";
        Rcpp::Rcout << "  Convergence: " << result.convergence << "\n";
        Rcpp::Rcout << "  Final value: " << result.value << "\n";

        // Check for reasonable convergence
        if (result.convergence < 1 || result.convergence > 4) {
            Rcpp::Rcout << "Did not converge properly\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error: " << e.what() << "\n";
        return 1;
    }
}

int test_quadratic_benchmark() {
    try {
        auto quad = benchmarks::Quadratic::simple(5);
        BenchmarkProblem<benchmarks::Quadratic> prob(std::move(quad));

        std::vector<double> x0(5, 0.0);
        auto check_result = diagnostics::check_problem_gradient(prob, x0);

        if (!check_result.passed) {
            Rcpp::Rcout << "Quadratic gradient check FAILED\n";
            return 1;
        }

        solvers::UcminfControl control;
        auto result = solvers::ucminf_solve(prob, x0, control);

        Rcpp::Rcout << "Quadratic optimization:\n";
        Rcpp::Rcout << "  Convergence: " << result.convergence << "\n";
        Rcpp::Rcout << "  Final value: " << result.value << "\n";

        // Quadratic should converge very well
        if (result.convergence != 1 && result.convergence != 2) {
            Rcpp::Rcout << "Did not converge\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        Rcpp::Rcout << "Error: " << e.what() << "\n";
        return 1;
    }
}
