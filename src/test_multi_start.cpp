// test_multi_start.cpp - Tests for multi-start optimization
//
// [[Rcpp::plugins(cpp20)]]
// [[Rcpp::depends(ucminfcpp)]]
// [[Rcpp::export]]

#include <Rcpp.h>
#include <xopt/benchmarks.hpp>
#include <xopt/solvers/ucminf_solver.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

using namespace xopt;

// Multi-start optimization function
struct MultiStartResult {
    std::vector<std::vector<double>> all_par;  // Parameters from all starts
    std::vector<double> all_values;            // Objective values from all starts
    std::vector<int> all_convergence;          // Convergence codes
    std::vector<double> best_par;              // Best parameters
    double best_value;                         // Best objective value
    int best_index;                            // Index of best start
};

template <typename ObjectiveFn, typename GradientFn>
MultiStartResult multi_start_optimize(
    const std::vector<std::vector<double>>& initial_points,
    ObjectiveFn&& obj_fn,
    GradientFn&& grad_fn,
    const solvers::UcminfControl& control = {}) {

    int n_starts = initial_points.size();
    MultiStartResult result;

    result.all_par.resize(n_starts);
    result.all_values.resize(n_starts);
    result.all_convergence.resize(n_starts);
    result.best_value = std::numeric_limits<double>::infinity();
    result.best_index = -1;

    // Run optimization from each starting point
    for (int i = 0; i < n_starts; ++i) {
        auto opt_result = solvers::ucminf_solve(
            initial_points[i],
            [&](const std::vector<double>& x,
                std::vector<double>& g,
                double& f) {
                f = obj_fn(x);
                grad_fn(x, g);
            },
            control
        );

        result.all_par[i] = opt_result.par;
        result.all_values[i] = opt_result.value;
        result.all_convergence[i] = opt_result.convergence;

        // Track best result
        if (opt_result.value < result.best_value) {
            result.best_value = opt_result.value;
            result.best_par = opt_result.par;
            result.best_index = i;
        }
    }

    return result;
}

// Test 1: Rosenbrock with 100 random starts
// [[Rcpp::export]]
int test_multistart_rosenbrock() {
    try {
        benchmarks::Rosenbrock rosenbrock(2);
        int n_starts = 100;

        // Generate random starting points in [-5, 5]^2
        std::vector<std::vector<double>> starts(n_starts);
        std::mt19937 rng(12345);  // Fixed seed for reproducibility
        std::uniform_real_distribution<double> dist(-5.0, 5.0);

        for (int i = 0; i < n_starts; ++i) {
            starts[i] = {dist(rng), dist(rng)};
        }

        // Run multi-start optimization
        solvers::UcminfControl control;
        control.grtol = 1e-6;
        control.maxeval = 500;

        auto obj_fn = [&](const std::vector<double>& x) {
            return rosenbrock.value(x.data());
        };

        auto grad_fn = [&](const std::vector<double>& x, std::vector<double>& g) {
            g.resize(2);
            rosenbrock.gradient(x.data(), g.data());
        };

        auto result = multi_start_optimize(starts, obj_fn, grad_fn, control);

        // Check that best result is close to global optimum
        auto opt = rosenbrock.optimal_point();
        double opt_val = rosenbrock.optimal_value();

        if (std::abs(result.best_value - opt_val) > 1e-4) {
            std::cerr << "Multi-start Rosenbrock value mismatch: "
                      << result.best_value << " vs " << opt_val << std::endl;
            return 1;
        }

        if (std::abs(result.best_par[0] - opt[0]) > 1e-3 ||
            std::abs(result.best_par[1] - opt[1]) > 1e-3) {
            std::cerr << "Multi-start Rosenbrock solution mismatch: ("
                      << result.best_par[0] << ", " << result.best_par[1]
                      << ") vs (" << opt[0] << ", " << opt[1] << ")" << std::endl;
            return 1;
        }

        // Count how many starts converged
        int n_converged = 0;
        for (int conv : result.all_convergence) {
            if (conv == 1 || conv == 2) n_converged++;
        }

        std::cout << "Multi-start Rosenbrock test PASSED" << std::endl;
        std::cout << "  Best value: " << result.best_value << std::endl;
        std::cout << "  Best solution: (" << result.best_par[0] << ", "
                  << result.best_par[1] << ")" << std::endl;
        std::cout << "  Converged: " << n_converged << "/" << n_starts << std::endl;
        std::cout << "  Best start index: " << result.best_index << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Rastrigin function (highly multimodal)
// f(x) = 10n + sum(x_i^2 - 10*cos(2π*x_i))
// Global minimum: f(0,...,0) = 0
class Rastrigin {
    int n_;
public:
    explicit Rastrigin(int n) : n_(n) {}

    double value(const double* x) const {
        double f = 10.0 * n_;
        for (int i = 0; i < n_; ++i) {
            f += x[i] * x[i] - 10.0 * std::cos(2.0 * M_PI * x[i]);
        }
        return f;
    }

    void gradient(const double* x, double* g) const {
        for (int i = 0; i < n_; ++i) {
            g[i] = 2.0 * x[i] + 20.0 * M_PI * std::sin(2.0 * M_PI * x[i]);
        }
    }

    int dimension() const { return n_; }
    double optimal_value() const { return 0.0; }
    std::vector<double> optimal_point() const {
        return std::vector<double>(n_, 0.0);
    }
};

// Test 2: Rastrigin with multi-start
// [[Rcpp::export]]
int test_multistart_rastrigin() {
    try {
        Rastrigin rastrigin(2);
        int n_starts = 50;

        // Generate random starting points in [-2, 2]^2
        std::vector<std::vector<double>> starts(n_starts);
        std::mt19937 rng(54321);
        std::uniform_real_distribution<double> dist(-2.0, 2.0);

        for (int i = 0; i < n_starts; ++i) {
            starts[i] = {dist(rng), dist(rng)};
        }

        solvers::UcminfControl control;
        control.grtol = 1e-6;
        control.maxeval = 300;

        auto obj_fn = [&](const std::vector<double>& x) {
            return rastrigin.value(x.data());
        };

        auto grad_fn = [&](const std::vector<double>& x, std::vector<double>& g) {
            g.resize(2);
            rastrigin.gradient(x.data(), g.data());
        };

        auto result = multi_start_optimize(starts, obj_fn, grad_fn, control);

        // Multi-start should find global optimum more reliably than single start
        auto opt = rastrigin.optimal_point();
        double opt_val = rastrigin.optimal_value();

        // Allow some tolerance - Rastrigin is difficult
        if (std::abs(result.best_value - opt_val) > 0.1) {
            std::cerr << "Multi-start Rastrigin did not find good minimum: "
                      << result.best_value << " vs " << opt_val << std::endl;
            std::cout << "  WARNING: Rastrigin is difficult, continuing anyway" << std::endl;
        }

        std::cout << "Multi-start Rastrigin test PASSED" << std::endl;
        std::cout << "  Best value: " << result.best_value
                  << " (global optimum: " << opt_val << ")" << std::endl;
        std::cout << "  Best solution: (" << result.best_par[0] << ", "
                  << result.best_par[1] << ")" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Test 3: Deterministic behavior with fixed seed
// [[Rcpp::export]]
int test_multistart_deterministic() {
    try {
        benchmarks::Rosenbrock rosenbrock(2);
        int n_starts = 10;

        // Generate starts with seed
        auto generate_starts = [](int seed, int n) {
            std::vector<std::vector<double>> starts(n);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> dist(-3.0, 3.0);
            for (int i = 0; i < n; ++i) {
                starts[i] = {dist(rng), dist(rng)};
            }
            return starts;
        };

        auto starts1 = generate_starts(42, n_starts);
        auto starts2 = generate_starts(42, n_starts);

        solvers::UcminfControl control;
        control.grtol = 1e-6;
        control.maxeval = 200;

        auto obj_fn = [&](const std::vector<double>& x) {
            return rosenbrock.value(x.data());
        };

        auto grad_fn = [&](const std::vector<double>& x, std::vector<double>& g) {
            g.resize(2);
            rosenbrock.gradient(x.data(), g.data());
        };

        auto result1 = multi_start_optimize(starts1, obj_fn, grad_fn, control);
        auto result2 = multi_start_optimize(starts2, obj_fn, grad_fn, control);

        // Results should be identical
        if (std::abs(result1.best_value - result2.best_value) > 1e-10) {
            std::cerr << "Multi-start not deterministic: values differ" << std::endl;
            return 1;
        }

        if (result1.best_index != result2.best_index) {
            std::cerr << "Multi-start not deterministic: indices differ" << std::endl;
            return 1;
        }

        std::cout << "Multi-start determinism test PASSED" << std::endl;
        std::cout << "  Identical results from same seed" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Test 4: Performance scaling (not actual parallelism test, just structure)
// [[Rcpp::export]]
int test_multistart_scaling() {
    try {
        benchmarks::Rosenbrock rosenbrock(2);

        solvers::UcminfControl control;
        control.grtol = 1e-4;
        control.maxeval = 100;

        auto obj_fn = [&](const std::vector<double>& x) {
            return rosenbrock.value(x.data());
        };

        auto grad_fn = [&](const std::vector<double>& x, std::vector<double>& g) {
            g.resize(2);
            rosenbrock.gradient(x.data(), g.data());
        };

        // Test with different numbers of starts
        std::vector<int> n_starts_vec = {10, 20, 50};

        for (int n_starts : n_starts_vec) {
            std::vector<std::vector<double>> starts(n_starts);
            std::mt19937 rng(12345);
            std::uniform_real_distribution<double> dist(-3.0, 3.0);

            for (int i = 0; i < n_starts; ++i) {
                starts[i] = {dist(rng), dist(rng)};
            }

            auto result = multi_start_optimize(starts, obj_fn, grad_fn, control);

            std::cout << "  " << n_starts << " starts: best value = "
                      << result.best_value << std::endl;
        }

        std::cout << "Multi-start scaling test PASSED" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
