// test_param_spec.cpp - Tests for ParamSpec flatten/unflatten and transforms
//
// [[Rcpp::plugins(cpp20)]]
// [[Rcpp::depends(ucminfcpp)]]

#include <Rcpp.h>
#include <xopt/param_spec.hpp>
#include <xopt/benchmarks.hpp>
#include <xopt/solvers/ucminf_solver.hpp>
#include <iostream>
#include <cmath>

using namespace xopt;

// Test round-trip: structured params → flatten → unflatten → structured
int test_param_spec_roundtrip() {
    try {
        ParamSpec spec;
        spec.add_scalar("alpha");
        spec.add_vector("beta", 3);
        spec.add_matrix("Sigma", 2, 2);

        // Create structured parameters
        std::map<std::string, std::vector<double>> params;
        params["alpha"] = {1.5};
        params["beta"] = {-1.0, 2.5, 0.3};
        params["Sigma"] = {1.0, 0.5, 0.5, 2.0};  // row-major

        // Flatten
        std::vector<double> flat;
        spec.flatten(params, flat);

        // Check size
        if (flat.size() != 1 + 3 + 4) {
            std::cerr << "Wrong flat size: " << flat.size() << std::endl;
            return 1;
        }

        // Unflatten
        std::map<std::string, std::vector<double>> params2;
        spec.unflatten(flat, params2);

        // Check equality
        auto check_vec = [](const std::vector<double>& a,
                           const std::vector<double>& b,
                           const std::string& name) {
            if (a.size() != b.size()) {
                std::cerr << name << " size mismatch" << std::endl;
                return false;
            }
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::abs(a[i] - b[i]) > 1e-10) {
                    std::cerr << name << "[" << i << "] mismatch: "
                              << a[i] << " vs " << b[i] << std::endl;
                    return false;
                }
            }
            return true;
        };

        if (!check_vec(params["alpha"], params2["alpha"], "alpha")) return 1;
        if (!check_vec(params["beta"], params2["beta"], "beta")) return 1;
        if (!check_vec(params["Sigma"], params2["Sigma"], "Sigma")) return 1;

        std::cout << "ParamSpec round-trip test PASSED" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Test positive transform
int test_positive_transform() {
    try {
        auto trans = positive();

        // Test forward/inverse
        double x = 2.5;
        double y = trans->forward(x);
        double x2 = trans->inverse(y);

        if (std::abs(x - x2) > 1e-10) {
            std::cerr << "Positive transform round-trip failed: "
                      << x << " -> " << y << " -> " << x2 << std::endl;
            return 1;
        }

        // Test derivative
        double eps = 1e-8;
        double deriv_approx = (trans->forward(x + eps) - trans->forward(x - eps)) / (2 * eps);
        double deriv_exact = trans->forward_deriv(x);

        if (std::abs(deriv_approx - deriv_exact) > 1e-6) {
            std::cerr << "Positive transform derivative mismatch: "
                      << deriv_approx << " vs " << deriv_exact << std::endl;
            return 1;
        }

        std::cout << "Positive transform test PASSED" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Test bounded transform
int test_bounded_transform() {
    try {
        auto trans = bounded(-2.0, 5.0);

        // Test forward/inverse
        double x = 1.5;
        double y = trans->forward(x);
        double x2 = trans->inverse(y);

        if (std::abs(x - x2) > 1e-10) {
            std::cerr << "Bounded transform round-trip failed: "
                      << x << " -> " << y << " -> " << x2 << std::endl;
            return 1;
        }

        // Test derivative
        double eps = 1e-8;
        double deriv_approx = (trans->forward(x + eps) - trans->forward(x - eps)) / (2 * eps);
        double deriv_exact = trans->forward_deriv(x);

        if (std::abs(deriv_approx - deriv_exact) > 1e-6) {
            std::cerr << "Bounded transform derivative mismatch: "
                      << deriv_approx << " vs " << deriv_exact << std::endl;
            return 1;
        }

        std::cout << "Bounded transform test PASSED" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Test Rosenbrock with structured params
int test_rosenbrock_structured() {
    try {
        // Define Rosenbrock via ParamSpec
        ParamSpec spec;
        spec.add_scalar("x1");
        spec.add_scalar("x2");

        benchmarks::Rosenbrock rosenbrock(2);

        // Initial structured params
        std::map<std::string, std::vector<double>> params;
        params["x1"] = {-1.2};
        params["x2"] = {1.0};

        // Flatten
        std::vector<double> x0;
        spec.flatten(params, x0);

        // Optimize using flat representation
        solvers::UcminfControl control;
        control.grtol = 1e-6;
        control.maxeval = 500;

        auto result = solvers::ucminf_solve(
            x0,
            [&rosenbrock](const std::vector<double>& x,
                         std::vector<double>& g,
                         double& f) {
                f = rosenbrock.value(x.data());
                rosenbrock.gradient(x.data(), g.data());
            },
            control
        );

        // Unflatten result
        std::map<std::string, std::vector<double>> params_opt;
        spec.unflatten(result.par, params_opt);

        // Check convergence
        if (result.convergence != 1 && result.convergence != 2) {
            std::cerr << "Rosenbrock did not converge: " << result.message << std::endl;
            return 1;
        }

        // Check solution
        auto opt = rosenbrock.optimal_point();
        if (std::abs(params_opt["x1"][0] - opt[0]) > 1e-4 ||
            std::abs(params_opt["x2"][0] - opt[1]) > 1e-4) {
            std::cerr << "Rosenbrock solution mismatch: ("
                      << params_opt["x1"][0] << ", " << params_opt["x2"][0]
                      << ") vs (" << opt[0] << ", " << opt[1] << ")" << std::endl;
            return 1;
        }

        std::cout << "Rosenbrock structured params test PASSED" << std::endl;
        std::cout << "  Solution: x1=" << params_opt["x1"][0]
                  << ", x2=" << params_opt["x2"][0] << std::endl;
        std::cout << "  Value: " << result.value << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Test positive-constrained parameter on exponential MLE
int test_positive_constrained_mle() {
    try {
        // Exponential MLE: maximize L(λ) = n*log(λ) - λ*sum(x_i)
        // Equivalent to minimizing -L(λ)
        std::vector<double> data = {1.2, 0.8, 1.5, 2.1, 0.9, 1.3};
        double n = data.size();
        double sum_x = 0.0;
        for (double x : data) sum_x += x;

        // Use ParamSpec with positive transform
        ParamSpec spec;
        spec.add_scalar("lambda", positive());

        // Initial value (constrained space)
        std::map<std::string, std::vector<double>> params;
        params["lambda"] = {1.0};  // Must be positive

        // Flatten (transforms to unconstrained)
        std::vector<double> x0;
        spec.flatten(params, x0);

        // Define objective and gradient in unconstrained space
        auto objective_fn = [&](const std::vector<double>& x_flat,
                                std::vector<double>& g,
                                double& f) {
            // Transform back to constrained space
            std::map<std::string, std::vector<double>> p;
            spec.unflatten(x_flat, p);
            double lambda = p["lambda"][0];

            // Negative log-likelihood
            f = -n * std::log(lambda) + lambda * sum_x;

            // Gradient in constrained space: d(-L)/dλ = -n/λ + sum(x_i)
            double g_lambda = -n / lambda + sum_x;

            // Chain rule: transform to unconstrained space
            // If y = log(λ), then dλ/dy = λ
            // So d(-L)/dy = d(-L)/dλ * dλ/dy = g_lambda * λ
            g.resize(1);
            g[0] = g_lambda * lambda;
        };

        solvers::UcminfControl control;
        control.grtol = 1e-8;
        control.maxeval = 100;

        auto result = solvers::ucminf_solve(x0, objective_fn, control);

        // Unflatten
        std::map<std::string, std::vector<double>> params_opt;
        spec.unflatten(result.par, params_opt);

        // Check convergence
        if (result.convergence != 1 && result.convergence != 2) {
            std::cerr << "Exponential MLE did not converge: "
                      << result.message << std::endl;
            return 1;
        }

        // Analytical solution: λ_MLE = n / sum(x_i)
        double lambda_true = n / sum_x;
        if (std::abs(params_opt["lambda"][0] - lambda_true) > 1e-4) {
            std::cerr << "Exponential MLE mismatch: "
                      << params_opt["lambda"][0] << " vs " << lambda_true << std::endl;
            return 1;
        }

        // Check that λ is positive (it should be, given the transform)
        if (params_opt["lambda"][0] <= 0.0) {
            std::cerr << "Lambda not positive: " << params_opt["lambda"][0] << std::endl;
            return 1;
        }

        std::cout << "Positive-constrained MLE test PASSED" << std::endl;
        std::cout << "  MLE estimate: λ = " << params_opt["lambda"][0] << std::endl;
        std::cout << "  True value: λ = " << lambda_true << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
