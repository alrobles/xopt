// diagnostics.hpp - Gradient checking and diagnostic utilities
//
// This header provides tools for verifying gradient implementations
// via finite differences and other numerical diagnostics.

#ifndef XOPT_DIAGNOSTICS_HPP
#define XOPT_DIAGNOSTICS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <iomanip>

namespace xopt {
namespace diagnostics {

// Gradient check result
struct GradientCheckResult {
    bool passed;                      // Overall pass/fail
    double max_abs_error;             // Maximum absolute error
    double max_rel_error;             // Maximum relative error
    std::vector<double> abs_errors;   // Per-component absolute errors
    std::vector<double> rel_errors;   // Per-component relative errors
    std::vector<double> analytical;   // Analytical gradient
    std::vector<double> numerical;    // Numerical gradient
};

// Compute numerical gradient using central differences
template <typename Fn>
std::vector<double> numerical_gradient(
    Fn fn,
    const std::vector<double>& x,
    double epsilon = 1e-6) {

    int n = x.size();
    std::vector<double> grad(n);
    std::vector<double> x_plus = x;
    std::vector<double> x_minus = x;

    for (int i = 0; i < n; ++i) {
        // Central difference: [f(x+h) - f(x-h)] / (2h)
        x_plus[i] = x[i] + epsilon;
        x_minus[i] = x[i] - epsilon;

        double f_plus = fn(x_plus);
        double f_minus = fn(x_minus);

        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);

        // Reset for next iteration
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    return grad;
}

// Check gradient against numerical approximation
template <typename Fn, typename GradFn>
GradientCheckResult check_gradient(
    Fn fn,
    GradFn grad_fn,
    const std::vector<double>& x,
    double epsilon = 1e-6,
    double abs_tol = 1e-5,
    double rel_tol = 1e-4) {

    int n = x.size();
    GradientCheckResult result;

    // Compute analytical gradient
    result.analytical.resize(n);
    grad_fn(x, result.analytical);

    // Compute numerical gradient
    result.numerical = numerical_gradient(fn, x, epsilon);

    // Compute errors
    result.abs_errors.resize(n);
    result.rel_errors.resize(n);
    result.max_abs_error = 0.0;
    result.max_rel_error = 0.0;

    for (int i = 0; i < n; ++i) {
        double abs_err = std::abs(result.analytical[i] - result.numerical[i]);
        double rel_err = 0.0;

        if (std::abs(result.numerical[i]) > 1e-10) {
            rel_err = abs_err / std::abs(result.numerical[i]);
        }

        result.abs_errors[i] = abs_err;
        result.rel_errors[i] = rel_err;
        result.max_abs_error = std::max(result.max_abs_error, abs_err);
        result.max_rel_error = std::max(result.max_rel_error, rel_err);
    }

    // Determine pass/fail
    result.passed = (result.max_abs_error < abs_tol ||
                     result.max_rel_error < rel_tol);

    return result;
}

// Check gradient for a Problem object
template <typename Problem>
GradientCheckResult check_problem_gradient(
    const Problem& prob,
    const std::vector<double>& x,
    double epsilon = 1e-6,
    double abs_tol = 1e-5,
    double rel_tol = 1e-4) {

    static_assert(Problem::has_gradient(),
                  "Problem must provide gradients");

    auto fn = [&prob](const std::vector<double>& x_eval) {
        return prob.value(x_eval.data());
    };

    auto grad_fn = [&prob](const std::vector<double>& x_eval,
                           std::vector<double>& g) {
        prob.gradient(x_eval.data(), g.data());
    };

    return check_gradient(fn, grad_fn, x, epsilon, abs_tol, rel_tol);
}

// Print gradient check results
inline void print_gradient_check(const GradientCheckResult& result,
                                  std::ostream& os = std::cout) {
    os << "Gradient Check Results:\n";
    os << "  Status: " << (result.passed ? "PASSED" : "FAILED") << "\n";
    os << "  Max absolute error: " << std::scientific << std::setprecision(3)
       << result.max_abs_error << "\n";
    os << "  Max relative error: " << std::scientific << std::setprecision(3)
       << result.max_rel_error << "\n";

    if (!result.passed) {
        os << "\nDetailed errors:\n";
        os << std::setw(5) << "i"
           << std::setw(15) << "Analytical"
           << std::setw(15) << "Numerical"
           << std::setw(15) << "Abs Error"
           << std::setw(15) << "Rel Error" << "\n";

        for (size_t i = 0; i < result.analytical.size(); ++i) {
            os << std::setw(5) << i
               << std::setw(15) << std::scientific << std::setprecision(6)
               << result.analytical[i]
               << std::setw(15) << result.numerical[i]
               << std::setw(15) << result.abs_errors[i]
               << std::setw(15) << result.rel_errors[i] << "\n";
        }
    }
}

} // namespace diagnostics
} // namespace xopt

#endif // XOPT_DIAGNOSTICS_HPP
