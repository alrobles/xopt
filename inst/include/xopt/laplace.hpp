// laplace.hpp - Laplace approximation for marginal likelihoods

#ifndef XOPT_LAPLACE_HPP
#define XOPT_LAPLACE_HPP

#include <xopt/second_order.hpp>
#include <xopt/solvers/trust_region_newton.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

namespace xopt {

struct LaplaceResult {
    double log_marginal = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> mode;
    std::vector<double> hessian;
    int iterations = 0;
    int convergence = 0;
    std::string message = "Optimization failed";
};

inline bool cholesky_logdet(const std::vector<double>& A, int n, double& logdet) {
    std::vector<double> L(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i * n + j];
            for (int k = 0; k < j; ++k) {
                sum -= L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                if (sum <= 0.0) return false;
                L[i * n + j] = std::sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }

    logdet = 0.0;
    for (int i = 0; i < n; ++i) {
        logdet += 2.0 * std::log(L[i * n + i]);
    }
    return true;
}

inline LaplaceResult laplace_approximate(
    const std::vector<double>& x0,
    const second_order::ObjectiveFunction& nll,
    const second_order::GradientFunction& grad = nullptr,
    const second_order::HessianFunction& hess = nullptr,
    const solvers::TRNewtonControl& control = {}) {
    if (!nll) {
        throw std::invalid_argument("laplace_approximate: objective required");
    }

    LaplaceResult out;
    const int n = static_cast<int>(x0.size());
    if (n == 0) {
        throw std::invalid_argument("laplace_approximate: x0 must be non-empty");
    }

    solvers::TRNewtonResult mode_fit = solvers::trust_region_newton(x0, nll, grad, nullptr, control);
    out.mode = mode_fit.par;
    out.iterations = mode_fit.iterations;
    out.convergence = mode_fit.convergence;
    out.message = mode_fit.message;

    if (mode_fit.convergence == 0) {
        return out;
    }

    second_order::hessian_or_fallback(nll, out.mode, out.hessian, hess);
    double logdet = 0.0;
    if (!cholesky_logdet(out.hessian, n, logdet)) {
        out.convergence = 0;
        out.message = "Laplace Hessian not positive definite at mode";
        return out;
    }

    const double nll_mode = nll(out.mode);
    const double log_two_pi = std::log(2.0 * 3.14159265358979323846);
    out.log_marginal = -nll_mode + 0.5 * static_cast<double>(n) * log_two_pi - 0.5 * logdet;
    return out;
}

} // namespace xopt

#endif // XOPT_LAPLACE_HPP
