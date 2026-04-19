// test_phase3.cpp - Phase 3 second-order feature tests
//
// [[Rcpp::plugins(cpp20)]]
// [[Rcpp::depends(ucminfcpp)]]

#include <Rcpp.h>
#include <xopt/problem.hpp>
#include <xopt/second_order.hpp>
#include <xopt/solvers/trust_region_newton.hpp>
#include <xopt/solvers/ucminf_solver.hpp>
#include <xopt/laplace.hpp>
#include <vector>
#include <cmath>
#include <numbers>

using namespace xopt;

// [[Rcpp::export]]
int test_phase3_hessian_hvp() {
    try {
        // f(x) = 0.5 * x^T A x, A symmetric
        std::vector<double> A = {4.0, 1.0, 1.0, 3.0};
        auto fn = [&](const std::vector<double>& x) {
            return 0.5 * (A[0] * x[0] * x[0] + 2.0 * A[1] * x[0] * x[1] + A[3] * x[1] * x[1]);
        };
        std::vector<double> x = {0.5, -1.2};

        std::vector<double> H;
        second_order::finite_diff_hessian(fn, x, H);
        const double max_h_err = std::max(
            std::max(std::abs(H[0] - A[0]), std::abs(H[1] - A[1])),
            std::max(std::abs(H[2] - A[2]), std::abs(H[3] - A[3]))
        );
        if (max_h_err > 1e-4) return 1;

        std::vector<double> v = {1.5, -0.2};
        std::vector<double> hv;
        second_order::finite_diff_hvp(fn, x, v, hv);
        std::vector<double> hv_exact = {
            A[0] * v[0] + A[1] * v[1],
            A[2] * v[0] + A[3] * v[1]
        };
        const double max_hvp_err = std::max(
            std::abs(hv[0] - hv_exact[0]),
            std::abs(hv[1] - hv_exact[1])
        );
        if (max_hvp_err > 1e-4) return 1;

        return 0;
    } catch (...) {
        return 1;
    }
}

struct IllConditionedQuad : public xopt::ProblemBase<double> {
    IllConditionedQuad() : ProblemBase<double>(2) {}
    double value(const double* x) const {
        return 0.5 * (x[0] * x[0] + 1e4 * x[1] * x[1]);
    }
    void gradient(const double* x, double* g) const {
        g[0] = x[0];
        g[1] = 1e4 * x[1];
    }
    static constexpr bool has_gradient() { return true; }
    static constexpr GradKind gradient_kind() { return GradKind::UserFn; }
};

// [[Rcpp::export]]
int test_phase3_trust_region_newton() {
    try {
        std::vector<double> x0 = {10.0, 10.0};
        auto fn = [](const std::vector<double>& x) {
            return 0.5 * (x[0] * x[0] + 1e4 * x[1] * x[1]);
        };
        auto grad = [](const std::vector<double>& x, std::vector<double>& g) {
            g.resize(2);
            g[0] = x[0];
            g[1] = 1e4 * x[1];
        };
        auto hvp = [](const std::vector<double>&, const std::vector<double>& v, std::vector<double>& hv) {
            hv.resize(2);
            hv[0] = v[0];
            hv[1] = 1e4 * v[1];
        };

        solvers::TRNewtonControl ctrl;
        ctrl.maxiter = 200;
        ctrl.delta_init = 1.0;
        solvers::TRNewtonResult tr = solvers::trust_region_newton(x0, fn, grad, hvp, ctrl);

        IllConditionedQuad prob;
        solvers::UcminfControl uc_ctrl;
        uc_ctrl.maxeval = 200;
        solvers::UcminfResult bfgs = solvers::ucminf_solve(prob, x0, uc_ctrl);

        if (tr.convergence == 0) return 1;
        if (tr.value > bfgs.value + 1e-6) return 1;
        if (tr.value > 1e-8) return 1;
        return 0;
    } catch (...) {
        return 1;
    }
}

// [[Rcpp::export]]
int test_phase3_laplace() {
    try {
        const double mu = 0.7;
        const double sigma2 = 0.5;
        auto nll = [&](const std::vector<double>& u) {
            const double d = u[0] - mu;
            return 0.5 * d * d / sigma2;
        };
        auto grad = [&](const std::vector<double>& u, std::vector<double>& g) {
            g.resize(1);
            g[0] = (u[0] - mu) / sigma2;
        };
        auto hess = [&](const std::vector<double>&, std::vector<double>& H) {
            H = {1.0 / sigma2};
        };

        auto fit = laplace_approximate({0.0}, nll, grad, nullptr, hess);
        if (fit.convergence == 0) return 1;
        const double exact = 0.5 * std::log(2.0 * std::numbers::pi_v<double> * sigma2);
        if (std::abs(fit.log_marginal - exact) > 1e-8) return 1;
        return 0;
    } catch (...) {
        return 1;
    }
}
