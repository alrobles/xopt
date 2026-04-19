// trust_region_newton.hpp - Trust-region Newton with Steihaug CG

#ifndef XOPT_TRUST_REGION_NEWTON_HPP
#define XOPT_TRUST_REGION_NEWTON_HPP

#include <xopt/second_order.hpp>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace xopt {
namespace solvers {

struct TRNewtonControl {
    double gtol = 1e-8;
    double xtol = 1e-10;
    double ftol = 1e-12;
    int maxiter = 200;
    int cg_maxiter = 200;
    double delta_init = 1.0;
    double delta_max = 1000.0;
    double eta = 0.15;
    double boundary_tol = 1e-10;
};

struct TRNewtonResult {
    std::vector<double> par;
    double value = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> gradient;
    int iterations = 0;
    int convergence = 0;
    std::string message = "Maximum iterations reached";
};

inline std::vector<double> add_scaled(const std::vector<double>& x,
                                      const std::vector<double>& y,
                                      double alpha) {
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] + alpha * y[i];
    return out;
}

inline double steihaug_tau_to_boundary(const std::vector<double>& p,
                                       const std::vector<double>& d,
                                       double delta) {
    const double pp = second_order::dot(p, p);
    const double pd = second_order::dot(p, d);
    const double dd = second_order::dot(d, d);
    constexpr double dd_tol = 1e-24;
    if (dd <= dd_tol) {
        return 0.0;
    }
    const double c = pp - delta * delta;
    const double disc = std::max(0.0, pd * pd - dd * c);
    return (-pd + std::sqrt(disc)) / dd;
}

inline std::vector<double> steihaug_cg(const std::vector<double>& g,
                                       const second_order::ObjectiveFunction& fn,
                                       const std::vector<double>& x,
                                       const second_order::HvpFunction& hvp,
                                       double delta,
                                       int maxiter) {
    const int n = static_cast<int>(g.size());
    std::vector<double> p(n, 0.0);
    std::vector<double> r = g;
    std::vector<double> d(n);
    for (int i = 0; i < n; ++i) d[i] = -r[i];

    const double gnorm0 = second_order::norm2(g);
    if (gnorm0 == 0.0) return p;

    for (int k = 0; k < maxiter; ++k) {
        std::vector<double> Bd;
        second_order::hvp_or_fallback(fn, x, d, Bd, hvp, nullptr);
        const double dBd = second_order::dot(d, Bd);

        if (dBd <= 0.0) {
            const double tau = steihaug_tau_to_boundary(p, d, delta);
            return add_scaled(p, d, tau);
        }

        const double rr = second_order::dot(r, r);
        const double alpha = rr / dBd;
        std::vector<double> p_next = add_scaled(p, d, alpha);
        if (second_order::norm2(p_next) >= delta) {
            const double tau = steihaug_tau_to_boundary(p, d, delta);
            return add_scaled(p, d, tau);
        }

        std::vector<double> r_next(n);
        for (int i = 0; i < n; ++i) r_next[i] = r[i] + alpha * Bd[i];

        if (second_order::norm2(r_next) <= 1e-12 * gnorm0) {
            return p_next;
        }

        const double beta = second_order::dot(r_next, r_next) / rr;
        for (int i = 0; i < n; ++i) d[i] = -r_next[i] + beta * d[i];
        p = p_next;
        r = r_next;
    }

    return p;
}

inline TRNewtonResult trust_region_newton(
    const std::vector<double>& x0,
    const second_order::ObjectiveFunction& fn,
    const second_order::GradientFunction& grad = nullptr,
    const second_order::HvpFunction& hvp = nullptr,
    const TRNewtonControl& control = {}) {
    if (!fn) {
        throw std::invalid_argument("trust_region_newton: objective function required");
    }

    TRNewtonResult result;
    std::vector<double> x = x0;
    const int n = static_cast<int>(x.size());
    if (n == 0) {
        throw std::invalid_argument("trust_region_newton: x0 must be non-empty");
    }

    auto gradient_fn = grad;
    if (!gradient_fn) {
        gradient_fn = [&](const std::vector<double>& xv, std::vector<double>& gv) {
            second_order::finite_diff_gradient(fn, xv, gv);
        };
    }

    double f = fn(x);
    double delta = std::max(1e-12, control.delta_init);

    for (int iter = 0; iter < control.maxiter; ++iter) {
        std::vector<double> g;
        gradient_fn(x, g);
        const double gnorm = second_order::norm2(g);
        if (gnorm <= control.gtol) {
            result.convergence = 1;
            result.message = "Gradient below tolerance";
            result.iterations = iter;
            break;
        }

        std::vector<double> p = steihaug_cg(g, fn, x, hvp, delta, control.cg_maxiter);
        const double pnorm = second_order::norm2(p);
        if (pnorm <= control.xtol) {
            result.convergence = 2;
            result.message = "Step below tolerance";
            result.iterations = iter;
            break;
        }

        std::vector<double> Bp;
        second_order::hvp_or_fallback(fn, x, p, Bp, hvp, nullptr);
        const double pred = -(second_order::dot(g, p) + 0.5 * second_order::dot(p, Bp));
        if (pred <= 0.0) {
            delta *= 0.25;
            if (delta < control.xtol) {
                result.convergence = 3;
                result.message = "Predicted decrease non-positive";
                result.iterations = iter;
                break;
            }
            continue;
        }

        std::vector<double> x_trial = add_scaled(x, p, 1.0);
        const double f_trial = fn(x_trial);
        const double ared = f - f_trial;
        const double rho = ared / pred;

        if (rho < 0.25) {
            delta *= 0.25;
        } else if (rho > 0.75 &&
                   std::abs(pnorm - delta) <= control.boundary_tol * std::max(1.0, delta)) {
            delta = std::min(2.0 * delta, control.delta_max);
        }

        if (rho > control.eta) {
            x = std::move(x_trial);
            if (std::abs(ared) <= control.ftol * (1.0 + std::abs(f))) {
                result.convergence = 4;
                result.message = "Function change below tolerance";
                result.iterations = iter + 1;
                f = f_trial;
                break;
            }
            f = f_trial;
        }

        result.iterations = iter + 1;
    }

    if (result.convergence == 0) {
        result.message = "Maximum iterations reached";
    }

    std::vector<double> g_final;
    gradient_fn(x, g_final);
    result.par = std::move(x);
    result.value = f;
    result.gradient = std::move(g_final);
    return result;
}

} // namespace solvers
} // namespace xopt

#endif // XOPT_TRUST_REGION_NEWTON_HPP
