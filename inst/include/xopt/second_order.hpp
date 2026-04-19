// second_order.hpp - Hessian and Hessian-vector product utilities
//
// This header provides second-order derivative helpers used by trust-region
// and Laplace approximation routines. It supports user callbacks and robust
// finite-difference fallbacks when exact AD Hessians are unavailable.

#ifndef XOPT_SECOND_ORDER_HPP
#define XOPT_SECOND_ORDER_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace xopt {
namespace second_order {

using ObjectiveFunction = std::function<double(const std::vector<double>&)>;
using GradientFunction = std::function<void(const std::vector<double>&, std::vector<double>&)>;
using HessianFunction = std::function<void(const std::vector<double>&, std::vector<double>&)>;
using HvpFunction = std::function<void(const std::vector<double>&,
                                       const std::vector<double>&,
                                       std::vector<double>&)>;

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot: size mismatch");
    }
    double out = 0.0;
    for (size_t i = 0; i < a.size(); ++i) out += a[i] * b[i];
    return out;
}

inline double norm2(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}

inline void finite_diff_gradient(const ObjectiveFunction& fn,
                                 const std::vector<double>& x,
                                 std::vector<double>& g,
                                 double eps = 1e-6) {
    const int n = static_cast<int>(x.size());
    g.assign(n, 0.0);
    std::vector<double> xp = x;
    std::vector<double> xm = x;

    for (int j = 0; j < n; ++j) {
        const double h = eps * std::max(1.0, std::abs(x[j]));
        xp[j] = x[j] + h;
        xm[j] = x[j] - h;
        const double fp = fn(xp);
        const double fm = fn(xm);
        g[j] = (fp - fm) / (2.0 * h);
        xp[j] = x[j];
        xm[j] = x[j];
    }
}

inline void finite_diff_hessian(const ObjectiveFunction& fn,
                                const std::vector<double>& x,
                                std::vector<double>& H,
                                double eps = 1e-4) {
    const int n = static_cast<int>(x.size());
    H.assign(n * n, 0.0);
    std::vector<double> xpp = x, xpm = x, xmp = x, xmm = x;

    for (int i = 0; i < n; ++i) {
        const double hi = eps * std::max(1.0, std::abs(x[i]));
        for (int j = i; j < n; ++j) {
            const double hj = eps * std::max(1.0, std::abs(x[j]));

            xpp[i] = x[i] + hi; xpp[j] = x[j] + hj;
            xpm[i] = x[i] + hi; xpm[j] = x[j] - hj;
            xmp[i] = x[i] - hi; xmp[j] = x[j] + hj;
            xmm[i] = x[i] - hi; xmm[j] = x[j] - hj;

            const double fpp = fn(xpp);
            const double fpm = fn(xpm);
            const double fmp = fn(xmp);
            const double fmm = fn(xmm);
            const double hij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);

            H[i * n + j] = hij;
            H[j * n + i] = hij;

            xpp[i] = x[i]; xpp[j] = x[j];
            xpm[i] = x[i]; xpm[j] = x[j];
            xmp[i] = x[i]; xmp[j] = x[j];
            xmm[i] = x[i]; xmm[j] = x[j];
        }
    }
}

inline void dense_hessian_to_hvp(const std::vector<double>& H,
                                 const std::vector<double>& v,
                                 std::vector<double>& hv) {
    const int n = static_cast<int>(v.size());
    if (H.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("dense_hessian_to_hvp: Hessian size mismatch");
    }
    hv.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            hv[i] += H[i * n + j] * v[j];
        }
    }
}

inline void finite_diff_hvp(const ObjectiveFunction& fn,
                            const std::vector<double>& x,
                            const std::vector<double>& v,
                            std::vector<double>& hv,
                            double eps = 1e-6) {
    if (x.size() != v.size()) {
        throw std::invalid_argument("finite_diff_hvp: size mismatch");
    }

    std::vector<double> xp = x;
    std::vector<double> xm = x;
    const double scale = eps / std::max(1.0, norm2(v));
    for (size_t i = 0; i < x.size(); ++i) {
        xp[i] += scale * v[i];
        xm[i] -= scale * v[i];
    }

    std::vector<double> gp;
    std::vector<double> gm;
    finite_diff_gradient(fn, xp, gp, eps);
    finite_diff_gradient(fn, xm, gm, eps);

    hv.assign(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        hv[i] = (gp[i] - gm[i]) / (2.0 * scale);
    }
}

inline void hessian_or_fallback(const ObjectiveFunction& fn,
                                const std::vector<double>& x,
                                std::vector<double>& H,
                                const HessianFunction& hess = nullptr) {
    if (hess) {
        hess(x, H);
    } else {
        finite_diff_hessian(fn, x, H);
    }
}

inline void hvp_or_fallback(const ObjectiveFunction& fn,
                            const std::vector<double>& x,
                            const std::vector<double>& v,
                            std::vector<double>& hv,
                            const HvpFunction& hvp = nullptr,
                            const HessianFunction& hess = nullptr) {
    if (hvp) {
        hvp(x, v, hv);
        return;
    }

    if (hess) {
        std::vector<double> H;
        hess(x, H);
        dense_hessian_to_hvp(H, v, hv);
        return;
    }

    finite_diff_hvp(fn, x, v, hv);
}

} // namespace second_order
} // namespace xopt

#endif // XOPT_SECOND_ORDER_HPP
