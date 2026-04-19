#ifndef XOPT_PHASE4_HPP
#define XOPT_PHASE4_HPP

#include <xopt/second_order.hpp>
#include <xopt/solvers/trust_region_newton.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xopt {
namespace phase4 {

using ObjectiveFunction = second_order::ObjectiveFunction;
using GradientFunction = second_order::GradientFunction;
using ConstraintFunction = std::function<void(const std::vector<double>&,
                                              std::vector<double>&,
                                              std::vector<double>&)>;

struct ALControl {
    double tol = 1e-6;
    double rho_init = 10.0;
    double rho_max = 1e8;
    int outer_maxiter = 50;
    solvers::TRNewtonControl inner_control{};
};

struct ALResult {
    std::vector<double> par;
    double value = std::numeric_limits<double>::quiet_NaN();
    double constraint_violation = std::numeric_limits<double>::infinity();
    int outer_iterations = 0;
    int convergence = 0;
    std::string message = "Maximum outer iterations reached";
};

inline double max_violation(const std::vector<double>& c_eq,
                            const std::vector<double>& c_ineq) {
    double v = 0.0;
    for (double ci : c_eq) v = std::max(v, std::abs(ci));
    for (double ci : c_ineq) v = std::max(v, std::max(0.0, ci));
    return v;
}

inline double augmented_objective(const std::vector<double>& x,
                                  const ObjectiveFunction& objective,
                                  const ConstraintFunction& constraints,
                                  const std::vector<double>& lambda_eq,
                                  const std::vector<double>& lambda_ineq,
                                  double rho) {
    std::vector<double> c_eq;
    std::vector<double> c_ineq;
    constraints(x, c_eq, c_ineq);

    double out = objective(x);
    for (size_t i = 0; i < c_eq.size(); ++i) {
        out += lambda_eq[i] * c_eq[i] + 0.5 * rho * c_eq[i] * c_eq[i];
    }
    for (size_t i = 0; i < c_ineq.size(); ++i) {
        const double shifted = lambda_ineq[i] / rho + c_ineq[i];
        const double pos = std::max(0.0, shifted);
        out += 0.5 * rho * (pos * pos - (lambda_ineq[i] / rho) * (lambda_ineq[i] / rho));
    }
    return out;
}

inline ALResult augmented_lagrangian_solve(const std::vector<double>& x0,
                                           const ObjectiveFunction& objective,
                                           const ConstraintFunction& constraints,
                                           const GradientFunction& gradient = nullptr,
                                           const ALControl& control = {}) {
    ALResult result;
    std::vector<double> x = x0;

    std::vector<double> c_eq;
    std::vector<double> c_ineq;
    constraints(x, c_eq, c_ineq);

    std::vector<double> lambda_eq(c_eq.size(), 0.0);
    std::vector<double> lambda_ineq(c_ineq.size(), 0.0);

    double rho = std::max(1e-12, control.rho_init);
    double best_violation = std::numeric_limits<double>::infinity();

    for (int outer = 0; outer < control.outer_maxiter; ++outer) {
        auto aug_fn = [&](const std::vector<double>& xv) {
            return augmented_objective(xv, objective, constraints, lambda_eq, lambda_ineq, rho);
        };

        second_order::GradientFunction aug_grad;
        if (gradient) {
            aug_grad = [&](const std::vector<double>& xv, std::vector<double>& g) {
                gradient(xv, g);
                std::vector<double> g_aug;
                second_order::finite_diff_gradient(aug_fn, xv, g_aug);
                if (g_aug.size() != g.size()) {
                    g = std::move(g_aug);
                    return;
                }
                for (size_t i = 0; i < g.size(); ++i) {
                    g[i] = g_aug[i];
                }
            };
        } else {
            aug_grad = [&](const std::vector<double>& xv, std::vector<double>& g) {
                second_order::finite_diff_gradient(aug_fn, xv, g);
            };
        }

        auto inner = solvers::trust_region_newton(x, aug_fn, aug_grad, nullptr, control.inner_control);
        x = inner.par;

        constraints(x, c_eq, c_ineq);
        const double violation = max_violation(c_eq, c_ineq);

        for (size_t i = 0; i < c_eq.size(); ++i) {
            lambda_eq[i] += rho * c_eq[i];
        }
        for (size_t i = 0; i < c_ineq.size(); ++i) {
            lambda_ineq[i] = std::max(0.0, lambda_ineq[i] + rho * c_ineq[i]);
        }

        result.outer_iterations = outer + 1;
        result.constraint_violation = violation;
        if (violation < best_violation * 0.5) {
            best_violation = violation;
        } else {
            rho = std::min(control.rho_max, rho * 5.0);
        }

        if (violation <= control.tol) {
            result.convergence = 1;
            result.message = "Constraint violation below tolerance";
            break;
        }
    }

    result.par = x;
    result.value = objective(x);
    if (result.convergence == 0) {
        constraints(x, c_eq, c_ineq);
        result.constraint_violation = max_violation(c_eq, c_ineq);
    }
    return result;
}

template <typename SolveFn, typename Result>
struct MultiStartResult {
    std::vector<Result> all_results;
    size_t best_index = 0;
};

template <typename SolveFn, typename Result>
inline MultiStartResult<SolveFn, Result> parallel_multi_start(
    const std::vector<std::vector<double>>& starts,
    SolveFn solve_fn,
    std::function<double(const Result&)> value_fn,
    size_t n_threads = std::thread::hardware_concurrency()) {

    MultiStartResult<SolveFn, Result> out;
    const size_t n = starts.size();
    out.all_results.resize(n);

    if (n == 0) {
        out.best_index = 0;
        return out;
    }

    const size_t workers = std::max<size_t>(1, std::min(n, n_threads == 0 ? 1 : n_threads));
    std::atomic<size_t> next{0};

    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (size_t t = 0; t < workers; ++t) {
        threads.emplace_back([&]() {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= n) break;
                out.all_results[i] = solve_fn(starts[i]);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    double best_value = value_fn(out.all_results[0]);
    size_t best_index = 0;
    for (size_t i = 1; i < n; ++i) {
        const double val = value_fn(out.all_results[i]);
        if (val < best_value) {
            best_value = val;
            best_index = i;
        }
    }
    out.best_index = best_index;
    return out;
}

using SparsePattern = std::vector<std::vector<int>>;  // column -> rows

inline bool intersects(const std::vector<int>& a, const std::vector<int>& b) {
    size_t i = 0;
    size_t j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) return true;
        if (a[i] < b[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return false;
}

inline SparsePattern detect_jacobian_sparsity(
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& residual_fn,
    const std::vector<double>& x,
    double eps = 1e-6,
    double tol = 1e-10) {

    std::vector<double> r0;
    residual_fn(x, r0);
    const int m = static_cast<int>(r0.size());
    const int n = static_cast<int>(x.size());

    SparsePattern pattern(static_cast<size_t>(n));
    std::vector<double> xp = x;

    for (int j = 0; j < n; ++j) {
        const double h = eps * std::max(1.0, std::abs(x[j]));
        xp[j] += h;
        std::vector<double> rp;
        residual_fn(xp, rp);
        xp[j] = x[j];

        for (int i = 0; i < m; ++i) {
            const double d = (rp[i] - r0[i]) / h;
            if (std::abs(d) > tol) {
                pattern[j].push_back(i);
            }
        }
        std::sort(pattern[j].begin(), pattern[j].end());
    }

    return pattern;
}

inline std::vector<int> greedy_column_coloring(const SparsePattern& pattern) {
    const int n = static_cast<int>(pattern.size());
    std::vector<int> colors(static_cast<size_t>(n), -1);

    for (int j = 0; j < n; ++j) {
        std::vector<bool> used(static_cast<size_t>(n), false);
        for (int k = 0; k < j; ++k) {
            if (colors[k] >= 0 && intersects(pattern[j], pattern[k])) {
                used[static_cast<size_t>(colors[k])] = true;
            }
        }

        int c = 0;
        while (c < n && used[static_cast<size_t>(c)]) {
            ++c;
        }
        colors[j] = c;
    }

    return colors;
}

inline std::vector<double> compressed_fd_jacobian(
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& residual_fn,
    const std::vector<double>& x,
    const SparsePattern& pattern,
    const std::vector<int>& colors,
    double eps = 1e-6) {

    std::vector<double> r0;
    residual_fn(x, r0);
    const int m = static_cast<int>(r0.size());
    const int n = static_cast<int>(x.size());
    std::vector<double> J(static_cast<size_t>(m * n), 0.0);

    int max_color = 0;
    for (int c : colors) max_color = std::max(max_color, c);

    std::vector<double> xp = x;
    for (int color = 0; color <= max_color; ++color) {
        std::vector<int> cols;
        cols.reserve(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) {
            if (colors[static_cast<size_t>(j)] == color) {
                cols.push_back(j);
                xp[static_cast<size_t>(j)] += eps * std::max(1.0, std::abs(x[static_cast<size_t>(j)]));
            }
        }

        if (cols.empty()) continue;

        std::vector<double> rp;
        residual_fn(xp, rp);

        for (int j : cols) {
            const double h = eps * std::max(1.0, std::abs(x[static_cast<size_t>(j)]));
            for (int i : pattern[static_cast<size_t>(j)]) {
                J[static_cast<size_t>(i * n + j)] = (rp[static_cast<size_t>(i)] - r0[static_cast<size_t>(i)]) / h;
            }
            xp[static_cast<size_t>(j)] = x[static_cast<size_t>(j)];
        }
    }

    return J;
}

struct CscMatrix {
    int nrow = 0;
    int ncol = 0;
    std::vector<int> p;
    std::vector<int> i;
    std::vector<double> x;
};

inline CscMatrix dense_to_csc(const std::vector<double>& dense,
                              int nrow,
                              int ncol,
                              double tol = 0.0) {
    CscMatrix out;
    out.nrow = nrow;
    out.ncol = ncol;
    out.p.resize(static_cast<size_t>(ncol + 1), 0);

    for (int col = 0; col < ncol; ++col) {
        out.p[static_cast<size_t>(col)] = static_cast<int>(out.x.size());
        for (int row = 0; row < nrow; ++row) {
            const double v = dense[static_cast<size_t>(row * ncol + col)];
            if (std::abs(v) > tol) {
                out.i.push_back(row);
                out.x.push_back(v);
            }
        }
    }
    out.p[static_cast<size_t>(ncol)] = static_cast<int>(out.x.size());
    return out;
}

class JitPrototype {
public:
    JitPrototype() = default;

    explicit JitPrototype(ObjectiveFunction fn) : fn_(std::move(fn)) {}

    void record(ObjectiveFunction fn) {
        fn_ = std::move(fn);
        recorded_ = true;
        compiled_ = false;
    }

    bool compile() {
        compiled_ = recorded_;
        return compiled_;
    }

    double evaluate(const std::vector<double>& x) const {
        if (!fn_) {
            throw std::invalid_argument("JitPrototype::evaluate called before record");
        }
        return fn_(x);
    }

    bool is_recorded() const { return recorded_; }
    bool is_compiled() const { return compiled_; }

private:
    ObjectiveFunction fn_;
    bool recorded_ = false;
    bool compiled_ = false;
};

using CheckpointCallback = std::function<bool(int, const std::vector<double>&, double)>;

inline int replay_with_checkpoint(const ObjectiveFunction& fn,
                                  std::vector<double> x,
                                  int n_steps,
                                  const CheckpointCallback& checkpoint,
                                  double shrink = 0.99) {
    int executed = 0;
    for (int step = 0; step < n_steps; ++step) {
        const double f = fn(x);
        ++executed;
        if (checkpoint && !checkpoint(step, x, f)) {
            break;
        }
        for (double& xi : x) {
            xi *= shrink;
        }
    }
    return executed;
}

} // namespace phase4
} // namespace xopt

#endif // XOPT_PHASE4_HPP
