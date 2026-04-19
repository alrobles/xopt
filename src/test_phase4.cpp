// test_phase4.cpp - Phase 4 constraints/sparsity/parallel/JIT prototype tests
//
// [[Rcpp::plugins(cpp20)]]
// [[Rcpp::depends(ucminfcpp)]]

#include <Rcpp.h>
#include <xopt/phase4.hpp>
#include <xopt/solvers/ucminf_solver.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace xopt;

namespace {

struct MinResult {
    std::vector<double> par;
    double value = 0.0;
    int convergence = 0;
};

} // namespace

// [[Rcpp::export]]
int test_phase4_constraints() {
    try {
        auto objective = [](const std::vector<double>& x) {
            return std::pow(x[0] - 1.0, 2.0) + std::pow(x[1] - 2.0, 2.0);
        };

        auto gradient = [](const std::vector<double>& x, std::vector<double>& g) {
            g.resize(2);
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = 2.0 * (x[1] - 2.0);
        };

        auto constraints = [](const std::vector<double>& x,
                              std::vector<double>& c_eq,
                              std::vector<double>& c_ineq) {
            c_eq = {x[0] + x[1] - 3.0};
            c_ineq = {-x[0]}; // x0 >= 0
        };

        phase4::ALControl control;
        control.tol = 1e-6;
        control.outer_maxiter = 30;
        control.rho_init = 10.0;
        control.inner_control.maxiter = 100;

        auto fit = phase4::augmented_lagrangian_solve({3.0, -1.0}, objective, constraints, gradient, control);

        if (fit.convergence == 0) return 1;
        if (fit.constraint_violation > 1e-4) return 1;
        if (std::abs(fit.par[0] - 1.0) > 2e-2 || std::abs(fit.par[1] - 2.0) > 2e-2) return 1;
        return 0;
    } catch (...) {
        return 1;
    }
}

// [[Rcpp::export]]
int test_phase4_multistart_parallel() {
    try {
        auto solve_one = [](const std::vector<double>& start) {
            auto fdf = [](const std::vector<double>& x, std::vector<double>& g, double& f) {
                f = std::pow(1.0 - x[0], 2.0) + 100.0 * std::pow(x[1] - x[0] * x[0], 2.0);
                g.resize(2);
                g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
                g[1] = 200.0 * (x[1] - x[0] * x[0]);
            };

            solvers::UcminfControl ctrl;
            ctrl.grtol = 1e-6;
            ctrl.maxeval = 400;
            auto r = solvers::ucminf_solve(start, fdf, ctrl);
            return MinResult{r.par, r.value, r.convergence};
        };

        std::mt19937 rng(1234);
        std::uniform_real_distribution<double> dist(-3.0, 3.0);

        std::vector<std::vector<double>> starts(128, std::vector<double>(2, 0.0));
        for (auto& s : starts) {
            s[0] = dist(rng);
            s[1] = dist(rng);
        }

        auto serial = phase4::parallel_multi_start<decltype(solve_one), MinResult>(
            starts,
            solve_one,
            [](const MinResult& r) { return r.value; },
            1
        );

        auto parallel = phase4::parallel_multi_start<decltype(solve_one), MinResult>(
            starts,
            solve_one,
            [](const MinResult& r) { return r.value; },
            4
        );

        if (serial.all_results.size() != starts.size()) return 1;
        if (parallel.all_results.size() != starts.size()) return 1;

        const double serial_best = serial.all_results[serial.best_index].value;
        const double parallel_best = parallel.all_results[parallel.best_index].value;

        if (std::abs(serial_best - parallel_best) > 1e-10) return 1;
        if (parallel_best > 1e-5) return 1;
        return 0;
    } catch (...) {
        return 1;
    }
}

// [[Rcpp::export]]
int test_phase4_sparse() {
    try {
        const int n = 40;
        std::vector<double> x(static_cast<size_t>(n), 1.0);

        auto residual_fn = [n](const std::vector<double>& xv, std::vector<double>& r) {
            r.assign(static_cast<size_t>(n), 0.0);
            for (int i = 0; i < n; ++i) {
                r[static_cast<size_t>(i)] = 2.0 * xv[static_cast<size_t>(i)];
                if (i > 0) r[static_cast<size_t>(i)] -= xv[static_cast<size_t>(i - 1)];
                if (i < n - 1) r[static_cast<size_t>(i)] -= xv[static_cast<size_t>(i + 1)];
            }
        };

        auto pattern = phase4::detect_jacobian_sparsity(residual_fn, x);
        auto colors = phase4::greedy_column_coloring(pattern);
        auto J = phase4::compressed_fd_jacobian(residual_fn, x, pattern, colors);
        auto csc = phase4::dense_to_csc(J, n, n, 1e-12);

        int max_color = 0;
        for (int c : colors) max_color = std::max(max_color, c);
        if (max_color + 1 > 3) return 1;

        // Tridiagonal matrix nnz: n diagonal + (n - 1) lower + (n - 1) upper.
        const int expected_nnz = 3 * n - 2;
        if (static_cast<int>(csc.x.size()) != expected_nnz) return 1;

        for (int i = 0; i < n; ++i) {
            if (std::abs(J[static_cast<size_t>(i * n + i)] - 2.0) > 1e-4) return 1;
            if (i > 0 && std::abs(J[static_cast<size_t>(i * n + i - 1)] + 1.0) > 1e-4) return 1;
            if (i < n - 1 && std::abs(J[static_cast<size_t>(i * n + i + 1)] + 1.0) > 1e-4) return 1;
        }

        return 0;
    } catch (...) {
        return 1;
    }
}

// [[Rcpp::export]]
int test_phase4_jit_checkpoint() {
    try {
        auto fn = [](const std::vector<double>& x) {
            return x[0] * x[0] + 2.0 * x[1] * x[1] + x[0] * x[1];
        };

        phase4::JitPrototype jit;
        jit.record(fn);
        if (!jit.is_recorded()) return 1;
        if (!jit.compile()) return 1;
        if (!jit.is_compiled()) return 1;

        std::vector<double> x = {0.4, -0.8};
        const double direct = fn(x);
        const double compiled = jit.evaluate(x);
        if (std::abs(direct - compiled) > 1e-12) return 1;

        int callbacks = 0;
        auto checkpoint = [&](int step, const std::vector<double>&, double) {
            if ((step % 5) == 0) {
                ++callbacks;
            }
            return step < 24;
        };

        const int executed = phase4::replay_with_checkpoint(fn, {2.0, -1.0}, 100, checkpoint);
        if (executed != 25) return 1;
        if (callbacks != 5) return 1;

        return 0;
    } catch (...) {
        return 1;
    }
}
