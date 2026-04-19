// lbfgsb.hpp - L-BFGS-B (L-BFGS with box constraints) solver
//
// Implementation of the L-BFGS-B algorithm for large-scale box-constrained
// optimization. Handles lower and upper bounds on parameters using an
// active set strategy with subspace minimization.
//
// Reference: Byrd, Lu, Nocedal, "A Limited Memory Algorithm for Bound
// Constrained Optimization" (1995), SIAM J. Sci. Comput.

#ifndef XOPT_SOLVERS_LBFGSB_HPP
#define XOPT_SOLVERS_LBFGSB_HPP

#include "../problem.hpp"
#include "lbfgs.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace xopt {
namespace solvers {

// L-BFGS-B control parameters
struct LBFGSBControl {
    int m = 10;                     // History size
    double gtol = 1e-5;             // Projected gradient tolerance
    double ftol = 1e-8;             // Function tolerance
    double xtol = 1e-8;             // Parameter tolerance
    int max_iter = 1000;            // Maximum iterations
    int max_linesearch = 20;        // Maximum line search iterations
    double c1 = 1e-4;               // Armijo parameter
    double c2 = 0.9;                // Curvature parameter
    double factr = 1e7;             // Function tolerance factor
    double pgtol = 1e-5;            // Projected gradient tolerance
    bool trace = false;             // Print trace
};

// L-BFGS-B result
template <typename Scalar = double>
struct LBFGSBResult {
    std::vector<Scalar> par;        // Optimal parameters
    Scalar value;                   // Objective value
    std::vector<Scalar> gradient;   // Final gradient
    int iterations;                 // Number of iterations
    int function_evals;             // Function evaluations
    int gradient_evals;             // Gradient evaluations
    int convergence;                // Convergence code (0 = success)
    std::string message;            // Convergence message
};

// Bound type for each parameter
enum class BoundType {
    None,       // Unbounded
    Lower,      // Lower bound only
    Upper,      // Upper bound only
    Both        // Both bounds
};

// L-BFGS-B solver implementation
template <typename Problem, typename Scalar = double>
class LBFGSB {
private:
    Problem& problem_;
    LBFGSBControl control_;

    // Bounds
    std::vector<Scalar> lower_;
    std::vector<Scalar> upper_;
    std::vector<BoundType> bound_type_;

    // History for L-BFGS
    std::vector<std::vector<Scalar>> s_history_;
    std::vector<std::vector<Scalar>> y_history_;
    std::vector<Scalar> rho_history_;

    int n_par_;
    int history_size_;

    // Statistics
    int n_feval_;
    int n_geval_;

    static constexpr Scalar inf_ = std::numeric_limits<Scalar>::infinity();

public:
    LBFGSB(Problem& prob, const LBFGSBControl& ctrl = LBFGSBControl())
        : problem_(prob), control_(ctrl), n_par_(prob.n_par),
          history_size_(0), n_feval_(0), n_geval_(0) {

        if (control_.m < 1) {
            throw std::invalid_argument("History size m must be >= 1");
        }

        // Initialize bounds from problem
        if (problem_.has_bounds()) {
            lower_ = problem_.lower;
            upper_ = problem_.upper;
        } else {
            lower_.resize(n_par_, -inf_);
            upper_.resize(n_par_, inf_);
        }

        // Classify bound types
        bound_type_.resize(n_par_);
        for (int i = 0; i < n_par_; ++i) {
            bool has_lower = std::isfinite(lower_[i]);
            bool has_upper = std::isfinite(upper_[i]);

            if (has_lower && has_upper) {
                bound_type_[i] = BoundType::Both;
            } else if (has_lower) {
                bound_type_[i] = BoundType::Lower;
            } else if (has_upper) {
                bound_type_[i] = BoundType::Upper;
            } else {
                bound_type_[i] = BoundType::None;
            }
        }
    }

    // Solve the optimization problem
    LBFGSBResult<Scalar> solve(const std::vector<Scalar>& x0) {
        if (static_cast<int>(x0.size()) != n_par_) {
            throw std::invalid_argument("Initial point dimension mismatch");
        }

        // Project initial point onto feasible region
        std::vector<Scalar> x = x0;
        project_bounds(x);

        std::vector<Scalar> g(n_par_);
        std::vector<Scalar> g_prev(n_par_);
        std::vector<Scalar> x_prev(n_par_);
        std::vector<Scalar> search_dir(n_par_);

        // Reset statistics
        n_feval_ = 0;
        n_geval_ = 0;
        history_size_ = 0;
        s_history_.clear();
        y_history_.clear();
        rho_history_.clear();

        // Initial function and gradient
        Scalar f = evaluate_function(x.data());
        evaluate_gradient(x.data(), g.data());

        if (control_.trace) {
            print_header();
        }

        int iter = 0;
        int convergence = 1;
        std::string message = "Maximum iterations reached";

        for (iter = 0; iter < control_.max_iter; ++iter) {
            // Compute projected gradient
            std::vector<Scalar> pg = compute_projected_gradient(x, g);
            Scalar pgnorm = compute_norm(pg);

            if (control_.trace) {
                print_iteration(iter, f, pgnorm);
            }

            // Check convergence
            if (pgnorm < control_.pgtol) {
                convergence = 0;
                message = "Projected gradient tolerance satisfied";
                break;
            }

            // Compute search direction using L-BFGS in subspace
            compute_search_direction(x, g, search_dir);

            // Project search direction
            for (int i = 0; i < n_par_; ++i) {
                // If at bound and direction points outside, zero it out
                if (at_lower_bound(x[i], i) && search_dir[i] < 0) {
                    search_dir[i] = 0;
                }
                if (at_upper_bound(x[i], i) && search_dir[i] > 0) {
                    search_dir[i] = 0;
                }
            }

            // Line search with projection
            x_prev = x;
            g_prev = g;
            Scalar f_prev = f;

            Scalar step = 1.0;
            if (iter == 0) {
                step = std::min(Scalar(1.0), Scalar(1.0) / pgnorm);
            }

            bool ls_success = line_search_projected(x, g, search_dir, f, step);

            if (!ls_success) {
                convergence = 2;
                message = "Line search failed";
                break;
            }

            // Check function convergence
            Scalar fdiff = std::abs(f - f_prev);
            if (fdiff < control_.ftol) {
                convergence = 0;
                message = "Function tolerance satisfied";
                break;
            }

            // Update history
            update_history(x, x_prev, g, g_prev);
        }

        LBFGSBResult<Scalar> result;
        result.par = x;
        result.value = f;
        result.gradient = g;
        result.iterations = iter;
        result.function_evals = n_feval_;
        result.gradient_evals = n_geval_;
        result.convergence = convergence;
        result.message = message;

        return result;
    }

private:
    Scalar evaluate_function(const Scalar* x) {
        ++n_feval_;
        return problem_.value(x);
    }

    void evaluate_gradient(const Scalar* x, Scalar* g) {
        ++n_geval_;
        problem_.gradient(x, g);
    }

    Scalar compute_norm(const std::vector<Scalar>& v) const {
        Scalar sum = 0.0;
        for (const auto& val : v) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    Scalar dot_product(const std::vector<Scalar>& a,
                       const std::vector<Scalar>& b) const {
        Scalar sum = 0.0;
        for (int i = 0; i < n_par_; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // Project point onto feasible region
    void project_bounds(std::vector<Scalar>& x) const {
        for (int i = 0; i < n_par_; ++i) {
            x[i] = std::max(lower_[i], std::min(upper_[i], x[i]));
        }
    }

    // Check if parameter is at lower bound
    bool at_lower_bound(Scalar x, int i) const {
        return std::isfinite(lower_[i]) && x <= lower_[i] + Scalar(1e-10);
    }

    // Check if parameter is at upper bound
    bool at_upper_bound(Scalar x, int i) const {
        return std::isfinite(upper_[i]) && x >= upper_[i] - Scalar(1e-10);
    }

    // Compute projected gradient
    std::vector<Scalar> compute_projected_gradient(
        const std::vector<Scalar>& x,
        const std::vector<Scalar>& g) const {

        std::vector<Scalar> pg(n_par_);
        for (int i = 0; i < n_par_; ++i) {
            if (at_lower_bound(x[i], i) && g[i] > 0) {
                pg[i] = 0;  // At lower bound with positive gradient
            } else if (at_upper_bound(x[i], i) && g[i] < 0) {
                pg[i] = 0;  // At upper bound with negative gradient
            } else {
                pg[i] = g[i];
            }
        }
        return pg;
    }

    // L-BFGS two-loop recursion
    void compute_search_direction(const std::vector<Scalar>& x,
                                   const std::vector<Scalar>& g,
                                   std::vector<Scalar>& dir) {
        // Start with negative gradient
        for (int i = 0; i < n_par_; ++i) {
            dir[i] = -g[i];
        }

        if (history_size_ == 0) {
            return;
        }

        // First loop
        std::vector<Scalar> alpha(history_size_);
        for (int i = history_size_ - 1; i >= 0; --i) {
            alpha[i] = rho_history_[i] * dot_product(s_history_[i], dir);
            for (int j = 0; j < n_par_; ++j) {
                dir[j] -= alpha[i] * y_history_[i][j];
            }
        }

        // Apply initial Hessian approximation
        if (history_size_ > 0) {
            int last = history_size_ - 1;
            Scalar gamma = dot_product(s_history_[last], y_history_[last]) /
                          dot_product(y_history_[last], y_history_[last]);
            for (int i = 0; i < n_par_; ++i) {
                dir[i] *= gamma;
            }
        }

        // Second loop
        for (int i = 0; i < history_size_; ++i) {
            Scalar beta = rho_history_[i] * dot_product(y_history_[i], dir);
            for (int j = 0; j < n_par_; ++j) {
                dir[j] += s_history_[i][j] * (alpha[i] - beta);
            }
        }
    }

    void update_history(const std::vector<Scalar>& x,
                       const std::vector<Scalar>& x_prev,
                       const std::vector<Scalar>& g,
                       const std::vector<Scalar>& g_prev) {
        std::vector<Scalar> s(n_par_);
        std::vector<Scalar> y(n_par_);

        for (int i = 0; i < n_par_; ++i) {
            s[i] = x[i] - x_prev[i];
            y[i] = g[i] - g_prev[i];
        }

        Scalar ys = dot_product(y, s);
        if (ys <= Scalar(1e-10)) {
            return;
        }

        Scalar rho = Scalar(1.0) / ys;

        if (history_size_ < control_.m) {
            s_history_.push_back(s);
            y_history_.push_back(y);
            rho_history_.push_back(rho);
            ++history_size_;
        } else {
            for (int i = 0; i < control_.m - 1; ++i) {
                s_history_[i] = s_history_[i + 1];
                y_history_[i] = y_history_[i + 1];
                rho_history_[i] = rho_history_[i + 1];
            }
            s_history_[control_.m - 1] = s;
            y_history_[control_.m - 1] = y;
            rho_history_[control_.m - 1] = rho;
        }
    }

    bool line_search_projected(std::vector<Scalar>& x,
                               std::vector<Scalar>& g,
                               const std::vector<Scalar>& dir,
                               Scalar& f,
                               Scalar& step) {
        Scalar f0 = f;
        Scalar dg0 = dot_product(g, dir);

        if (dg0 >= 0) {
            return false;
        }

        std::vector<Scalar> x_new(n_par_);
        std::vector<Scalar> g_new(n_par_);

        for (int ls_iter = 0; ls_iter < control_.max_linesearch; ++ls_iter) {
            // Compute new point and project
            for (int i = 0; i < n_par_; ++i) {
                x_new[i] = x[i] + step * dir[i];
            }
            project_bounds(x_new);

            Scalar f_new = evaluate_function(x_new.data());
            evaluate_gradient(x_new.data(), g_new.data());

            // Armijo condition
            if (f_new <= f0 + control_.c1 * step * dg0) {
                x = x_new;
                g = g_new;
                f = f_new;
                return true;
            }

            step *= 0.5;
        }

        return false;
    }

    void print_header() const {
        std::printf("\n%5s %15s %15s\n", "Iter", "f(x)", "||pg||");
        std::printf("----------------------------------------------\n");
    }

    void print_iteration(int iter, Scalar f, Scalar pgnorm) const {
        std::printf("%5d %15.8e %15.8e\n", iter, f, pgnorm);
    }
};

// Convenience function
template <typename Problem, typename Scalar = double>
LBFGSBResult<Scalar> minimize_lbfgsb(
    Problem& problem,
    const std::vector<Scalar>& x0,
    const LBFGSBControl& control = LBFGSBControl()) {

    LBFGSB<Problem, Scalar> solver(problem, control);
    return solver.solve(x0);
}

} // namespace solvers
} // namespace xopt

#endif // XOPT_SOLVERS_LBFGSB_HPP
