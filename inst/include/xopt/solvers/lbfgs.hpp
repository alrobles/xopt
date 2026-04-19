// lbfgs.hpp - L-BFGS (Limited-memory BFGS) solver
//
// Implementation of the L-BFGS algorithm for large-scale unconstrained
// optimization. Uses two-loop recursion to efficiently approximate the
// inverse Hessian using a limited history of gradient differences.
//
// Reference: Nocedal & Wright, "Numerical Optimization" (2006), Algorithm 7.4

#ifndef XOPT_SOLVERS_LBFGS_HPP
#define XOPT_SOLVERS_LBFGS_HPP

#include "../problem.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace xopt {
namespace solvers {

// L-BFGS control parameters
struct LBFGSControl {
    int m = 10;                     // History size (number of correction pairs)
    double gtol = 1e-6;             // Gradient tolerance for convergence
    double ftol = 1e-8;             // Function tolerance for convergence
    double xtol = 1e-8;             // Parameter tolerance for convergence
    int max_iter = 1000;            // Maximum iterations
    int max_linesearch = 20;        // Maximum line search iterations
    double c1 = 1e-4;               // Armijo condition parameter
    double c2 = 0.9;                // Curvature condition parameter
    double min_step = 1e-20;        // Minimum step size
    double max_step = 1e20;         // Maximum step size
    bool trace = false;             // Print iteration trace
};

// L-BFGS result
template <typename Scalar = double>
struct LBFGSResult {
    std::vector<Scalar> par;        // Optimal parameters
    Scalar value;                   // Objective value at optimum
    std::vector<Scalar> gradient;   // Final gradient
    int iterations;                 // Number of iterations
    int function_evals;             // Number of function evaluations
    int gradient_evals;             // Number of gradient evaluations
    int convergence;                // Convergence code (0 = success)
    std::string message;            // Convergence message
};

// L-BFGS solver implementation
template <typename Problem, typename Scalar = double>
class LBFGS {
private:
    Problem& problem_;
    LBFGSControl control_;

    // History storage for L-BFGS two-loop recursion
    std::vector<std::vector<Scalar>> s_history_;  // Parameter differences
    std::vector<std::vector<Scalar>> y_history_;  // Gradient differences
    std::vector<Scalar> rho_history_;             // 1 / (y^T s)

    int n_par_;
    int history_size_;

    // Statistics
    int n_feval_;
    int n_geval_;

public:
    LBFGS(Problem& prob, const LBFGSControl& ctrl = LBFGSControl())
        : problem_(prob), control_(ctrl), n_par_(prob.n_par),
          history_size_(0), n_feval_(0), n_geval_(0) {

        if (control_.m < 1) {
            throw std::invalid_argument("History size m must be >= 1");
        }
    }

    // Solve the optimization problem
    LBFGSResult<Scalar> solve(const std::vector<Scalar>& x0) {
        if (static_cast<int>(x0.size()) != n_par_) {
            throw std::invalid_argument("Initial point dimension mismatch");
        }

        // Initialize
        std::vector<Scalar> x = x0;
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

        // Main optimization loop
        int iter = 0;
        int convergence = 1;  // Not converged yet
        std::string message = "Maximum iterations reached";

        for (iter = 0; iter < control_.max_iter; ++iter) {
            // Check gradient convergence
            Scalar gnorm = compute_norm(g);

            if (control_.trace) {
                print_iteration(iter, f, gnorm);
            }

            if (gnorm < control_.gtol) {
                convergence = 0;
                message = "Gradient tolerance satisfied";
                break;
            }

            // Compute search direction using L-BFGS two-loop recursion
            compute_search_direction(g, search_dir);

            // Line search along search_dir
            Scalar step = 1.0;
            if (iter == 0) {
                // First iteration: use gradient norm to scale initial step
                step = 1.0 / std::max(gnorm, Scalar(1.0));
            }

            x_prev = x;
            g_prev = g;
            Scalar f_prev = f;

            bool ls_success = line_search(x, g, search_dir, f, step);

            if (!ls_success) {
                convergence = 2;
                message = "Line search failed";
                break;
            }

            // Check function convergence
            Scalar fdiff = std::abs(f - f_prev);
            if (fdiff < control_.ftol * (std::abs(f_prev) + control_.ftol)) {
                convergence = 0;
                message = "Function tolerance satisfied";
                break;
            }

            // Check parameter convergence
            Scalar xdiff = 0.0;
            for (int i = 0; i < n_par_; ++i) {
                xdiff = std::max(xdiff, std::abs(x[i] - x_prev[i]));
            }
            if (xdiff < control_.xtol) {
                convergence = 0;
                message = "Parameter tolerance satisfied";
                break;
            }

            // Update history for L-BFGS
            update_history(x, x_prev, g, g_prev);
        }

        // Prepare result
        LBFGSResult<Scalar> result;
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
    // Evaluate objective function
    Scalar evaluate_function(const Scalar* x) {
        ++n_feval_;
        return problem_.value(x);
    }

    // Evaluate gradient
    void evaluate_gradient(const Scalar* x, Scalar* g) {
        ++n_geval_;
        problem_.gradient(x, g);
    }

    // Compute L2 norm of vector
    Scalar compute_norm(const std::vector<Scalar>& v) const {
        Scalar sum = 0.0;
        for (const auto& val : v) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    // Compute dot product
    Scalar dot_product(const std::vector<Scalar>& a, const std::vector<Scalar>& b) const {
        Scalar sum = 0.0;
        for (int i = 0; i < n_par_; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // L-BFGS two-loop recursion to compute search direction
    void compute_search_direction(const std::vector<Scalar>& g,
                                   std::vector<Scalar>& dir) {
        // Start with negative gradient
        for (int i = 0; i < n_par_; ++i) {
            dir[i] = -g[i];
        }

        if (history_size_ == 0) {
            return;  // Steepest descent for first iteration
        }

        // First loop: from newest to oldest
        std::vector<Scalar> alpha(history_size_);
        for (int i = history_size_ - 1; i >= 0; --i) {
            alpha[i] = rho_history_[i] * dot_product(s_history_[i], dir);
            for (int j = 0; j < n_par_; ++j) {
                dir[j] -= alpha[i] * y_history_[i][j];
            }
        }

        // Apply initial Hessian approximation: H0 = (s^T y / y^T y) * I
        if (history_size_ > 0) {
            int last = history_size_ - 1;
            Scalar gamma = dot_product(s_history_[last], y_history_[last]) /
                          dot_product(y_history_[last], y_history_[last]);
            for (int i = 0; i < n_par_; ++i) {
                dir[i] *= gamma;
            }
        }

        // Second loop: from oldest to newest
        for (int i = 0; i < history_size_; ++i) {
            Scalar beta = rho_history_[i] * dot_product(y_history_[i], dir);
            for (int j = 0; j < n_par_; ++j) {
                dir[j] += s_history_[i][j] * (alpha[i] - beta);
            }
        }
    }

    // Update history with new s and y vectors
    void update_history(const std::vector<Scalar>& x,
                       const std::vector<Scalar>& x_prev,
                       const std::vector<Scalar>& g,
                       const std::vector<Scalar>& g_prev) {
        // Compute s = x - x_prev and y = g - g_prev
        std::vector<Scalar> s(n_par_);
        std::vector<Scalar> y(n_par_);

        for (int i = 0; i < n_par_; ++i) {
            s[i] = x[i] - x_prev[i];
            y[i] = g[i] - g_prev[i];
        }

        // Compute rho = 1 / (y^T s)
        Scalar ys = dot_product(y, s);

        // Skip update if curvature condition not satisfied
        if (ys <= Scalar(1e-10)) {
            return;
        }

        Scalar rho = Scalar(1.0) / ys;

        // Add to history (or replace oldest if full)
        if (history_size_ < control_.m) {
            s_history_.push_back(s);
            y_history_.push_back(y);
            rho_history_.push_back(rho);
            ++history_size_;
        } else {
            // Shift history and add newest
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

    // Backtracking line search with strong Wolfe conditions
    bool line_search(std::vector<Scalar>& x,
                    std::vector<Scalar>& g,
                    const std::vector<Scalar>& dir,
                    Scalar& f,
                    Scalar& step) {

        // Initial function value and directional derivative
        Scalar f0 = f;
        Scalar dg0 = dot_product(g, dir);

        // Search direction must be descent direction
        if (dg0 >= 0) {
            return false;
        }

        std::vector<Scalar> x_new(n_par_);
        std::vector<Scalar> g_new(n_par_);

        // Try steps with backtracking
        for (int ls_iter = 0; ls_iter < control_.max_linesearch; ++ls_iter) {
            // Compute new point
            for (int i = 0; i < n_par_; ++i) {
                x_new[i] = x[i] + step * dir[i];
            }

            // Evaluate function and gradient at new point
            Scalar f_new = evaluate_function(x_new.data());
            evaluate_gradient(x_new.data(), g_new.data());

            // Check Armijo condition (sufficient decrease)
            if (f_new > f0 + control_.c1 * step * dg0) {
                step *= 0.5;  // Backtrack
                continue;
            }

            // Check curvature condition
            Scalar dg_new = dot_product(g_new, dir);
            if (dg_new < control_.c2 * dg0) {
                step *= 2.1;  // Try larger step
                if (step > control_.max_step) {
                    step = control_.max_step;
                }
                continue;
            }

            // Both conditions satisfied
            x = x_new;
            g = g_new;
            f = f_new;
            return true;
        }

        // Line search failed - use current best
        return false;
    }

    void print_header() const {
        std::printf("\n%5s %15s %15s\n", "Iter", "f(x)", "||g||");
        std::printf("----------------------------------------------\n");
    }

    void print_iteration(int iter, Scalar f, Scalar gnorm) const {
        std::printf("%5d %15.8e %15.8e\n", iter, f, gnorm);
    }
};

// Convenience function to minimize with L-BFGS
template <typename Problem, typename Scalar = double>
LBFGSResult<Scalar> minimize_lbfgs(
    Problem& problem,
    const std::vector<Scalar>& x0,
    const LBFGSControl& control = LBFGSControl()) {

    LBFGS<Problem, Scalar> solver(problem, control);
    return solver.solve(x0);
}

} // namespace solvers
} // namespace xopt

#endif // XOPT_SOLVERS_LBFGS_HPP
