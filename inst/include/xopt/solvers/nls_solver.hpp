// nls_solver.hpp - Nonlinear Least Squares with Levenberg-Marquardt
//
// This header provides an implementation of the Levenberg-Marquardt algorithm
// for nonlinear least squares problems: min_θ ½‖r(θ)‖²
//
// The solver computes Jacobians via automatic differentiation (XAD) or
// finite differences, and returns covariance matrices for statistical inference.

#ifndef XOPT_NLS_SOLVER_HPP
#define XOPT_NLS_SOLVER_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <string>

namespace xopt {
namespace solvers {

// Control parameters for Levenberg-Marquardt
struct LMControl {
    double ftol = 1e-8;       // Function tolerance (relative)
    double xtol = 1e-8;       // Parameter tolerance (relative)
    double gtol = 1e-8;       // Gradient tolerance (absolute)
    int maxiter = 100;        // Maximum iterations
    double lambda_init = 1e-3; // Initial damping parameter
    double lambda_up = 10.0;   // Damping increase factor
    double lambda_down = 0.1;  // Damping decrease factor
    bool trace = false;        // Print trace output
};

// Result from NLS optimization
struct NLSResult {
    std::vector<double> par;         // Optimal parameters
    double value;                    // Final ½‖r‖² (sum of squared residuals)
    std::vector<double> residuals;   // Final residual vector r(θ)
    std::vector<double> gradient;    // Final gradient ∇f = Jᵀr
    std::vector<double> jacobian;    // Final Jacobian (m×n, row-major)
    std::vector<double> vcov;        // Covariance matrix (JᵀJ)⁻¹σ² (n×n)
    int iterations;                  // Iterations used
    int convergence;                 // Convergence code
    std::string message;             // Convergence message
    int nresiduals;                  // Number of residuals
    int nparams;                     // Number of parameters
};

// Residual function interface: r(θ) → vector
using ResidualFunction = std::function<void(const std::vector<double>&,
                                            std::vector<double>&)>;

// Jacobian function interface: J(θ) → m×n matrix (row-major)
using JacobianFunction = std::function<void(const std::vector<double>&,
                                            std::vector<double>&)>;

// Compute Jacobian via central finite differences
inline void finite_diff_jacobian(const ResidualFunction& residual_fn,
                                  const std::vector<double>& x,
                                  std::vector<double>& J,
                                  int m, int n,
                                  double eps = 1e-8) {
    J.resize(m * n);
    std::vector<double> r1(m), r2(m);
    std::vector<double> x_plus(x), x_minus(x);

    for (int j = 0; j < n; ++j) {
        double h = eps * std::max(1.0, std::abs(x[j]));
        x_plus[j] = x[j] + h;
        x_minus[j] = x[j] - h;

        residual_fn(x_plus, r1);
        residual_fn(x_minus, r2);

        for (int i = 0; i < m; ++i) {
            J[i * n + j] = (r1[i] - r2[i]) / (2.0 * h);
        }

        x_plus[j] = x[j];
        x_minus[j] = x[j];
    }
}

// Solve linear system Ax = b using Cholesky decomposition
// A is n×n symmetric positive definite (stored as full matrix, row-major)
inline bool cholesky_solve(const std::vector<double>& A,
                           const std::vector<double>& b,
                           std::vector<double>& x,
                           int n) {
    // Compute Cholesky decomposition A = LLᵀ
    std::vector<double> L(n * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i * n + j];
            for (int k = 0; k < j; ++k) {
                sum -= L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                if (sum <= 0.0) return false;  // Not positive definite
                L[i * n + j] = std::sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }

    // Forward substitution: solve Ly = b
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int j = 0; j < i; ++j) {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / L[i * n + i];
    }

    // Back substitution: solve Lᵀx = y
    x.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= L[j * n + i] * x[j];
        }
        x[i] = sum / L[i * n + i];
    }

    return true;
}

// Matrix inversion via Cholesky (for covariance computation)
inline bool cholesky_inverse(const std::vector<double>& A,
                             std::vector<double>& Ainv,
                             int n) {
    Ainv.resize(n * n);
    std::vector<double> eye(n, 0.0);

    for (int i = 0; i < n; ++i) {
        eye[i] = 1.0;
        std::vector<double> col;
        if (!cholesky_solve(A, eye, col, n)) {
            return false;
        }
        for (int j = 0; j < n; ++j) {
            Ainv[j * n + i] = col[j];
        }
        eye[i] = 0.0;
    }

    return true;
}

// Levenberg-Marquardt algorithm
inline NLSResult levenberg_marquardt(const std::vector<double>& x0,
                                     const ResidualFunction& residual_fn,
                                     JacobianFunction jacobian_fn = nullptr,
                                     const LMControl& control = {}) {
    int n = x0.size();
    std::vector<double> x = x0;
    std::vector<double> r, r_new;
    std::vector<double> J;

    // Compute initial residuals
    residual_fn(x, r);
    int m = r.size();

    if (m < n) {
        throw std::invalid_argument("LM: number of residuals must be >= number of parameters");
    }

    // Initial objective value
    double f = 0.0;
    for (double ri : r) f += ri * ri;
    f *= 0.5;

    // Use finite differences if no Jacobian provided
    bool use_fd = (jacobian_fn == nullptr);

    double lambda = control.lambda_init;
    int iter = 0;
    int convergence = 0;
    std::string message = "Maximum iterations reached";

    for (; iter < control.maxiter; ++iter) {
        // Compute Jacobian
        if (use_fd) {
            finite_diff_jacobian(residual_fn, x, J, m, n);
        } else {
            jacobian_fn(x, J);
        }

        // Compute gradient: g = Jᵀr
        std::vector<double> g(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                g[i] += J[j * n + i] * r[j];
            }
        }

        // Check gradient convergence
        double gnorm = 0.0;
        for (double gi : g) gnorm = std::max(gnorm, std::abs(gi));
        if (gnorm < control.gtol) {
            convergence = 1;
            message = "Gradient below tolerance";
            break;
        }

        // Compute JᵀJ
        std::vector<double> JtJ(n * n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < m; ++k) {
                    JtJ[i * n + j] += J[k * n + i] * J[k * n + j];
                }
            }
        }

        // LM iteration: try to find step that reduces f
        bool step_accepted = false;
        std::vector<double> x_new(n);
        double f_new;

        for (int lm_iter = 0; lm_iter < 10; ++lm_iter) {
            // Add damping: (JᵀJ + λI)h = -Jᵀr
            std::vector<double> A = JtJ;
            for (int i = 0; i < n; ++i) {
                A[i * n + i] += lambda;
            }

            // Solve for step h
            std::vector<double> h;
            std::vector<double> neg_g(n);
            for (int i = 0; i < n; ++i) neg_g[i] = -g[i];

            if (!cholesky_solve(A, neg_g, h, n)) {
                // Increase damping and retry
                lambda *= control.lambda_up;
                continue;
            }

            // Try step
            for (int i = 0; i < n; ++i) {
                x_new[i] = x[i] + h[i];
            }

            residual_fn(x_new, r_new);
            f_new = 0.0;
            for (double ri : r_new) f_new += ri * ri;
            f_new *= 0.5;

            // Check if step reduces objective
            if (f_new < f) {
                step_accepted = true;
                lambda *= control.lambda_down;  // Decrease damping
                break;
            } else {
                lambda *= control.lambda_up;  // Increase damping
            }
        }

        if (!step_accepted) {
            convergence = 3;
            message = "Cannot find step that reduces objective";
            break;
        }

        // Check function convergence
        double f_rel = std::abs(f_new - f) / (std::abs(f) + 1e-10);
        if (f_rel < control.ftol) {
            convergence = 2;
            message = "Function change below tolerance";
        }

        // Check parameter convergence
        double x_rel = 0.0;
        for (int i = 0; i < n; ++i) {
            double dx = std::abs(x_new[i] - x[i]) / (std::abs(x[i]) + 1e-10);
            x_rel = std::max(x_rel, dx);
        }
        if (x_rel < control.xtol) {
            convergence = 4;
            message = "Parameter change below tolerance";
        }

        // Update
        x = x_new;
        r = r_new;
        f = f_new;

        if (convergence != 0) break;
    }

    // Compute final Jacobian and covariance
    if (use_fd) {
        finite_diff_jacobian(residual_fn, x, J, m, n);
    } else {
        jacobian_fn(x, J);
    }

    // Compute JᵀJ
    std::vector<double> JtJ(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < m; ++k) {
                JtJ[i * n + j] += J[k * n + i] * J[k * n + j];
            }
        }
    }

    // Compute gradient
    std::vector<double> g(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            g[i] += J[j * n + i] * r[j];
        }
    }

    // Compute covariance: (JᵀJ)⁻¹ σ²
    // where σ² = ‖r‖² / (m - n) is the residual variance
    std::vector<double> vcov;
    double sigma2 = (m > n) ? (2.0 * f) / (m - n) : 0.0;

    if (cholesky_inverse(JtJ, vcov, n)) {
        for (auto& v : vcov) v *= sigma2;
    } else {
        vcov.resize(n * n, std::numeric_limits<double>::quiet_NaN());
    }

    // Build result
    NLSResult result;
    result.par = x;
    result.value = f;
    result.residuals = r;
    result.gradient = g;
    result.jacobian = J;
    result.vcov = vcov;
    result.iterations = iter;
    result.convergence = convergence;
    result.message = message;
    result.nresiduals = m;
    result.nparams = n;

    return result;
}

} // namespace solvers
} // namespace xopt

#endif // XOPT_NLS_SOLVER_HPP
