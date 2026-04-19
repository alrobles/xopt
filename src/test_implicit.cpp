// test_implicit.cpp — Rcpp-exported unit tests for xopt::implicit_::implicit_spd.
//
// Exposes:
//   xopt_implicit_spd_grad_ridge(X, y, lambda)
//     — builds the ridge-regression fixed point
//           g(β, λ) = (X^T X + λI) β − X^T y = 0
//       solves β*(λ) on plain double, installs the IFT callback, records the
//       loss L(β) = ‖β‖² on the tape, and returns dL/dλ computed via the
//       callback's cotangent. Used to verify against the closed-form analytic
//       derivative and a central finite-difference check in R.
//
//   xopt_implicit_spd_grad_generic(A, B, x, dLdx)
//     — direct-form check: given A (SPD n×n), B (n×p), x* (length n), and an
//       arbitrary x̄ = ∂L/∂x* (length n), return θ̄ = −B^T A^{-1} x̄.

#include <Rcpp.h>
#include <XAD/XAD.hpp>

#include <xopt/implicit/ift.hpp>
#include <xopt/linalg/solve.hpp>

#include <cstddef>
#include <vector>

using AD = xad::AReal<double>;
using Tape = xad::Tape<double>;

namespace {

std::vector<double> mat_to_vec(const Rcpp::NumericMatrix& M) {
    const int n = M.nrow();
    const int m = M.ncol();
    std::vector<double> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            v[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n] = M(i, j);
        }
    }
    return v;
}

std::vector<double> numvec_to_vec(const Rcpp::NumericVector& v) {
    return std::vector<double>(v.begin(), v.end());
}

// Build A = X^T X + λ I  (p × p, column-major) and b = X^T y (length p).
void build_ridge_normal(const std::vector<double>& X, int n, int p,
                        const std::vector<double>& y, double lambda,
                        std::vector<double>& A_out,
                        std::vector<double>& b_out) {
    A_out.assign(static_cast<std::size_t>(p) * static_cast<std::size_t>(p), 0.0);
    b_out.assign(static_cast<std::size_t>(p), 0.0);
    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < p; ++i) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k) {
                acc += X[static_cast<std::size_t>(k)
                         + static_cast<std::size_t>(i) * n]
                     * X[static_cast<std::size_t>(k)
                         + static_cast<std::size_t>(j) * n];
            }
            A_out[static_cast<std::size_t>(i)
                 + static_cast<std::size_t>(j) * p] = acc;
        }
        A_out[static_cast<std::size_t>(j)
             + static_cast<std::size_t>(j) * p] += lambda;
    }
    for (int j = 0; j < p; ++j) {
        double acc = 0.0;
        for (int k = 0; k < n; ++k) {
            acc += X[static_cast<std::size_t>(k)
                     + static_cast<std::size_t>(j) * n] * y[static_cast<std::size_t>(k)];
        }
        b_out[static_cast<std::size_t>(j)] = acc;
    }
}

}  // namespace

//' Adjoint dL/dλ for L(β*(λ)) = ‖β*(λ)‖² via IFT
//' @param X numeric matrix n × p
//' @param y numeric vector length n
//' @param lambda scalar ridge penalty (λ > 0)
//' @return scalar dL/dλ computed through xopt::implicit_::implicit_spd
// [[Rcpp::export]]
double xopt_implicit_spd_grad_ridge(Rcpp::NumericMatrix X, Rcpp::NumericVector y,
                                    double lambda) {
    const int n = X.nrow();
    const int p = X.ncol();
    if (y.size() != n) Rcpp::stop("length(y) must equal nrow(X)");
    if (!(lambda > 0.0)) Rcpp::stop("lambda must be positive");

    const std::vector<double> Xv = mat_to_vec(X);
    const std::vector<double> yv = numvec_to_vec(y);

    std::vector<double> A, b;
    build_ridge_normal(Xv, n, p, yv, lambda, A, b);
    const std::vector<double> beta_star = xopt::linalg::solve_spd(A, p, b);

    // B = ∂g/∂λ = β*  (n_rows = p, n_cols = 1 in this scalar-θ case).
    const std::vector<double> B = beta_star;

    Tape tape;
    std::vector<AD> lambda_ad(1);
    lambda_ad[0] = lambda;
    tape.registerInput(lambda_ad[0]);
    tape.newRecording();

    auto beta_ad = xopt::implicit_::implicit_spd(beta_star, A, p, B, /*p_theta*/ 1,
                                                 lambda_ad);

    AD loss = AD(0.0);
    for (std::size_t i = 0; i < beta_ad.size(); ++i) loss = loss + beta_ad[i] * beta_ad[i];
    tape.registerOutput(loss);
    xad::derivative(loss) = 1.0;
    tape.computeAdjoints();
    return xad::derivative(lambda_ad[0]);
}

//' Direct IFT cotangent: θ̄ = −B^T A^{-1} x̄
//' @param A numeric SPD matrix n × n
//' @param B numeric matrix n × p
//' @param x_star numeric vector length n (used as the forward x*, but the
//'   cotangent formula depends only on A, B, x_bar)
//' @param x_bar numeric vector length n (upstream gradient ∂L/∂x*)
//' @return numeric vector length p: θ̄
// [[Rcpp::export]]
Rcpp::NumericVector xopt_implicit_spd_grad_generic(Rcpp::NumericMatrix A,
                                                   Rcpp::NumericMatrix B,
                                                   Rcpp::NumericVector x_star,
                                                   Rcpp::NumericVector x_bar) {
    const int n = A.nrow();
    const int p = B.ncol();
    if (A.ncol() != n) Rcpp::stop("A must be square");
    if (B.nrow() != n) Rcpp::stop("nrow(B) must equal nrow(A)");
    if (x_star.size() != n) Rcpp::stop("length(x_star) must equal nrow(A)");
    if (x_bar.size() != n) Rcpp::stop("length(x_bar) must equal nrow(A)");

    const std::vector<double> Av = mat_to_vec(A);
    const std::vector<double> Bv = mat_to_vec(B);
    const std::vector<double> xv = numvec_to_vec(x_star);

    Tape tape;
    std::vector<AD> theta(static_cast<std::size_t>(p));
    for (int j = 0; j < p; ++j) {
        theta[j] = 0.0;
        tape.registerInput(theta[j]);
    }
    tape.newRecording();

    auto xad_out = xopt::implicit_::implicit_spd(xv, Av, n, Bv, p, theta);

    AD loss = AD(0.0);
    for (int i = 0; i < n; ++i) loss = loss + xad_out[i] * x_bar[i];
    tape.registerOutput(loss);
    xad::derivative(loss) = 1.0;
    tape.computeAdjoints();

    Rcpp::NumericVector out(p);
    for (int j = 0; j < p; ++j) out[j] = xad::derivative(theta[j]);
    return out;
}
