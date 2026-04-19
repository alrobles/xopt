// test_linalg.cpp — Rcpp-exported unit tests for xopt::linalg::{chol, solve,
// logdet, inv}. Two layers of tests per primitive:
//
//   (a) forward-mode correctness on plain double, compared against R base
//       (`base::chol`, `base::solve`, `base::determinant`).
//
//   (b) adjoint-mode correctness: use xad::AReal<double> to tape the linalg
//       op under a scalar loss L(A) = f(primitive(A)), compute ∇_A L via
//       XAD's adjoint sweep, and compare component-wise against a central
//       finite-difference gradient.
//
// All matrices are column-major std::vector<Scalar> (matching R convention).

#include <Rcpp.h>
#include <XAD/XAD.hpp>

#include <xopt/linalg/ad.hpp>
#include <xopt/linalg/chol.hpp>
#include <xopt/linalg/inv.hpp>
#include <xopt/linalg/logdet.hpp>
#include <xopt/linalg/solve.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

using AD = xad::AReal<double>;
using Tape = xad::Tape<double>;

namespace {

// Copy an R NumericMatrix into a column-major std::vector<double>.
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

Rcpp::NumericMatrix vec_to_mat(const std::vector<double>& v, int n, int m) {
    Rcpp::NumericMatrix M(n, m);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            M(i, j) = v[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n];
        }
    }
    return M;
}

Rcpp::NumericVector vec_to_numvec(const std::vector<double>& v) {
    return Rcpp::NumericVector(v.begin(), v.end());
}

std::vector<double> numvec_to_vec(const Rcpp::NumericVector& v) {
    return std::vector<double>(v.begin(), v.end());
}

// Copy a column-major matrix of doubles into active AD.
std::vector<AD> to_ad_mat(const std::vector<double>& v, Tape& tape) {
    std::vector<AD> a(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        a[i] = v[i];
        tape.registerInput(a[i]);
    }
    return a;
}

std::vector<AD> to_ad_vec(const std::vector<double>& v) {
    std::vector<AD> a(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) a[i] = v[i];
    return a;
}

}  // namespace

//' Forward-mode Cholesky on double
//' @param A numeric SPD matrix (n x n)
//' @return lower-triangular Cholesky factor L (n x n) with A = L L^T
// [[Rcpp::export]]
Rcpp::NumericMatrix xopt_chol_impl(Rcpp::NumericMatrix A) {
    const int n = A.nrow();
    if (A.ncol() != n) Rcpp::stop("xopt_chol: A must be square");
    auto L = xopt::linalg::chol(mat_to_vec(A), n);
    return vec_to_mat(L, n, n);
}

//' Forward-mode SPD solve on double
//' @param A numeric SPD matrix (n x n)
//' @param b numeric vector of length n (single RHS)
//' @return numeric vector x satisfying A x = b
// [[Rcpp::export]]
Rcpp::NumericVector xopt_solve_impl(Rcpp::NumericMatrix A, Rcpp::NumericVector b) {
    const int n = A.nrow();
    if (A.ncol() != n) Rcpp::stop("xopt_solve: A must be square");
    if (b.size() != n) Rcpp::stop("xopt_solve: length(b) must equal nrow(A)");
    auto x = xopt::linalg::solve_spd(mat_to_vec(A), n, numvec_to_vec(b));
    return vec_to_numvec(x);
}

//' Forward-mode log-determinant on double
//' @param A numeric SPD matrix (n x n)
//' @return scalar log|A|
// [[Rcpp::export]]
double xopt_logdet_impl(Rcpp::NumericMatrix A) {
    const int n = A.nrow();
    if (A.ncol() != n) Rcpp::stop("xopt_logdet: A must be square");
    return xopt::linalg::logdet_spd(mat_to_vec(A), n);
}

//' Forward-mode SPD inverse on double
//' @param A numeric SPD matrix (n x n)
//' @return numeric matrix A^{-1}
// [[Rcpp::export]]
Rcpp::NumericMatrix xopt_inv_impl(Rcpp::NumericMatrix A) {
    const int n = A.nrow();
    if (A.ncol() != n) Rcpp::stop("xopt_inv: A must be square");
    auto Inv = xopt::linalg::inv_spd(mat_to_vec(A), n);
    return vec_to_mat(Inv, n, n);
}

// ---------------------------------------------------------------------------
// Adjoint-mode gradients: scalar loss L(A) = sum-reduction over the output.
//
// The loss is chosen so every output entry influences L, exercising the full
// adjoint path. Component weights are fixed (1.0) rather than random so the
// test is reproducible across platforms and XAD tape orderings.
// ---------------------------------------------------------------------------

//' Adjoint gradient of sum(L) where L = chol(A)
//' @param A numeric SPD matrix
//' @return numeric matrix of ∂(sum L) / ∂A
// [[Rcpp::export]]
Rcpp::NumericMatrix xopt_chol_grad(Rcpp::NumericMatrix A) {
    const int n = A.nrow();
    Tape tape;
    auto Av = mat_to_vec(A);
    auto Aa = to_ad_mat(Av, tape);
    tape.newRecording();
    auto La = xopt::linalg::chol(Aa, n);
    AD loss = AD(0.0);
    for (std::size_t i = 0; i < La.size(); ++i) loss = loss + La[i];
    tape.registerOutput(loss);
    xad::derivative(loss) = 1.0;
    tape.computeAdjoints();
    std::vector<double> grad(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < grad.size(); ++i) grad[i] = xad::derivative(Aa[i]);
    return vec_to_mat(grad, n, n);
}

//' Adjoint gradient of sum(x) where x = solve(A, b)
//' @param A numeric SPD matrix
//' @param b numeric RHS
//' @param wrt "A" or "b" — which variable to differentiate w.r.t.
//' @return numeric matrix (wrt=A) or vector (wrt=b) of partials
// [[Rcpp::export]]
SEXP xopt_solve_grad(Rcpp::NumericMatrix A, Rcpp::NumericVector b, std::string wrt) {
    const int n = A.nrow();
    Tape tape;
    auto Av = mat_to_vec(A);
    auto bv = numvec_to_vec(b);
    std::vector<AD> Aa, ba;
    if (wrt == "A") {
        Aa = to_ad_mat(Av, tape);
        ba = to_ad_vec(bv);
    } else if (wrt == "b") {
        Aa = to_ad_vec(Av);
        ba.resize(bv.size());
        for (std::size_t i = 0; i < bv.size(); ++i) {
            ba[i] = bv[i];
            tape.registerInput(ba[i]);
        }
    } else {
        Rcpp::stop("xopt_solve_grad: wrt must be 'A' or 'b'");
    }
    tape.newRecording();
    auto xa = xopt::linalg::solve_spd(Aa, n, ba);
    AD loss = AD(0.0);
    for (std::size_t i = 0; i < xa.size(); ++i) loss = loss + xa[i];
    tape.registerOutput(loss);
    xad::derivative(loss) = 1.0;
    tape.computeAdjoints();
    if (wrt == "A") {
        std::vector<double> grad(Aa.size());
        for (std::size_t i = 0; i < grad.size(); ++i) grad[i] = xad::derivative(Aa[i]);
        return vec_to_mat(grad, n, n);
    }
    std::vector<double> grad(ba.size());
    for (std::size_t i = 0; i < grad.size(); ++i) grad[i] = xad::derivative(ba[i]);
    return vec_to_numvec(grad);
}

//' Adjoint gradient of logdet(A)
//' @param A numeric SPD matrix
//' @return numeric matrix of ∂ log|A| / ∂A (expected A^{-1})
// [[Rcpp::export]]
Rcpp::NumericMatrix xopt_logdet_grad(Rcpp::NumericMatrix A) {
    const int n = A.nrow();
    Tape tape;
    auto Av = mat_to_vec(A);
    auto Aa = to_ad_mat(Av, tape);
    tape.newRecording();
    AD loss = xopt::linalg::logdet_spd(Aa, n);
    tape.registerOutput(loss);
    xad::derivative(loss) = 1.0;
    tape.computeAdjoints();
    std::vector<double> grad(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < grad.size(); ++i) grad[i] = xad::derivative(Aa[i]);
    return vec_to_mat(grad, n, n);
}

//' Adjoint gradient of sum(A^{-1})
//' @param A numeric SPD matrix
//' @return numeric matrix of ∂ sum(A^{-1}) / ∂A
// [[Rcpp::export]]
Rcpp::NumericMatrix xopt_inv_grad(Rcpp::NumericMatrix A) {
    const int n = A.nrow();
    Tape tape;
    auto Av = mat_to_vec(A);
    auto Aa = to_ad_mat(Av, tape);
    tape.newRecording();
    auto Ia = xopt::linalg::inv_spd(Aa, n);
    AD loss = AD(0.0);
    for (std::size_t i = 0; i < Ia.size(); ++i) loss = loss + Ia[i];
    tape.registerOutput(loss);
    xad::derivative(loss) = 1.0;
    tape.computeAdjoints();
    std::vector<double> grad(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < grad.size(); ++i) grad[i] = xad::derivative(Aa[i]);
    return vec_to_mat(grad, n, n);
}

// ---------------------------------------------------------------------------
// Tape-size benchmark.
//
// The CheckpointCallback path records a bounded number of tape slots per
// call: one output slot for logdet, n for solve, and n^2 for inv. The generic
// elementary-op path records every arithmetic operation inside chol +
// triangular solves, scaling as O(n^3). This benchmark returns
// `tape.getMemory()` (bytes) for each primitive on a random SPD matrix of
// size n, so the caller can compare growth rates across n.
// ---------------------------------------------------------------------------

//' Tape memory (bytes) for one adjoint sweep of each xopt::linalg primitive.
//' @param A numeric SPD matrix
//' @param op one of "logdet", "solve", "inv"
//' @return scalar tape memory in bytes (std::size_t → double)
// [[Rcpp::export]]
double xopt_linalg_tape_bytes(Rcpp::NumericMatrix A, std::string op) {
    const int n = A.nrow();
    Tape tape;
    auto Av = mat_to_vec(A);
    auto Aa = to_ad_mat(Av, tape);
    tape.newRecording();
    if (op == "logdet") {
        AD loss = xopt::linalg::logdet_spd(Aa, n);
        tape.registerOutput(loss);
        xad::derivative(loss) = 1.0;
    } else if (op == "solve") {
        std::vector<double> bv(static_cast<std::size_t>(n), 1.0);
        auto ba = to_ad_vec(bv);
        auto xa = xopt::linalg::solve_spd(Aa, n, ba);
        AD loss = AD(0.0);
        for (std::size_t i = 0; i < xa.size(); ++i) loss = loss + xa[i];
        tape.registerOutput(loss);
        xad::derivative(loss) = 1.0;
    } else if (op == "inv") {
        auto Ia = xopt::linalg::inv_spd(Aa, n);
        AD loss = AD(0.0);
        for (std::size_t i = 0; i < Ia.size(); ++i) loss = loss + Ia[i];
        tape.registerOutput(loss);
        xad::derivative(loss) = 1.0;
    } else {
        Rcpp::stop("xopt_linalg_tape_bytes: op must be 'logdet', 'solve', or 'inv'");
    }
    tape.computeAdjoints();
    return static_cast<double>(tape.getMemory());
}
