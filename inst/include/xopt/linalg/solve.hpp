// xopt/linalg/solve.hpp - SPD linear solve via Cholesky with generic AD support
//
// Solves A * x = b where A is symmetric positive-definite (SPD), n x n.
// Strategy: A = L L^T (Cholesky), then forward-solve L y = b, back-solve L^T x = y.
//
// Matrices are column-major, vectors are flat std::vector<Scalar>.
// For multi-RHS, pass B as a flat column-major vector of size n*nrhs; the
// returned X has the same shape.
//
// Generic template records O(n^3) elementary XAD ops and adjoints flow
// automatically. A fast LAPACK-backed specialization for Scalar = double
// is a planned follow-up.

#ifndef XOPT_LINALG_SOLVE_HPP
#define XOPT_LINALG_SOLVE_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <xopt/linalg/chol.hpp>

namespace xopt {
namespace linalg {

// Forward-solve L y = b (L lower triangular, column-major n x n).
// y is written into y (which may alias b in the caller via copy).
template <class Scalar>
void trisolve_L(const std::vector<Scalar>& L, int n,
                const std::vector<Scalar>& b,
                std::vector<Scalar>& y) {
    y = b;
    for (int i = 0; i < n; ++i) {
        Scalar s = y[i];
        for (int k = 0; k < i; ++k) {
            s = s - L[i + k * n] * y[k];
        }
        y[i] = s / L[i + i * n];
    }
}

// Back-solve L^T x = y (L lower triangular so L^T is upper triangular).
template <class Scalar>
void trisolve_Lt(const std::vector<Scalar>& L, int n,
                 const std::vector<Scalar>& y,
                 std::vector<Scalar>& x) {
    x = y;
    for (int i = n - 1; i >= 0; --i) {
        Scalar s = x[i];
        for (int k = i + 1; k < n; ++k) {
            s = s - L[k + i * n] * x[k];
        }
        x[i] = s / L[i + i * n];
    }
}

// Solve A x = b for SPD A. Returns x.
// b has length n (single RHS). For multi-RHS see `solve_spd_multi`.
template <class Scalar>
std::vector<Scalar> solve_spd(const std::vector<Scalar>& A, int n,
                              const std::vector<Scalar>& b) {
    if (static_cast<std::size_t>(n) != b.size()) {
        throw std::invalid_argument("solve_spd: length(b) must equal n");
    }
    std::vector<Scalar> L = chol(A, n);  // throws if not SPD
    std::vector<Scalar> y, x;
    trisolve_L(L, n, b, y);
    trisolve_Lt(L, n, y, x);
    return x;
}

// Solve A X = B for SPD A with multiple RHS. B has shape n x nrhs (column-major).
// Returns X with the same shape.
template <class Scalar>
std::vector<Scalar> solve_spd_multi(const std::vector<Scalar>& A, int n,
                                    const std::vector<Scalar>& B, int nrhs) {
    if (static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs) != B.size()) {
        throw std::invalid_argument("solve_spd_multi: B size must equal n*nrhs");
    }
    std::vector<Scalar> L = chol(A, n);
    std::vector<Scalar> X(static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs));
    std::vector<Scalar> col(n), y(n), x(n);
    for (int k = 0; k < nrhs; ++k) {
        for (int i = 0; i < n; ++i) col[i] = B[i + k * n];
        trisolve_L(L, n, col, y);
        trisolve_Lt(L, n, y, x);
        for (int i = 0; i < n; ++i) X[i + k * n] = x[i];
    }
    return X;
}

}  // namespace linalg
}  // namespace xopt

#endif  // XOPT_LINALG_SOLVE_HPP
