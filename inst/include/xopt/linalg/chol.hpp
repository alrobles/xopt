// xopt/linalg/chol.hpp - Cholesky decomposition with generic AD support
//
// Computes the Cholesky factor L of a symmetric positive-definite (SPD)
// matrix A, i.e. A = L * L^T, where L is lower triangular.
//
// Matrices are stored as std::vector<Scalar> in column-major order (size n*n),
// matching R/BLAS/LAPACK convention.
//
// The generic template uses only elementary operations (sqrt, subtract,
// multiply, divide). When Scalar is a taped AD type (xad::AReal<double>),
// XAD records O(n^3) elementary ops and the adjoint flows automatically.
// This is correct but slower than a custom CheckpointCallback rule; the
// latter is a planned follow-up optimization.
//
// A fast LAPACK-backed specialization for Scalar = double is provided
// separately; see inst/include/xopt/linalg/chol_lapack.hpp.

#ifndef XOPT_LINALG_CHOL_HPP
#define XOPT_LINALG_CHOL_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace xopt {
namespace linalg {

// Generic templated Cholesky factorization (in-place).
//
// Input:  A[i + j*n] = A_{i,j}, SPD, n x n (column-major).
// Output: A overwritten with L (lower triangular; upper triangle zeroed).
// Returns: true if SPD (all pivots positive), false if non-SPD.
//
// Throws std::invalid_argument on size mismatch.
template <class Scalar>
bool chol_inplace(std::vector<Scalar>& A, int n) {
    using std::sqrt;
    if (n <= 0 || static_cast<std::size_t>(n) * static_cast<std::size_t>(n) != A.size()) {
        throw std::invalid_argument("chol_inplace: A must be n*n");
    }
    for (int j = 0; j < n; ++j) {
        // Compute diagonal pivot: A_{jj} - sum_{k<j} L_{jk}^2
        Scalar d = A[j + j * n];
        for (int k = 0; k < j; ++k) {
            const Scalar& ljk = A[j + k * n];
            d = d - ljk * ljk;
        }
        // Non-SPD guard uses overloaded operator<= on the forward value.
        if (!(d > Scalar(0))) {
            return false;
        }
        d = sqrt(d);
        A[j + j * n] = d;
        // Column below diagonal: L_{ij} = (A_{ij} - sum_{k<j} L_{ik} L_{jk}) / d
        for (int i = j + 1; i < n; ++i) {
            Scalar s = A[i + j * n];
            for (int k = 0; k < j; ++k) {
                s = s - A[i + k * n] * A[j + k * n];
            }
            A[i + j * n] = s / d;
        }
    }
    // Zero the strict upper triangle so callers see a clean L.
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            A[i + j * n] = Scalar(0);
        }
    }
    return true;
}

// Convenience wrapper that returns a new vector.
template <class Scalar>
std::vector<Scalar> chol(const std::vector<Scalar>& A, int n) {
    std::vector<Scalar> L = A;
    if (!chol_inplace(L, n)) {
        throw std::runtime_error("xopt::linalg::chol: matrix is not symmetric positive-definite");
    }
    return L;
}

}  // namespace linalg
}  // namespace xopt

#endif  // XOPT_LINALG_CHOL_HPP
