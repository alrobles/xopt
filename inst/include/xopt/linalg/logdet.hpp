// xopt/linalg/logdet.hpp - log-determinant of an SPD matrix via Cholesky
//
// log|A| = 2 * sum_i log(L_ii) where A = L L^T (Cholesky).
//
// Generic template over Scalar; uses chol_inplace + elementary ops.

#ifndef XOPT_LINALG_LOGDET_HPP
#define XOPT_LINALG_LOGDET_HPP

#include <cmath>
#include <vector>

#include <xopt/linalg/chol.hpp>

namespace xopt {
namespace linalg {

// log-determinant of an SPD matrix A (n x n, column-major).
// Throws std::runtime_error if A is not SPD.
template <class Scalar>
Scalar logdet_spd(const std::vector<Scalar>& A, int n) {
    using std::log;
    std::vector<Scalar> L = A;
    if (!chol_inplace(L, n)) {
        throw std::runtime_error("xopt::linalg::logdet_spd: matrix is not SPD");
    }
    Scalar acc = Scalar(0);
    for (int i = 0; i < n; ++i) {
        acc = acc + log(L[i + i * n]);
    }
    return Scalar(2) * acc;
}

}  // namespace linalg
}  // namespace xopt

#endif  // XOPT_LINALG_LOGDET_HPP
