// xopt/linalg/inv.hpp - SPD matrix inverse via Cholesky + triangular solves
//
// A^{-1} is computed column-by-column as the solution to A X = I.
// Same AD story as solve.hpp.

#ifndef XOPT_LINALG_INV_HPP
#define XOPT_LINALG_INV_HPP

#include <cstddef>
#include <vector>

#include <xopt/linalg/chol.hpp>
#include <xopt/linalg/solve.hpp>

namespace xopt {
namespace linalg {

// Invert an SPD matrix A (n x n, column-major). Returns A^{-1}.
template <class Scalar>
std::vector<Scalar> inv_spd(const std::vector<Scalar>& A, int n) {
    std::vector<Scalar> L = chol(A, n);  // throws if not SPD
    std::vector<Scalar> Inv(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                            Scalar(0));
    std::vector<Scalar> e(n), y(n), x(n);
    for (int k = 0; k < n; ++k) {
        // e_k = k-th column of identity
        for (int i = 0; i < n; ++i) e[i] = (i == k) ? Scalar(1) : Scalar(0);
        trisolve_L(L, n, e, y);
        trisolve_Lt(L, n, y, x);
        for (int i = 0; i < n; ++i) Inv[i + k * n] = x[i];
    }
    return Inv;
}

}  // namespace linalg
}  // namespace xopt

#endif  // XOPT_LINALG_INV_HPP
