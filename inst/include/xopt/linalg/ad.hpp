// xopt/linalg/ad.hpp - CheckpointCallback-based custom adjoints for xopt::linalg.
//
// Including this header activates, by overload resolution, O(n^2) tape
// footprint for the AReal<double> variants of
//   logdet_spd(A)
//   solve_spd(A, b)
//   inv_spd(A)
//
// The generic templates in chol.hpp / solve.hpp / logdet.hpp / inv.hpp record
// O(n^3) elementary operations onto the XAD tape. That is correct but quickly
// exhausts tape memory at n = a few hundred and is wasteful for compositional
// AD pipelines (e.g., Laplace over a Gaussian likelihood).
//
// The overloads below use xad::CheckpointCallback to:
//   (a) run the forward computation with passive doubles,
//   (b) register only the scalar / vector / matrix output(s) as active on the
//       tape,
//   (c) on the reverse sweep, apply the closed-form cotangent
//       (Giles "Collected matrix derivative results", 2008):
//
//           logdet_spd:  dA_math = y_bar * A^{-1}
//           solve_spd:   db      = A^{-1} * x_bar; dA_math = -(db) * x^T
//           inv_spd:     dA_math = -A^{-1} * B_bar * A^{-1}
//
// ---------------------------------------------------------------------------
// "chol reads lower only" convention.
//
// xopt::linalg::chol_inplace reads strictly the lower triangle + diagonal of
// A; the upper triangle is never read. The generic-template adjoint sweep
// therefore deposits zero gradient on upper-triangle slots and the *full*
// contribution on each lower-triangle slot (since each A_{ij} with i>j is
// read exactly once by chol_inplace — not twice for the symmetric pair).
//
// To match that convention (and the testthat FD checks against it), each
// custom-adjoint callback below:
//   1. Computes the mathematical dA_math matrix (from the symmetric
//      Giles formulas above; symmetrized where the raw formula isn't
//      already symmetric, e.g. solve's rank-1 -db * x^T).
//   2. Writes to tape slots as:
//        upper (i < j): 0
//        diagonal    : dA_sym[i,i]
//        strict lower: 2 * dA_sym[i,j]
// The factor of 2 folds in the symmetric pair (i,j)+(j,i) → (i,j) slot.
// ---------------------------------------------------------------------------
//
// Cholesky's own cotangent (Giles' Phi(L^T L_bar) / L^{-1} formula) is
// deferred to a follow-up PR. Calling chol() on AReal inputs still works —
// the generic template records O(n^3) elementary ops, same as before.

#ifndef XOPT_LINALG_AD_HPP
#define XOPT_LINALG_AD_HPP

#include <cstddef>
#include <vector>

#include <XAD/XAD.hpp>

#include <xopt/linalg/chol.hpp>
#include <xopt/linalg/inv.hpp>
#include <xopt/linalg/logdet.hpp>
#include <xopt/linalg/solve.hpp>

namespace xopt {
namespace linalg {
namespace detail {

using ad_real = xad::AReal<double>;
using ad_tape = xad::Tape<double>;
using ad_slot = ad_tape::slot_type;
static constexpr ad_slot AD_INVALID_SLOT = ad_tape::INVALID_SLOT;

inline void snapshot_slots_and_values(const std::vector<ad_real>& v,
                                      std::vector<ad_slot>& slots,
                                      std::vector<double>& values) {
    slots.resize(v.size());
    values.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        slots[i] = v[i].getSlot();
        values[i] = value(v[i]);
    }
}

// Deposit a symmetric cotangent matrix dA_sym (column-major, n x n) onto the
// tape following the "chol reads lower only" convention (see header comment).
// Invalid slots (passive inputs) are silently skipped.
inline void deposit_spd_cotangent(ad_tape* tape,
                                  const std::vector<ad_slot>& A_slots,
                                  const std::vector<double>& dA_sym,
                                  int n) {
    for (int j = 0; j < n; ++j) {
        // diagonal: i == j, factor 1
        {
            const std::size_t k = static_cast<std::size_t>(j)
                                 + static_cast<std::size_t>(j) * n;
            const ad_slot slot = A_slots[k];
            const double g = dA_sym[k];
            if (slot != AD_INVALID_SLOT && g != 0.0) {
                tape->incrementAdjoint(slot, g);
            }
        }
        // strict lower: i > j, factor 2
        for (int i = j + 1; i < n; ++i) {
            const std::size_t k = static_cast<std::size_t>(i)
                                 + static_cast<std::size_t>(j) * n;
            const ad_slot slot = A_slots[k];
            const double g = 2.0 * dA_sym[k];
            if (slot != AD_INVALID_SLOT && g != 0.0) {
                tape->incrementAdjoint(slot, g);
            }
        }
        // strict upper: i < j, skip (deposits zero)
    }
}

// ---------------------------------------------------------------------------
// logdet_spd:  y = log det A.   dA_math = y_bar * A^{-1}  (symmetric).
// ---------------------------------------------------------------------------
class LogdetSpdCallback : public xad::CheckpointCallback<ad_tape> {
  public:
    LogdetSpdCallback(std::vector<double> A_val, int n,
                      std::vector<ad_slot> input_slots, ad_slot output_slot)
        : A_(std::move(A_val)), n_(n),
          input_slots_(std::move(input_slots)), output_slot_(output_slot) {}

    void computeAdjoint(ad_tape* tape) override {
        const double y_bar = tape->getAndResetOutputAdjoint(output_slot_);
        if (y_bar == 0.0) return;
        const std::vector<double> Ainv = xopt::linalg::inv_spd(A_, n_);
        std::vector<double> dA_sym(Ainv.size());
        for (std::size_t i = 0; i < Ainv.size(); ++i) dA_sym[i] = y_bar * Ainv[i];
        deposit_spd_cotangent(tape, input_slots_, dA_sym, n_);
    }

  private:
    std::vector<double> A_;
    int n_;
    std::vector<ad_slot> input_slots_;
    ad_slot output_slot_;
};

// ---------------------------------------------------------------------------
// solve_spd:  x = A^{-1} b.
//   db_bar      = A^{-1} x_bar                    (passed straight through)
//   dA_math     = -(db_bar) * x^T                 (rank-1, NOT symmetric)
//   dA_sym_{ij} = -(db_bar[i]*x[j] + db_bar[j]*x[i]) / 2
// ---------------------------------------------------------------------------
class SolveSpdCallback : public xad::CheckpointCallback<ad_tape> {
  public:
    SolveSpdCallback(std::vector<double> A_val, int n,
                     std::vector<double> x_val,
                     std::vector<ad_slot> A_slots,
                     std::vector<ad_slot> b_slots,
                     std::vector<ad_slot> x_slots)
        : A_(std::move(A_val)), n_(n), x_(std::move(x_val)),
          A_slots_(std::move(A_slots)), b_slots_(std::move(b_slots)),
          x_slots_(std::move(x_slots)) {}

    void computeAdjoint(ad_tape* tape) override {
        std::vector<double> x_bar(static_cast<std::size_t>(n_));
        bool any_nonzero = false;
        for (int i = 0; i < n_; ++i) {
            x_bar[i] = tape->getAndResetOutputAdjoint(x_slots_[i]);
            if (x_bar[i] != 0.0) any_nonzero = true;
        }
        if (!any_nonzero) return;

        const std::vector<double> db_bar = xopt::linalg::solve_spd(A_, n_, x_bar);

        for (int i = 0; i < n_; ++i) {
            if (b_slots_[i] != AD_INVALID_SLOT && db_bar[i] != 0.0) {
                tape->incrementAdjoint(b_slots_[i], db_bar[i]);
            }
        }
        // dA_sym = -(db_bar * x^T + x * db_bar^T) / 2 as a full n x n matrix.
        std::vector<double> dA_sym(static_cast<std::size_t>(n_)
                                   * static_cast<std::size_t>(n_));
        for (int j = 0; j < n_; ++j) {
            for (int i = 0; i < n_; ++i) {
                dA_sym[static_cast<std::size_t>(i)
                      + static_cast<std::size_t>(j) * n_] =
                    -0.5 * (db_bar[i] * x_[j] + db_bar[j] * x_[i]);
            }
        }
        deposit_spd_cotangent(tape, A_slots_, dA_sym, n_);
    }

  private:
    std::vector<double> A_;
    int n_;
    std::vector<double> x_;
    std::vector<ad_slot> A_slots_;
    std::vector<ad_slot> b_slots_;
    std::vector<ad_slot> x_slots_;
};

// ---------------------------------------------------------------------------
// inv_spd:  B = A^{-1}.
//   dA_math = -A^{-1} * B_bar * A^{-1}
//   dA_sym  = -A^{-1} * (B_bar + B_bar^T) / 2 * A^{-1}
// Implemented via two back-to-back solve_spd passes; inner work is O(n^3).
// ---------------------------------------------------------------------------
class InvSpdCallback : public xad::CheckpointCallback<ad_tape> {
  public:
    InvSpdCallback(std::vector<double> A_val, int n,
                   std::vector<ad_slot> A_slots,
                   std::vector<ad_slot> B_slots)
        : A_(std::move(A_val)), n_(n),
          A_slots_(std::move(A_slots)), B_slots_(std::move(B_slots)) {}

    void computeAdjoint(ad_tape* tape) override {
        const std::size_t N = static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_);
        std::vector<double> Bbar(N);
        bool any_nonzero = false;
        for (std::size_t i = 0; i < N; ++i) {
            Bbar[i] = tape->getAndResetOutputAdjoint(B_slots_[i]);
            if (Bbar[i] != 0.0) any_nonzero = true;
        }
        if (!any_nonzero) return;

        // Symmetrize B_bar: 0.5 * (Bbar + Bbar^T).
        std::vector<double> Bsym(N);
        for (int j = 0; j < n_; ++j) {
            for (int i = 0; i < n_; ++i) {
                const std::size_t ij = static_cast<std::size_t>(i)
                                      + static_cast<std::size_t>(j) * n_;
                const std::size_t ji = static_cast<std::size_t>(j)
                                      + static_cast<std::size_t>(i) * n_;
                Bsym[ij] = 0.5 * (Bbar[ij] + Bbar[ji]);
            }
        }

        // M = A^{-1} * Bsym (column-by-column).
        std::vector<double> M(N);
        std::vector<double> col(n_);
        for (int k = 0; k < n_; ++k) {
            for (int i = 0; i < n_; ++i) {
                col[static_cast<std::size_t>(i)] = Bsym[static_cast<std::size_t>(i)
                                                        + static_cast<std::size_t>(k) * n_];
            }
            const std::vector<double> y = xopt::linalg::solve_spd(A_, n_, col);
            for (int i = 0; i < n_; ++i) {
                M[static_cast<std::size_t>(i)
                  + static_cast<std::size_t>(k) * n_] = y[static_cast<std::size_t>(i)];
            }
        }

        // dA_sym = -M * A^{-1}. For each row i of M, solve A z = row_i^T
        // (since A is symmetric), giving z_j = (M A^{-1})_{ij}.
        std::vector<double> dA_sym(N);
        std::vector<double> row(n_);
        for (int i = 0; i < n_; ++i) {
            for (int k = 0; k < n_; ++k) {
                row[static_cast<std::size_t>(k)] = M[static_cast<std::size_t>(i)
                                                    + static_cast<std::size_t>(k) * n_];
            }
            const std::vector<double> z = xopt::linalg::solve_spd(A_, n_, row);
            for (int j = 0; j < n_; ++j) {
                dA_sym[static_cast<std::size_t>(i)
                      + static_cast<std::size_t>(j) * n_] =
                    -z[static_cast<std::size_t>(j)];
            }
        }

        deposit_spd_cotangent(tape, A_slots_, dA_sym, n_);
    }

  private:
    std::vector<double> A_;
    int n_;
    std::vector<ad_slot> A_slots_;
    std::vector<ad_slot> B_slots_;
};

}  // namespace detail

// ---------------------------------------------------------------------------
// Overloads for std::vector<xad::AReal<double>> that install custom
// CheckpointCallback cotangents in place of O(n^3) elementary-op recording.
// ---------------------------------------------------------------------------

inline xad::AReal<double> logdet_spd(const std::vector<xad::AReal<double>>& A, int n) {
    using detail::ad_real;
    using detail::ad_slot;
    using detail::ad_tape;

    std::vector<ad_slot> input_slots;
    std::vector<double> A_val;
    detail::snapshot_slots_and_values(A, input_slots, A_val);

    const double y_val = xopt::linalg::logdet_spd(A_val, n);

    ad_tape* tape = ad_tape::getActive();
    ad_real y = y_val;
    if (tape == nullptr) {
        return y;
    }
    tape->registerOutput(y);
    const ad_slot output_slot = y.getSlot();

    auto* cb = new detail::LogdetSpdCallback(std::move(A_val), n,
                                             std::move(input_slots), output_slot);
    tape->pushCallback(cb);
    tape->insertCallback(cb);
    return y;
}

inline std::vector<xad::AReal<double>> solve_spd(
    const std::vector<xad::AReal<double>>& A, int n,
    const std::vector<xad::AReal<double>>& b) {
    using detail::ad_real;
    using detail::ad_slot;
    using detail::ad_tape;

    std::vector<ad_slot> A_slots, b_slots;
    std::vector<double> A_val, b_val;
    detail::snapshot_slots_and_values(A, A_slots, A_val);
    detail::snapshot_slots_and_values(b, b_slots, b_val);

    const std::vector<double> x_val = xopt::linalg::solve_spd(A_val, n, b_val);

    ad_tape* tape = ad_tape::getActive();
    std::vector<ad_real> x(x_val.size());
    for (std::size_t i = 0; i < x_val.size(); ++i) x[i] = x_val[i];
    if (tape == nullptr) {
        return x;
    }
    for (auto& xi : x) tape->registerOutput(xi);

    std::vector<ad_slot> x_slots(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) x_slots[i] = x[i].getSlot();

    auto* cb = new detail::SolveSpdCallback(std::move(A_val), n,
                                            x_val,
                                            std::move(A_slots), std::move(b_slots),
                                            std::move(x_slots));
    tape->pushCallback(cb);
    tape->insertCallback(cb);
    return x;
}

inline std::vector<xad::AReal<double>> inv_spd(
    const std::vector<xad::AReal<double>>& A, int n) {
    using detail::ad_real;
    using detail::ad_slot;
    using detail::ad_tape;

    std::vector<ad_slot> A_slots;
    std::vector<double> A_val;
    detail::snapshot_slots_and_values(A, A_slots, A_val);

    const std::vector<double> B_val = xopt::linalg::inv_spd(A_val, n);

    ad_tape* tape = ad_tape::getActive();
    std::vector<ad_real> B(B_val.size());
    for (std::size_t i = 0; i < B_val.size(); ++i) B[i] = B_val[i];
    if (tape == nullptr) {
        return B;
    }
    for (auto& Bi : B) tape->registerOutput(Bi);

    std::vector<ad_slot> B_slots(B.size());
    for (std::size_t i = 0; i < B.size(); ++i) B_slots[i] = B[i].getSlot();

    auto* cb = new detail::InvSpdCallback(std::move(A_val), n,
                                          std::move(A_slots), std::move(B_slots));
    tape->pushCallback(cb);
    tape->insertCallback(cb);
    return B;
}

}  // namespace linalg
}  // namespace xopt

#endif  // XOPT_LINALG_AD_HPP
