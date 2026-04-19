// xopt/implicit/ift.hpp — implicit differentiation via the implicit function
// theorem, with a custom XAD CheckpointCallback so the O(n) output slots and
// O(n) θ-slots are the only things that touch the tape.
//
// Given a fixed-point equation
//     g(x, θ) = 0
// with a unique locally differentiable solution x*(θ) and ∂g/∂x invertible at
// (x*(θ), θ), the IFT gives
//     dx*/dθ = -(∂g/∂x)^{-1} (∂g/∂θ)
// so the vector-Jacobian product is
//     θ̄ = -(∂g/∂θ)^T (∂g/∂x)^{-T} x̄
//
// This header ships the SPD specialization (∂g/∂x symmetric positive
// definite), which covers the two most common cases in xopt:
//
//   (1) KKT stationarity of a convex objective: g = ∇_x f(x, θ), ∂g/∂x = H(x, θ).
//       H is SPD at a strict local minimum.
//
//   (2) Linear least squares / ridge / GLS: g = Aβ − b + λβ, ∂g/∂β = A + λI.
//
// The callback:
//   * runs the forward pass on plain double (caller supplies x* — typically
//     from xopt_minimize — plus the two Jacobians ∂g/∂x and ∂g/∂θ evaluated at
//     (x*, θ));
//   * registers x* as a fresh active output on the tape;
//   * on reverse sweep, reads x̄ from the output slots, computes the SPD solve
//     λ = A^{-1} x̄ once (O(n²) tape footprint via xopt::linalg::solve_spd),
//     and deposits θ̄ += −B^T λ onto the θ input slots.
//
// Notes:
//   * `A` is assumed SPD; the caller is responsible for symmetrising /
//     Cholesky-readability. `xopt::linalg::solve_spd` reads the lower triangle
//     only, so A's upper triangle is never used.
//   * `B` (n × p, column-major) is arbitrary — no symmetry assumption.
//   * If the active tape is null, the function returns `x*` as passive AReal
//     with no callback registered.

#ifndef XOPT_IMPLICIT_IFT_HPP
#define XOPT_IMPLICIT_IFT_HPP

#include <cstddef>
#include <utility>
#include <vector>

#include <XAD/XAD.hpp>

#include <xopt/linalg/solve.hpp>

namespace xopt {
namespace implicit_ {
namespace detail {

using ad_real = xad::AReal<double>;
using ad_tape = xad::Tape<double>;
using ad_slot = ad_tape::slot_type;
static constexpr ad_slot AD_INVALID_SLOT = ad_tape::INVALID_SLOT;

class ImplicitSpdCallback : public xad::CheckpointCallback<ad_tape> {
  public:
    ImplicitSpdCallback(std::vector<double> A, int n,
                        std::vector<double> B, int p,
                        std::vector<ad_slot> x_slots,
                        std::vector<ad_slot> theta_slots)
        : A_(std::move(A)), n_(n),
          B_(std::move(B)), p_(p),
          x_slots_(std::move(x_slots)),
          theta_slots_(std::move(theta_slots)) {}

    void computeAdjoint(ad_tape* tape) override {
        std::vector<double> x_bar(static_cast<std::size_t>(n_));
        bool any_nonzero = false;
        for (int i = 0; i < n_; ++i) {
            x_bar[i] = tape->getAndResetOutputAdjoint(x_slots_[i]);
            if (x_bar[i] != 0.0) any_nonzero = true;
        }
        if (!any_nonzero) return;

        // λ = A^{-1} x̄  (A assumed SPD).
        const std::vector<double> lambda = xopt::linalg::solve_spd(A_, n_, x_bar);

        // θ̄ += −B^T λ.  B is n×p column-major, so B[i + j*n_] is (i, j).
        for (int j = 0; j < p_; ++j) {
            if (theta_slots_[j] == AD_INVALID_SLOT) continue;
            double acc = 0.0;
            for (int i = 0; i < n_; ++i) {
                acc += B_[static_cast<std::size_t>(i)
                         + static_cast<std::size_t>(j) * n_] * lambda[i];
            }
            if (acc != 0.0) {
                tape->incrementAdjoint(theta_slots_[j], -acc);
            }
        }
    }

  private:
    std::vector<double> A_;
    int n_;
    std::vector<double> B_;
    int p_;
    std::vector<ad_slot> x_slots_;
    std::vector<ad_slot> theta_slots_;
};

}  // namespace detail

// SPD implicit-function lift.
//
//   x_star  — solution of g(x*, θ) = 0, as plain double (n-vector).
//   A       — ∂g/∂x evaluated at (x*, θ), SPD, column-major n×n.
//   B       — ∂g/∂θ evaluated at (x*, θ), column-major n×p.
//   theta   — active AReal input vector (length p). The adjoint will be
//             accumulated onto these slots in reverse sweep.
//
// Returns `x*` lifted as active AReal output vector, with the IFT cotangent
// rule installed via xad::CheckpointCallback. Downstream tape recording on
// these outputs flows correctly all the way back to θ.
inline std::vector<xad::AReal<double>> implicit_spd(
    const std::vector<double>& x_star,
    const std::vector<double>& A, int n,
    const std::vector<double>& B, int p,
    const std::vector<xad::AReal<double>>& theta) {
    using detail::ad_real;
    using detail::ad_slot;
    using detail::ad_tape;

    std::vector<ad_real> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) x[i] = x_star[i];

    ad_tape* tape = ad_tape::getActive();
    if (tape == nullptr) return x;

    std::vector<ad_slot> x_slots(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        tape->registerOutput(x[i]);
        x_slots[i] = x[i].getSlot();
    }
    std::vector<ad_slot> theta_slots(static_cast<std::size_t>(p));
    for (int j = 0; j < p; ++j) theta_slots[j] = theta[j].getSlot();

    auto* cb = new detail::ImplicitSpdCallback(A, n, B, p,
                                               std::move(x_slots),
                                               std::move(theta_slots));
    tape->pushCallback(cb);
    tape->insertCallback(cb);
    return x;
}

}  // namespace implicit_
}  // namespace xopt

#endif  // XOPT_IMPLICIT_IFT_HPP
