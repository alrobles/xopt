// ucminf_solver.hpp - Integration of ucminfcpp with xopt Problem abstraction
//
// This header provides a bridge between xopt's Problem trait and ucminfcpp's
// minimize_direct<F>() template API, enabling zero-overhead optimization with
// automatic gradient dispatch.

#ifndef XOPT_UCMINF_SOLVER_HPP
#define XOPT_UCMINF_SOLVER_HPP

#include <ucminf_core.hpp>
#include <vector>
#include <stdexcept>
#include <string>

namespace xopt {
namespace solvers {

// Control parameters for ucminf solver (mirrors ucminf::Control)
struct UcminfControl {
    double grtol = 1e-6;      // Gradient tolerance
    double xtol = 1e-12;      // Step tolerance
    double stepmax = 1.0;     // Initial trust region radius
    int maxeval = 500;        // Maximum function evaluations
    bool trace = false;       // Print trace output

    // Optional initial inverse Hessian (packed lower-triangle)
    std::vector<double> inv_hessian_lt;

    // Convert to ucminf::Control
    ucminf::Control to_ucminf() const {
        ucminf::Control ctrl;
        ctrl.grtol = grtol;
        ctrl.xtol = xtol;
        ctrl.stepmax = stepmax;
        ctrl.maxeval = maxeval;
        ctrl.inv_hessian_lt = inv_hessian_lt;
        return ctrl;
    }
};

// Result from ucminf optimization (mirrors xopt::OptimResult)
struct UcminfResult {
    std::vector<double> par;           // Optimal parameters
    double value;                      // Objective value at optimum
    std::vector<double> gradient;      // Final gradient
    int iterations;                    // Function evaluations used
    int convergence;                   // Convergence code
    std::string message;               // Convergence message
    std::vector<double> inv_hessian_lt;  // Final inverse Hessian (packed)

    // Convert ucminf::Result to UcminfResult
    static UcminfResult from_ucminf(const ucminf::Result& res) {
        UcminfResult result;
        result.par = res.x;
        result.value = res.f;
        result.iterations = res.n_eval;
        result.convergence = static_cast<int>(res.status);
        result.message = ucminf::status_message(res.status);
        result.inv_hessian_lt = res.inv_hessian_lt;

        // Compute final gradient
        result.gradient.resize(res.x.size());
        // Note: ucminf doesn't return final gradient directly,
        // but we can store max_gradient
        // For now, leave gradient empty or recompute if needed
        return result;
    }
};

// Solve using ucminf with a Problem that has analytical gradients
template <typename Problem>
UcminfResult ucminf_solve(const Problem& prob,
                          const std::vector<double>& x0,
                          const UcminfControl& control = {}) {
    static_assert(Problem::has_gradient(),
                  "Problem must provide gradients for ucminf");

    if (x0.size() != static_cast<size_t>(prob.n_par)) {
        throw std::invalid_argument("x0 size must match problem dimension");
    }

    // Create objective+gradient callback for ucminf
    auto fdf = [&prob](const std::vector<double>& x,
                       std::vector<double>& g,
                       double& f) {
        f = prob.value(x.data());
        prob.gradient(x.data(), g.data());
    };

    // Call ucminf::minimize_direct with zero-overhead template dispatch
    ucminf::Control uc_ctrl = control.to_ucminf();
    ucminf::Result res = ucminf::minimize_direct(x0, fdf, uc_ctrl);

    return UcminfResult::from_ucminf(res);
}

// Solve using ucminf with a raw objective/gradient function
// This overload is for when you don't have a Problem object
inline UcminfResult ucminf_solve(
    const std::vector<double>& x0,
    std::function<void(const std::vector<double>&,
                       std::vector<double>&,
                       double&)> fdf,
    const UcminfControl& control = {}) {

    ucminf::Control uc_ctrl = control.to_ucminf();
    // Use the header-only minimize_direct<F> rather than the non-inline
    // minimize() overload so LinkingTo: ucminfcpp (headers only) is sufficient.
    ucminf::Result res = ucminf::minimize_direct(x0, fdf, uc_ctrl);

    return UcminfResult::from_ucminf(res);
}

} // namespace solvers
} // namespace xopt

#endif // XOPT_UCMINF_SOLVER_HPP
