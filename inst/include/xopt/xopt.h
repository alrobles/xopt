// xopt - Main header file
//
// This package integrates:
// - ucminfcpp: Quasi-Newton BFGS optimizer
// - XAD: Automatic differentiation library (AGPL-3)
// - xtensor-r: Zero-copy R <-> C++ tensor bridge
//
// Note: Dependencies will be linked via LinkingTo in DESCRIPTION
// For Phase 0, we establish the structure and prove compilation works

#ifndef XOPT_H
#define XOPT_H

#include <vector>
#include <functional>

namespace xopt {

// Version information
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 1;

// Forward declarations for future implementation
// These will be filled in subsequent phases

// Phase 1: Basic optimization interface
struct OptimControl {
    double gtol = 1e-6;      // Gradient tolerance
    double xtol = 1e-6;      // Parameter tolerance
    int maxiter = 1000;       // Maximum iterations
    bool trace = false;       // Print trace
};

struct OptimResult {
    std::vector<double> par;  // Optimal parameters
    double value;             // Objective value at optimum
    std::vector<double> gradient;  // Final gradient
    int iterations;           // Number of iterations
    int convergence;          // Convergence code (0 = success)
    std::string message;      // Convergence message
};

// Objective function signature
using ObjectiveFn = std::function<void(const std::vector<double>&, std::vector<double>&, double&)>;

// Placeholder for minimize function
// Will integrate ucminfcpp in Phase 1
OptimResult minimize(
    const std::vector<double>& x0,
    ObjectiveFn fn,
    const OptimControl& control = OptimControl()
);

} // namespace xopt

#endif // XOPT_H
