// Probe: verify that XAD is wired into xopt's compiled C++ build via
// LinkingTo: xadr. This file instantiates xad::AReal<double>, records a
// simple forward sweep (f(x) = sum(x_i^2)), and reads back the exact
// adjoint gradient.
//
// Replaces the earlier placeholder that hand-coded a gradient on plain
// doubles. If this probe returns 0 and prints the expected values, the
// full XAD tape + adjoint machinery is available to the rest of xopt's
// compiled C++ sources (including inst/include/xopt/linalg/*).

#include <Rcpp.h>
#include <XAD/XAD.hpp>

#include <vector>

using AD = xad::AReal<double>;
using Tape = xad::Tape<double>;

//' @title Probe XAD integration
//' @description Instantiates `xad::AReal<double>`, records `f(x) = sum(x^2)`
//'   with 3 active inputs, and reports the value and adjoint gradient.
//'   Returns 0 on success (printed gradients match `2 * x` to machine
//'   precision), nonzero otherwise.
//' @return Integer status code (0 = pass).
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
int probe_xad_xtensor() {
    Tape tape;
    std::vector<AD> x{1.0, 2.0, 3.0};
    for (auto& xi : x) tape.registerInput(xi);
    tape.newRecording();

    AD y = 0.0;
    for (auto& xi : x) y = y + xi * xi;  // f(x) = sum(x_i^2)

    tape.registerOutput(y);
    xad::derivative(y) = 1.0;
    tape.computeAdjoints();

    const double fval = xad::value(y);
    Rcpp::Rcout << "probe_xad_xtensor: f(x) = " << fval
                << " (expected 14)\n";
    int status = (fval == 14.0) ? 0 : 1;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double g = xad::derivative(x[i]);
        const double expected = 2.0 * xad::value(x[i]);
        Rcpp::Rcout << "probe_xad_xtensor: df/dx" << i << " = " << g
                    << " (expected " << expected << ")\n";
        if (g != expected) status = 2;
    }
    return status;
}
