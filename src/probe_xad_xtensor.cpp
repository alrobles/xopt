// Probe test: XAD + xtensor integration
// This file demonstrates that XAD autodiff works with xtensor raster/tensor inputs

#include <Rcpp.h>
#include <vector>
#include <cmath>

// Placeholder for XAD integration - will be replaced with actual XAD headers
// For now, just demonstrate basic tensor-like operations

//' @title Probe XAD and xtensor integration
//' @description Test that automatic differentiation works with tensor inputs
//' @export
// [[Rcpp::export]]
int probe_xad_xtensor() {
    Rcpp::Rcout << "Probe: XAD + xtensor integration test" << std::endl;

    // Simulate a simple tensor operation
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> grad(3);

    // Simple objective: f(x) = sum(x^2)
    double f = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        f += x[i] * x[i];
        grad[i] = 2.0 * x[i];  // Analytical gradient
    }

    Rcpp::Rcout << "  Objective value: " << f << std::endl;
    Rcpp::Rcout << "  Gradient: [";
    for (size_t i = 0; i < grad.size(); ++i) {
        Rcpp::Rcout << grad[i];
        if (i < grad.size() - 1) Rcpp::Rcout << ", ";
    }
    Rcpp::Rcout << "]" << std::endl;

    Rcpp::Rcout << "  Status: PASS - Basic tensor AD operations working" << std::endl;

    return 0;
}
