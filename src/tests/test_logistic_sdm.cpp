// test_logistic_sdm.cpp - Unit tests for logistic SDM model
//
// Tests gradient accuracy via numerical differentiation and
// validates the logistic SDM implementation

#include <Rcpp.h>
#include "../inst/include/xopt/models/logistic_sdm.hpp"
#include "../inst/include/xopt/ad_reduce.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace xopt;
using namespace xopt::models;

// Numerical gradient via finite differences
std::vector<double> numerical_gradient(
    const std::vector<double>& x,
    const std::vector<std::vector<double>>& covariates,
    const std::vector<double>& response,
    const RasterMask& mask,
    double eps = 1e-6) {

    LogisticSDM<double> model;
    size_t n_par = x.size();
    std::vector<double> grad(n_par);
    std::vector<double> x_plus = x;
    std::vector<double> x_minus = x;

    for (size_t i = 0; i < n_par; ++i) {
        // f(x + eps)
        x_plus[i] = x[i] + eps;
        double f_plus = model.value(x_plus.data(), covariates, response, mask);

        // f(x - eps)
        x_minus[i] = x[i] - eps;
        double f_minus = model.value(x_minus.data(), covariates, response, mask);

        // Central difference
        grad[i] = (f_plus - f_minus) / (2.0 * eps);

        // Reset
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    return grad;
}

//' @title Test logistic SDM gradient accuracy
//' @description Numerical gradient check for small raster stack
//' @export
// [[Rcpp::export]]
int test_logistic_sdm_gradient() {
    Rcpp::Rcout << "Test: Logistic SDM gradient accuracy" << std::endl;

    // Create small test dataset
    const size_t n_rows = 10;
    const size_t n_cols = 10;
    const size_t n_cells = n_rows * n_cols;
    const size_t n_layers = 3;

    RasterDims dims(n_rows, n_cols, n_layers);

    // Generate synthetic covariate data
    std::vector<std::vector<double>> covariates(n_layers);
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t j = 0; j < n_layers; ++j) {
        covariates[j].resize(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            covariates[j][i] = dist(rng);
        }
    }

    // Generate synthetic response data (presence/absence)
    std::vector<double> response(n_cells);
    std::vector<double> true_beta = {-0.5, 0.8, -0.3, 0.4};  // True parameters

    for (size_t i = 0; i < n_cells; ++i) {
        double linear_pred = true_beta[0];
        for (size_t j = 0; j < n_layers; ++j) {
            linear_pred += true_beta[j + 1] * covariates[j][i];
        }
        double prob = 1.0 / (1.0 + std::exp(-linear_pred));

        // Generate binary response
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        response[i] = (unif(rng) < prob) ? 1.0 : 0.0;
    }

    // Create mask (all valid)
    RasterMask mask(n_cells, true);

    // Test parameters
    std::vector<double> beta = {0.0, 0.0, 0.0, 0.0};

    // Compute analytical gradient
    LogisticSDM<double> model;
    std::vector<double> grad_analytical(beta.size());
    model.gradient(beta.data(), grad_analytical.data(), covariates, response, mask);

    // Compute numerical gradient
    std::vector<double> grad_numerical = numerical_gradient(beta, covariates, response, mask);

    // Check gradient accuracy
    Rcpp::Rcout << "  Gradient comparison:" << std::endl;
    double max_error = 0.0;
    for (size_t i = 0; i < beta.size(); ++i) {
        double error = std::abs(grad_analytical[i] - grad_numerical[i]);
        max_error = std::max(max_error, error);

        Rcpp::Rcout << "    Parameter " << i << ": "
                    << "analytical = " << grad_analytical[i]
                    << ", numerical = " << grad_numerical[i]
                    << ", error = " << error << std::endl;
    }

    Rcpp::Rcout << "  Max gradient error: " << max_error << std::endl;

    // Check if gradient is accurate (within tolerance)
    const double tolerance = 1e-5;
    if (max_error < tolerance) {
        Rcpp::Rcout << "  Status: PASS - Gradient matches numerical derivative" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL - Gradient error exceeds tolerance" << std::endl;
        return 1;
    }
}

//' @title Test logistic SDM end-to-end
//' @description Full test of model value, gradient, and predictions
//' @export
// [[Rcpp::export]]
int test_logistic_sdm_endtoend() {
    Rcpp::Rcout << "Test: Logistic SDM end-to-end" << std::endl;

    // Create test dataset
    const size_t n_rows = 5;
    const size_t n_cols = 5;
    const size_t n_cells = n_rows * n_cols;
    const size_t n_layers = 2;

    RasterDims dims(n_rows, n_cols, n_layers);

    // Simple linear covariates
    std::vector<std::vector<double>> covariates(n_layers);
    for (size_t j = 0; j < n_layers; ++j) {
        covariates[j].resize(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            covariates[j][i] = static_cast<double>(i) / 10.0 + static_cast<double>(j);
        }
    }

    // Deterministic response
    std::vector<double> response(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        response[i] = (i % 2 == 0) ? 1.0 : 0.0;
    }

    // All valid
    RasterMask mask(n_cells, true);

    // Create problem
    auto problem = make_logistic_sdm_problem(dims, covariates, response);

    // Test parameters
    std::vector<double> beta = {0.1, -0.2, 0.3};

    // Evaluate objective
    double nll = problem.value(beta.data());
    Rcpp::Rcout << "  Negative log-likelihood: " << nll << std::endl;

    // Evaluate gradient
    std::vector<double> grad(beta.size());
    problem.gradient(beta.data(), grad.data());
    Rcpp::Rcout << "  Gradient: [";
    for (size_t i = 0; i < grad.size(); ++i) {
        Rcpp::Rcout << grad[i];
        if (i < grad.size() - 1) Rcpp::Rcout << ", ";
    }
    Rcpp::Rcout << "]" << std::endl;

    // Make predictions
    auto predictions = predict_logistic_sdm(beta.data(), covariates, mask);
    Rcpp::Rcout << "  Number of predictions: " << predictions.size() << std::endl;

    // Compute deviance
    double deviance = compute_deviance(beta.data(), covariates, response, mask);
    Rcpp::Rcout << "  Deviance: " << deviance << std::endl;

    // Compute AUC
    double auc = compute_auc(beta.data(), covariates, response, mask);
    Rcpp::Rcout << "  AUC: " << auc << std::endl;

    // Basic sanity checks
    bool pass = true;
    if (!std::isfinite(nll)) {
        Rcpp::Rcout << "  ERROR: Non-finite objective value" << std::endl;
        pass = false;
    }
    if (predictions.size() != n_cells) {
        Rcpp::Rcout << "  ERROR: Wrong number of predictions" << std::endl;
        pass = false;
    }
    if (auc < 0.0 || auc > 1.0) {
        Rcpp::Rcout << "  ERROR: AUC out of [0, 1] range" << std::endl;
        pass = false;
    }

    if (pass) {
        Rcpp::Rcout << "  Status: PASS - All checks successful" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL - Some checks failed" << std::endl;
        return 1;
    }
}
