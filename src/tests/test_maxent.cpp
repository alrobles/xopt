// test_maxent.cpp - Unit tests for MaxEnt model
//
// Tests the MaxEnt implementation and chunked raster processing

#include <Rcpp.h>
#include "../inst/include/xopt/models/maxent.hpp"
#include "../inst/include/xopt/solvers/lbfgs.hpp"
#include <vector>
#include <cmath>
#include <random>

using namespace xopt;
using namespace xopt::models;
using namespace xopt::solvers;

//' @title Test MaxEnt gradient accuracy
//' @description Numerical gradient check for MaxEnt model
//' @export
// [[Rcpp::export]]
int test_maxent_gradient() {
    Rcpp::Rcout << "Test: MaxEnt gradient accuracy" << std::endl;

    // Small test dataset
    const size_t n_rows = 10;
    const size_t n_cols = 10;
    const size_t n_cells = n_rows * n_cols;
    const size_t n_features = 3;

    RasterDims dims(n_rows, n_cols, n_features);

    // Generate synthetic covariate data
    std::vector<std::vector<double>> covariates(n_features);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t j = 0; j < n_features; ++j) {
        covariates[j].resize(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            covariates[j][i] = dist(rng);
        }
    }

    // Generate presence/background data
    std::vector<double> response(n_cells, 0.0);
    // Mark 20 random cells as presence
    std::uniform_int_distribution<size_t> cell_dist(0, n_cells - 1);
    for (int i = 0; i < 20; ++i) {
        response[cell_dist(rng)] = 1.0;
    }

    // Create mask (all valid)
    RasterMask mask(n_cells, true);

    // Test parameters
    std::vector<double> beta(n_features, 0.0);

    // Compute analytical gradient
    MaxEnt<double> model;
    std::vector<double> grad_analytical(n_features);
    model.gradient(beta.data(), grad_analytical.data(), covariates, response, mask);

    // Compute numerical gradient
    const double eps = 1e-6;
    std::vector<double> grad_numerical(n_features);
    std::vector<double> beta_plus = beta;
    std::vector<double> beta_minus = beta;

    for (size_t i = 0; i < n_features; ++i) {
        beta_plus[i] = beta[i] + eps;
        double f_plus = model.value(beta_plus.data(), covariates, response, mask);

        beta_minus[i] = beta[i] - eps;
        double f_minus = model.value(beta_minus.data(), covariates, response, mask);

        grad_numerical[i] = (f_plus - f_minus) / (2.0 * eps);

        beta_plus[i] = beta[i];
        beta_minus[i] = beta[i];
    }

    // Check gradient accuracy
    Rcpp::Rcout << "  Gradient comparison:" << std::endl;
    double max_error = 0.0;
    for (size_t i = 0; i < n_features; ++i) {
        double error = std::abs(grad_analytical[i] - grad_numerical[i]);
        max_error = std::max(max_error, error);

        Rcpp::Rcout << "    Feature " << i << ": "
                    << "analytical = " << grad_analytical[i]
                    << ", numerical = " << grad_numerical[i]
                    << ", error = " << error << std::endl;
    }

    Rcpp::Rcout << "  Max gradient error: " << max_error << std::endl;

    const double tolerance = 1e-5;
    if (max_error < tolerance) {
        Rcpp::Rcout << "  Status: PASS" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        return 1;
    }
}

//' @title Test MaxEnt end-to-end
//' @description Full test of MaxEnt model with optimization
//' @export
// [[Rcpp::export]]
int test_maxent_endtoend() {
    Rcpp::Rcout << "Test: MaxEnt end-to-end optimization" << std::endl;

    const size_t n_rows = 20;
    const size_t n_cols = 20;
    const size_t n_cells = n_rows * n_cols;
    const size_t n_features = 4;

    RasterDims dims(n_rows, n_cols, n_features);

    // Generate covariates
    std::vector<std::vector<double>> covariates(n_features);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t j = 0; j < n_features; ++j) {
        covariates[j].resize(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            covariates[j][i] = dist(rng);
        }
    }

    // Generate presence points using known weights
    std::vector<double> true_beta = {0.5, -0.3, 0.8, -0.2};
    std::vector<double> response(n_cells, 0.0);

    // Compute probabilities
    std::vector<double> prob(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        double linear_pred = 0.0;
        for (size_t j = 0; j < n_features; ++j) {
            linear_pred += true_beta[j] * covariates[j][i];
        }
        prob[i] = std::exp(linear_pred);
    }

    // Normalize
    double sum_prob = 0.0;
    for (auto p : prob) sum_prob += p;
    for (auto& p : prob) p /= sum_prob;

    // Sample presence points
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int sample = 0; sample < 50; ++sample) {
        double r = unif(rng);
        double cumsum = 0.0;
        for (size_t i = 0; i < n_cells; ++i) {
            cumsum += prob[i];
            if (r < cumsum) {
                response[i] = 1.0;
                break;
            }
        }
    }

    // Create problem
    auto problem = make_maxent_problem(dims, covariates, response);

    // Optimize
    std::vector<double> beta0(n_features, 0.0);

    LBFGSControl control;
    control.gtol = 1e-4;
    control.max_iter = 200;
    control.trace = false;

    auto result = minimize_lbfgs(problem, beta0, control);

    Rcpp::Rcout << "  Iterations: " << result.iterations << std::endl;
    Rcpp::Rcout << "  Final value: " << result.value << std::endl;
    Rcpp::Rcout << "  Solution: [";
    for (size_t i = 0; i < n_features; ++i) {
        Rcpp::Rcout << result.par[i];
        if (i < n_features - 1) Rcpp::Rcout << ", ";
    }
    Rcpp::Rcout << "]" << std::endl;

    // Compute AUC
    double auc = compute_maxent_auc(result.par.data(), covariates, response, problem.mask);
    Rcpp::Rcout << "  AUC: " << auc << std::endl;

    // Check convergence and reasonable AUC
    bool converged = (result.convergence == 0);
    bool auc_ok = (auc > 0.6);  // Should be better than random
    bool finite_value = std::isfinite(result.value);

    if (converged && auc_ok && finite_value) {
        Rcpp::Rcout << "  Status: PASS" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        return 1;
    }
}

//' @title Test chunked raster processing
//' @description Verify chunking works on large rasters
//' @export
// [[Rcpp::export]]
int test_chunked_processing() {
    Rcpp::Rcout << "Test: Chunked raster processing (large dataset)" << std::endl;

    // Create a moderately large raster (simulating >1M cells case)
    const size_t n_rows = 500;
    const size_t n_cols = 500;
    const size_t n_cells = n_rows * n_cols;  // 250,000 cells
    const size_t n_features = 3;

    Rcpp::Rcout << "  Raster size: " << n_rows << " x " << n_cols
                << " = " << n_cells << " cells" << std::endl;

    RasterDims dims(n_rows, n_cols, n_features);

    // Generate simple covariates
    std::vector<std::vector<double>> covariates(n_features);
    for (size_t j = 0; j < n_features; ++j) {
        covariates[j].resize(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            // Simple gradient pattern
            size_t row = i / n_cols;
            size_t col = i % n_cols;
            covariates[j][i] = (row + col * j) / static_cast<double>(n_rows + n_cols);
        }
    }

    // Create sparse presence data
    std::vector<double> response(n_cells, 0.0);
    std::mt19937 rng(456);
    std::uniform_int_distribution<size_t> cell_dist(0, n_cells - 1);
    for (int i = 0; i < 100; ++i) {
        response[cell_dist(rng)] = 1.0;
    }

    // Create problem with chunking (10k cells per chunk)
    const size_t chunk_size = 10000;
    auto problem = make_maxent_problem(dims, covariates, response, chunk_size);

    Rcpp::Rcout << "  Chunk size: " << chunk_size << std::endl;
    Rcpp::Rcout << "  Number of chunks: " << problem.chunking.n_chunks << std::endl;

    // Quick optimization
    std::vector<double> beta0(n_features, 0.0);

    LBFGSControl control;
    control.gtol = 1e-3;
    control.max_iter = 50;
    control.trace = false;

    auto result = minimize_lbfgs(problem, beta0, control);

    Rcpp::Rcout << "  Iterations: " << result.iterations << std::endl;
    Rcpp::Rcout << "  Final value: " << result.value << std::endl;
    Rcpp::Rcout << "  Convergence: " << result.convergence << std::endl;

    bool finite_value = std::isfinite(result.value);
    bool reasonable_iterations = (result.iterations > 0 && result.iterations <= control.max_iter);

    if (finite_value && reasonable_iterations) {
        Rcpp::Rcout << "  Status: PASS - Chunking works on large dataset" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        return 1;
    }
}

//' @title Test raster processing with NA values
//' @description Verify masking works with chunked processing
//' @export
// [[Rcpp::export]]
int test_chunked_with_na() {
    Rcpp::Rcout << "Test: Chunked processing with NA values" << std::endl;

    const size_t n_rows = 100;
    const size_t n_cols = 100;
    const size_t n_cells = n_rows * n_cols;
    const size_t n_features = 2;

    RasterDims dims(n_rows, n_cols, n_features);

    // Generate covariates with NAs
    std::vector<std::vector<double>> covariates(n_features);
    std::mt19937 rng(789);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    for (size_t j = 0; j < n_features; ++j) {
        covariates[j].resize(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            // 20% NA values
            if (unif(rng) < 0.2) {
                covariates[j][i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                covariates[j][i] = dist(rng);
            }
        }
    }

    // Response with some NAs
    std::vector<double> response(n_cells, 0.0);
    std::uniform_int_distribution<size_t> cell_dist(0, n_cells - 1);
    for (int i = 0; i < 50; ++i) {
        size_t idx = cell_dist(rng);
        if (!std::isnan(covariates[0][idx]) && !std::isnan(covariates[1][idx])) {
            response[idx] = 1.0;
        }
    }

    // Add some NAs to response
    for (size_t i = 0; i < n_cells; ++i) {
        if (unif(rng) < 0.1) {
            response[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Create problem (auto-detects NAs)
    auto problem = make_maxent_problem(dims, covariates, response, 5000);

    size_t n_valid = problem.mask.n_valid();
    Rcpp::Rcout << "  Total cells: " << n_cells << std::endl;
    Rcpp::Rcout << "  Valid cells: " << n_valid << std::endl;
    Rcpp::Rcout << "  Masked cells: " << (n_cells - n_valid) << std::endl;

    // Check that some cells were masked
    bool masking_worked = (n_valid < n_cells);

    // Verify function evaluation works
    std::vector<double> beta(n_features, 0.0);
    double value = problem.value(beta.data());

    bool finite_value = std::isfinite(value);

    if (masking_worked && finite_value) {
        Rcpp::Rcout << "  Status: PASS - NA masking works with chunking" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL" << std::endl;
        return 1;
    }
}
