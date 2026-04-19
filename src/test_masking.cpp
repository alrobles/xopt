// test_masking.cpp - Unit tests for NA/masking functionality
//
// Tests masked reductions and NA handling in raster problems

#include <Rcpp.h>
#include "../inst/include/xopt/raster_problem.hpp"
#include "../inst/include/xopt/ad_reduce.hpp"
#include "../inst/include/xopt/models/logistic_sdm.hpp"
#include <vector>
#include <cmath>
#include <limits>

using namespace xopt;

//' @title Test masked sum reduction
//' @description Test sum_masked function with various masks
//' @export
// [[Rcpp::export]]
int test_masked_sum() {
    Rcpp::Rcout << "Test: Masked sum reduction" << std::endl;

    // Test data
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    // Test 1: All valid
    {
        std::vector<bool> mask(10, true);
        double result = sum_masked(data, mask);
        double expected = 55.0;  // 1 + 2 + ... + 10

        Rcpp::Rcout << "  Test 1 (all valid): result = " << result
                    << ", expected = " << expected << std::endl;

        if (std::abs(result - expected) > 1e-10) {
            Rcpp::Rcout << "  Status: FAIL - Sum mismatch" << std::endl;
            return 1;
        }
    }

    // Test 2: Half masked
    {
        std::vector<bool> mask = {true, false, true, false, true, false, true, false, true, false};
        double result = sum_masked(data, mask);
        double expected = 25.0;  // 1 + 3 + 5 + 7 + 9

        Rcpp::Rcout << "  Test 2 (half masked): result = " << result
                    << ", expected = " << expected << std::endl;

        if (std::abs(result - expected) > 1e-10) {
            Rcpp::Rcout << "  Status: FAIL - Sum mismatch" << std::endl;
            return 1;
        }
    }

    // Test 3: All masked except one
    {
        std::vector<bool> mask(10, false);
        mask[5] = true;
        double result = sum_masked(data, mask);
        double expected = 6.0;

        Rcpp::Rcout << "  Test 3 (single valid): result = " << result
                    << ", expected = " << expected << std::endl;

        if (std::abs(result - expected) > 1e-10) {
            Rcpp::Rcout << "  Status: FAIL - Sum mismatch" << std::endl;
            return 1;
        }
    }

    // Test 4: All masked
    {
        std::vector<bool> mask(10, false);
        double result = sum_masked(data, mask);
        double expected = 0.0;

        Rcpp::Rcout << "  Test 4 (all masked): result = " << result
                    << ", expected = " << expected << std::endl;

        if (std::abs(result - expected) > 1e-10) {
            Rcpp::Rcout << "  Status: FAIL - Sum mismatch" << std::endl;
            return 1;
        }
    }

    Rcpp::Rcout << "  Status: PASS - All masked sum tests passed" << std::endl;
    return 0;
}

//' @title Test raster mask from NA values
//' @description Test automatic mask creation from NA-containing data
//' @export
// [[Rcpp::export]]
int test_raster_mask_na() {
    Rcpp::Rcout << "Test: Raster mask from NA values" << std::endl;

    const size_t n_cells = 10;

    // Create data with some NAs
    std::vector<double> data = {1.0, 2.0, NAN, 4.0, 5.0, NAN, 7.0, 8.0, NAN, 10.0};

    // Create mask from NA values
    RasterMask mask(n_cells, true);
    mask.from_na_values(data);

    // Check mask
    Rcpp::Rcout << "  Mask: [";
    for (size_t i = 0; i < n_cells; ++i) {
        Rcpp::Rcout << (mask.is_valid(i) ? "1" : "0");
        if (i < n_cells - 1) Rcpp::Rcout << ", ";
    }
    Rcpp::Rcout << "]" << std::endl;

    // Expected: indices 2, 5, 8 should be invalid
    bool pass = true;
    if (mask.is_valid(2) || mask.is_valid(5) || mask.is_valid(8)) {
        Rcpp::Rcout << "  ERROR: NA cells not masked" << std::endl;
        pass = false;
    }
    if (!mask.is_valid(0) || !mask.is_valid(1) || !mask.is_valid(3)) {
        Rcpp::Rcout << "  ERROR: Valid cells incorrectly masked" << std::endl;
        pass = false;
    }

    // Check valid count
    size_t n_valid = mask.n_valid();
    Rcpp::Rcout << "  Valid cells: " << n_valid << " (expected 7)" << std::endl;
    if (n_valid != 7) {
        Rcpp::Rcout << "  ERROR: Wrong number of valid cells" << std::endl;
        pass = false;
    }

    if (pass) {
        Rcpp::Rcout << "  Status: PASS - NA mask creation works correctly" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL - Mask creation errors" << std::endl;
        return 1;
    }
}

//' @title Test logistic SDM with NA values
//' @description Test that logistic SDM handles NA values correctly
//' @export
// [[Rcpp::export]]
int test_logistic_sdm_with_na() {
    Rcpp::Rcout << "Test: Logistic SDM with NA values" << std::endl;

    const size_t n_rows = 5;
    const size_t n_cols = 5;
    const size_t n_cells = n_rows * n_cols;
    const size_t n_layers = 2;

    RasterDims dims(n_rows, n_cols, n_layers);

    // Create covariates with some NA values
    std::vector<std::vector<double>> covariates(n_layers);
    for (size_t j = 0; j < n_layers; ++j) {
        covariates[j].resize(n_cells);
        for (size_t i = 0; i < n_cells; ++i) {
            if (i % 7 == 0) {
                covariates[j][i] = NAN;  // Insert some NAs
            } else {
                covariates[j][i] = static_cast<double>(i) / 10.0;
            }
        }
    }

    // Create response with some NA values
    std::vector<double> response(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        if (i % 5 == 0) {
            response[i] = NAN;  // Insert some NAs
        } else {
            response[i] = (i % 2 == 0) ? 1.0 : 0.0;
        }
    }

    // Create problem (should auto-detect NAs)
    auto problem = make_logistic_sdm_problem(dims, covariates, response);

    // Check mask
    size_t n_valid = problem.mask.n_valid();
    Rcpp::Rcout << "  Valid cells after NA detection: " << n_valid
                << " out of " << n_cells << std::endl;

    // Should have fewer valid cells due to NAs
    if (n_valid >= n_cells) {
        Rcpp::Rcout << "  ERROR: NAs not detected" << std::endl;
        return 1;
    }

    // Test that objective can be evaluated
    std::vector<double> beta = {0.0, 0.0, 0.0};
    double nll = 0.0;

    try {
        nll = problem.value(beta.data());
        Rcpp::Rcout << "  Negative log-likelihood: " << nll << std::endl;

        if (!std::isfinite(nll)) {
            Rcpp::Rcout << "  ERROR: Non-finite objective with NAs" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        Rcpp::Rcout << "  ERROR: Exception during evaluation: " << e.what() << std::endl;
        return 1;
    }

    // Test gradient
    std::vector<double> grad(beta.size());
    try {
        problem.gradient(beta.data(), grad.data());
        Rcpp::Rcout << "  Gradient computed successfully" << std::endl;

        for (double g : grad) {
            if (!std::isfinite(g)) {
                Rcpp::Rcout << "  ERROR: Non-finite gradient with NAs" << std::endl;
                return 1;
            }
        }
    } catch (const std::exception& e) {
        Rcpp::Rcout << "  ERROR: Exception during gradient: " << e.what() << std::endl;
        return 1;
    }

    Rcpp::Rcout << "  Status: PASS - NA handling works correctly" << std::endl;
    return 0;
}

//' @title Test mask intersection
//' @description Test combining multiple masks
//' @export
// [[Rcpp::export]]
int test_mask_intersection() {
    Rcpp::Rcout << "Test: Mask intersection" << std::endl;

    const size_t n_cells = 10;

    // Create two masks
    RasterMask mask1(n_cells, true);
    RasterMask mask2(n_cells, true);

    // Mask some cells in each
    mask1.set_invalid(2);
    mask1.set_invalid(5);
    mask1.set_invalid(8);

    mask2.set_invalid(1);
    mask2.set_invalid(5);
    mask2.set_invalid(7);

    // Intersect
    mask1.intersect(mask2);

    // Check result - should have cells 1, 2, 5, 7, 8 invalid
    Rcpp::Rcout << "  Combined mask: [";
    for (size_t i = 0; i < n_cells; ++i) {
        Rcpp::Rcout << (mask1.is_valid(i) ? "1" : "0");
        if (i < n_cells - 1) Rcpp::Rcout << ", ";
    }
    Rcpp::Rcout << "]" << std::endl;

    // Verify
    bool pass = true;
    std::vector<size_t> should_be_invalid = {1, 2, 5, 7, 8};
    for (size_t idx : should_be_invalid) {
        if (mask1.is_valid(idx)) {
            Rcpp::Rcout << "  ERROR: Cell " << idx << " should be invalid" << std::endl;
            pass = false;
        }
    }

    size_t n_valid = mask1.n_valid();
    Rcpp::Rcout << "  Valid cells: " << n_valid << " (expected 5)" << std::endl;
    if (n_valid != 5) {
        Rcpp::Rcout << "  ERROR: Wrong number of valid cells" << std::endl;
        pass = false;
    }

    if (pass) {
        Rcpp::Rcout << "  Status: PASS - Mask intersection works correctly" << std::endl;
        return 0;
    } else {
        Rcpp::Rcout << "  Status: FAIL - Mask intersection errors" << std::endl;
        return 1;
    }
}
