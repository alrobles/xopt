// maxent.hpp - Maximum Entropy (MaxEnt) Species Distribution Model
//
// Implements a MaxEnt model for species presence-only data, based on
// a Poisson point process formulation. This is a reference implementation
// for testing L-BFGS solvers on ecological applications.
//
// Reference: Phillips, Anderson & Schapire (2006), "Maximum entropy modeling
// of species geographic distributions", Ecological Modelling.

#ifndef XOPT_MODELS_MAXENT_HPP
#define XOPT_MODELS_MAXENT_HPP

#include "../problem.hpp"
#include "../raster_problem.hpp"
#include "../ad_reduce.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace xopt {
namespace models {

// MaxEnt objective function
// Implements a Poisson point process model for presence-only data
// Parameters: beta = [coef_1, coef_2, ..., coef_p] (no intercept in standard MaxEnt)
//
// The model maximizes:
//   L = (1/n) Σ_presence [Σ_i β_i f_i(x)] - log(Σ_background exp(Σ_i β_i f_i(x)))
//
// Where f_i are features (transformations of covariates), presence points are
// where species was observed, and background is a sample of available habitat.
template <typename Scalar = double>
class MaxEnt {
public:
    MaxEnt() = default;

    // Evaluate negative log-likelihood (for minimization)
    // x: parameter vector [beta_1, ..., beta_p]
    // covariates: [n_layers][n_cells] - environmental covariates
    // response: [n_cells] - presence (1) or background (0) indicator
    // mask: valid cells indicator
    Scalar value(const Scalar* x,
                 const std::vector<std::vector<Scalar>>& covariates,
                 const std::vector<Scalar>& response,
                 const RasterMask& mask) const {

        size_t n_cells = response.size();
        size_t n_features = covariates.size();

        if (n_cells == 0 || n_features == 0) {
            throw std::invalid_argument("Empty data");
        }

        // Compute linear predictor for all cells
        std::vector<Scalar> linear_pred(n_cells, 0.0);
        for (size_t i = 0; i < n_cells; ++i) {
            if (!mask.is_valid(i)) {
                continue;
            }

            for (size_t j = 0; j < n_features; ++j) {
                linear_pred[i] += x[j] * covariates[j][i];
            }
        }

        // Compute log partition function: log(Σ exp(linear_pred))
        // Use log-sum-exp trick for numerical stability
        Scalar max_pred = -std::numeric_limits<Scalar>::infinity();
        for (size_t i = 0; i < n_cells; ++i) {
            if (mask.is_valid(i)) {
                max_pred = std::max(max_pred, linear_pred[i]);
            }
        }

        Scalar sum_exp = 0.0;
        size_t n_valid = 0;
        for (size_t i = 0; i < n_cells; ++i) {
            if (mask.is_valid(i)) {
                sum_exp += std::exp(linear_pred[i] - max_pred);
                ++n_valid;
            }
        }

        Scalar log_partition = max_pred + std::log(sum_exp);

        // Compute average linear predictor at presence points
        Scalar sum_presence = 0.0;
        size_t n_presence = 0;
        for (size_t i = 0; i < n_cells; ++i) {
            if (mask.is_valid(i) && response[i] > Scalar(0.5)) {
                sum_presence += linear_pred[i];
                ++n_presence;
            }
        }

        if (n_presence == 0) {
            throw std::runtime_error("No presence points found");
        }

        Scalar avg_presence = sum_presence / static_cast<Scalar>(n_presence);

        // Negative log-likelihood (we minimize this)
        // MaxEnt maximizes: avg_presence - log_partition
        // So we minimize: -avg_presence + log_partition
        Scalar nll = -avg_presence + log_partition;

        return nll;
    }

    // Compute gradient analytically
    void gradient(const Scalar* x,
                  Scalar* g,
                  const std::vector<std::vector<Scalar>>& covariates,
                  const std::vector<Scalar>& response,
                  const RasterMask& mask) const {

        size_t n_cells = response.size();
        size_t n_features = covariates.size();

        // Initialize gradient to zero
        for (size_t j = 0; j < n_features; ++j) {
            g[j] = 0.0;
        }

        // Compute linear predictor and probabilities
        std::vector<Scalar> linear_pred(n_cells, 0.0);
        for (size_t i = 0; i < n_cells; ++i) {
            if (!mask.is_valid(i)) {
                continue;
            }
            for (size_t j = 0; j < n_features; ++j) {
                linear_pred[i] += x[j] * covariates[j][i];
            }
        }

        // Compute probabilities (normalized exponentials)
        Scalar max_pred = -std::numeric_limits<Scalar>::infinity();
        for (size_t i = 0; i < n_cells; ++i) {
            if (mask.is_valid(i)) {
                max_pred = std::max(max_pred, linear_pred[i]);
            }
        }

        std::vector<Scalar> prob(n_cells, 0.0);
        Scalar sum_exp = 0.0;
        for (size_t i = 0; i < n_cells; ++i) {
            if (mask.is_valid(i)) {
                prob[i] = std::exp(linear_pred[i] - max_pred);
                sum_exp += prob[i];
            }
        }

        // Normalize probabilities
        for (size_t i = 0; i < n_cells; ++i) {
            if (mask.is_valid(i)) {
                prob[i] /= sum_exp;
            }
        }

        // Count presence points
        size_t n_presence = 0;
        for (size_t i = 0; i < n_cells; ++i) {
            if (mask.is_valid(i) && response[i] > Scalar(0.5)) {
                ++n_presence;
            }
        }

        // Gradient: E_background[f] - E_presence[f]
        // Where E_background is expectation under exp(linear_pred) distribution
        // and E_presence is empirical average over presence points
        for (size_t j = 0; j < n_features; ++j) {
            // Background expectation (weighted by probabilities)
            Scalar background_mean = 0.0;
            for (size_t i = 0; i < n_cells; ++i) {
                if (mask.is_valid(i)) {
                    background_mean += prob[i] * covariates[j][i];
                }
            }

            // Presence mean
            Scalar presence_mean = 0.0;
            for (size_t i = 0; i < n_cells; ++i) {
                if (mask.is_valid(i) && response[i] > Scalar(0.5)) {
                    presence_mean += covariates[j][i];
                }
            }
            presence_mean /= static_cast<Scalar>(n_presence);

            // Gradient component
            g[j] = background_mean - presence_mean;
        }
    }

    // Hessian not implemented (use L-BFGS approximation)
    void hessian(const Scalar* x,
                 Scalar* H,
                 const std::vector<std::vector<Scalar>>& covariates,
                 const std::vector<Scalar>& response,
                 const RasterMask& mask) const {
        (void)x; (void)H; (void)covariates; (void)response; (void)mask;
        throw std::runtime_error("Exact Hessian not implemented for MaxEnt");
    }

    // Get number of parameters (one coefficient per feature)
    static int n_parameters(size_t n_features) {
        return static_cast<int>(n_features);
    }
};

// Helper to create MaxEnt problem
template <typename Scalar = double>
RasterProblem<MaxEnt<Scalar>, GradKind::UserFn, HessKind::LbfgsApprox, Scalar>
make_maxent_problem(
    RasterDims dims,
    std::vector<std::vector<Scalar>> covariates,
    std::vector<Scalar> response,
    size_t chunk_size = 10000) {

    int n_par = MaxEnt<Scalar>::n_parameters(covariates.size());

    return RasterProblem<MaxEnt<Scalar>, GradKind::UserFn, HessKind::LbfgsApprox, Scalar>(
        n_par,
        dims,
        std::move(covariates),
        std::move(response),
        MaxEnt<Scalar>(),
        chunk_size
    );
}

// Predict relative probability of occurrence
template <typename Scalar = double>
std::vector<Scalar> predict_maxent(
    const Scalar* beta,
    const std::vector<std::vector<Scalar>>& covariates,
    const RasterMask& mask) {

    size_t n_cells = covariates[0].size();
    size_t n_features = covariates.size();
    std::vector<Scalar> predictions(n_cells, 0.0);

    // Compute raw predictions (linear predictor)
    for (size_t i = 0; i < n_cells; ++i) {
        if (!mask.is_valid(i)) {
            predictions[i] = std::numeric_limits<Scalar>::quiet_NaN();
            continue;
        }

        Scalar linear_pred = 0.0;
        for (size_t j = 0; j < n_features; ++j) {
            linear_pred += beta[j] * covariates[j][i];
        }
        predictions[i] = linear_pred;
    }

    // Convert to probabilities via exponential and normalize
    Scalar max_pred = -std::numeric_limits<Scalar>::infinity();
    for (size_t i = 0; i < n_cells; ++i) {
        if (mask.is_valid(i)) {
            max_pred = std::max(max_pred, predictions[i]);
        }
    }

    Scalar sum_exp = 0.0;
    for (size_t i = 0; i < n_cells; ++i) {
        if (mask.is_valid(i)) {
            predictions[i] = std::exp(predictions[i] - max_pred);
            sum_exp += predictions[i];
        }
    }

    // Normalize to get probability distribution
    for (size_t i = 0; i < n_cells; ++i) {
        if (mask.is_valid(i)) {
            predictions[i] /= sum_exp;
        }
    }

    return predictions;
}

// Compute feature importance (contribution to model)
template <typename Scalar = double>
std::vector<Scalar> compute_feature_importance(
    const Scalar* beta,
    const std::vector<std::vector<Scalar>>& covariates,
    const RasterMask& mask) {

    size_t n_features = covariates.size();
    std::vector<Scalar> importance(n_features, 0.0);

    // Compute predictions
    auto predictions = predict_maxent(beta, covariates, mask);

    // For each feature, compute weighted variance
    for (size_t j = 0; j < n_features; ++j) {
        Scalar weighted_mean = 0.0;
        Scalar weighted_sq_mean = 0.0;

        for (size_t i = 0; i < covariates[j].size(); ++i) {
            if (mask.is_valid(i)) {
                Scalar val = covariates[j][i];
                Scalar prob = predictions[i];
                weighted_mean += prob * val;
                weighted_sq_mean += prob * val * val;
            }
        }

        // Importance is the weighted variance times |coefficient|
        Scalar variance = weighted_sq_mean - weighted_mean * weighted_mean;
        importance[j] = std::abs(beta[j]) * variance;
    }

    return importance;
}

// Compute AUC for MaxEnt model evaluation
template <typename Scalar = double>
Scalar compute_maxent_auc(
    const Scalar* beta,
    const std::vector<std::vector<Scalar>>& covariates,
    const std::vector<Scalar>& response,
    const RasterMask& mask) {

    auto predictions = predict_maxent(beta, covariates, mask);

    // Collect valid predictions and responses
    std::vector<std::pair<Scalar, bool>> pairs;
    for (size_t i = 0; i < response.size(); ++i) {
        if (mask.is_valid(i)) {
            pairs.emplace_back(predictions[i], response[i] > Scalar(0.5));
        }
    }

    if (pairs.empty()) {
        return Scalar(0);
    }

    // Sort by prediction descending
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Compute AUC
    size_t n_pos = 0, n_neg = 0;
    for (const auto& p : pairs) {
        if (p.second) ++n_pos;
        else ++n_neg;
    }

    if (n_pos == 0 || n_neg == 0) {
        return Scalar(0.5);
    }

    Scalar auc = 0.0;
    size_t tp = 0;

    for (const auto& p : pairs) {
        if (p.second) {
            ++tp;
        } else {
            auc += static_cast<Scalar>(tp);
        }
    }

    auc /= static_cast<Scalar>(n_pos * n_neg);
    return auc;
}

} // namespace models
} // namespace xopt

#endif // XOPT_MODELS_MAXENT_HPP
