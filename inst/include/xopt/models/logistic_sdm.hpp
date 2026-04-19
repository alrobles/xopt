// logistic_sdm.hpp - Logistic Species Distribution Model
//
// Implements a logistic regression model for species presence/absence data
// with support for raster covariate stacks and masked/NA-aware reductions.
// This serves as a reference implementation and test case for the raster
// optimization framework.

#ifndef XOPT_MODELS_LOGISTIC_SDM_HPP
#define XOPT_MODELS_LOGISTIC_SDM_HPP

#include "../problem.hpp"
#include "../raster_problem.hpp"
#include "../ad_reduce.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace xopt {
namespace models {

// Logistic SDM objective function
// Parameters: beta = [intercept, coef_1, coef_2, ..., coef_p]
// Model: P(presence | covariates) = logistic(intercept + sum(beta_i * covariate_i))
template <typename Scalar = double>
class LogisticSDM {
public:
    // Constructor
    LogisticSDM() = default;

    // Evaluate negative log-likelihood (for minimization)
    // x: parameter vector [intercept, beta_1, ..., beta_p]
    // covariates: [n_layers][n_cells] - covariate raster stack
    // response: [n_cells] - presence/absence (1/0) or binary response
    // mask: valid cells indicator
    Scalar value(const Scalar* x,
                 const std::vector<std::vector<Scalar>>& covariates,
                 const std::vector<Scalar>& response,
                 const RasterMask& mask) const {

        size_t n_cells = response.size();
        size_t n_layers = covariates.size();

        if (n_cells == 0 || n_layers == 0) {
            throw std::invalid_argument("Empty data");
        }

        Scalar nll = 0.0;  // Negative log-likelihood
        size_t n_valid = 0;

        // Loop over cells
        for (size_t i = 0; i < n_cells; ++i) {
            if (!mask.is_valid(i)) {
                continue;  // Skip masked cells
            }

            // Compute linear predictor: intercept + sum(beta * covariate)
            Scalar linear_pred = x[0];  // Intercept
            for (size_t j = 0; j < n_layers; ++j) {
                linear_pred += x[j + 1] * covariates[j][i];
            }

            // Compute probability via logistic function
            Scalar prob = logistic(linear_pred);

            // Add log-likelihood contribution
            // L = y * log(p) + (1 - y) * log(1 - p)
            Scalar y = response[i];

            // Numerical stability: clamp probability away from 0 and 1
            constexpr Scalar eps = 1e-10;
            prob = std::max(eps, std::min(Scalar(1) - eps, prob));

            if (y > Scalar(0.5)) {
                // Presence
                nll -= std::log(prob);
            } else {
                // Absence
                nll -= std::log(Scalar(1) - prob);
            }

            ++n_valid;
        }

        if (n_valid == 0) {
            throw std::runtime_error("No valid cells for likelihood evaluation");
        }

        return nll;
    }

    // Compute gradient using autodiff or finite differences
    // For now, this is a placeholder - gradient computation will be done via XAD
    void gradient(const Scalar* x,
                  Scalar* g,
                  const std::vector<std::vector<Scalar>>& covariates,
                  const std::vector<Scalar>& response,
                  const RasterMask& mask) const {

        // This would be implemented using XAD for automatic differentiation
        // For now, we provide analytical gradient as reference

        size_t n_cells = response.size();
        size_t n_layers = covariates.size();
        size_t n_par = n_layers + 1;  // intercept + coefficients

        // Initialize gradient to zero
        for (size_t j = 0; j < n_par; ++j) {
            g[j] = 0.0;
        }

        // Compute gradient analytically
        for (size_t i = 0; i < n_cells; ++i) {
            if (!mask.is_valid(i)) {
                continue;
            }

            // Linear predictor
            Scalar linear_pred = x[0];
            for (size_t j = 0; j < n_layers; ++j) {
                linear_pred += x[j + 1] * covariates[j][i];
            }

            // Probability
            Scalar prob = logistic(linear_pred);

            // Residual: predicted - observed
            Scalar residual = prob - response[i];

            // Gradient of intercept
            g[0] += residual;

            // Gradient of coefficients
            for (size_t j = 0; j < n_layers; ++j) {
                g[j + 1] += residual * covariates[j][i];
            }
        }
    }

    // Compute Hessian (not implemented - would use BFGS approximation)
    void hessian(const Scalar* x,
                 Scalar* H,
                 const std::vector<std::vector<Scalar>>& covariates,
                 const std::vector<Scalar>& response,
                 const RasterMask& mask) const {
        (void)x; (void)H; (void)covariates; (void)response; (void)mask;
        throw std::runtime_error("Exact Hessian not implemented for LogisticSDM");
    }

    // Get number of parameters for given number of covariates
    static int n_parameters(size_t n_covariates) {
        return static_cast<int>(n_covariates + 1);  // intercept + coefficients
    }
};

// Helper function to create a LogisticSDM RasterProblem
template <typename Scalar = double>
RasterProblem<LogisticSDM<Scalar>, GradKind::UserFn, HessKind::BfgsApprox, Scalar>
make_logistic_sdm_problem(
    RasterDims dims,
    std::vector<std::vector<Scalar>> covariates,
    std::vector<Scalar> response,
    size_t chunk_size = 10000) {

    int n_par = LogisticSDM<Scalar>::n_parameters(covariates.size());

    return RasterProblem<LogisticSDM<Scalar>, GradKind::UserFn, HessKind::BfgsApprox, Scalar>(
        n_par,
        dims,
        std::move(covariates),
        std::move(response),
        LogisticSDM<Scalar>(),
        chunk_size
    );
}

// Predict probabilities for new data
template <typename Scalar = double>
std::vector<Scalar> predict_logistic_sdm(
    const Scalar* beta,
    const std::vector<std::vector<Scalar>>& covariates,
    const RasterMask& mask) {

    size_t n_cells = covariates[0].size();
    size_t n_layers = covariates.size();
    std::vector<Scalar> predictions(n_cells, 0.0);

    for (size_t i = 0; i < n_cells; ++i) {
        if (!mask.is_valid(i)) {
            predictions[i] = std::numeric_limits<Scalar>::quiet_NaN();
            continue;
        }

        // Linear predictor
        Scalar linear_pred = beta[0];  // Intercept
        for (size_t j = 0; j < n_layers; ++j) {
            linear_pred += beta[j + 1] * covariates[j][i];
        }

        // Probability
        predictions[i] = logistic(linear_pred);
    }

    return predictions;
}

// Compute deviance (goodness of fit measure)
template <typename Scalar = double>
Scalar compute_deviance(
    const Scalar* beta,
    const std::vector<std::vector<Scalar>>& covariates,
    const std::vector<Scalar>& response,
    const RasterMask& mask) {

    auto predictions = predict_logistic_sdm(beta, covariates, mask);

    Scalar deviance = 0.0;
    size_t n_valid = 0;

    for (size_t i = 0; i < response.size(); ++i) {
        if (!mask.is_valid(i)) {
            continue;
        }

        Scalar y = response[i];
        Scalar p = predictions[i];

        // Clamp probability
        constexpr Scalar eps = 1e-10;
        p = std::max(eps, std::min(Scalar(1) - eps, p));

        // Deviance contribution
        if (y > Scalar(0.5)) {
            deviance -= Scalar(2) * std::log(p);
        } else {
            deviance -= Scalar(2) * std::log(Scalar(1) - p);
        }

        ++n_valid;
    }

    return deviance;
}

// Compute AUC (Area Under ROC Curve) for model evaluation
template <typename Scalar = double>
Scalar compute_auc(
    const Scalar* beta,
    const std::vector<std::vector<Scalar>>& covariates,
    const std::vector<Scalar>& response,
    const RasterMask& mask) {

    auto predictions = predict_logistic_sdm(beta, covariates, mask);

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

    // Compute AUC using trapezoidal rule
    size_t n_pos = 0, n_neg = 0;
    for (const auto& p : pairs) {
        if (p.second) ++n_pos;
        else ++n_neg;
    }

    if (n_pos == 0 || n_neg == 0) {
        return Scalar(0.5);  // No discrimination possible
    }

    Scalar auc = 0.0;
    size_t tp = 0, fp = 0;

    for (const auto& p : pairs) {
        if (p.second) {
            ++tp;
        } else {
            auc += static_cast<Scalar>(tp);
            ++fp;
        }
    }

    auc /= static_cast<Scalar>(n_pos * n_neg);
    return auc;
}

} // namespace models
} // namespace xopt

#endif // XOPT_MODELS_LOGISTIC_SDM_HPP
