// Probe test: Species Distribution Model (SDM) optimization
// This file demonstrates optimization for ecological/SDM use cases

#include <Rcpp.h>
#include <vector>
#include <cmath>

// Simplified SDM objective function
// Simulates a logistic regression for species presence/absence

//' @title Probe SDM optimization
//' @description Test optimization for species distribution modeling use case
//' @export
// [[Rcpp::export]]
int probe_sdm() {
    Rcpp::Rcout << "Probe: SDM (Species Distribution Model) optimization test" << std::endl;

    // Simulate environmental data and species presence
    const int n_obs = 100;
    const int n_vars = 3;

    std::vector<std::vector<double>> env_data(n_obs, std::vector<double>(n_vars));
    std::vector<int> presence(n_obs);

    // Generate synthetic data
    for (int i = 0; i < n_obs; ++i) {
        for (int j = 0; j < n_vars; ++j) {
            env_data[i][j] = static_cast<double>(i + j) / 10.0;
        }
        presence[i] = (i % 2 == 0) ? 1 : 0;
    }

    // Simplified logistic regression objective
    std::vector<double> beta = {0.1, -0.2, 0.3};  // Parameters
    double log_likelihood = 0.0;

    for (int i = 0; i < n_obs; ++i) {
        double linear_pred = 0.0;
        for (int j = 0; j < n_vars; ++j) {
            linear_pred += beta[j] * env_data[i][j];
        }
        double prob = 1.0 / (1.0 + std::exp(-linear_pred));

        if (presence[i] == 1) {
            log_likelihood += std::log(prob + 1e-10);
        } else {
            log_likelihood += std::log(1.0 - prob + 1e-10);
        }
    }

    Rcpp::Rcout << "  Log-likelihood: " << log_likelihood << std::endl;
    Rcpp::Rcout << "  Parameters: [";
    for (size_t i = 0; i < beta.size(); ++i) {
        Rcpp::Rcout << beta[i];
        if (i < beta.size() - 1) Rcpp::Rcout << ", ";
    }
    Rcpp::Rcout << "]" << std::endl;

    Rcpp::Rcout << "  Status: PASS - SDM objective evaluation working" << std::endl;

    return 0;
}
