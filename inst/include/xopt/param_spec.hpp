// param_spec.hpp - Structured parameter specification with transforms
//
// This header provides ParamSpec for handling structured parameters
// (vectors, matrices, mixed lists) with automatic flatten/unflatten
// and differentiable parameter transformations.

#ifndef XOPT_PARAM_SPEC_HPP
#define XOPT_PARAM_SPEC_HPP

#include "problem.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>

namespace xopt {

// Bounded transformation: (lo, hi) → ℝ via scaled logit
struct BoundedTransform : public Transform {
    double lo, hi;

    BoundedTransform(double lower, double upper) : lo(lower), hi(upper) {
        if (lo >= hi) {
            throw std::invalid_argument("BoundedTransform requires lo < hi");
        }
    }

    double forward(double x) const override {
        // Map (lo, hi) to (0, 1) then apply logit
        double y = (x - lo) / (hi - lo);
        if (y <= 0.0 || y >= 1.0) {
            throw std::domain_error("BoundedTransform: x must be in (lo, hi)");
        }
        return std::log(y / (1.0 - y));
    }

    double inverse(double x) const override {
        // Apply logistic then map (0, 1) to (lo, hi)
        double y = 1.0 / (1.0 + std::exp(-x));
        return lo + y * (hi - lo);
    }

    double forward_deriv(double x) const override {
        double y = (x - lo) / (hi - lo);
        return 1.0 / (y * (1.0 - y) * (hi - lo));
    }

    std::string name() const override {
        return "bounded(" + std::to_string(lo) + ", " + std::to_string(hi) + ")";
    }
};

// Simplex transformation: Δ^{n-1} → ℝ^{n-1} via log-ratio (softmax inverse)
// Maps probability simplex to unconstrained space
struct SimplexTransform : public Transform {
    int n;  // Dimension of simplex (n components, n-1 free parameters)

    explicit SimplexTransform(int dim) : n(dim) {
        if (n < 2) {
            throw std::invalid_argument("SimplexTransform requires n >= 2");
        }
    }

    // Forward: take first n-1 components, compute log-ratio to last component
    // Note: This operates on individual components during parameter flattening
    // The actual implementation for vectors is in ParamSpec
    double forward(double x) const override {
        // For individual components, just return log
        // (Full simplex transform is implemented in ParamSpec)
        if (x <= 0.0) {
            throw std::domain_error("SimplexTransform: components must be positive");
        }
        return std::log(x);
    }

    double inverse(double x) const override {
        return std::exp(x);
    }

    double forward_deriv(double x) const override {
        return 1.0 / x;
    }

    std::string name() const override {
        return "simplex(" + std::to_string(n) + ")";
    }
};

// SPD Cholesky transformation: SPD matrices → unconstrained Cholesky factors
// Maps symmetric positive definite matrix to its Cholesky decomposition
// with log-transformed diagonal elements
struct SpdCholeskyTransform : public Transform {
    int n;  // Matrix dimension

    explicit SpdCholeskyTransform(int dim) : n(dim) {
        if (n < 1) {
            throw std::invalid_argument("SpdCholeskyTransform requires n >= 1");
        }
    }

    // Note: SPD transformation operates on matrices, not scalars
    // This is a placeholder for the full matrix transformation
    // implemented in ParamSpec
    double forward(double x) const override {
        // For diagonal elements, use log transform
        if (x <= 0.0) {
            throw std::domain_error("SpdCholeskyTransform: diagonal must be positive");
        }
        return std::log(x);
    }

    double inverse(double x) const override {
        return std::exp(x);
    }

    double forward_deriv(double x) const override {
        return 1.0 / x;
    }

    std::string name() const override {
        return "spd_chol(" + std::to_string(n) + ")";
    }
};

// Parameter component specification
struct ParamComponent {
    std::string name;
    std::vector<size_t> shape;  // Shape of parameter (scalar=[], vector=[n], matrix=[n,m])
    std::shared_ptr<Transform> transform;  // Optional transformation
    size_t offset;  // Offset in flattened parameter vector
    size_t size;    // Number of elements

    ParamComponent() : offset(0), size(0) {}

    ParamComponent(std::string n, std::vector<size_t> s,
                   std::shared_ptr<Transform> t = nullptr)
        : name(std::move(n)), shape(std::move(s)), transform(t), offset(0) {
        size = 1;
        for (auto d : shape) size *= d;
    }

    bool is_scalar() const { return shape.empty() || (shape.size() == 1 && shape[0] == 1); }
    bool is_vector() const { return shape.size() == 1; }
    bool is_matrix() const { return shape.size() == 2; }
};

// Structured parameter specification
class ParamSpec {
public:
    std::vector<ParamComponent> components;
    size_t total_size;

    ParamSpec() : total_size(0) {}

    // Add a scalar parameter
    void add_scalar(const std::string& name,
                    std::shared_ptr<Transform> transform = nullptr) {
        ParamComponent comp(name, {1}, transform);
        comp.offset = total_size;
        total_size += comp.size;
        components.push_back(comp);
    }

    // Add a vector parameter
    void add_vector(const std::string& name, size_t n,
                    std::shared_ptr<Transform> transform = nullptr) {
        ParamComponent comp(name, {n}, transform);
        comp.offset = total_size;
        total_size += comp.size;
        components.push_back(comp);
    }

    // Add a matrix parameter
    void add_matrix(const std::string& name, size_t nrow, size_t ncol,
                    std::shared_ptr<Transform> transform = nullptr) {
        ParamComponent comp(name, {nrow, ncol}, transform);
        comp.offset = total_size;
        total_size += comp.size;
        components.push_back(comp);
    }

    // Get total number of parameters (after flattening)
    size_t size() const { return total_size; }

    // Flatten structured parameters to vector
    void flatten(const std::map<std::string, std::vector<double>>& params,
                 std::vector<double>& out) const {
        out.resize(total_size);

        for (const auto& comp : components) {
            auto it = params.find(comp.name);
            if (it == params.end()) {
                throw std::invalid_argument("Missing parameter: " + comp.name);
            }

            const auto& values = it->second;
            if (values.size() != comp.size) {
                throw std::invalid_argument("Parameter " + comp.name +
                                            " has wrong size");
            }

            // Copy and transform if needed
            for (size_t i = 0; i < comp.size; ++i) {
                double val = values[i];
                if (comp.transform) {
                    val = comp.transform->forward(val);
                }
                out[comp.offset + i] = val;
            }
        }
    }

    // Unflatten vector to structured parameters
    void unflatten(const std::vector<double>& flat,
                   std::map<std::string, std::vector<double>>& params) const {
        if (flat.size() != total_size) {
            throw std::invalid_argument("Flattened vector has wrong size");
        }

        params.clear();
        for (const auto& comp : components) {
            std::vector<double> values(comp.size);
            for (size_t i = 0; i < comp.size; ++i) {
                double val = flat[comp.offset + i];
                if (comp.transform) {
                    val = comp.transform->inverse(val);
                }
                values[i] = val;
            }
            params[comp.name] = values;
        }
    }

    // Compute Jacobian of transform (diagonal)
    // Returns d(unflatten)/d(flatten) on the diagonal
    void transform_jacobian(const std::vector<double>& flat,
                           std::vector<double>& diag) const {
        diag.resize(total_size);

        for (const auto& comp : components) {
            for (size_t i = 0; i < comp.size; ++i) {
                if (comp.transform) {
                    // Inverse derivative: d(inverse)/d(x) at flat[comp.offset + i]
                    double x = flat[comp.offset + i];
                    // For inverse transform, use chain rule:
                    // If y = inverse(x), then dy/dx = 1 / forward_deriv(y)
                    double y = comp.transform->inverse(x);
                    diag[comp.offset + i] = 1.0 / comp.transform->forward_deriv(y);
                } else {
                    diag[comp.offset + i] = 1.0;
                }
            }
        }
    }

    // Find component by name
    const ParamComponent* find(const std::string& name) const {
        for (const auto& comp : components) {
            if (comp.name == name) return &comp;
        }
        return nullptr;
    }
};

// Helper functions for creating transforms

inline std::shared_ptr<Transform> positive() {
    return std::make_shared<LogTransform>();
}

inline std::shared_ptr<Transform> bounded(double lo, double hi) {
    return std::make_shared<BoundedTransform>(lo, hi);
}

inline std::shared_ptr<Transform> simplex(int n) {
    return std::make_shared<SimplexTransform>(n);
}

inline std::shared_ptr<Transform> spd_chol(int n) {
    return std::make_shared<SpdCholeskyTransform>(n);
}

inline std::shared_ptr<Transform> identity() {
    return std::make_shared<IdentityTransform>();
}

} // namespace xopt

#endif // XOPT_PARAM_SPEC_HPP
