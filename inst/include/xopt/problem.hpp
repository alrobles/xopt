// problem.hpp - Generic optimization problem interface
//
// This header defines the core abstractions for xopt's optimization framework.
// It provides a minimal but extensible Problem trait for vector/tensor optimization.

#ifndef XOPT_PROBLEM_HPP
#define XOPT_PROBLEM_HPP

#include <vector>
#include <array>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <cmath>
#include <type_traits>

namespace xopt {

// Gradient computation policies
enum class GradKind {
    XadAdj,       // Automatic differentiation using XAD adjoint mode
    XadFwd,       // Automatic differentiation using XAD forward mode
    UserFn,       // User-provided gradient function
    FiniteDiff,   // Finite differences
    None          // No gradient available
};

// Hessian computation policies
enum class HessKind {
    XadFwdAdj,    // Exact Hessian via forward-over-adjoint
    BfgsApprox,   // BFGS quasi-Newton approximation
    LbfgsApprox,  // L-BFGS limited-memory approximation
    UserFn,       // User-provided Hessian function
    None          // No Hessian available
};

// Base problem interface
// This is a minimal trait that all problems must satisfy
template <typename Scalar = double>
struct ProblemBase {
    int n_par;  // Number of parameters

    explicit ProblemBase(int n) : n_par(n) {}
    virtual ~ProblemBase() = default;

    // Parameter bounds (optional)
    std::vector<Scalar> lower;
    std::vector<Scalar> upper;

    // Check if bounds are active
    bool has_bounds() const {
        return !lower.empty() || !upper.empty();
    }

    // Initialize bounds to unbounded
    void set_unbounded() {
        lower.clear();
        upper.clear();
    }

    // Set box constraints
    void set_bounds(const std::vector<Scalar>& lb, const std::vector<Scalar>& ub) {
        if (lb.size() != static_cast<size_t>(n_par) ||
            ub.size() != static_cast<size_t>(n_par)) {
            throw std::invalid_argument("Bounds size must match n_par");
        }
        lower = lb;
        upper = ub;
    }
};

// Generic Problem abstraction with gradient and Hessian policies
template <typename UserObj,
          GradKind Grad = GradKind::XadAdj,
          HessKind Hess = HessKind::BfgsApprox,
          typename Scalar = double>
struct Problem : public ProblemBase<Scalar> {
    UserObj obj;  // User's objective function object

    explicit Problem(int n, UserObj&& user_obj)
        : ProblemBase<Scalar>(n), obj(std::forward<UserObj>(user_obj)) {}

    // Evaluate objective value at x
    Scalar value(const Scalar* x) const {
        return obj.value(x);
    }

    // Compute gradient at x (if available)
    void gradient(const Scalar* x, Scalar* g) const {
        if constexpr (Grad == GradKind::None) {
            throw std::runtime_error("Gradient not available for this problem");
        } else {
            obj.gradient(x, g);
        }
    }

    // Compute Hessian at x (if available)
    void hessian(const Scalar* x, Scalar* H) const {
        if constexpr (Hess == HessKind::None) {
            throw std::runtime_error("Hessian not available for this problem");
        } else {
            obj.hessian(x, H);
        }
    }

    // Compute Hessian-vector product at x in direction v (if available)
    // Fallback: dense Hessian-vector multiply when only hessian() is provided
    void hessian_vector_product(const Scalar* x, const Scalar* v, Scalar* hv) const {
        if constexpr (Hess == HessKind::None) {
            throw std::runtime_error("Hessian-vector product not available for this problem");
        } else if constexpr (requires(const UserObj& o, const Scalar* px,
                                      const Scalar* pv, Scalar* phv) { o.hessian_vector_product(px, pv, phv); }) {
            obj.hessian_vector_product(x, v, hv);
        } else if constexpr (requires(const UserObj& o, const Scalar* px, Scalar* pH) { o.hessian(px, pH); }) {
            std::vector<Scalar> H(static_cast<size_t>(this->n_par * this->n_par));
            obj.hessian(x, H.data());
            for (int i = 0; i < this->n_par; ++i) {
                hv[i] = 0;
                for (int j = 0; j < this->n_par; ++j) {
                    hv[i] += H[i * this->n_par + j] * v[j];
                }
            }
        } else {
            throw std::runtime_error("Hessian-vector product not implemented by objective");
        }
    }

    // Query gradient availability
    static constexpr bool has_gradient() {
        return Grad != GradKind::None;
    }

    // Query Hessian availability
    static constexpr bool has_hessian() {
        return Hess != HessKind::None;
    }

    // Query HVP availability
    static constexpr bool has_hvp() {
        return Hess != HessKind::None;
    }

    // Query gradient kind
    static constexpr GradKind gradient_kind() {
        return Grad;
    }

    // Query Hessian kind
    static constexpr HessKind hessian_kind() {
        return Hess;
    }
};

// Tensor shape metadata for problems that operate on shaped parameters
// (e.g., matrix parameters, raster stacks)
struct TensorShape {
    std::vector<size_t> dims;

    TensorShape() = default;

    explicit TensorShape(std::vector<size_t> d) : dims(std::move(d)) {}

    // Total number of elements
    size_t size() const {
        size_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }

    // Number of dimensions
    size_t ndim() const {
        return dims.size();
    }

    // Check if shape is scalar (empty or 1D with size 1)
    bool is_scalar() const {
        return dims.empty() || (dims.size() == 1 && dims[0] == 1);
    }

    // Check if shape is vector (1D)
    bool is_vector() const {
        return dims.size() == 1;
    }

    // Check if shape is matrix (2D)
    bool is_matrix() const {
        return dims.size() == 2;
    }
};

// Problem with tensor-shaped parameters
template <typename UserObj,
          GradKind Grad = GradKind::XadAdj,
          HessKind Hess = HessKind::BfgsApprox,
          typename Scalar = double>
struct TensorProblem : public Problem<UserObj, Grad, Hess, Scalar> {
    TensorShape par_shape;  // Parameter tensor shape

    TensorProblem(TensorShape shape, UserObj&& user_obj)
        : Problem<UserObj, Grad, Hess, Scalar>(
            static_cast<int>(shape.size()),
            std::forward<UserObj>(user_obj)),
          par_shape(std::move(shape)) {}

    // Get parameter shape
    const TensorShape& shape() const {
        return par_shape;
    }
};

// Parameter transformation interface
// Allows transforming constrained parameters to unconstrained space
struct Transform {
    virtual ~Transform() = default;

    // Transform from constrained to unconstrained space
    virtual double forward(double x) const = 0;

    // Transform from unconstrained to constrained space
    virtual double inverse(double x) const = 0;

    // Derivative of forward transformation
    virtual double forward_deriv(double x) const = 0;

    // Name of transformation
    virtual std::string name() const = 0;
};

// Log transformation: (0, ∞) → ℝ
struct LogTransform : public Transform {
    double forward(double x) const override {
        return std::log(x);
    }

    double inverse(double x) const override {
        return std::exp(x);
    }

    double forward_deriv(double x) const override {
        return 1.0 / x;
    }

    std::string name() const override {
        return "log";
    }
};

// Logit transformation: (0, 1) → ℝ
struct LogitTransform : public Transform {
    double forward(double x) const override {
        return std::log(x / (1.0 - x));
    }

    double inverse(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double forward_deriv(double x) const override {
        return 1.0 / (x * (1.0 - x));
    }

    std::string name() const override {
        return "logit";
    }
};

// Identity transformation: ℝ → ℝ
struct IdentityTransform : public Transform {
    double forward(double x) const override {
        return x;
    }

    double inverse(double x) const override {
        return x;
    }

    double forward_deriv(double x) const override {
        (void)x;  // Suppress unused parameter warning
        return 1.0;
    }

    std::string name() const override {
        return "identity";
    }
};

} // namespace xopt

#endif // XOPT_PROBLEM_HPP
