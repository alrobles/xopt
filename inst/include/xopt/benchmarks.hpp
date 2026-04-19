// benchmarks.hpp - Classical optimization benchmark functions
//
// This header provides standard test problems from the optimization literature.
// All benchmarks include analytical gradients for verification purposes.
//
// References:
// - Moré, Garbow, Hillstrom (1981): "Testing Unconstrained Optimization Software"
// - CUTEst test problem collection

#ifndef XOPT_BENCHMARKS_HPP
#define XOPT_BENCHMARKS_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

namespace xopt {
namespace benchmarks {

// Base benchmark problem interface
struct Benchmark {
    virtual ~Benchmark() = default;
    virtual std::string name() const = 0;
    virtual int dimension() const = 0;
    virtual std::vector<double> initial_point() const = 0;
    virtual double value(const double* x) const = 0;
    virtual void gradient(const double* x, double* g) const = 0;
    virtual double optimal_value() const { return 0.0; }
    virtual std::vector<double> optimal_point() const = 0;
};

// Rosenbrock function (banana valley)
// f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
// Global minimum: f(1,1,...,1) = 0
class Rosenbrock : public Benchmark {
    int n_;
public:
    explicit Rosenbrock(int n = 2) : n_(n) {
        if (n < 2) throw std::invalid_argument("Rosenbrock: n must be >= 2");
    }

    std::string name() const override { return "Rosenbrock"; }
    int dimension() const override { return n_; }

    std::vector<double> initial_point() const override {
        std::vector<double> x(n_);
        x[0] = -1.2;
        for (int i = 1; i < n_; ++i) x[i] = 1.0;
        return x;
    }

    double value(const double* x) const override {
        double f = 0.0;
        for (int i = 0; i < n_ - 1; ++i) {
            double t1 = x[i + 1] - x[i] * x[i];
            double t2 = 1.0 - x[i];
            f += 100.0 * t1 * t1 + t2 * t2;
        }
        return f;
    }

    void gradient(const double* x, double* g) const override {
        for (int i = 0; i < n_; ++i) g[i] = 0.0;

        for (int i = 0; i < n_ - 1; ++i) {
            double t1 = x[i + 1] - x[i] * x[i];
            double t2 = 1.0 - x[i];
            g[i] += -400.0 * x[i] * t1 - 2.0 * t2;
            g[i + 1] += 200.0 * t1;
        }
    }

    std::vector<double> optimal_point() const override {
        return std::vector<double>(n_, 1.0);
    }
};

// Quadratic function (elliptic paraboloid)
// f(x) = 0.5 * x^T A x + b^T x + c
// For SPD matrix A, global minimum at x* = -A^{-1}b
class Quadratic : public Benchmark {
    int n_;
    std::vector<double> A_;  // stored as n x n row-major
    std::vector<double> b_;
    double c_;
    std::vector<double> x_opt_;
    double f_opt_;

public:
    Quadratic(int n, const std::vector<double>& A,
              const std::vector<double>& b, double c = 0.0)
        : n_(n), A_(A), b_(b), c_(c) {
        if (A.size() != static_cast<size_t>(n * n))
            throw std::invalid_argument("A must be n x n");
        if (b.size() != static_cast<size_t>(n))
            throw std::invalid_argument("b must have length n");

        // Compute optimal point: x* = -A^{-1}b
        // For simplicity, assume A is diagonal or identity for benchmarks
        x_opt_.resize(n);
        for (int i = 0; i < n; ++i) {
            x_opt_[i] = -b[i] / A[i * n + i];
        }
        f_opt_ = value(x_opt_.data());
    }

    // Simple quadratic with identity Hessian
    static Quadratic simple(int n) {
        std::vector<double> A(n * n, 0.0);
        std::vector<double> b(n);
        for (int i = 0; i < n; ++i) {
            A[i * n + i] = 1.0;
            b[i] = static_cast<double>(i + 1);
        }
        return Quadratic(n, A, b);
    }

    std::string name() const override { return "Quadratic"; }
    int dimension() const override { return n_; }

    std::vector<double> initial_point() const override {
        return std::vector<double>(n_, 0.0);
    }

    double value(const double* x) const override {
        double f = c_;
        // f += 0.5 * x^T A x
        for (int i = 0; i < n_; ++i) {
            double Ax_i = 0.0;
            for (int j = 0; j < n_; ++j) {
                Ax_i += A_[i * n_ + j] * x[j];
            }
            f += 0.5 * x[i] * Ax_i;
        }
        // f += b^T x
        for (int i = 0; i < n_; ++i) {
            f += b_[i] * x[i];
        }
        return f;
    }

    void gradient(const double* x, double* g) const override {
        // g = A*x + b
        for (int i = 0; i < n_; ++i) {
            g[i] = b_[i];
            for (int j = 0; j < n_; ++j) {
                g[i] += A_[i * n_ + j] * x[j];
            }
        }
    }

    double optimal_value() const override { return f_opt_; }

    std::vector<double> optimal_point() const override {
        return x_opt_;
    }
};

// Sphere function (sum of squares)
// f(x) = sum_{i=1}^{n} x_i^2
// Global minimum: f(0,...,0) = 0
class Sphere : public Benchmark {
    int n_;
public:
    explicit Sphere(int n) : n_(n) {
        if (n < 1) throw std::invalid_argument("Sphere: n must be >= 1");
    }

    std::string name() const override { return "Sphere"; }
    int dimension() const override { return n_; }

    std::vector<double> initial_point() const override {
        return std::vector<double>(n_, 0.5);
    }

    double value(const double* x) const override {
        double f = 0.0;
        for (int i = 0; i < n_; ++i) {
            f += x[i] * x[i];
        }
        return f;
    }

    void gradient(const double* x, double* g) const override {
        for (int i = 0; i < n_; ++i) {
            g[i] = 2.0 * x[i];
        }
    }

    std::vector<double> optimal_point() const override {
        return std::vector<double>(n_, 0.0);
    }
};

// Powell Singular function
// f(x) = sum_{i=1}^{n/4} [(x_{4i-3} + 10*x_{4i-2})^2 +
//                         5*(x_{4i-1} - x_{4i})^2 +
//                         (x_{4i-2} - 2*x_{4i-1})^4 +
//                         10*(x_{4i-3} - x_{4i})^4]
// Dimension must be multiple of 4
// Global minimum: f(0,...,0) = 0
class PowellSingular : public Benchmark {
    int n_;
public:
    explicit PowellSingular(int n = 4) : n_(n) {
        if (n % 4 != 0)
            throw std::invalid_argument("PowellSingular: n must be multiple of 4");
    }

    std::string name() const override { return "Powell Singular"; }
    int dimension() const override { return n_; }

    std::vector<double> initial_point() const override {
        std::vector<double> x(n_);
        for (int i = 0; i < n_ / 4; ++i) {
            x[4*i]     = 3.0;
            x[4*i + 1] = -1.0;
            x[4*i + 2] = 0.0;
            x[4*i + 3] = 1.0;
        }
        return x;
    }

    double value(const double* x) const override {
        double f = 0.0;
        for (int i = 0; i < n_ / 4; ++i) {
            double t1 = x[4*i] + 10.0 * x[4*i + 1];
            double t2 = x[4*i + 2] - x[4*i + 3];
            double t3 = x[4*i + 1] - 2.0 * x[4*i + 2];
            double t4 = x[4*i] - x[4*i + 3];
            f += t1 * t1 + 5.0 * t2 * t2 + t3 * t3 * t3 * t3 +
                 10.0 * t4 * t4 * t4 * t4;
        }
        return f;
    }

    void gradient(const double* x, double* g) const override {
        for (int i = 0; i < n_; ++i) g[i] = 0.0;

        for (int i = 0; i < n_ / 4; ++i) {
            double t1 = x[4*i] + 10.0 * x[4*i + 1];
            double t2 = x[4*i + 2] - x[4*i + 3];
            double t3 = x[4*i + 1] - 2.0 * x[4*i + 2];
            double t4 = x[4*i] - x[4*i + 3];

            g[4*i]     += 2.0 * t1 + 40.0 * t4 * t4 * t4;
            g[4*i + 1] += 20.0 * t1 + 4.0 * t3 * t3 * t3;
            g[4*i + 2] += 10.0 * t2 - 8.0 * t3 * t3 * t3;
            g[4*i + 3] += -10.0 * t2 - 40.0 * t4 * t4 * t4;
        }
    }

    std::vector<double> optimal_point() const override {
        return std::vector<double>(n_, 0.0);
    }
};

// Beale function (2D only)
// f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
// Global minimum: f(3, 0.5) = 0
class Beale : public Benchmark {
public:
    std::string name() const override { return "Beale"; }
    int dimension() const override { return 2; }

    std::vector<double> initial_point() const override {
        return {1.0, 1.0};
    }

    double value(const double* x) const override {
        double t1 = 1.5 - x[0] + x[0] * x[1];
        double t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
        double t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
        return t1 * t1 + t2 * t2 + t3 * t3;
    }

    void gradient(const double* x, double* g) const override {
        double t1 = 1.5 - x[0] + x[0] * x[1];
        double t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
        double t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];

        g[0] = 2.0 * t1 * (-1.0 + x[1]) +
               2.0 * t2 * (-1.0 + x[1] * x[1]) +
               2.0 * t3 * (-1.0 + x[1] * x[1] * x[1]);

        g[1] = 2.0 * t1 * x[0] +
               2.0 * t2 * 2.0 * x[0] * x[1] +
               2.0 * t3 * 3.0 * x[0] * x[1] * x[1];
    }

    std::vector<double> optimal_point() const override {
        return {3.0, 0.5};
    }
};

// Brown badly scaled function (2D only)
// f(x,y) = (x - 10^6)^2 + (y - 2*10^{-6})^2 + (xy - 2)^2
// Global minimum: f(10^6, 2*10^{-6}) = 0 (approximately)
class BrownBadlyScaled : public Benchmark {
public:
    std::string name() const override { return "Brown Badly Scaled"; }
    int dimension() const override { return 2; }

    std::vector<double> initial_point() const override {
        return {1.0, 1.0};
    }

    double value(const double* x) const override {
        double t1 = x[0] - 1e6;
        double t2 = x[1] - 2e-6;
        double t3 = x[0] * x[1] - 2.0;
        return t1 * t1 + t2 * t2 + t3 * t3;
    }

    void gradient(const double* x, double* g) const override {
        double t1 = x[0] - 1e6;
        double t2 = x[1] - 2e-6;
        double t3 = x[0] * x[1] - 2.0;

        g[0] = 2.0 * t1 + 2.0 * t3 * x[1];
        g[1] = 2.0 * t2 + 2.0 * t3 * x[0];
    }

    std::vector<double> optimal_point() const override {
        return {1e6, 2e-6};
    }
};

// Broyden tridiagonal function
// f(x) = sum_{i=1}^{n} [(3 - 2x_i)*x_i - x_{i-1} - 2*x_{i+1} + 1]^2
// (with x_0 = x_{n+1} = 0)
// Global minimum near x* = (-0.5714,...,-0.5714)
class BroydenTridiagonal : public Benchmark {
    int n_;
public:
    explicit BroydenTridiagonal(int n) : n_(n) {
        if (n < 2) throw std::invalid_argument("BroydenTridiagonal: n must be >= 2");
    }

    std::string name() const override { return "Broyden Tridiagonal"; }
    int dimension() const override { return n_; }

    std::vector<double> initial_point() const override {
        return std::vector<double>(n_, -1.0);
    }

    double value(const double* x) const override {
        double f = 0.0;
        for (int i = 0; i < n_; ++i) {
            double x_prev = (i > 0) ? x[i - 1] : 0.0;
            double x_next = (i < n_ - 1) ? x[i + 1] : 0.0;
            double r = (3.0 - 2.0 * x[i]) * x[i] - x_prev - 2.0 * x_next + 1.0;
            f += r * r;
        }
        return f;
    }

    void gradient(const double* x, double* g) const override {
        for (int i = 0; i < n_; ++i) g[i] = 0.0;

        for (int i = 0; i < n_; ++i) {
            double x_prev = (i > 0) ? x[i - 1] : 0.0;
            double x_next = (i < n_ - 1) ? x[i + 1] : 0.0;
            double r = (3.0 - 2.0 * x[i]) * x[i] - x_prev - 2.0 * x_next + 1.0;
            double dr_dxi = 3.0 - 4.0 * x[i];

            g[i] += 2.0 * r * dr_dxi;
            if (i > 0) g[i - 1] += 2.0 * r * (-1.0);
            if (i < n_ - 1) g[i + 1] += 2.0 * r * (-2.0);
        }
    }

    std::vector<double> optimal_point() const override {
        // Approximate solution
        return std::vector<double>(n_, -0.5714);
    }
};

} // namespace benchmarks
} // namespace xopt

#endif // XOPT_BENCHMARKS_HPP
