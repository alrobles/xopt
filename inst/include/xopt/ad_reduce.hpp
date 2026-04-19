// ad_reduce.hpp - XAD-aware reduction operations
//
// This header provides reduction functions that work with XAD active types
// to enable sum and masked reductions for autodiff-typed tensor expressions.
// This is a crucial workaround for expression-template limitations with reducers.

#ifndef XOPT_AD_REDUCE_HPP
#define XOPT_AD_REDUCE_HPP

#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>

namespace xopt {

// Type trait to detect if a type is XAD active
template <typename T>
struct is_xad_active : std::false_type {};

// Forward declaration for XAD types (will be specialized when XAD is included)
// This allows the code to compile without XAD while maintaining the interface

// Generic sum reduction that works with both plain and XAD types
template <typename T>
T sum_active(const std::vector<T>& vec) {
    T result = T(0);
    for (const auto& val : vec) {
        result += val;
    }
    return result;
}

// Sum reduction for iterators (useful for expression templates)
template <typename Iter>
auto sum_active(Iter begin, Iter end) -> typename std::iterator_traits<Iter>::value_type {
    using T = typename std::iterator_traits<Iter>::value_type;
    T result = T(0);
    for (auto it = begin; it != end; ++it) {
        result += *it;
    }
    return result;
}

// Sum reduction for array-like containers
template <typename Container>
auto sum_active(const Container& container)
    -> decltype(sum_active(std::begin(container), std::end(container))) {
    return sum_active(std::begin(container), std::end(container));
}

// Masked sum reduction - sum only where mask is true
template <typename T>
T sum_masked(const std::vector<T>& vec, const std::vector<bool>& mask) {
    if (vec.size() != mask.size()) {
        throw std::invalid_argument("Vector and mask sizes must match");
    }

    T result = T(0);
    for (size_t i = 0; i < vec.size(); ++i) {
        if (mask[i]) {
            result += vec[i];
        }
    }
    return result;
}

// Masked sum with iterator interface
template <typename Iter, typename MaskIter>
auto sum_masked(Iter begin, Iter end, MaskIter mask_begin)
    -> typename std::iterator_traits<Iter>::value_type {
    using T = typename std::iterator_traits<Iter>::value_type;
    T result = T(0);
    auto mask_it = mask_begin;
    for (auto it = begin; it != end; ++it, ++mask_it) {
        if (*mask_it) {
            result += *it;
        }
    }
    return result;
}

// Masked sum for containers
template <typename Container, typename MaskContainer>
auto sum_masked(const Container& container, const MaskContainer& mask)
    -> decltype(sum_masked(std::begin(container), std::end(container), std::begin(mask))) {
    if (std::distance(std::begin(container), std::end(container)) !=
        std::distance(std::begin(mask), std::end(mask))) {
        throw std::invalid_argument("Container and mask sizes must match");
    }
    return sum_masked(std::begin(container), std::end(container), std::begin(mask));
}

// Mean reduction (active type aware)
template <typename T>
T mean_active(const std::vector<T>& vec) {
    if (vec.empty()) {
        throw std::invalid_argument("Cannot compute mean of empty vector");
    }
    return sum_active(vec) / static_cast<double>(vec.size());
}

// Masked mean reduction
template <typename T>
T mean_masked(const std::vector<T>& vec, const std::vector<bool>& mask) {
    if (vec.size() != mask.size()) {
        throw std::invalid_argument("Vector and mask sizes must match");
    }

    size_t count = 0;
    for (bool m : mask) {
        if (m) ++count;
    }

    if (count == 0) {
        throw std::invalid_argument("Cannot compute mean with all values masked");
    }

    return sum_masked(vec, mask) / static_cast<double>(count);
}

// Product reduction (active type aware)
template <typename T>
T prod_active(const std::vector<T>& vec) {
    T result = T(1);
    for (const auto& val : vec) {
        result *= val;
    }
    return result;
}

// Masked product reduction
template <typename T>
T prod_masked(const std::vector<T>& vec, const std::vector<bool>& mask) {
    if (vec.size() != mask.size()) {
        throw std::invalid_argument("Vector and mask sizes must match");
    }

    T result = T(1);
    for (size_t i = 0; i < vec.size(); ++i) {
        if (mask[i]) {
            result *= vec[i];
        }
    }
    return result;
}

// Logistic function (active type aware)
template <typename T>
T logistic(const T& x) {
    return T(1) / (T(1) + std::exp(-x));
}

// Logit function (active type aware)
template <typename T>
T logit(const T& x) {
    return std::log(x / (T(1) - x));
}

// Log-sum-exp for numerical stability (active type aware)
template <typename T>
T log_sum_exp(const std::vector<T>& vec) {
    if (vec.empty()) {
        throw std::invalid_argument("Cannot compute log-sum-exp of empty vector");
    }

    // Find maximum for numerical stability
    T max_val = vec[0];
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }

    // Compute log-sum-exp
    T sum = T(0);
    for (const auto& val : vec) {
        sum += std::exp(val - max_val);
    }

    return max_val + std::log(sum);
}

// Masked log-sum-exp
template <typename T>
T log_sum_exp_masked(const std::vector<T>& vec, const std::vector<bool>& mask) {
    if (vec.size() != mask.size()) {
        throw std::invalid_argument("Vector and mask sizes must match");
    }

    // Find maximum among valid values
    bool found_first = false;
    T max_val = T(0);
    for (size_t i = 0; i < vec.size(); ++i) {
        if (mask[i]) {
            if (!found_first) {
                max_val = vec[i];
                found_first = true;
            } else if (vec[i] > max_val) {
                max_val = vec[i];
            }
        }
    }

    if (!found_first) {
        throw std::invalid_argument("Cannot compute log-sum-exp with all values masked");
    }

    // Compute log-sum-exp over valid values
    T sum = T(0);
    for (size_t i = 0; i < vec.size(); ++i) {
        if (mask[i]) {
            sum += std::exp(vec[i] - max_val);
        }
    }

    return max_val + std::log(sum);
}

// Count valid (non-masked) elements
inline size_t count_valid(const std::vector<bool>& mask) {
    size_t count = 0;
    for (bool m : mask) {
        if (m) ++count;
    }
    return count;
}

// Check if any element is valid
inline bool any_valid(const std::vector<bool>& mask) {
    for (bool m : mask) {
        if (m) return true;
    }
    return false;
}

// Check if all elements are valid
inline bool all_valid(const std::vector<bool>& mask) {
    for (bool m : mask) {
        if (!m) return false;
    }
    return true;
}

} // namespace xopt

#endif // XOPT_AD_REDUCE_HPP
