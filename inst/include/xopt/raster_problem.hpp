// raster_problem.hpp - Raster-centric optimization problem interface
//
// This header provides abstractions for optimization over raster stacks,
// supporting chunking for memory-bounded operations and NA/masking.

#ifndef XOPT_RASTER_PROBLEM_HPP
#define XOPT_RASTER_PROBLEM_HPP

#include "problem.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <functional>

namespace xopt {

// Chunk specification for processing large rasters in blocks
struct Chunk {
    size_t offset;  // Starting index in flattened raster
    size_t size;    // Number of cells in this chunk

    Chunk(size_t off, size_t sz) : offset(off), size(sz) {}
};

// Chunking strategy for raster processing
struct ChunkingStrategy {
    size_t n_cells;        // Total number of raster cells
    size_t chunk_size;     // Target size per chunk
    size_t n_chunks;       // Number of chunks

    explicit ChunkingStrategy(size_t total_cells, size_t target_chunk_size = 10000)
        : n_cells(total_cells),
          chunk_size(target_chunk_size),
          n_chunks((total_cells + target_chunk_size - 1) / target_chunk_size) {}

    // Get chunk specification by index
    Chunk get_chunk(size_t idx) const {
        if (idx >= n_chunks) {
            throw std::out_of_range("Chunk index out of range");
        }
        size_t offset = idx * chunk_size;
        size_t size = std::min(chunk_size, n_cells - offset);
        return Chunk(offset, size);
    }

    // Iterate over all chunks
    template <typename Fn>
    void for_each_chunk(Fn&& fn) const {
        for (size_t i = 0; i < n_chunks; ++i) {
            fn(get_chunk(i));
        }
    }
};

// Raster mask for handling NA values and subsetting
struct RasterMask {
    std::vector<bool> mask;  // true = valid cell, false = NA/masked

    explicit RasterMask(size_t n_cells, bool default_valid = true)
        : mask(n_cells, default_valid) {}

    // Number of cells
    size_t size() const {
        return mask.size();
    }

    // Number of valid (unmasked) cells
    size_t n_valid() const {
        size_t count = 0;
        for (bool m : mask) {
            if (m) ++count;
        }
        return count;
    }

    // Check if cell is valid
    bool is_valid(size_t idx) const {
        return idx < mask.size() && mask[idx];
    }

    // Mark cell as invalid (NA/masked)
    void set_invalid(size_t idx) {
        if (idx < mask.size()) {
            mask[idx] = false;
        }
    }

    // Mark cell as valid
    void set_valid(size_t idx) {
        if (idx < mask.size()) {
            mask[idx] = true;
        }
    }

    // Set mask from NA values in data
    template <typename T>
    void from_na_values(const std::vector<T>& data) {
        if (data.size() != mask.size()) {
            throw std::invalid_argument("Data size must match mask size");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            mask[i] = !std::isnan(data[i]);
        }
    }

    // Combine with another mask (logical AND)
    void intersect(const RasterMask& other) {
        if (other.size() != size()) {
            throw std::invalid_argument("Mask sizes must match");
        }
        for (size_t i = 0; i < mask.size(); ++i) {
            mask[i] = mask[i] && other.mask[i];
        }
    }
};

// Raster stack dimensions
struct RasterDims {
    size_t n_rows;      // Number of rows
    size_t n_cols;      // Number of columns
    size_t n_layers;    // Number of layers (covariates)

    RasterDims(size_t rows, size_t cols, size_t layers)
        : n_rows(rows), n_cols(cols), n_layers(layers) {}

    // Total number of cells
    size_t n_cells() const {
        return n_rows * n_cols;
    }

    // Total number of values (cells × layers)
    size_t n_values() const {
        return n_cells() * n_layers;
    }

    // Convert 2D coordinates to flat index
    size_t cell_index(size_t row, size_t col) const {
        return row * n_cols + col;
    }

    // Convert flat index to 2D coordinates
    std::pair<size_t, size_t> cell_coords(size_t idx) const {
        return {idx / n_cols, idx % n_cols};
    }
};

// Base class for raster-based optimization problems
template <typename Scalar = double>
struct RasterProblemBase : public ProblemBase<Scalar> {
    RasterDims dims;           // Raster dimensions
    RasterMask mask;           // Valid cells mask
    ChunkingStrategy chunking; // Chunking strategy

    RasterProblemBase(int n_par, RasterDims d, size_t chunk_size = 10000)
        : ProblemBase<Scalar>(n_par),
          dims(d),
          mask(d.n_cells(), true),
          chunking(d.n_cells(), chunk_size) {}

    // Get number of valid cells
    size_t n_valid_cells() const {
        return mask.n_valid();
    }

    // Set mask from response data (NA handling)
    template <typename T>
    void set_mask_from_response(const std::vector<T>& response) {
        if (response.size() != dims.n_cells()) {
            throw std::invalid_argument("Response size must match number of cells");
        }
        mask.from_na_values(response);
    }

    // Set mask from covariate data (NA handling)
    template <typename T>
    void set_mask_from_covariates(const std::vector<std::vector<T>>& covariates) {
        RasterMask combined_mask(dims.n_cells(), true);
        for (const auto& layer : covariates) {
            RasterMask layer_mask(dims.n_cells(), true);
            layer_mask.from_na_values(layer);
            combined_mask.intersect(layer_mask);
        }
        mask = combined_mask;
    }
};

// Raster problem with covariate tensors and response raster
template <typename UserObj,
          GradKind Grad = GradKind::XadAdj,
          HessKind Hess = HessKind::BfgsApprox,
          typename Scalar = double>
struct RasterProblem : public RasterProblemBase<Scalar> {
    UserObj obj;  // User's objective function

    // Covariate data: [n_cells, n_layers] stored as vector of layers
    std::vector<std::vector<Scalar>> covariates;

    // Response data: [n_cells]
    std::vector<Scalar> response;

    RasterProblem(int n_par,
                  RasterDims dims,
                  std::vector<std::vector<Scalar>> cov,
                  std::vector<Scalar> resp,
                  UserObj&& user_obj,
                  size_t chunk_size = 10000)
        : RasterProblemBase<Scalar>(n_par, dims, chunk_size),
          obj(std::forward<UserObj>(user_obj)),
          covariates(std::move(cov)),
          response(std::move(resp)) {

        // Validate dimensions
        if (response.size() != dims.n_cells()) {
            throw std::invalid_argument("Response size must match number of cells");
        }
        if (covariates.size() != dims.n_layers) {
            throw std::invalid_argument("Number of covariate layers must match dims.n_layers");
        }
        for (const auto& layer : covariates) {
            if (layer.size() != dims.n_cells()) {
                throw std::invalid_argument("Covariate layer size must match number of cells");
            }
        }

        // Auto-detect and set mask from NA values
        auto_detect_mask();
    }

    // Automatically detect NA values and create mask
    void auto_detect_mask() {
        // Start with all cells valid
        RasterMask combined_mask(this->dims.n_cells(), true);

        // Check response for NAs
        for (size_t i = 0; i < response.size(); ++i) {
            if (std::isnan(response[i])) {
                combined_mask.set_invalid(i);
            }
        }

        // Check each covariate layer for NAs
        for (const auto& layer : covariates) {
            for (size_t i = 0; i < layer.size(); ++i) {
                if (std::isnan(layer[i])) {
                    combined_mask.set_invalid(i);
                }
            }
        }

        this->mask = combined_mask;
    }

    // Evaluate objective value at parameters x
    Scalar value(const Scalar* x) const {
        return obj.value(x, covariates, response, this->mask);
    }

    // Compute gradient at x
    void gradient(const Scalar* x, Scalar* g) const {
        if constexpr (Grad == GradKind::None) {
            throw std::runtime_error("Gradient not available for this problem");
        } else {
            obj.gradient(x, g, covariates, response, this->mask);
        }
    }

    // Compute Hessian at x
    void hessian(const Scalar* x, Scalar* H) const {
        if constexpr (Hess == HessKind::None) {
            throw std::runtime_error("Hessian not available for this problem");
        } else {
            obj.hessian(x, H, covariates, response, this->mask);
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

    // Chunked evaluation for memory-bounded operations
    template <typename ChunkFn, typename ReduceFn>
    Scalar evaluate_chunked(const Scalar* x, ChunkFn&& chunk_fn, ReduceFn&& reduce_fn) const {
        Scalar result = 0;
        this->chunking.for_each_chunk([&](const Chunk& chunk) {
            Scalar chunk_result = chunk_fn(x, chunk);
            result = reduce_fn(result, chunk_result);
        });
        return result;
    }
};

} // namespace xopt

#endif // XOPT_RASTER_PROBLEM_HPP
