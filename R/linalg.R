#' Differentiable linear algebra primitives for SPD matrices
#'
#' `xopt_chol`, `xopt_solve`, `xopt_logdet`, and `xopt_inv` wrap the four
#' differentiable linear algebra building blocks in `xopt::linalg`. Each is
#' templated on a scalar type in C++ and records every elementary operation
#' on the active XAD tape when invoked through `xad::AReal<double>`, so
#' composing them inside a user-provided objective makes the entire chain
#' differentiable end-to-end without hand-written gradients.
#'
#' These are building blocks for downstream statistical workflows that need
#' derivatives through linear algebra: Gaussian and Laplace likelihoods,
#' Gaussian-process hyperparameters, Kalman filters/smoothers, and implicit
#' differentiation of KKT systems.
#'
#' All input matrices must be symmetric positive definite (SPD). No
#' pivoting is performed; a non-SPD matrix will cause an error.
#'
#' @param A symmetric positive-definite numeric matrix (n x n).
#' @param b numeric vector of length n (right-hand side of `A x = b`).
#'
#' @return
#'   * `xopt_chol(A)` — lower-triangular Cholesky factor `L` with `A = L L^T`.
#'   * `xopt_solve(A, b)` — vector `x` satisfying `A x = b`.
#'   * `xopt_logdet(A)` — scalar `log|A|`.
#'   * `xopt_inv(A)` — matrix `A^{-1}`.
#'
#' @examples
#' A <- crossprod(matrix(rnorm(9), 3, 3)) + diag(3)
#' L <- xopt_chol(A)
#' all.equal(L %*% t(L), A)
#'
#' b <- c(1, 2, 3)
#' x <- xopt_solve(A, b)
#' all.equal(A %*% x, matrix(b, ncol = 1))
#'
#' xopt_logdet(A)      # == log(det(A))
#' xopt_inv(A) %*% A   # == identity(3)
#'
#' @name xopt_linalg
NULL

#' @rdname xopt_linalg
#' @export
xopt_chol <- function(A) {
  A <- .validate_spd(A)
  xopt_chol_impl(A)
}

#' @rdname xopt_linalg
#' @export
xopt_solve <- function(A, b) {
  A <- .validate_spd(A)
  if (!is.numeric(b)) stop("xopt_solve: b must be numeric", call. = FALSE)
  if (length(b) != nrow(A)) {
    stop("xopt_solve: length(b) must equal nrow(A)", call. = FALSE)
  }
  xopt_solve_impl(A, as.numeric(b))
}

#' @rdname xopt_linalg
#' @export
xopt_logdet <- function(A) {
  A <- .validate_spd(A)
  xopt_logdet_impl(A)
}

#' @rdname xopt_linalg
#' @export
xopt_inv <- function(A) {
  A <- .validate_spd(A)
  xopt_inv_impl(A)
}

# Coerce input to an unambiguous symmetric-looking matrix before passing to
# the C++ layer. We do NOT enforce exact symmetry here (call sites sometimes
# pass nearly-symmetric matrices from numerical pipelines); the C++ Cholesky
# only reads the lower triangle and will fail on the first non-positive
# pivot if the input is genuinely indefinite.
.validate_spd <- function(A) {
  if (!is.matrix(A)) {
    A <- as.matrix(A)
  }
  if (!is.numeric(A)) stop("xopt linalg: A must be numeric", call. = FALSE)
  if (nrow(A) != ncol(A)) stop("xopt linalg: A must be square", call. = FALSE)
  if (nrow(A) < 1L) stop("xopt linalg: A must have at least one row", call. = FALSE)
  storage.mode(A) <- "double"
  A
}
