#' Implicit differentiation via the implicit function theorem
#'
#' Given a fixed-point equation `g(x, theta) = 0` solved for `x*(theta)`,
#' compute the vector-Jacobian product
#'   \deqn{\bar\theta = -(\partial g / \partial \theta)^T (\partial g / \partial x)^{-T} \bar x}
#' using the implicit function theorem. The SPD specialization requires
#' `A = d g / d x` symmetric positive definite at `(x*, theta)`.
#'
#' @param A numeric `n x n` SPD matrix, the Jacobian of `g` w.r.t. `x` at
#'   `(x*, theta)`.
#' @param B numeric `n x p` matrix, the Jacobian of `g` w.r.t. `theta` at
#'   `(x*, theta)`. No symmetry assumption.
#' @param x_star numeric vector of length `n`, the solution of
#'   `g(x*, theta) = 0`.
#' @param x_bar numeric vector of length `n`, the upstream gradient
#'   `dL/dx*` coming into this IFT node.
#' @return numeric vector of length `p`: the theta-cotangent
#'   `theta_bar = - B^T A^{-1} x_bar`.
#' @details The computation uses one SPD solve (`xopt::linalg::solve_spd`)
#'   and records O(n) tape slots via [xad::CheckpointCallback], so the tape
#'   footprint is independent of how `x*` was computed. The SPD solve is the
#'   same primitive shipped by xopt PR #34; callers can differentiate through
#'   arbitrarily many IFT nodes without elementary-op tape blow-up.
#' @examples
#' set.seed(1L)
#' n <- 5L; p <- 2L
#' A <- crossprod(matrix(rnorm(n * n), n, n)) + n * diag(n)
#' B <- matrix(rnorm(n * p), n, p)
#' x_star <- rnorm(n)
#' x_bar  <- rnorm(n)
#' xopt_implicit_spd(A, B, x_star, x_bar)
#' @export
xopt_implicit_spd <- function(A, B, x_star, x_bar) {
  stopifnot(is.matrix(A), is.matrix(B), is.numeric(x_star), is.numeric(x_bar))
  n <- nrow(A)
  if (ncol(A) != n) stop("A must be square")
  if (nrow(B) != n) stop("nrow(B) must equal nrow(A)")
  if (length(x_star) != n) stop("length(x_star) must equal nrow(A)")
  if (length(x_bar)  != n) stop("length(x_bar) must equal nrow(A)")
  xopt_implicit_spd_grad_generic(A, B, x_star, x_bar)
}
