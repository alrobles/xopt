#' Laplace approximation to a negative log-posterior
#'
#' Compute the Laplace approximation to an intractable integral of the form
#' \deqn{Z = \int_{\mathbb{R}^p} \exp(-f(\theta)) \, \mathrm{d}\theta}
#' where \code{fn(theta) = f(theta)} is a smooth function playing the role
#' of a negative log-posterior (or negative log-likelihood + negative
#' log-prior, up to an additive constant). Writing \eqn{\theta^*} for the
#' mode of \eqn{\exp(-f)} and \eqn{H} for the Hessian of \eqn{f} at that
#' mode, the Laplace approximation is
#' \deqn{\log Z \approx -f(\theta^*) + \frac{p}{2}\log(2\pi) - \frac{1}{2}\log\det H.}
#'
#' \code{xopt_laplace} locates \eqn{\theta^*} with \code{\link{xopt_minimize}}
#' (or uses the mode the user passes in, skipping the optimization),
#' computes \eqn{H} via central finite differences (or via a user-supplied
#' \code{hessian} function), and evaluates \eqn{\log\det H} with
#' \code{\link{xopt_logdet}} — which is differentiable end-to-end through
#' the XAD tape, so downstream marginal-likelihood gradients w.r.t.
#' hyperparameters flow automatically when the linear-algebra path is
#' instrumented.
#'
#' @param fn Negative log-posterior (or negative log-likelihood + negative
#'   log-prior). Must be smooth and attain a strict local minimum at the
#'   mode.
#' @param par Starting point for the optimizer, or — when
#'   \code{optimize = FALSE} — the mode itself.
#' @param gr Optional gradient of \code{fn}, forwarded to
#'   \code{xopt_minimize}.
#' @param hessian Optional function taking a parameter vector and returning
#'   the Hessian of \code{fn} at that point. If \code{NULL}, a central
#'   finite-difference Hessian is used.
#' @param optimize Logical. If \code{TRUE} (default), run \code{xopt_minimize}
#'   to locate the mode. If \code{FALSE}, treat \code{par} as the mode.
#' @param gradient Gradient mode forwarded to \code{xopt_minimize}; see
#'   \code{\link{xopt_minimize}}.
#' @param control Control parameters forwarded to \code{xopt_minimize}.
#'
#' @return A list of class \code{"xopt_laplace"} with components:
#' \describe{
#'   \item{log_marginal}{The Laplace-approximated \eqn{\log Z}.}
#'   \item{par}{The mode \eqn{\theta^*}.}
#'   \item{value}{\eqn{f(\theta^*)} (the minimum of \code{fn}).}
#'   \item{hessian}{The Hessian of \code{fn} at the mode.}
#'   \item{logdet_hessian}{\eqn{\log\det H} computed via
#'     \code{\link{xopt_logdet}}.}
#'   \item{fit}{The underlying \code{xopt_result} from \code{xopt_minimize},
#'     or \code{NULL} when \code{optimize = FALSE}.}
#' }
#'
#' @details
#' The Hessian is forced to be exactly symmetric (\eqn{(H + H^\top)/2})
#' before being handed to \code{xopt_logdet}, so that symmetric-rounded
#' inputs coming from a finite-difference estimator satisfy the SPD
#' precondition required by Cholesky. If the Hessian is not positive
#' definite at \eqn{\theta^*} — i.e., \eqn{\theta^*} is not a local
#' minimum — \code{xopt_logdet} will raise an error from the C++ Cholesky
#' step.
#'
#' @seealso \code{\link{xopt_minimize}}, \code{\link{xopt_logdet}},
#'   \code{\link{xopt_chol}}.
#'
#' @examples
#' \dontrun{
#' # Laplace approximation to a univariate Gaussian.
#' # Target: integral of exp(-0.5 * (theta - 3)^2 / 2) d theta = sqrt(2*pi*2).
#' # So log Z should be 0.5 * log(2*pi*2) = 1.2655.
#' fn <- function(theta) 0.5 * (theta - 3)^2 / 2
#' res <- xopt_laplace(fn, par = 0)
#' stopifnot(abs(res$log_marginal - 0.5 * log(2 * pi * 2)) < 1e-4)
#' }
#'
#' @export
xopt_laplace <- function(fn,
                         par,
                         gr = NULL,
                         hessian = NULL,
                         optimize = TRUE,
                         gradient = NULL,
                         control = xopt_control()) {
  if (!is.function(fn)) {
    stop("fn must be a function", call. = FALSE)
  }
  if (!is.numeric(par)) {
    stop("par must be numeric", call. = FALSE)
  }
  if (!is.null(hessian) && !is.function(hessian)) {
    stop("hessian must be a function or NULL", call. = FALSE)
  }

  if (optimize) {
    fit <- xopt_minimize(par = par, fn = fn, gr = gr,
                        gradient = gradient, control = control)
    mode <- fit$par
    value_at_mode <- fit$value
  } else {
    fit <- NULL
    mode <- as.numeric(par)
    value_at_mode <- fn(mode)
  }

  H <- if (is.null(hessian)) {
    .xopt_fd_hessian(fn, mode)
  } else {
    as.matrix(hessian(mode))
  }

  if (!is.numeric(H) || nrow(H) != ncol(H) || nrow(H) != length(mode)) {
    stop(sprintf(
      "Hessian at mode is not a square numeric matrix of size %d x %d.",
      length(mode), length(mode)
    ), call. = FALSE)
  }

  # Symmetrize (FD Hessians are only approximately symmetric; xopt_logdet
  # requires SPD input, and the mathematical object is symmetric by
  # construction).
  H <- 0.5 * (H + t(H))
  ldet_H <- xopt_logdet(H)

  p <- length(mode)
  log_marginal <- -value_at_mode + 0.5 * p * log(2 * pi) - 0.5 * ldet_H

  structure(
    list(
      log_marginal = log_marginal,
      par = mode,
      value = value_at_mode,
      hessian = H,
      logdet_hessian = ldet_H,
      fit = fit
    ),
    class = "xopt_laplace"
  )
}

#' @export
print.xopt_laplace <- function(x, ...) {
  cat("xopt Laplace approximation:\n")
  cat(sprintf("  log Z (approx): %.6f\n", x$log_marginal))
  cat(sprintf("  Mode value:     %.6e\n", x$value))
  cat(sprintf("  log det H:      %.6f\n", x$logdet_hessian))
  cat(sprintf("  Dimension:      %d\n", length(x$par)))
  cat(sprintf("  Mode:           [%s]\n",
              paste(sprintf("%.6f", x$par), collapse = ", ")))
  invisible(x)
}
