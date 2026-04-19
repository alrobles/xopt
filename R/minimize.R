#' Minimize an objective function using xopt
#'
#' Minimize a scalar-valued objective function using the UCMINF quasi-Newton algorithm.
#'
#' @param par Initial parameter vector (numeric).
#' @param fn Objective function to minimize. Should take a numeric vector and return a scalar.
#' @param gr Optional gradient function. Should take a numeric vector and return a gradient vector.
#'   If NULL, numerical gradients will be computed.
#' @param control Control parameters (see \code{\link{xopt_control}}).
#'
#' @return A list with components:
#' \item{par}{The optimal parameter vector.}
#' \item{value}{The objective function value at the optimum.}
#' \item{gradient}{The gradient at the optimum (if available).}
#' \item{convergence}{Convergence code (1 = small gradient, 2 = small step, etc.).}
#' \item{message}{Convergence message.}
#' \item{iterations}{Number of function evaluations used.}
#'
#' @details
#' This function provides a high-level interface to optimization using the UCMINF
#' quasi-Newton BFGS algorithm. If a gradient function is provided, it will be used
#' for optimization. Otherwise, numerical gradients will be computed using finite
#' differences.
#'
#' @examples
#' \dontrun{
#' # Rosenbrock function
#' rosenbrock <- function(x) {
#'   (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
#' }
#'
#' # With analytical gradient
#' rosenbrock_grad <- function(x) {
#'   c(-400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
#'     200 * (x[2] - x[1]^2))
#' }
#'
#' result <- xopt_minimize(c(-1.2, 1), rosenbrock, rosenbrock_grad)
#' print(result$par)  # Should be close to c(1, 1)
#'
#' # Without gradient (numerical gradients)
#' result2 <- xopt_minimize(c(-1.2, 1), rosenbrock)
#' }
#'
#' @export
xopt_minimize <- function(par, fn, gr = NULL, control = xopt_control()) {
  # Validate inputs
  if (!is.numeric(par)) {
    stop("par must be a numeric vector")
  }
  if (!is.function(fn)) {
    stop("fn must be a function")
  }
  if (!is.null(gr) && !is.function(gr)) {
    stop("gr must be a function or NULL")
  }
  if (!inherits(control, "xopt_control")) {
    control <- do.call(xopt_control, control)
  }

  # For now, dispatch to ucminfcpp directly
  # In the future, this will support multiple backends and gradient modes
  if (!requireNamespace("ucminfcpp", quietly = TRUE)) {
    stop("ucminfcpp package is required but not installed")
  }

  # Create ucminf control object
  uc_control <- list(
    grtol = control$grtol,
    xtol = control$xtol,
    maxeval = control$maxiter,
    stepmax = control$stepmax
  )

  # Call ucminf
  if (is.null(gr)) {
    # Use numerical gradients
    result <- ucminfcpp::ucminf(par, fn, control = uc_control)
  } else {
    # Use analytical gradients
    result <- ucminfcpp::ucminf(par, fn, gr, control = uc_control)
  }

  # Convert to xopt result format
  structure(
    list(
      par = result$par,
      value = result$value,
      gradient = if (!is.null(result$gradient)) result$gradient else numeric(0),
      convergence = result$convergence,
      message = result$message,
      iterations = result$neval
    ),
    class = "xopt_result"
  )
}

#' @export
print.xopt_result <- function(x, ...) {
  cat("xopt optimization result:\n")
  cat(sprintf("  Convergence: %d - %s\n", x$convergence, x$message))
  cat(sprintf("  Iterations:  %d\n", x$iterations))
  cat(sprintf("  Final value: %.6e\n", x$value))
  cat(sprintf("  Parameters:  [%s]\n",
              paste(sprintf("%.6f", x$par), collapse = ", ")))
  invisible(x)
}
