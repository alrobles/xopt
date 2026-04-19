#' Optimization control parameters
#'
#' Create a list of control parameters for xopt optimization.
#'
#' @param grtol Gradient tolerance. Optimization stops when max|gradient| <= grtol.
#' @param xtol Step tolerance. Optimization stops when step length <= xtol.
#' @param maxiter Maximum number of function evaluations.
#' @param trace Logical; if TRUE, print trace output during optimization.
#' @param stepmax Initial trust region radius.
#'
#' @return A list of control parameters.
#' @export
xopt_control <- function(grtol = 1e-6,
                         xtol = 1e-12,
                         maxiter = 500,
                         trace = FALSE,
                         stepmax = 1.0) {
  structure(
    list(
      grtol = grtol,
      xtol = xtol,
      maxiter = maxiter,
      trace = trace,
      stepmax = stepmax
    ),
    class = "xopt_control"
  )
}

#' @export
print.xopt_control <- function(x, ...) {
  cat("xopt control parameters:\n")
  cat(sprintf("  grtol:   %g\n", x$grtol))
  cat(sprintf("  xtol:    %g\n", x$xtol))
  cat(sprintf("  maxiter: %d\n", x$maxiter))
  cat(sprintf("  trace:   %s\n", x$trace))
  cat(sprintf("  stepmax: %g\n", x$stepmax))
  invisible(x)
}
