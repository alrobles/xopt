#' Minimize an objective function using xopt
#'
#' Minimize a scalar-valued objective function using the UCMINF quasi-Newton algorithm.
#'
#' @param par Initial parameter vector (numeric).
#' @param fn Objective function to minimize. Should take a numeric vector and return a scalar.
#' @param gr Optional gradient function. Should take a numeric vector and return a gradient vector.
#'   If NULL, numerical gradients will be computed.
#' @param method Optimization method: \code{"bfgs"} (default) or
#'   \code{"tr_newton"} (trust-region Newton).
#' @param gradient Gradient mode: \code{"auto"}, \code{"user"}, or \code{"finite"}.
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
xopt_minimize <- function(par,
                          fn,
                          gr = NULL,
                          method = c("bfgs", "tr_newton"),
                          gradient = c("auto", "user", "finite"),
                          control = xopt_control()) {
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
  method <- match.arg(method)
  gradient <- match.arg(gradient)
  if (!inherits(control, "xopt_control")) {
    control <- do.call(xopt_control, control)
  }

  gr_use <- gr
  if (is.null(gr_use)) {
    if (gradient == "user") {
      stop("gradient='user' requires gr")
    }
    if (gradient == "auto") {
      gr_use <- function(x) xopt_auto_gradient(fn, x, tracer = TRUE)
    } else {
      gr_use <- function(x) .xopt_fd_gradient(fn, x)
    }
  }

  if (method == "tr_newton") {
    return(.xopt_trust_region_newton(par, fn, gr_use, control))
  }

  # BFGS path via ucminfcpp
  if (!requireNamespace("ucminfcpp", quietly = TRUE)) {
    stop("ucminfcpp package is required but not installed")
  }

  # Call ucminf
  result <- ucminfcpp::ucminf(
    par, fn, gr_use,
    control = list(
      grtol = control$grtol,
      xtol = control$xtol,
      maxeval = control$maxiter,
      stepmax = control$stepmax
    )
  )

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

.xopt_trust_region_newton <- function(par, fn, gr, control) {
  x <- as.numeric(par)
  f <- fn(x)
  delta <- control$stepmax
  maxiter <- control$maxiter
  g <- gr(x)
  message <- "Maximum iterations reached"
  convergence <- 0L
  iterations <- 0L

  for (iter in seq_len(maxiter)) {
    g <- gr(x)
    if (max(abs(g)) <= control$grtol) {
      convergence <- 1L
      message <- "Gradient below tolerance"
      iterations <- iter - 1L
      break
    }

    h <- .xopt_fd_hessian(fn, x)
    step <- tryCatch(
      as.numeric(solve(h, -g)),
      error = function(e) {
        lambda <- 1e-10
        repeat {
          trial <- tryCatch(
            as.numeric(solve(h + diag(lambda, length(x)), -g)),
            error = function(err) NULL
          )
          if (!is.null(trial) || lambda > 1) {
            return(if (is.null(trial)) rep(0, length(x)) else trial)
          }
          lambda <- lambda * 10
        }
      }
    )
    step_norm <- sqrt(sum(step^2))
    if (step_norm > delta && step_norm > 0) {
      step <- step * (delta / step_norm)
      step_norm <- delta
    }
    if (step_norm <= control$xtol) {
      convergence <- 2L
      message <- "Step below tolerance"
      iterations <- iter - 1L
      break
    }

    x_trial <- x + step
    f_trial <- fn(x_trial)
    pred <- -sum(g * step) - 0.5 * sum(step * as.numeric(h %*% step))
    if (pred <= 0) {
      delta <- max(delta * 0.25, control$xtol)
      next
    }

    rho <- (f - f_trial) / pred
    if (rho < 0.25) {
      delta <- delta * 0.25
    } else if (rho > 0.75 && abs(step_norm - delta) <= 1e-12 * max(1, delta)) {
      delta <- min(2 * delta, 1e6)
    }

    if (rho > 0.15) {
      x <- x_trial
      if (abs(f - f_trial) <= control$ftol * (1 + abs(f))) {
        convergence <- 4L
        message <- "Function change below tolerance"
        iterations <- iter
        f <- f_trial
        break
      }
      f <- f_trial
    }

    iterations <- iter
  }

  structure(
    list(
      par = x,
      value = f,
      gradient = gr(x),
      convergence = convergence,
      message = message,
      iterations = iterations
    ),
    class = "xopt_result"
  )
}

.xopt_fd_hessian <- function(fn, par, eps = 1e-4) {
  n <- length(par)
  H <- matrix(0, n, n)
  xpp <- xpm <- xmp <- xmm <- par
  for (i in seq_len(n)) {
    hi <- eps * max(1, abs(par[[i]]))
    for (j in i:n) {
      hj <- eps * max(1, abs(par[[j]]))
      xpp[[i]] <- par[[i]] + hi; xpp[[j]] <- par[[j]] + hj
      xpm[[i]] <- par[[i]] + hi; xpm[[j]] <- par[[j]] - hj
      xmp[[i]] <- par[[i]] - hi; xmp[[j]] <- par[[j]] + hj
      xmm[[i]] <- par[[i]] - hi; xmm[[j]] <- par[[j]] - hj
      hij <- (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * hi * hj)
      H[[i, j]] <- hij
      H[[j, i]] <- hij
      xpp[[i]] <- par[[i]]; xpp[[j]] <- par[[j]]
      xpm[[i]] <- par[[i]]; xpm[[j]] <- par[[j]]
      xmp[[i]] <- par[[i]]; xmp[[j]] <- par[[j]]
      xmm[[i]] <- par[[i]]; xmm[[j]] <- par[[j]]
    }
  }
  H
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
