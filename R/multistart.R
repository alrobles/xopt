#' Multi-start optimization
#'
#' Run optimization from multiple starting points and return the best result.
#'
#' @param starts Either a matrix where each row is a starting point, or a list of
#'   starting parameter vectors.
#' @param fn Objective function to minimize.
#' @param gr Optional gradient function.
#' @param control Control parameters (see \code{\link{xopt_control}}).
#' @param return_all Logical; if TRUE, return results from all starts.
#'
#' @return If \code{return_all = FALSE} (default), returns the best result as an
#'   \code{xopt_result} object. If \code{return_all = TRUE}, returns a list with
#'   components:
#' \item{all_results}{List of all optimization results.}
#' \item{all_values}{Vector of final objective values.}
#' \item{best_result}{The best result.}
#' \item{best_index}{Index of the best start.}
#'
#' @details
#' This function runs \code{xopt_minimize} from each starting point independently
#' and returns the result with the lowest objective value. This is particularly
#' useful for multimodal functions where different starting points may converge
#' to different local minima.
#'
#' @examples
#' \dontrun{
#' # Multimodal Rosenbrock with random starts
#' rosenbrock <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
#'
#' # Generate 10 random starting points
#' set.seed(42)
#' starts <- matrix(runif(20, -3, 3), ncol = 2)
#'
#' result <- xopt_multistart(starts, rosenbrock)
#' print(result$par)  # Should be close to c(1, 1)
#' }
#'
#' @export
xopt_multistart <- function(starts, fn, gr = NULL, control = xopt_control(),
                           return_all = FALSE) {
  # Convert starts to list if matrix
  if (is.matrix(starts)) {
    starts_list <- lapply(1:nrow(starts), function(i) starts[i, ])
  } else if (is.list(starts)) {
    starts_list <- starts
  } else {
    stop("starts must be a matrix or list")
  }

  n_starts <- length(starts_list)
  if (n_starts == 0) {
    stop("Must provide at least one starting point")
  }

  # Run optimization from each start
  all_results <- vector("list", n_starts)
  all_values <- numeric(n_starts)

  for (i in seq_len(n_starts)) {
    result <- xopt_minimize(starts_list[[i]], fn, gr, control)
    all_results[[i]] <- result
    all_values[i] <- result$value
  }

  # Find best result
  best_idx <- which.min(all_values)
  best_result <- all_results[[best_idx]]

  if (return_all) {
    list(
      all_results = all_results,
      all_values = all_values,
      best_result = best_result,
      best_index = best_idx
    )
  } else {
    best_result
  }
}

#' Nonlinear least squares optimization
#'
#' Minimize a nonlinear least squares objective using the Levenberg-Marquardt algorithm.
#'
#' @param par Initial parameter vector.
#' @param residual_fn Residual function that takes parameters and returns a vector
#'   of residuals.
#' @param jacobian_fn Optional Jacobian function. If NULL, uses finite differences.
#' @param control List of control parameters:
#'   \itemize{
#'     \item \code{ftol}: Function tolerance (default: 1e-8)
#'     \item \code{xtol}: Parameter tolerance (default: 1e-8)
#'     \item \code{gtol}: Gradient tolerance (default: 1e-8)
#'     \item \code{maxiter}: Maximum iterations (default: 100)
#'     \item \code{trace}: Print trace output (default: FALSE)
#'   }
#'
#' @return A list with components:
#' \item{par}{Optimal parameters.}
#' \item{value}{Final sum of squared residuals (multiplied by 0.5).}
#' \item{residuals}{Final residual vector.}
#' \item{jacobian}{Final Jacobian matrix (m x n).}
#' \item{vcov}{Covariance matrix (n x n).}
#' \item{iterations}{Number of iterations used.}
#' \item{convergence}{Convergence code (1 = gradient, 2 = function, 4 = parameters).}
#' \item{message}{Convergence message.}
#'
#' @details
#' The Levenberg-Marquardt algorithm is specifically designed for nonlinear least
#' squares problems of the form: minimize ½‖r(θ)‖². It adaptively interpolates
#' between gradient descent and Gauss-Newton steps.
#'
#' The covariance matrix is computed as (J'J)^{-1} σ², where σ² = ‖r‖²/(m-n) is
#' the residual variance estimate.
#'
#' @examples
#' \dontrun{
#' # Exponential decay model: y = a * exp(-b * t)
#' t <- seq(0, 5, by = 0.5)
#' y_true <- 5 * exp(-0.5 * t)
#' y_obs <- y_true + rnorm(length(t), sd = 0.1)
#'
#' residual_fn <- function(par) {
#'   a <- par[1]
#'   b <- par[2]
#'   y_pred <- a * exp(-b * t)
#'   return(y_obs - y_pred)
#' }
#'
#' result <- xopt_nls(c(1, 0.1), residual_fn)
#' print(result$par)  # Should be close to c(5, 0.5)
#' print(sqrt(diag(result$vcov)))  # Standard errors
#' }
#'
#' @export
xopt_nls <- function(par, residual_fn, jacobian_fn = NULL,
                     control = list()) {
  # Default control parameters
  default_control <- list(
    ftol = 1e-8,
    xtol = 1e-8,
    gtol = 1e-8,
    maxiter = 100,
    trace = FALSE
  )
  control <- utils::modifyList(default_control, control)

  # Validate inputs
  if (!is.numeric(par)) {
    stop("par must be a numeric vector")
  }
  if (!is.function(residual_fn)) {
    stop("residual_fn must be a function")
  }
  if (!is.null(jacobian_fn) && !is.function(jacobian_fn)) {
    stop("jacobian_fn must be a function or NULL")
  }

  if (exists("xopt_nls_cpp", mode = "function")) {
    return(xopt_nls_cpp(par, residual_fn, jacobian_fn, control))
  }

  n <- length(par)

  # Test residual function to get dimensions
  r_test <- residual_fn(par)
  if (!is.numeric(r_test)) {
    stop("residual_fn must return a numeric vector")
  }
  m <- length(r_test)

  if (m < n) {
    stop("Number of residuals must be >= number of parameters")
  }

  # Simple R implementation of Levenberg-Marquardt
  # (In practice, this would call the C++ implementation)

  x <- par
  lambda <- if (is.null(control$lambda_init)) 1e-3 else control$lambda_init
  lambda_up <- 10.0
  lambda_down <- 0.1
  converged <- FALSE

  for (iter in seq_len(control$maxiter)) {
    # Compute residuals
    r <- residual_fn(x)
    f <- 0.5 * sum(r^2)

    # Compute Jacobian
    if (!is.null(jacobian_fn)) {
      J <- jacobian_fn(x)
      if (!is.matrix(J) || nrow(J) != m || ncol(J) != n) {
        stop("jacobian_fn must return an m x n matrix")
      }
    } else {
      # Finite differences
      J <- matrix(0, m, n)
      eps <- 1e-8
      for (j in seq_len(n)) {
        x_plus <- x
        x_minus <- x
        x_plus[j] <- x[j] + eps
        x_minus[j] <- x[j] - eps
        r_plus <- residual_fn(x_plus)
        r_minus <- residual_fn(x_minus)
        J[, j] <- (r_plus - r_minus) / (2 * eps)
      }
    }

    # Compute gradient: g = J'r
    g <- as.vector(t(J) %*% r)

    # Check convergence
    if (max(abs(g)) < control$gtol) {
      convergence <- 1
      message <- "Gradient below tolerance"
      converged <- TRUE
      break
    }

    # Compute J'J
    JtJ <- t(J) %*% J

    # Try to find step
    step_found <- FALSE
    for (lm_iter in 1:10) {
      # Add damping: (J'J + λI)
      A <- JtJ + diag(lambda, n)

      # Solve for step
      h <- tryCatch(
        solve(A, -g),
        error = function(e) NULL
      )

      if (is.null(h)) {
        lambda <- lambda * lambda_up
        next
      }

      # Try step
      x_new <- x + h
      r_new <- residual_fn(x_new)
      f_new <- 0.5 * sum(r_new^2)

      if (f_new < f) {
        step_found <- TRUE
        lambda <- lambda * lambda_down
        break
      } else {
        lambda <- lambda * lambda_up
      }
    }

    if (!step_found) {
      convergence <- 3
      message <- "Cannot find step that reduces objective"
      converged <- TRUE
      break
    }

    # Check function convergence
    if (abs(f_new - f) / (abs(f) + 1e-10) < control$ftol) {
      convergence <- 2
      message <- "Function change below tolerance"
      converged <- TRUE
    }

    # Check parameter convergence
    if (max(abs(x_new - x) / (abs(x) + 1e-10)) < control$xtol) {
      convergence <- 4
      message <- "Parameter change below tolerance"
      converged <- TRUE
    }

    x <- x_new
    f <- f_new

    if (converged) break
  }

  if (!converged) {
    convergence <- 0
    message <- "Maximum iterations reached"
  }

  # Final evaluation
  r <- residual_fn(x)
  f <- 0.5 * sum(r^2)

  # Compute final Jacobian
  if (!is.null(jacobian_fn)) {
    J <- jacobian_fn(x)
  } else {
    J <- matrix(0, m, n)
    eps <- 1e-8
    for (j in seq_len(n)) {
      x_plus <- x
      x_minus <- x
      x_plus[j] <- x[j] + eps
      x_minus[j] <- x[j] - eps
      r_plus <- residual_fn(x_plus)
      r_minus <- residual_fn(x_minus)
      J[, j] <- (r_plus - r_minus) / (2 * eps)
    }
  }

  # Compute covariance
  JtJ <- t(J) %*% J
  sigma2 <- if (m > n) sum(r^2) / (m - n) else 0
  vcov <- tryCatch(
    solve(JtJ) * sigma2,
    error = function(e) matrix(NA, n, n)
  )

  list(
    par = x,
    value = f,
    residuals = r,
    jacobian = J,
    vcov = vcov,
    iterations = iter,
    convergence = convergence,
    message = message
  )
}
