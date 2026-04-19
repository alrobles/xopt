#' Minimize an objective function using xopt
#'
#' Minimize a scalar-valued objective function using the UCMINF quasi-Newton
#' algorithm or a trust-region Newton solver, with explicit control over how
#' the gradient is obtained.
#'
#' @param par Initial parameter vector (numeric).
#' @param fn Objective function to minimize. Should take a numeric vector and
#'   return a scalar.
#' @param gr Optional gradient function. Should take a numeric vector and
#'   return a gradient vector. If supplied, the \code{gradient} argument
#'   defaults to \code{"user"}.
#' @param method Optimization method: \code{"bfgs"} (default) or
#'   \code{"tr_newton"} (trust-region Newton).
#' @param gradient Gradient source; one of \code{"auto"}, \code{"user"},
#'   \code{"traced"}, \code{"fd"}, or \code{"compiled"}. See \strong{Gradient
#'   modes} below. \code{"finite"} is accepted as a backward-compatible alias
#'   of \code{"fd"}.
#' @param control Control parameters (see \code{\link{xopt_control}}).
#'
#' @return A list of class \code{"xopt_result"} with components:
#' \describe{
#'   \item{par}{The optimal parameter vector.}
#'   \item{value}{The objective function value at the optimum.}
#'   \item{gradient}{The gradient at the optimum (if available).}
#'   \item{convergence}{Convergence code (1 = small gradient, 2 = small step,
#'     etc.).}
#'   \item{message}{Convergence message.}
#'   \item{iterations}{Number of function evaluations used.}
#'   \item{gradient_mode}{Which gradient source actually produced the
#'     derivatives used during optimization. See below.}
#' }
#'
#' @section Gradient modes:
#'
#' \code{xopt_minimize} distinguishes four mechanically different ways to
#' obtain a gradient, plus \code{"auto"} which picks among them. The
#' distinction matters because their accuracy, speed, and failure modes are
#' very different, and silently falling off an AD path onto finite
#' differences (as earlier versions did) can mask real bugs.
#'
#' \describe{
#'   \item{\code{"user"}}{Use the supplied \code{gr} directly. Fails if
#'     \code{gr = NULL}. This is the highest-accuracy, highest-speed path
#'     when you already have a hand-written or codegen'd gradient.}
#'   \item{\code{"compiled"}}{Reserved for user-supplied gradients produced
#'     by a C++-level XAD adjoint sweep (e.g., a \code{Problem<Scalar>}
#'     template instantiated with \code{xad::AReal<double>}). The
#'     \code{xopt_minimize} R wrapper does not compile R functions to XAD,
#'     so this mode behaves exactly like \code{"user"} — you must still
#'     supply \code{gr}. The label exists to make the API truthful about
#'     where the derivatives came from.}
#'   \item{\code{"traced"}}{Evaluate \code{fn} inside an R environment where
#'     a whitelisted subset of base-R math (\code{sin}, \code{cos},
#'     \code{exp}, \code{log}, \code{sqrt}, inverse trig, \code{gamma},
#'     \code{lgamma}, \code{digamma}, \code{trigamma}) is masked with the
#'     corresponding \pkg{xadr} adjoint implementations, and compute the
#'     gradient via XAD's reverse sweep. This gives machine-precision
#'     derivatives \emph{only} for objectives whose computation stays
#'     inside that whitelist; anything calling through \code{.Call},
#'     \code{optim}, \code{integrate}, or user C code silently falls
#'     outside the trace. Errors are raised — not silently swallowed —
#'     when \pkg{xadr} is unavailable.}
#'   \item{\code{"fd"}}{Central finite differences, step \eqn{h = 10^{-6}
#'     \max(1, |x|)}. Universal but O(\eqn{p}) evaluations per gradient and
#'     O(\eqn{\sqrt{\epsilon}}) accuracy. Accepts the legacy alias
#'     \code{"finite"}.}
#'   \item{\code{"auto"}}{Try \code{"user"} → \code{"traced"} → \code{"fd"}
#'     in that order. Unlike previous releases, \code{"auto"} emits a
#'     one-time warning when \pkg{xadr}'s adjoint path throws and execution
#'     falls back to finite differences, so users notice when they lose
#'     AD accuracy instead of getting silent numerical gradients.}
#' }
#'
#' @details
#' If \code{gr} is supplied and \code{gradient} is not explicitly set, the
#' default is \code{"user"}. If \code{gr} is \code{NULL} and \code{gradient}
#' is not explicitly set, the default is \code{"auto"}.
#'
#' @examples
#' \dontrun{
#' rosenbrock <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
#' rosenbrock_grad <- function(x) {
#'   c(-400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
#'     200 * (x[2] - x[1]^2))
#' }
#'
#' # Hand-written gradient (fastest, exact)
#' xopt_minimize(c(-1.2, 1), rosenbrock, rosenbrock_grad)
#'
#' # R-side AD tracer (exact for base-R math)
#' xopt_minimize(c(-1.2, 1), rosenbrock, gradient = "traced")
#'
#' # Finite differences (universal fallback)
#' xopt_minimize(c(-1.2, 1), rosenbrock, gradient = "fd")
#'
#' # Auto: tries user → traced → fd, warns on fallback
#' xopt_minimize(c(-1.2, 1), rosenbrock, gradient = "auto")
#' }
#'
#' @export
xopt_minimize <- function(par,
                          fn,
                          gr = NULL,
                          method = c("bfgs", "tr_newton"),
                          gradient = NULL,
                          control = xopt_control()) {
  if (!is.numeric(par)) {
    stop("par must be a numeric vector", call. = FALSE)
  }
  if (!is.function(fn)) {
    stop("fn must be a function", call. = FALSE)
  }
  if (!is.null(gr) && !is.function(gr)) {
    stop("gr must be a function or NULL", call. = FALSE)
  }
  method <- match.arg(method)

  gradient <- .resolve_gradient_mode(gradient, has_gr = !is.null(gr))

  if (!inherits(control, "xopt_control")) {
    control <- do.call(xopt_control, control)
  }

  gr_spec <- .build_gradient(fn, gr, mode = gradient)
  gr_use <- gr_spec$fn
  mode_label <- gr_spec$mode

  if (method == "tr_newton") {
    if (!exists("xopt_tr_newton_cpp", mode = "function")) {
      stop(
        "method = 'tr_newton' requires xopt_tr_newton_cpp(), but the C++ wrapper is not available.",
        call. = FALSE
      )
    }
    res <- xopt_tr_newton_cpp(
      par = par,
      fn = fn,
      gr = gr_use,
      hvp = NULL,
      control = list(
        gtol = control$grtol,
        xtol = control$xtol,
        ftol = control$ftol,
        maxiter = control$maxiter,
        delta_init = control$stepmax
      )
    )
    res$gradient_mode <- mode_label
    return(res)
  }

  if (!requireNamespace("ucminfcpp", quietly = TRUE)) {
    stop("ucminfcpp package is required but not installed", call. = FALSE)
  }

  result <- ucminfcpp::ucminf(
    par, fn, gr_use,
    control = list(
      grtol = control$grtol,
      xtol = control$xtol,
      maxeval = control$maxiter,
      stepmax = control$stepmax
    )
  )

  structure(
    list(
      par = result$par,
      value = result$value,
      gradient = if (!is.null(result$gradient)) result$gradient else numeric(0),
      convergence = result$convergence,
      message = result$message,
      iterations = result$neval,
      gradient_mode = mode_label
    ),
    class = "xopt_result"
  )
}

# Canonicalize and validate the user-facing gradient mode string. Accepts the
# legacy "finite" alias and a NULL that picks a sensible default based on
# whether the user supplied gr.
.resolve_gradient_mode <- function(gradient, has_gr) {
  if (is.null(gradient)) {
    return(if (has_gr) "user" else "auto")
  }
  if (!is.character(gradient) || length(gradient) == 0L) {
    stop("gradient must be a character string", call. = FALSE)
  }
  gradient <- gradient[[1L]]
  if (identical(gradient, "finite")) gradient <- "fd"
  allowed <- c("auto", "user", "compiled", "traced", "fd")
  if (!gradient %in% allowed) {
    stop(sprintf(
      "Invalid gradient mode '%s'. Must be one of: %s (or legacy 'finite').",
      gradient, paste(allowed, collapse = ", ")
    ), call. = FALSE)
  }
  gradient
}

# Resolve a gradient closure and report back the mode label that actually
# provided the derivatives. Modes "user" / "compiled" require gr; "traced"
# requires xadr; "fd" has no dependencies; "auto" picks user > traced > fd.
.build_gradient <- function(fn, gr, mode) {
  if (mode == "user" || mode == "compiled") {
    if (is.null(gr)) {
      stop(sprintf(
        "gradient = '%s' requires a gradient function (gr).", mode
      ), call. = FALSE)
    }
    return(list(fn = gr, mode = mode))
  }

  if (mode == "traced") {
    if (!requireNamespace("xadr", quietly = TRUE)) {
      stop("gradient = 'traced' requires the xadr package (not installed).",
           call. = FALSE)
    }
    traced_fn <- xopt_ad_trace(fn)
    traced_grad <- function(x) {
      res <- xadr::gradient_adjoint(traced_fn, x)
      if (!is.numeric(res)) {
        stop("xadr::gradient_adjoint did not return a numeric gradient.",
             call. = FALSE)
      }
      as.numeric(res)
    }
    return(list(fn = traced_grad, mode = "traced"))
  }

  if (mode == "fd") {
    return(list(fn = function(x) .xopt_fd_gradient(fn, x), mode = "fd"))
  }

  # mode == "auto": user first (already handled above via has_gr), then
  # traced with a one-time fallback warning, then fd.
  if (!is.null(gr)) {
    return(list(fn = gr, mode = "user"))
  }

  xadr_ready <- requireNamespace("xadr", quietly = TRUE) &&
    exists("gradient_adjoint", envir = asNamespace("xadr"), inherits = FALSE)

  if (!xadr_ready) {
    return(list(
      fn = function(x) .xopt_fd_gradient(fn, x),
      mode = "fd"
    ))
  }

  traced_fn <- xopt_ad_trace(fn)
  warned <- FALSE
  auto_grad <- function(x) {
    res <- tryCatch(xadr::gradient_adjoint(traced_fn, x),
                    error = function(e) e)
    if (inherits(res, "error") || !is.numeric(res)) {
      if (!warned) {
        msg <- if (inherits(res, "error")) {
          conditionMessage(res)
        } else {
          "xadr::gradient_adjoint returned a non-numeric result"
        }
        warning(sprintf(
          paste0("gradient = 'auto' falling back to finite differences: ",
                 "xadr adjoint path failed (%s). Pass gradient = 'fd' to ",
                 "silence this warning or gradient = 'user' with a ",
                 "hand-written gr for exact derivatives."),
          msg
        ), call. = FALSE)
        warned <<- TRUE
      }
      return(.xopt_fd_gradient(fn, x))
    }
    as.numeric(res)
  }
  list(fn = auto_grad, mode = "auto")
}

#' @export
print.xopt_result <- function(x, ...) {
  cat("xopt optimization result:\n")
  cat(sprintf("  Convergence: %d - %s\n", x$convergence, x$message))
  cat(sprintf("  Iterations:  %d\n", x$iterations))
  cat(sprintf("  Final value: %.6e\n", x$value))
  if (!is.null(x$gradient_mode)) {
    cat(sprintf("  Grad source: %s\n", x$gradient_mode))
  }
  cat(sprintf("  Parameters:  [%s]\n",
              paste(sprintf("%.6f", x$par), collapse = ", ")))
  invisible(x)
}
