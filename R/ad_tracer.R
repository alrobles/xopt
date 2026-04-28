#' Create an AD-traced version of an R objective closure
#'
#' Builds a masked evaluation environment that maps selected base/stats
#' functions to xad-r adjoint implementations when available.
#'
#' @param fn Objective function closure.
#' @return A function with traced environment.
#' @keywords internal
#' @noRd
xopt_ad_trace <- function(fn) {
  if (!is.function(fn)) {
    stop("fn must be a function")
  }
  if (!requireNamespace("xadr", quietly = TRUE)) {
    return(fn)
  }

  mask <- new.env(parent = environment(fn))
  alias <- c(
    sin = "adj_sin", cos = "adj_cos", tan = "adj_tan",
    exp = "adj_exp", log = "adj_log", sqrt = "adj_sqrt",
    asin = "adj_asin", acos = "adj_acos", atan = "adj_atan",
    abs = "adj_abs", gamma = "adj_gamma", lgamma = "adj_lgamma",
    digamma = "adj_digamma", trigamma = "adj_trigamma"
  )

  tryCatch({
    for (nm in names(alias)) {
      adj_nm <- alias[[nm]]
      if (exists(adj_nm, envir = asNamespace("xadr"), inherits = FALSE)) {
        base_fn <- get(nm, envir = baseenv(), inherits = FALSE)
        adj_fn  <- get(adj_nm, envir = asNamespace("xadr"), inherits = FALSE)
        dispatcher <- local({
          base_fn <- base_fn
          adj_fn  <- adj_fn
          function(...) if (is.numeric(..1)) base_fn(...) else adj_fn(...)
        })
        assign(nm, dispatcher, envir = mask)
      }
    }
  }, error = function(e) {
    NULL
  })

  traced <- fn
  environment(traced) <- mask
  traced
}

# Central-difference gradient fallback for plain R objectives.
.xopt_fd_gradient <- function(fn, par, eps = 1e-6) {
  g <- numeric(length(par))
  x_plus <- par
  x_minus <- par
  for (j in seq_along(par)) {
    h <- eps * max(1, abs(par[[j]]))
    x_plus[[j]] <- par[[j]] + h
    x_minus[[j]] <- par[[j]] - h
    g[[j]] <- (fn(x_plus) - fn(x_minus)) / (2 * h)
    x_plus[[j]] <- par[[j]]
    x_minus[[j]] <- par[[j]]
  }
  g
}

#' Automatic gradient dispatch with optional R-side AD tracing
#'
#' @param fn Objective function.
#' @param par Parameter vector where gradient is computed.
#' @param tracer Logical; if TRUE apply function masking prior to AD.
#' @return Numeric gradient vector.
#' @keywords internal
#' @noRd
xopt_auto_gradient <- function(fn, par, tracer = TRUE) {
  target_fn <- if (tracer) xopt_ad_trace(fn) else fn
  if (requireNamespace("xadr", quietly = TRUE) &&
      exists("gradient_adjoint", envir = asNamespace("xadr"), inherits = FALSE)) {
    res <- tryCatch(
      xadr::gradient_adjoint(target_fn, par),
      error = function(e) NULL
    )
    if (!is.null(res) && is.numeric(res)) {
      return(as.numeric(res))
    }
  }
  .xopt_fd_gradient(target_fn, par)
}
