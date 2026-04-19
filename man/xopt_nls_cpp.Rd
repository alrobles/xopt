\name{xopt_nls_cpp}
\alias{xopt_nls_cpp}
\title{Nonlinear least squares via C++}
\usage{
xopt_nls_cpp(par, residual_fn, jacobian_fn = NULL, control)
}
\arguments{
  \item{par}{Numeric vector of initial parameters.}
  \item{residual_fn}{Residual callback returning a numeric vector.}
  \item{jacobian_fn}{Optional Jacobian callback returning an \code{m x n} matrix.}
  \item{control}{Control list (\code{ftol}, \code{xtol}, \code{gtol}, \code{maxiter}, \code{lambda_init}, etc.).}
}
\value{
A list with NLS outputs (\code{par}, \code{value}, \code{residuals}, \code{gradient},
\code{jacobian}, \code{vcov}, \code{iterations}, \code{convergence}, \code{message}).
}
\description{
R wrapper for \code{xopt::solvers::levenberg_marquardt()}.
}
