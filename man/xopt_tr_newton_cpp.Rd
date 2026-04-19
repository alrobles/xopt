\name{xopt_tr_newton_cpp}
\alias{xopt_tr_newton_cpp}
\title{Trust-region Newton solver via C++}
\usage{
xopt_tr_newton_cpp(par, fn, gr = NULL, hvp = NULL, control)
}
\arguments{
  \item{par}{Numeric vector of initial parameters.}
  \item{fn}{Objective function returning a scalar.}
  \item{gr}{Optional gradient function returning a numeric vector.}
  \item{hvp}{Optional Hessian-vector product callback taking \code{(x, v)}.}
  \item{control}{Control list (\code{gtol}, \code{xtol}, \code{ftol}, \code{maxiter}, etc.).}
}
\value{
An object of class \code{"xopt_result"}.
}
\description{
R wrapper for \code{xopt::solvers::trust_region_newton()}.
}
